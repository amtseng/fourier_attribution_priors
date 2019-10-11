import torch
import math
import numpy as np
from model.util import sanitize_sacred_arguments, convolution_size

class ProfileTFBindingPredictor(torch.nn.Module):

    def __init__(
        self, input_length, input_depth, pred_length, num_tasks,
        num_dil_conv_layers, dil_conv_filter_sizes, dil_conv_stride,
        dil_conv_dilations, dil_conv_depths, prof_conv_kernel_size,
        prof_conv_stride
    ):
        """
        Creates a TF binding profile predictor from a DNA sequence.
        Arguments:
            `input_length`: length of the input sequences; each input sequence
                would be D x L, where L is the length
            `input_depth`: depth of the input sequences; each input sequence
                would be D x L, where D is the depth
            `pred_length`: length of the predicted profiles; it must be
                consistent with the convolutional layers specified
            `num_tasks`: number of tasks that are to be predicted; there will be
                two profiles and two read counts predicted for each task
            `num_dil_conv_layers`: number of dilating convolutional layers
            `dil_conv_filter_sizes`: sizes of the initial dilating convolutional
                filters; must have `num_conv_layers` entries
            `dil_conv_stride`: stride used for each dilating convolution
            `dil_conv_dilations`: dilations used for each layer of the dilating
                convolutional layers
            `dil_conv_depths`: depths of the dilating convolutional filters;
                must have `num_conv_layers` entries
            `prof_conv_kernel_size`: size of the large convolutional filter used
                for profile prediction
            `prof_conv_stride`: stride used for the large profile convolution

        Creates a close variant of the BPNet architecture, as described here:
            https://www.biorxiv.org/content/10.1101/737981v1.full
        """
        super().__init__()
        self.creation_args = locals()
        del self.creation_args["self"]
        del self.creation_args["__class__"]
        self.creation_args = sanitize_sacred_arguments(self.creation_args)
        
        assert len(dil_conv_filter_sizes) == num_dil_conv_layers
        assert len(dil_conv_dilations) == num_dil_conv_layers
        assert len(dil_conv_depths) == num_dil_conv_layers

        # Save some parameters
        self.input_depth = input_depth
        self.input_length = input_length
        self.pred_length = pred_length
        self.num_tasks = num_tasks
        self.num_dil_conv_layers = num_dil_conv_layers
        
        # 7 convolutional layers with increasing dilations
        self.dil_convs = torch.nn.ModuleList()
        last_out_size = input_length
        for i in range(num_dil_conv_layers):
            kernel_size = dil_conv_filter_sizes[i]
            in_channels = input_depth if i == 0 else dil_conv_depths[i - 1]
            out_channels = dil_conv_depths[i]
            dilation = dil_conv_dilations[i]
            padding = int(dilation * (kernel_size - 1) / 2)  # "same" padding,
                                                             # for easy adding
            self.dil_convs.append(
                torch.nn.Conv1d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=kernel_size, dilation=dilation, padding=padding
                )
            )
          
            last_out_size = last_out_size - (dilation * (kernel_size - 1))
        # The size of the final dilated convolution output, if there _weren't_
        # any padding (i.e. "valid" padding)
        self.last_dil_conv_size = last_out_size

        # ReLU activation for the convolutional layers
        self.relu = torch.nn.ReLU()

        # Profile prediction:
        # Convolutional layer with large kernel
        # TODO: Transposed convolution?
        # self.prof_first_conv = torch.nn.ConvTranspose1d(
        #     in_channels=64, out_channels=num_tasks, kernel_size=75
        # )
        self.prof_large_conv = torch.nn.Conv1d(
            in_channels=dil_conv_depths[-1], out_channels=(num_tasks * 2),
            kernel_size=prof_conv_kernel_size
        )

        self.prof_pred_size = self.last_dil_conv_size - \
            (prof_conv_kernel_size - 1)

        assert self.prof_pred_size == pred_length, \
            "Prediction length is specified to be %d, but with the given " +\
            "input length of %d and the given convolutions, the computed " +\
            "prediction length is %d" % \
            (pred_length, input_length, self.prof_pred_size)

        # Length-1 convolution over the convolutional output and controls to
        # get the final profile
        self.prof_one_conv = torch.nn.Conv1d(
            in_channels=(num_tasks * 4), out_channels=(num_tasks * 2),
            kernel_size=1, groups=num_tasks  # One set of filters over each task
        )
        
        # Counts prediction:
        # Global average pooling
        self.count_pool = torch.nn.AvgPool1d(
            kernel_size=self.last_dil_conv_size
        )

        # Dense layer to consolidate pooled result to small number of features
        self.count_dense = torch.nn.Linear(
            in_features=dil_conv_depths[-1], out_features=(num_tasks * 2)
        )

        # Dense layer over pooling features and controls to get the final
        # counts, implemented as grouped convolution with kernel size 1
        self.count_one_conv = torch.nn.Conv1d(
            in_channels=(num_tasks * 4), out_channels=(num_tasks * 2),
            kernel_size=1
        )

    def forward(self, input_seqs, cont_profs, cont_counts):
        """
        Computes a forward pass on a batch of sequences.
        Arguments:
            `inputs_seqs`: a B x D x I tensor, where B is the batch size, D is
                the number of channels in the input, and I is the input sequence
                length
            `cont_profs`: a B x T x 2 x O tensor, where T is the number of
                tasks, and O is the output sequence length
            `cont_counts`: a B x T x 2 tensor
        Returns the predicted (normalized) profiles for each task (both plus
        and minus strands) (a B x T x 2 x O tensor), and the predicted log
        counts for each task (both plus and minus strands) (a B x T x 2) tensor.
        """
        batch_size = input_seqs.size(0)
        input_length = input_seqs.size(2)
        assert input_length == self.input_length
        num_tasks = cont_profs.size(1)
        assert num_tasks == cont_counts.size(1)
        assert num_tasks == self.num_tasks
        pred_length = cont_profs.size(3)
        assert pred_length == self.pred_length
        
        # 1. Perform dilated convolutions on the input, each layer's input is
        # the sum of all previous layers' outputs
        dil_conv_out = None
        dil_conv_sum = 0
        for i, dil_conv in enumerate(self.dil_convs):
            if i == 0:
                dil_conv_out = self.relu(dil_conv(input_seqs))
            else:
                dil_conv_out = self.relu(dil_conv(dil_conv_sum))

            if i != self.num_dil_conv_layers - 1:
                dil_conv_sum += dil_conv_out

        # 2. Truncate the final dilated convolutional layer output so that it
        # only has entries that did not see padding; this is equivalent to
        # truncating it to the size it would be if no padding were ever added
        start = int((dil_conv_out.size(2) - self.last_dil_conv_size) / 2)
        end = start + self.last_dil_conv_size
        dil_conv_out_cut = dil_conv_out[:, :, start : end]

        # Branch A: profile prediction
        # A1. Perform convolution with a large kernel
        prof_large_conv_out = self.prof_large_conv(dil_conv_out_cut)
        assert prof_large_conv_out.size(2) == pred_length  # TODO: remove

        # A2. Concatenate with the control profiles
        # Reshaping is necessary to ensure the tasks are paired adjacently
        prof_large_conv_out = prof_large_conv_out.view(
            batch_size, num_tasks, 2, -1
        )
        prof_with_cont = torch.cat([prof_large_conv_out, cont_profs], dim=2)
        prof_with_cont = prof_with_cont.view(batch_size, num_tasks * 4, -1)

        # A3. Perform length-1 convolutions over the concatenated profiles with
        # controls; there are T convolutions, each one is done over one pair of
        # prof_first_conv_out, and a pair of controls
        prof_one_conv_out = self.prof_one_conv(prof_with_cont)
        prof_pred = prof_one_conv_out.view(batch_size, num_tasks, 2, -1)
        
        # Branch B: read count prediction
        # B1. Global average pooling across the output of dilated convolutions
        count_pool_out = self.count_pool(dil_conv_out_cut)
        count_pool_out = torch.squeeze(count_pool_out, dim=2)

        # B2. Reduce pooling output to fewer features, a pair for each task
        count_dense_out = self.count_dense(count_pool_out)

        # B3. Concatenate with the control counts
        # Reshaping is necessary to ensure the tasks are paired adjacently
        count_dense_out = count_dense_out.view(batch_size, num_tasks, 2)
        count_with_cont = torch.cat([count_dense_out, cont_counts], dim=2)
        count_with_cont = count_with_cont.view(batch_size, num_tasks * 4, -1)

        # B4. Dense layer over the concatenation with control counts; each set
        # of counts gets a different dense network (implemented as convolution
        # with kernel size 1)
        count_one_conv_out = self.count_one_conv(count_with_cont)
        count_pred = count_one_conv_out.view(batch_size, num_tasks, 2, -1)
        count_pred = torch.squeeze(count_pred, dim=3)

        return prof_pred, count_pred
