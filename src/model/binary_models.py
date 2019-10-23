import torch
import math
import numpy as np
from model.util import place_tensor, sanitize_sacred_arguments

class BinaryTFBindingPredictor(torch.nn.Module):

    def __init__(
        self, input_length, input_depth, num_conv_layers, conv_filter_sizes,
        conv_stride, conv_depths, max_pool_size, max_pool_stride, num_fc_layers,
        fc_sizes, num_outputs, batch_norm, conv_drop_rate, fc_drop_rate
    ):
        """
        Creates an binary TF binding site predictor from a DNA sequence.
        Arguments:
            `input_length`: length of the input sequences; each input sequence
                would be D x L, where L is the length
            `input_depth`: depth of the input sequences; each input sequence
                would be D x L, where D is the depth
            `num_conv_layers`: number of convolutional layers to apply
            `conv_filter_sizes`: sizes of the convolutional filters; must have
                `num_conv_layers` entries
            `conv_stride`: stride used for each convolutional kernel
            `conv_depths`: depths of the convolutional filters; must have
                `num_conv_layers` entries
            `max_pool_size`: size of the max pooling layer at the end of the
                convolutions
            `max_pool_stride`: stride of the max pooling operation
            `num_fc_layers`: number of fully connected layers after
                convolution/pooling
            `fc_sizes`: number of nodes in each fully connected layer; must have
                `num_fc` entries
            `num_outputs`: number of tasks the network outputs
            `batch_norm`: whether or not to use batch norm after each
                convolutional and fully connected layer
            `conv_drop_rate`: dropout rate for convolutional layers
            `fc_drop_rate`: dropout rate for fully connected layers
        """
        super().__init__()
        self.creation_args = locals()
        del self.creation_args["self"]
        del self.creation_args["__class__"]
        self.creation_args = sanitize_sacred_arguments(self.creation_args)

        assert len(conv_filter_sizes) == num_conv_layers
        assert len(conv_depths) == num_conv_layers
        assert len(fc_sizes) == num_fc_layers

        # Save some parameters
        self.input_depth = input_depth
        self.input_length = input_length
        input_size = (input_depth, input_length)
        self.num_conv_layers = num_conv_layers
        self.num_fc_layers = num_fc_layers
        self.batch_norm = batch_norm

        # Define the convolutional layers
        depths = [input_depth] + conv_depths
        self.conv_layers = torch.nn.ModuleList()
        for i in range(num_conv_layers):
            self.conv_layers.append(
                torch.nn.Conv1d(
                    in_channels=depths[i],
                    out_channels=depths[i + 1],
                    kernel_size=conv_filter_sizes[i],
                    stride=conv_stride,
                    padding=0  # No padding (AKA "valid")
                )
            )
            self.conv_layers.append(torch.nn.ReLU())
            if batch_norm:
                self.conv_layers.append(
                    torch.nn.BatchNorm1d(depths[i + 1])
                )
            self.conv_layers.append(torch.nn.Dropout(conv_drop_rate))

        # Compute sizes of the convolutional outputs
        conv_output_sizes = []
        last_size = input_size
        for i in range(num_conv_layers):
            next_length = math.floor(
                (last_size[1] - (conv_filter_sizes[i] - 1) - 1) / \
                conv_stride
            ) + 1
            next_size = (conv_depths[i], next_length)
            conv_output_sizes.append(next_size)
            last_size = next_size

        # Define the max pooling layer
        self.max_pool_layer = torch.nn.MaxPool1d(
            kernel_size=max_pool_size,
            stride=max_pool_stride,
            padding=0  # No padding (AKA "valid")
        )

        # Compute size of the pooling output
        pool_output_depth = conv_output_sizes[-1][0]
        pool_output_length = math.floor(
            (conv_output_sizes[-1][1] - (max_pool_size - 1) - 1) / \
            max_pool_stride
        ) + 1
        pool_output_size = (pool_output_depth, pool_output_length)
        
        # Define the fully connected layers
        dims = [pool_output_size[0] * pool_output_size[1]] + fc_sizes
        self.fc_layers = torch.nn.ModuleList()
        for i in range(num_fc_layers):
            self.fc_layers.append(
                torch.nn.Linear(dims[i], dims[i + 1])
            )
            self.fc_layers.append(torch.nn.ReLU())
            if batch_norm:
                self.fc_layers.append(
                    torch.nn.BatchNorm1d(dims[i + 1])
                )
            self.fc_layers.append(torch.nn.Dropout(fc_drop_rate))

        # Map last fully connected layer to final outputs
        self.out_map_fc = torch.nn.Linear(fc_sizes[-1], num_outputs)

        self.sigmoid = torch.nn.Sigmoid()

        self.bce_loss = torch.nn.BCELoss()


    def forward(self, input_seqs):
        """
        Computes a forward pass on a batch of sequences.
        Arguments:
            `inputs_seqs`: a B x D x L tensor, where B is the batch size, D is
                the number of channels in the input, and L is the sequence
                length
        Returns the probabilities of each output (i.e. output values passed
        through a sigmoid). Note that the probabilities are returned in the
        order according to the input sequences.
        """
        batch_size = input_seqs.size(0)
        input_depth = input_seqs.size(1)
        input_length = input_seqs.size(2)

        # Run through convolutions, activations, and batch norm
        x = input_seqs
        for layer in self.conv_layers:
            x = layer(x)
        conv_output = x

        # Perform max pooling
        pooled = self.max_pool_layer(conv_output)

        # Flatten
        flattened = pooled.view(batch_size, -1)

        # Run through fully connected layers, activations, and batch norm
        x = flattened
        for layer in self.fc_layers:
            x = layer(x)
        fc_output = x

        # Run through last layer to get logits
        logits = self.out_map_fc(fc_output)

        probs = self.sigmoid(logits)
        return probs


    def correctness_loss(self, true_vals, probs, average_classes=False):
        """
        Computes the binary cross-entropy loss.
        Arguments:
            `true_seqs`: a B x C tensor, where B is the batch size and C is the
                number of output tasks, containing the true binary values
            `probs`: a B x C tensor containing the predicted probabilities
            `average_classes`: if True, compute the loss separately for the
                positives and the negatives, and return their average; this
                weights the losses between imbalanced classes more evenly
        Returns a tensor scalar that is the loss for the batch.
        """
        assert true_vals.size() == probs.size()
        true_vals_flat = torch.flatten(true_vals)
        probs_flat = torch.flatten(probs)

        if average_classes:
            pos_mask = true_vals_flat == 1
            neg_mask = true_vals_flat == 0

            pos_true, pos_probs = true_vals_flat[pos_mask], probs_flat[pos_mask]
            neg_true, neg_probs = true_vals_flat[neg_mask], probs_flat[neg_mask]

            if not pos_true.nelement():
                return self.bce_loss(neg_probs, neg_true)
            elif not neg_true.nelement():
                return self.bce_loss(pos_probs, pos_true)
            else:
                pos_loss = self.bce_loss(pos_probs, pos_true)
                neg_loss = self.bce_loss(neg_probs, neg_true)
                return (pos_loss + neg_loss) / 2
        else:
            # Ignore anything that's not 0 or 1
            mask = (true_vals_flat == 0) | (true_vals_flat == 1)
            true_vals_flat = true_vals_flat[mask]
            probs_flat = probs_flat[mask]

            return self.bce_loss(probs_flat, true_vals_flat)


    def att_prior_loss(self, true_vals, input_grads, pos_limit, pos_weight):
        """
        Computes an attribution prior loss for some given training examples.
        Arguments:
            `true_vals`: a B x C tensor, where B is the batch size and C is the
                number of output tasks, containing the true binary values
            `input_grads`: a B x C x L x D tensor, where B is the batch size, C
                is the number of output tasks, L is the length of the input, and
                D is the dimensionality of each input base; this needs to be the
                gradients of the input with respect to each output task
            `pos_limit`: the maximum integer frequency index, k, to consider for
                the positive loss; this corresponds to a frequency cut-off of
                pi * k / L; k should be less than L / 2
            `pos_weight`: the amount to weight the positive loss by, to give it
                a similar scale as the negative loss
        Returns a single scalar Tensor consisting of the positive and negative
        losses together.
        """
        relu = torch.nn.ReLU()
        max_rect_grads = torch.max(relu(input_grads), dim=3)[0]

        neg_grads = max_rect_grads[true_vals == 0]
        pos_grads = max_rect_grads[true_vals == 1]

        # Loss for positives
        if pos_grads.nelement():
            pos_grads_comp = torch.stack(
                [pos_grads, place_tensor(torch.zeros(pos_grads.size()))], dim=2
            )
            # Magnitude of the Fourier coefficients, normalized:
            pos_fft = torch.fft(pos_grads_comp, 1)
            pos_mags = torch.norm(pos_fft, dim=2)
            pos_mags = pos_mags / torch.sum(pos_mags, dim=1, keepdim=True)
            # Cut off DC and high-frequency components:
            pos_mags = pos_mags[:, 1:pos_limit]
            pos_score = torch.sum(pos_mags, dim=1)
            pos_loss = 1 - pos_score
            pos_loss_mean = torch.mean(pos_loss)
        else:
            pos_loss_mean = 0
        # Loss for negatives
        if neg_grads.nelement():
            neg_loss = torch.sum(neg_grads, dim=1)
            neg_loss_mean = torch.mean(neg_loss)
        else:
            neg_loss_mean = 0

        return (pos_weight * pos_loss_mean) + neg_loss_mean
