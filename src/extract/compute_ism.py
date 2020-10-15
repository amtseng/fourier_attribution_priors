from model.util import place_tensor
import model.profile_models as profile_models
import model.binary_models as binary_models
import torch
import numpy as np
import tqdm

def compute_ism(input_seq, predict_func, slice_only=None):
    """
    Computes in-silico mutagenesis importance scores for a single input
    sequence.
    Arguments:
        `input_seq`: an I x 4 array of input sequences to explain
        `predict_func`: a function that takes in a set of input sequences,
            N x I x 4, and returns an N-array of output values (usually this is
            a logit, or an aggregate of logits); any batching must be done by
            this function
        `slice_only`: if provided, this is a slice along the input length
            dimension for which the ISM computation is limited to
    Returns an I x 4 array of ISM scores, which consists of the difference
    between the output values for each possible mutation made, and the output
    value of the original sequence.
    """
    input_length, num_bases = input_seq.shape
    if slice_only:
        start, end = slice_only.start, slice_only.stop
        if not start:
            start = 0
        elif start < 0:
            start = start % input_length
        if not end:
            end = input_length
        elif end < 0:
            end = (end % input_length) + 1
    else:
        start, end = 0, input_length

    # A dictionary mapping a base index to all the other base indices; this will
    # be useful when making mutations (e.g. 2 -> [0, 1, 3])
    non_base_indices = {}
    for base_index in range(num_bases):
        non_base_inds = np.arange(num_bases - 1)
        non_base_inds[base_index:] += 1
        non_base_indices[base_index] = non_base_inds

    # Allocate array to hold ISM scores, which are differences from original
    ism_scores = np.zeros_like(input_seq)  # Default 0, actual bases stay 0

    # Allocate array to hold the input sequences to feed in: original, plus
    # all mutations
    num_muts = (num_bases - 1) * (end - start)
    seqs_to_run = np.empty((num_muts + 1, input_length, num_bases))
    seqs_to_run[0] = input_seq  # Fill in original sequence
    i = 1  # Next location to fill in a sequence to `seqs_to_run`
    for pos_index in range(start, end):
        base_index = np.where(input_seq[pos_index])[0][0]
        # There should always be exactly 1 position that's a 1
        for mut_index in non_base_indices[base_index]:
            # For each base index that's not the actual base, make the mutation
            # and put it into `seqs_to_run`
            seqs_to_run[i] = input_seq
            seqs_to_run[i][pos_index][base_index] = 0  # Set actual base to 0
            seqs_to_run[i][pos_index][mut_index] = 1  # Set mutated base to 1
            i += 1
    
    # Make the predictions and get the outputs
    output_vals = predict_func(seqs_to_run)
   
    # Map back the output values to the proper location, and store
    # difference from original
    orig_val = output_vals[0]
    output_diffs = output_vals - orig_val
    i = 1  # Next location to read difference from `output_diffs`
    for pos_index in range(start, end):
        base_index = np.where(input_seq[pos_index])[0][0]
        for mut_index in non_base_indices[base_index]:
            # For each base index that's not the actual base, put the score
            # into the proper location; actual bases stay 0
            ism_scores[pos_index][mut_index] = output_diffs[i]
            i += 1
    
    return ism_scores


def get_profile_model_ism(
    model, input_seqs, cont_profs=None, task_index=None, slice_only=None,
    batch_size=128, show_progress=False
):
    """
    Computes in-silico mutagenesis importance scores for a profile model.
    Arguments:
        `model`: a trained `ProfilePredictorWithMatchedControls`,
            `ProfilePredictorWithSharedControls`, or
            `ProfilePredictorWithoutControls`
        `input_seqs`: an N x I x 4 array of input sequences to explain
        `cont_profs`: any control profiles needed for the model; this can be
            an N x (T or 1) x O x S array, or simply None
        `task_index`: a specific task index (0-indexed) to perform explanations
            from (i.e. explanations will only be from the specified outputs); by
            default explains all tasks
        `slice_only`: if provided, this is a slice along the input length
            dimension for which the ISM computation is limited to
        `batch_size`: batch size for running predictions
        `show_progress`: whether or not to show progress bar over input
            sequences
    Returns an N x I x 4 array of mean-normalized (along base dimension) ISM
    scores, as differences from reference.
    """
    all_ism_scores = np.empty_like(input_seqs)
    num_seqs = len(input_seqs)
    t_iter = tqdm.trange(num_seqs) if show_progress else range(num_seqs)

    for seq_index in t_iter:

        def predict_func(input_seq_batch):
            # Return the set of outputs for the batch of input sequences, all
            # using the same set of control profiles, if needed
            num_in_batch = len(input_seq_batch)
            if cont_profs is not None:
                cont_profs_batch = np.stack(
                    [cont_profs[seq_index]] * num_in_batch, axis=0
                )
            else:
                cont_profs_batch = None  # No controls

            output_vals = np.empty(num_in_batch)
            num_batches = int(np.ceil(num_in_batch / batch_size))
            for i in range(num_batches):
                batch_slice = slice(i * batch_size, (i + 1) * batch_size)
                logit_pred_profs, _ = model(
                    place_tensor(
                        torch.tensor(input_seq_batch[batch_slice])
                    ).float(),
                    place_tensor(
                        torch.tensor(cont_profs_batch[batch_slice])
                    ).float()
                )
                logit_pred_profs = logit_pred_profs.detach().cpu().numpy()
                
                if task_index is None:
                    output_vals[batch_slice] = np.sum(
                        logit_pred_profs, axis=(1, 2, 3)
                    )
                else:
                    output_vals[batch_slice] = np.sum(
                        logit_pred_profs[:, task_index], axis=(1, 2)
                    )

            return output_vals

        # Run ISM for this sequence
        ism_scores = compute_ism(
            input_seqs[seq_index], predict_func, slice_only=slice_only
        )
        all_ism_scores[seq_index] = ism_scores

    # Finally, mean-normalize the ISM scores along the base dimension
    return all_ism_scores - np.mean(all_ism_scores, axis=2, keepdims=True)


def get_binary_model_ism(
    model, input_seqs, task_index=None, slice_only=None, batch_size=128,
    show_progress=False
):
    """
    Computes in-silico mutagenesis importance scores for a binary model.
    Arguments:
        `model`: a trained `BinaryPredictor`
        `input_seqs`: an N x I x 4 array of input sequences to explain
        `task_index`: a specific task index (0-indexed) to perform explanations
            from (i.e. explanations will only be from the specified outputs); by
            default explains all tasks
        `slice_only`: if provided, this is a slice along the input length
            dimension for which the ISM computation is limited to
        `batch_size`: batch size for running predictions
        `show_progress`: whether or not to show progress bar over input
            sequences
    Returns an N x I x 4 array of mean-normalized (along base dimension) ISM
    scores, as differences from reference.
    """
    all_ism_scores = np.empty_like(input_seqs)
    num_seqs = len(input_seqs)
    t_iter = tqdm.trange(num_seqs) if show_progress else range(num_seqs)

    def predict_func(input_seq_batch):
        # Return the set of outputs for the batch of input sequences
        num_in_batch = len(input_seq_batch)
        output_vals = np.empty(num_in_batch)
        num_batches = int(np.ceil(num_in_batch / batch_size))
        for i in range(num_batches):
            batch_slice = slice(i * batch_size, (i + 1) * batch_size)
            logit_preds = model(
                place_tensor(
                    torch.tensor(input_seq_batch[batch_slice])
                ).float(),
            )
            logit_preds = logit_preds.detach().cpu().numpy()

            if task_index is None:
                output_vals[batch_slice] = np.sum(logit_preds, axis=1)
            else:
                output_vals[batch_slice] = logit_preds[:, task_index]

        return output_vals

    for seq_index in t_iter:
        # Run ISM for this sequence
        ism_scores = compute_ism(
            input_seqs[seq_index], predict_func, slice_only=slice_only
        )
        all_ism_scores[seq_index] = ism_scores

    # Finally, mean-normalize the ISM scores along the base dimension
    return all_ism_scores - np.mean(all_ism_scores, axis=2, keepdims=True)


if __name__ == "__main__":
    import extract.data_loading as data_loading
    import model.util as model_util
    import plot.viz_sequence as viz_sequence

    reference_fasta = "/users/amtseng/genomes/hg38.fasta"
    chrom_set = ["chr21"]

    print("Testing profile model")
    input_length = 1346
    profile_length = 1000
    controls = "matched"
    num_tasks = 4
    num_strands = 2
    
    files_spec_path = "/users/amtseng/att_priors/data/processed/ENCODE_TFChIP/profile/config/SPI1/SPI1_training_paths.json"
    model_class = profile_models.ProfilePredictorWithMatchedControls
    model_path = "/users/amtseng/att_priors/models/trained_models/profile/SPI1/1/model_ckpt_epoch_1.pt"

    input_func = data_loading.get_profile_input_func(
        files_spec_path, input_length, profile_length, reference_fasta,
    )
    pos_coords = data_loading.get_positive_profile_coords(
        files_spec_path, chrom_set=chrom_set
    )

    print("Loading model...")
    torch.set_grad_enabled(True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model_util.restore_model(model_class, model_path)
    model.eval()
    model = model.to(device)
    
    print("Computing ISM scores...")
    input_seqs, profiles = input_func(pos_coords[:10])
    hyp_scores = get_profile_model_ism(
        model, input_seqs, cont_profs=profiles[:, num_tasks:],
        show_progress=True
    )
    viz_sequence.plot_weights(hyp_scores[0][650:750], subticks_frequency=100)

    print("")

    print("Testing binary model")
    input_length = 1000
    
    files_spec_path = "/users/amtseng/att_priors/data/processed/ENCODE_TFChIP/binary/config/SPI1/SPI1_training_paths.json"
    model_class = binary_models.BinaryPredictor
    model_path = "/users/amtseng/att_priors/models/trained_models/binary/SPI1/1/model_ckpt_epoch_1.pt"

    input_func = data_loading.get_binary_input_func(
       files_spec_path, input_length, reference_fasta
    )
    pos_bins = data_loading.get_positive_binary_bins(
        files_spec_path, chrom_set=chrom_set
    )

    print("Loading model...")
    torch.set_grad_enabled(True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model_util.restore_model(model_class, model_path)
    model.eval()
    model = model.to(device)
    
    print("Computing importance scores...")
    input_seqs, output_vals, coords = input_func(pos_bins[:10])
    hyp_scores = get_binary_model_ism(
        model, input_seqs, show_progress=True
    )
    viz_sequence.plot_weights(hyp_scores[0][450:550], subticks_frequency=100)
