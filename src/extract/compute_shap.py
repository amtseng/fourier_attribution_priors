from model.util import place_tensor
import model.profile_models as profile_models
import model.binary_models as binary_models
from extract.dinuc_shuffle import dinuc_shuffle
import shap
import torch
import numpy as np
import os
import sys

DEVNULL = open(os.devnull, "w")
STDOUT = sys.stdout

def hide_stdout():
    sys.stdout = DEVNULL
def show_stdout():
    sys.stdout = STDOUT


def create_input_seq_background(
    input_seq, input_length, bg_size=10, seed=20200219
):
    """
    From the input sequence to a model, generates a set of background
    sequences to perform interpretation against.
    Arguments:
        `input_seq`: I x 4 tensor of one-hot encoded input sequence, or None
        `input_length`: length of input, I
        `bg_size`: the number of background examples to generate, G
    Returns a G x I x 4 tensor containing randomly dinucleotide-shuffles of the
    original input sequence. If `input_seq` is None, then a G x I x 4 tensor of
    all 0s is returned.
    """
    if input_seq is None:
        input_seq_bg_shape = (bg_size, input_length, 4)
        return place_tensor(torch.zeros(input_seq_bg_shape)).float()

    # Do dinucleotide shuffles
    input_seq_np = input_seq.cpu().numpy()
    rng = np.random.RandomState(seed)
    input_seq_bg_np = dinuc_shuffle(input_seq_np, bg_size, rng=rng)
    return place_tensor(torch.tensor(input_seq_bg_np)).float()


def create_profile_control_background(
    control_profs, profile_length, num_tasks, num_strands, controls="matched",
    bg_size=10
):
    """
    Generates a background for a set of profile controls. In general, this is
    the given control profiles, copied a number of times (i.e. the background
    for controls should always be the same). Note this is only used for profile
    models.
    Arguments:
        `control_profs`: (T or 1) x O x S tensor of control profiles,
            or None
        `profile_length`: length of profile, O
        `num_tasks`: number of tasks, T
        `num_strands`: number of strands, S
        `controls`: the kind of controls used: "matched" or "shared"; if
            "matched", the control profiles taken in and returned are
            T x O x S; if "shared", the profiles are 1 x O x S
        `bg_size`: the number of background examples to generate, G
    Returns the tensor of `control_profs`, replicated G times. If `controls` is
    "matched", this becomes a G x T x O x S tensor; if `controls` is "shared",
    this is a G x 1 x O x S tensor. If `control_profs` is None, then a tensor of
    all 0s is returned, whose shape is determined by `controls`.
    """
    assert controls in ("matched", "shared")

    if controls == "matched":
        control_profs_bg_shape = (
            bg_size, num_tasks, profile_length, num_strands
        )
    else:
        control_profs_bg_shape = (bg_size, 1, profile_length, num_strands)
    if control_profs is None:
        return place_tensor(torch.zeros(control_profs_bg_shape)).float()

    # Replicate `control_profs`
    return torch.stack([control_profs] * bg_size, dim=0)


def combine_input_seq_mult_and_diffref(mult, orig_inp, bg_data):
    """
    Computes the hypothetical contribution of any base along the input sequence
    to the final output, given the multipliers for the input sequence
    background. This will simulate all possible base identities as compute a
    "difference-from-reference" for each possible base, averaging the product
    of the multipliers with the differences, over the base identities.
    Arguments:
        `mult`: a G x I x 4 array containing multipliers for the background
            input sequences
        `orig_inp`: the target input sequence to compute contributions for, an
            I x 4 array
        `bg_data`: a G x I x 4 array containing the actual background sequences
    Returns the hypothetical importance scores in an I x 4 array.
    This function is necessary for this specific implementation of DeepSHAP. In
    the original DeepSHAP, the final step is to take the difference of the input
    sequence to each background sequence, and weight this difference by the
    contribution multipliers for the background sequence. However, all
    differences to the background would be only for the given input sequence
    (i.e. the actual importance scores). To get the hypothetical importance
    scores efficiently, we try every possible base for the input sequence, and
    for each one, compute the difference-from-reference and weight by the
    multipliers separately. This allows us to compute the hypothetical scores
    in just one pass, instead of running DeepSHAP many times. To get the actual
    scores for the original input, simply extract the entries for the bases in
    the real input sequence.
    """
    # Reassign arguments to better names; this specific implementation of
    # DeepSHAP requires the arguments to have the above names
    bg_mults, input_seq, bg_seqs = mult, orig_inp, bg_data

    # Allocate array to store hypothetical scores, one set for each background
    # reference (i.e. each difference-from-reference)
    input_seq_hyp_scores_eachdiff = np.empty_like(bg_seqs)
    
    # Loop over the 4 input bases
    for i in range(input_seq.shape[-1]):
        # Create hypothetical input of all one type of base
        hyp_input_seq = np.zeros_like(input_seq)
        hyp_input_seq[:, i] = 1

        # Compute difference from reference for each reference
        diff_from_ref = np.expand_dims(hyp_input_seq, axis=0) - bg_seqs
        # Shape: G x I x 4

        # Weight difference-from-reference by multipliers
        contrib = diff_from_ref * bg_mults

        # Sum across bases axis; this computes the actual importance score AS IF
        # the target sequence were all that base
        input_seq_hyp_scores_eachdiff[:, :, i] = np.sum(contrib, axis=-1)

    # Average hypothetical scores across background
    # references/diff-from-references
    return np.mean(input_seq_hyp_scores_eachdiff, axis=0)


class WrapperProfileModel(torch.nn.Module):
    def __init__(self, inner_model, task_index=None):
        """
        Takes a profile model and constructs wrapper model around it. This model
        takes in the same inputs (i.e. input tensor of shape B x I x 4 and
        perhaps a set of control profiles of shape B x (T or 1) x O x S). The
        model will return an output of B x 1, which is the profile logits
        (weighted), aggregated to a scalar for each input.
        Arguments:
            `inner_model`: a trained `ProfilePredictorWithMatchedControls`,
                `ProfilePredictorWithSharedControls`, or
                `ProfilePredictorWithoutControls`
            `task_index`: a specific task index (0-indexed) to perform
                explanations from (i.e. explanations will only be from the
                specified outputs); by default explains all tasks in aggregate
        """
        super().__init__()
        self.inner_model = inner_model
        self.task_index = task_index
        
    def forward(self, input_seqs, cont_profs=None):
        # Run through inner model, disregarding the predicted counts
        logit_pred_profs, _ = self.inner_model(input_seqs, cont_profs)
       
        # As with the computation of the gradients, instead of explaining the
        # logits, explain the mean-normalized logits, weighted by the final
        # probabilities after passing through the softmax; this exponentially
        # increases the weight for high-probability positions, and exponentially
        # reduces the weight for low-probability positions, resulting in a
        # cleaner signal

        # Subtract mean along output profile dimension; this wouldn't change
        # softmax probabilities, but normalizes the magnitude of the logits
        norm_logit_pred_profs = logit_pred_profs - \
            torch.mean(logit_pred_profs, dim=2, keepdim=True)

        # Weight by post-softmax probabilities, but detach it from the graph to
        # avoid explaining those
        pred_prof_probs = profile_models.profile_logits_to_log_probs(
            logit_pred_profs
        ).detach()
        weighted_norm_logits = norm_logit_pred_profs * pred_prof_probs

        if self.task_index is not None:
            # Subset to specific task
            weighted_norm_logits = \
                weighted_norm_logits[:, self.task_index : (self.task_index + 1)]
        prof_sum = torch.sum(weighted_norm_logits, dim=(1, 2, 3))

        # DeepSHAP requires the shape to be B x 1
        return torch.unsqueeze(prof_sum, dim=1)


class WrapperBinaryModel(torch.nn.Module):
    def __init__(self, inner_model, task_index=None):
        """
        Takes a binary model and constructs wrapper model around it. This model
        takes in the same inputs (i.e. an input tensor of shape B x I x 4). The
        model will return an output of B x 1, which is the prediction logits
        aggregated to a scalar for each input.
        Arguments:
            `inner_model`: a trained `BinaryPredictor`
            `task_index`: a specific task index (0-indexed) to perform
                explanations from (i.e. explanations will only be from the
                specified outputs); by default explains all tasks in aggregate
        """
        super().__init__()
        self.inner_model = inner_model
        self.task_index = task_index
        
    def forward(self, input_seqs):
        pred_logits = self.inner_model(input_seqs)
        
        if self.task_index is not None:
            # Subset to specific task
            pred_logits = \
                pred_logits[:, self.task_index : (self.task_index + 1)]
        # DeepSHAP requires the shape to be B x 1
        return torch.sum(pred_logits, dim=1, keepdim=True)


def create_profile_explainer(
    model, input_length, profile_length, num_tasks, num_strands, controls,
    task_index=None, bg_size=10, seed=20200219
):
    """
    Given a trained `ProfilePredictor` model, creates a Shap DeepExplainer that
    returns hypothetical scores for given input sequences.
    Arguments:
        `model`: a trained `ProfilePredictorWithMatchedControls`,
            `ProfilePredictorWithSharedControls`, or
            `ProfilePredictorWithoutControls`
        `input_length`: length of input sequence, I
        `profile_length`: length of output profiles, O
        `num_tasks`: number of tasks in model, T
        `num_strands`: number of strands in model, T
        `controls`: the kind of controls used: "matched", "shared", or None;
            if "matched", the control profiles taken in and returned are
            T x O x S; if "shared", the profiles are 1 x O x S; if None, no
            controls need to be provided
        `task_index`: a specific task index (0-indexed) to perform explanations
            from (i.e. explanations will only be from the specified outputs); by
            default explains all tasks
        `bg_size`: the number of background examples to generate
    Returns a function that takes in input sequences (B x I x 4 array) and
    control profiles (B x (T or 1) x O x S array, or nothing at all), and
    outputs hypothetical scores for the input sequences (B x I x 4 array).
    """
    wrapper_model = WrapperProfileModel(model, task_index=task_index)

    def bg_func(model_inputs):
        """
        Given a pair of inputs to the wrapper model, returns the backgrounds
        for the model.
        Arguments:
            `model_inputs`: a list of either the input sequence (I x 4 tensor)
                alone, or a pair of the input sequence and control profiles 
                ((T or 1) x O x S tensor), depending on `controls`
        If `controls` is None, the `model_inputs` must be just the input
        sequence, and this returns a list of just the input sequence background
        (G x I x 4 tensor). Otherwise, `model_inputs` must be the pair of the
        input sequence and control profiles, and this function will return a
        pair of tensors: the G x I x 4 tensor for the background input
        sequences, and a G x (T or 1) x O x S tensor of background control
        profiles.
        Note that in the PyTorch implementation of DeepSHAP, `model_inputs` may
        be None, in which case this function still needs to return tensors, but
        of the right shapes.
        """
        if controls is None:
            if model_inputs is None:
                input_seq = None
            else:
                input_seq = model_inputs[0]
            return [create_input_seq_background(
                input_seq, input_length, bg_size=bg_size, seed=seed
            )]
        else:
            if model_inputs is None:
                input_seq, control_profs = None, None
            else:
                input_seq, control_profs = model_inputs
            return [
                create_input_seq_background(
                    input_seq, input_length, bg_size=bg_size, seed=seed
                ),
                create_profile_control_background(
                    control_profs, profile_length, num_tasks, num_strands,
                    controls=controls, bg_size=bg_size
                )
            ]

    def combine_mult_and_diffref_func(mult, orig_inp, bg_data):
        """
        Computes the hypothetical contribution of any base along the inputs
        to the final output. This is a wrapper function around
        `combine_input_seq_mult_and_diffref` (further information can be found
        there). Note that the arguments must be named as such in this
        implementation of DeepSHAP.
        Arguments:
            `mult`: the multipliers for the background data; if `controls` is
                None, then this a list of just the G x I x 4 array of input
                sequence backgrounds; otherwise, this is a pair of input
                sequence and control profile backgrounds (a G x (T or 1) x O x S
                array)
            `orig_inp`: the original target inputs; if `controls` is None, then
                this is a list of just the I x 4 array of input sequence;
                otherwise, it is a pair of the input sequence and a
                (T or 1) x O x S array of control profiles
            `bg_data`: the backgrounds themselves; if `controls` is None, then
                this is a list of just the G x I x 4 array of input sequence
                backgrounds; otherwise, it is a pair of the input sequence and
                control profile backgrounds (the latter is a
                G x (T or 1) x O x S) array
        If `controls` is None, returns a list of just the hypothetical
        importance scores of the input sequence; if `controls` is not None (to
        be consistent with other profile model explaining functions), this will
        return a pair of the input sequence scores, and an array of all zeros,
        of shape (T or 1) x O x S.
        """
        if controls is None:
            input_seq_bg_mult = mult[0]
            input_seq = orig_inp[0]
            input_seq_bg = bg_data[0]
        else:
            input_seq_bg_mult, cont_profs_bg_mult = mult
            input_seq, cont_profs = orig_inp
            input_seq_bg, cont_profs_bg = bg_data
        
        input_seq_scores = combine_input_seq_mult_and_diffref(
            input_seq_bg_mult, input_seq, input_seq_bg
        )
        if controls is None:
            return [input_seq_scores]
        else:
            return [input_seq_scores, np.zeros_like(cont_profs)]

    explainer = shap.DeepExplainer(
        model=wrapper_model,
        data=bg_func,
        combine_mult_and_diffref=combine_mult_and_diffref_func
    )

    def explain_fn(
        input_seqs, cont_profs=None, batch_size=128, hide_shap_output=False
    ):
        """
        Given input sequences and control profiles, returns hypothetical scores
        for the input sequences.
        Arguments:
            `input_seqs`: a B x I x 4 array
            `cont_profs`: a B x (T or 1) x O x S array, or None
            `batch_size`: batch size for computation
            `hide_shap_output`: if True, do not show any warnings from DeepSHAP
        Returns a B x I x 4 array containing hypothetical importance scores for
        each of the B input sequences.
        """
        scores = np.empty_like(input_seqs)
        input_seqs_t = place_tensor(torch.tensor(input_seqs)).float()
        try:
            if hide_shap_output:
                hide_stdout()
            if controls is None:
                return explainer.shap_values(
                    [input_seqs_t], progress_message=None
                )[0]
            else:
                cont_profs_t = place_tensor(torch.tensor(cont_profs)).float()
                return explainer.shap_values(
                    [input_seqs_t, cont_profs_t], progress_message=None
                )[0]
        except Exception as e:
            raise e
        finally:
            show_stdout()

    return explain_fn


def create_binary_explainer(
    model, input_length, task_index=None, bg_size=10, seed=20200219
):
    """
    Given a trained `BinaryPredictor` model, creates a Shap DeepExplainer that
    returns hypothetical scores for given input sequences.
    Arguments:
        `model`: a trained `BinaryPredictor`
        `input_length`: length of input sequence, I
        `task_index`: a specific task index (0-indexed) to perform explanations
            from (i.e. explanations will only be from the specified outputs); by
            default explains all tasks
        `bg_size`: the number of background examples to generate
    Returns a function that takes in input sequences (B x I x 4 array) and
    outputs hypothetical scores for the input sequences (B x I x 4 array).
    """
    wrapper_model = WrapperBinaryModel(model, task_index=task_index)

    def bg_func(model_inputs):
        """
        Given the inputs to the wrapper model, returns the backgrounds for the
        model.
        Arguments:
            `model_inputs`: a list of just the input sequence (I x 4 tensor)
                alone
        Returns a list of just the input sequence background (G x I x 4 tensor).
        Note that in the PyTorch implementation of DeepSHAP, `model_inputs` may
        be None, in which case this function still needs to return a tensor, but
        of the right shape.
        """
        if model_inputs is None:
            input_seq = None
        else:
            input_seq = model_inputs[0]
        return [create_input_seq_background(
            input_seq, input_length, bg_size=bg_size, seed=seed
        )]

    def combine_mult_and_diffref_func(mult, orig_inp, bg_data):
        """
        Computes the hypothetical contribution of any base along the inputs
        to the final output. This is a wrapper function around
        `combine_input_seq_mult_and_diffref` (further information can be found
        there). Note that the arguments must be named as such in this
        implementation of DeepSHAP.
        Arguments:
            `mult`: the multipliers for the background data; this is a list of 
                just the G x I x 4 array of input sequence backgrounds
            `orig_inp`: the original target inputs; this is a list of just the 
                I x 4 array for the input sequence
            `bg_data`: the backgrounds themselves; this is a list of just the
                G x I x 4 array of input sequence backgrounds
        Returns a list of just the hypothetical importance scores of the input
        sequence.
        """
        input_seq_bg_mult = mult[0]
        input_seq = orig_inp[0]
        input_seq_bg = bg_data[0]
        return [combine_input_seq_mult_and_diffref(
            input_seq_bg_mult, input_seq, input_seq_bg
        )]

    explainer = shap.DeepExplainer(
        model=wrapper_model,
        data=bg_func,
        combine_mult_and_diffref=combine_mult_and_diffref_func
    )

    def explain_fn(input_seqs, hide_shap_output):
        """
        Given input sequences, returns hypothetical.
        Arguments:
            `input_seqs`: a B x I x 4 array
            `hide_shap_output`: if True, do not show any warnings from DeepSHAP
        Returns a B x I x 4 array containing hypothetical importance scores for
        each of the B input sequences.
        """
        input_seqs_t = place_tensor(torch.tensor(input_seqs)).float()
        try:
            if hide_shap_output:
                hide_stdout()
            return explainer.shap_values(
                [input_seqs_t], progress_message=None
            )[0]
        except Exception as e:
            raise e
        finally:
            show_stdout()

    return explain_fn


if __name__ == "__main__":
    import extract.data_loading as data_loading
    import model.util as model_util
    from deeplift.visualization import viz_sequence

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
    
    print("Creating explainer...")
    explainer = create_profile_explainer(
        model, input_length, profile_length, num_tasks, num_strands, controls
    )
    print("Computing importance scores...")
    input_seqs, profiles = input_func(pos_coords[:10])
    hyp_scores = explainer(
        input_seqs, profiles[:, num_tasks:], hide_shap_output=True
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
    
    print("Creating explainer...")
    explainer = create_binary_explainer(model, input_length)

    print("Computing importance scores...")
    input_seqs, output_vals, coords = input_func(pos_bins[:10])
    hyp_scores = explainer(input_seqs, hide_shap_output=True)
    viz_sequence.plot_weights(hyp_scores[0][450:550], subticks_frequency=100)
