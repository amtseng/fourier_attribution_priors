from model.util import place_tensor, restore_model
from deeplift.dinuc_shuffle import dinuc_shuffle
import shap
import torch
import numpy as np

def create_background(
    model_inputs, input_length, profile_length, num_tasks, use_controls=True,
    bg_size=10, seed=20191206
):
    """
    From a pair of single inputs to the model, generates the set of background
    inputs to perform interpretation against.
    Arguments:
        `model_inputs`: a pair of two tensors ; the first is a single one-hot
            encoded input sequence of shape I x 4; the second is the set of
            control profiles for the model, shaped T x O x 2
        `input_length`: length of input, I
        `profile_length`: length of profiles, O
        `num_tasks`: number of tasks, T
        `use_controls`: if False, does not expect or return controls (the inputs
            and outputs will be singleton lists)
        `bg_size`: the number of background examples to generate.
    Returns a pair of tensors as a list, where the first tensor is G x I x 4,
    and the second tensor is G x T x O x 2; these are the background inputs. The
    background for the input sequences is randomly dinuceotide-shuffles of the
    original sequence. The background for the control profiles is the same as
    the originals. If the control profiles are empty (i.e. because the model
    does not take in control profiles), then empty profiles are also returned.
    """
    input_seq_bg_shape = (bg_size, input_length, 4)
    cont_prof_bg_shape = (bg_size, num_tasks, profile_length, 2)
    if model_inputs is None:
        # For DeepSHAP PyTorch, the model inputs could be None, but something
        # of the right shapes need to be returned
        if use_controls:
            return [
                place_tensor(torch.zeros(input_seq_bg_shape)).float(),
                place_tensor(torch.zeros(cont_prof_bg_shape)).float()
            ]
        else:
            return [
                place_tensor(torch.zeros(input_seq_bg_shape)).float()
            ]
    else:
        if use_controls:
            input_seq, cont_profs = model_inputs
            cont_profs_np = cont_profs.cpu().numpy()
            cont_prof_bg = np.empty(cont_prof_bg_shape)
        else:
            input_seq = model_inputs[0]
        input_seq_np = input_seq.cpu().numpy()
        input_seq_bg = np.empty(input_seq_bg_shape)
        rng = np.random.RandomState(seed)
        for i in range(bg_size):
            input_seq_shuf = dinuc_shuffle(input_seq_np, rng=rng)
            input_seq_bg[i] = input_seq_shuf
            if use_controls:
                cont_prof_bg[i] = cont_profs_np
        if use_controls:
            return [
                place_tensor(torch.tensor(input_seq_bg)).float(),
                place_tensor(torch.tensor(cont_prof_bg)).float()
            ]
        else:
            return [
                place_tensor(torch.tensor(input_seq_bg)).float()
            ]


def combine_mult_and_diffref(mult, orig_inp, bg_data, use_controls=True):
    """
    Computes the hypothetical contribution of any base in the input to the
    output, given the multipliers for the background data. This will simulate
    all possible base identities and compute a separate "difference-from-
    reference" for each, averaging the product of the multipliers with these
    differences, over the base identities. For the control profiles, the
    returned contribution is 0.
    Arguments:
        `mult`: multipliers for the background data; a pair of a G x I x 4 array
            and a G x T x O x 2 array
        `orig_inp`: the target inputs to compute contributions for; a pair of an
            I x 4 array and a T x O x 2 array
        `bg_data`: the background data; a pair of a G x I x 4 array and a
            G x T x O x 2 array
        `use_controls`: if False, expects singleton lists for the input and
            returns a singleton list
    Returns a pair of importance scores as a list: an I x 4 array and a
    T x O x 2 zero-array.
    Note that this rule is necessary because by default, the multipliers are
    multiplied by the difference-from-reference (for each reference in the
    background set). However, using the actual sequence as the target would not
    allow getting hypothetical scores, as the resulting attributions use a
    difference-from-reference wherever the target does _not_ have that base.
    Thus, we compute the hypothetical scores manually by trying every possible
    base as the target (instead of using the actual original target input).
    To back-out the actual scores for the original target input, simply extract
    the entries for the bases in the real input.
    """
    if use_controls:
        input_mult, cont_profs_mult = mult
        input_seq, cont_profs = orig_inp
        input_seq_bg, cont_profs_bg = bg_data
        cont_profs_hyp_scores = np.zeros_like(cont_profs)
    else:
        input_mult = mult[0]
        input_seq = orig_inp[0]
        input_seq_bg = bg_data[0]

    # Allocate array to store hypothetical scores, one set for each background
    # reference
    input_seq_hyp_scores_eachref = np.empty_like(input_seq_bg)
    
    # Loop over input bases
    for i in range(input_seq.shape[-1]):
        # Create hypothetical input of all one type of base
        hyp_input_seq = np.zeros_like(input_seq)
        hyp_input_seq[:, i] = 1

        # Compute difference from reference for each reference
        diff_from_ref = np.expand_dims(hyp_input_seq, axis=0) - input_seq_bg
        # Shape: G x I x 4

        # Weight difference-from-reference by multipliers
        contrib = diff_from_ref * input_mult

        # Sum across bases axis; this computes the hypothetical score AS IF the
        # the target sequence were all that base
        input_seq_hyp_scores_eachref[:, :, i] = np.sum(contrib, axis=-1)

    # Average hypothetical scores across background references
    input_seq_hyp_scores = np.mean(input_seq_hyp_scores_eachref, axis=0)

    if use_controls:
        return [input_seq_hyp_scores, cont_profs_hyp_scores]
    else:
        return [input_seq_hyp_scores]


class WrapperModel(torch.nn.Module):
    def __init__(self, inner_model, task_index, output_type):
        """
        Takes a profile model and constructs wrapper model around it. This model
        takes in the same inputs (i.e. input tensor of shape B x I x 4 and
        control profile of shape B x T x O x 2). The model will return an
        output of shape B x 1, which can be either the profiles or the counts,
        aggregated to this shape.
        Arguments:
            `inner_model`: an instantiated or loaded model from
                `profile_model.profile_tf_binding_predictor`
            `task_index`: a specific task index (0-indexed) to perform
                explanations from (i.e. explanations will only be from the
                specified outputs); by default explains all tasks
            `output_type`: if "profile", utilizes the profile output to compute
                the importance scores; if "count", utilizes the counts output;
                in either case, the importance scores are for the ouputs summed
                across strand and tasks.
        """
        super().__init__()
        self.inner_model = inner_model
        assert output_type in ("profile", "count")
        self.output_type = output_type
        self.task_index = task_index
        
    def forward(self, input_seqs, cont_profs=None):
        prof_output, count_output = self.inner_model(input_seqs, cont_profs)
        
        if self.output_type == "profile":
            # As a slight optimization, instead of explaining the logits,
            # explain the logits weighted by the probabilities after passing
            # through the softmax; this exponentially increases the weight for
            # high-probability positions, and exponentially reduces the weight
            # for low-probability positions, resulting in a more cleaner signal

            # First, center/mean-normalize the logits so the contributions are
            # normalized, as a softmax would do
            logits = prof_output - torch.mean(prof_output, dim=2, keepdim=True)

            # Compute softmax probabilities, but detach it from the graph to
            # avoid explaining those
            probs = torch.nn.Softmax(dim=2)(logits).detach()

            logits_weighted = logits * probs  # Shape: B x T x O x 2
            if self.task_index:
                logits_weighted = \
                    logits_weighted[:, self.task_index : (self.task_index + 1)]
            prof_sum = torch.sum(logits_weighted, dim=(1, 2, 3))

            # DeepSHAP requires the shape to be B x 1
            return torch.unsqueeze(prof_sum, dim=1)
        else:
            if self.task_index:
                count_output = \
                    count_output[:, self.task_index : (self.task_index + 1)]
            count_sum = torch.sum(count_output, dim=(1, 2))
            
            # DeepSHAP requires the shape to be B x 1
            return torch.unsqueeze(count_sum, dim=1)


def create_explainer(
    model, input_length, profile_length, num_tasks, bg_size=10, task_index=None,
    output_type="profile", use_controls=True
):
    """
    Given a trained Keras model, creates a Shap DeepExplainer that returns
    hypothetical scores for the input sequence.
    Arguments:
        `model`: a model from `profile_model.profile_tf_binding_predictor`
        `input_length`: length of input, I
        `profile_length`: length of profiles, O
        `num_tasks`: number of tasks, T
        `bg_size`: the number of background examples to generate.
        `task_index`: a specific task index (0-indexed) to perform explanations
            from (i.e. explanations will only be from the specified outputs); by
            default explains all tasks
        `output_type`: if "profile", utilizes the profile output to compute the
            importance scores; if "count", utilizes the counts output; in either
            case, the importance scores are for the ouputs summed across strands
            and tasks.
    Returns a function that takes in input sequences and control profiles, and
    outputs hypothetical scores for the input sequences.
    """
    wrapper_model = WrapperModel(model, task_index, output_type)

    bg_func = lambda model_inputs: create_background(
        model_inputs, input_length, profile_length, num_tasks,
        use_controls=use_controls, bg_size=bg_size, seed=None
    )
    combine_func = lambda mult, orig_inp, bg_data: combine_mult_and_diffref(
        mult, orig_inp, bg_data, use_controls=use_controls
    )

    explainer = shap.DeepExplainer(
        model=wrapper_model,
        data=bg_func,
        combine_mult_and_diffref=combine_func
    )

    def explain_fn(input_seqs, cont_profs=None):
        """
        Given input sequences and control profiles, returns hypothetical scores
        for the input sequences.
        Arguments:
            `input_seqs`: a B x I x 4 array
            `cont_profs`: a B x T x O x 4 array
        Returns a B x I x 4 array containing hypothetical importance scores for
        each of the B input sequences.
        """
        # Convert to tensors
        input_seqs_t = place_tensor(torch.tensor(input_seqs)).float()

        if use_controls:
            cont_profs_t = place_tensor(torch.tensor(cont_profs)).float()
            inputs = [input_seqs_t, cont_profs_t]
        else:
            inputs = [input_seqs_t]
        return explainer.shap_values(inputs, progress_message=None)[0]

    return explain_fn


if __name__ == "__main__":
    import json
    import model.profile_models as profile_models
    import feature.util as feature_util
    import feature.make_profile_dataset as make_profile_dataset
    import torch
    import tqdm
    from deeplift.visualization import viz_sequence
    import pandas as pd

    files_spec_path = "/users/amtseng/att_priors/data/processed/ENCODE/profile/config/SPI1/SPI1_training_paths.json"
    model_path = "/users/amtseng/att_priors/models/trained_models/profile_models/SPI1/1/model_ckpt_epoch_10.pt"

    reference_fasta = "/users/amtseng/genomes/hg38.fasta"
    chrom_sizes = "/users/amtseng/genomes/hg38.canon.chrom.sizes"
    input_length = 1346
    profile_length = 1000
    num_tasks = 4

    # Extract files
    with open(files_spec_path, "r") as f:
        files_spec = json.load(f)
    peak_beds = files_spec["peak_beds"][0]  # First peaks BED, arbitrarily
    profile_hdf5 = files_spec["profile_hdf5"]
    pos_coords = pd.read_csv(
        peak_beds, sep="\t", header=None, compression="gzip"
        ).values[:, :3]

    # Import model
    model = restore_model(
        profile_models.ProfileTFBindingPredictor, model_path
    )
    torch.set_grad_enabled(True)
    device = torch.device("cuda") if torch.cuda.is_available() \
        else torch.device("cpu")
    model = model.to(device) 

    # Maps coordinates to 1-hot encoded sequence
    coords_to_seq = feature_util.CoordsToSeq(reference_fasta, center_size_to_use=input_length)
    
    # Maps coordinates to profiles
    coords_to_vals = make_profile_dataset.CoordsToVals(profile_hdf5, profile_length)
    
    # Maps many coordinates to inputs sequences and profiles for the network
    def coords_to_network_inputs(coords):
        input_seq = coords_to_seq(coords)
        profs = coords_to_vals(coords)
        return input_seq, np.swapaxes(profs, 1, 2)

    # Make explainers
    prof_explainer = create_explainer(
        model, input_length, profile_length, num_tasks, output_type="profile",
        use_controls=True
    )
    count_explainer = create_explainer(
        model, input_length, profile_length, num_tasks, output_type="count",
        use_controls=True
    )

    # Compute importance scores
    prof_scores = []
    count_scores = []
    all_input_seqs, all_coords = [], []
    
    batch_size = 128
    num_batches = int(np.ceil(len(pos_coords) / batch_size))
    for i in tqdm.trange(num_batches):
        coords = pos_coords[(i * batch_size) : ((i + 1) * batch_size)]
        input_seqs, profs = coords_to_network_inputs(coords)
        cont_profs = profs[:, num_tasks:, :, :]

        prof_scores.append(prof_explainer(input_seqs, cont_profs))
        count_scores.append(count_explainer(input_seqs, cont_profs))
        all_input_seqs.append(input_seqs)
        break
 
    prof_scores = np.concatenate(prof_scores, axis=0)  
    count_scores = np.concatenate(count_scores, axis=0)
    input_seqs = np.concatenate(all_input_seqs, axis=0)

    # Plot a pair of hypothetical and actual importance scores
    viz_sequence.plot_weights(prof_scores[0], subticks_frequency=100)
    viz_sequence.plot_weights(
        prof_scores[0] * input_seqs[0], subticks_frequency=100
    )
