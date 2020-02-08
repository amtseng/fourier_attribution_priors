import os
import numpy as np
import tqdm
import sknetwork as skn
import click
import model.util as model_util
import model.profile_models as profile_models
import model.binary_models as binary_models
import extract.compute_gradients as compute_gradients
import torch

def find_seqlets(grad, seqlet_size=10, thresh=0.7):
    """
    Finds seqlets by picking intervals centered at "peaks" in the gradients.
    Peaks are centered at the highest point, or anything at least `thresh` the
    size of this highest height.
    Arguments:
        `grad_x_seq`: an L x 4 array of actual importance scores (i.e. gradient
            times sequence)
        `seqlet_size`: size of seqlet to consider
        `thresh`: threshold for what other peaks to consider in the gradient
    Returns a list of peak centers as indices along the `L` dimension
    (0-indexed). Seqlets will not overlap
    """
    heights = np.sum(grad, axis=1)
    mode = np.max(heights)
    
    # All locations where peaks could be centered
    peak_centers = np.where(heights > (thresh * mode))[0]
    
    # Sort peak centers in decreasing order of height
    inds = np.flip(np.argsort(heights[peak_centers]))
    peak_centers = peak_centers[inds]
    # Traverse the list, adding non-overlapping intervals to the list
    half_size = seqlet_size // 2
    final_peak_centers = []
    for center in peak_centers:
        if center < 0:
            # This center has been invalidated (i.e. lies within a
            # previously found interval)
            continue
        left = center - half_size
        right = left + seqlet_size
        final_peak_centers.append(center)
        # Invalidate all centers that lie within this interval
        peak_centers[(peak_centers >= left) & (peak_centers < right)] = -1
    return final_peak_centers


def window_similarities(seq_1, seq_2):
    """
    Takes two windows (W x 4 arrays) and computes a similarity between them,
    using a continuous Jaccard metric.
    """
    ab_1, ab_2 = np.abs(seq_1), np.abs(seq_2)
    inter = np.minimum(ab_1, ab_2) * np.sign(seq_1) * np.sign(seq_2)
    union = np.maximum(ab_1, ab_2)
    cont_jaccard = np.sum(inter, axis=1) / np.sum(union, axis=1)
    return np.sum(cont_jaccard)


def max_seqlet_similarity(seq_1, seq_2, window_size=8):
    """
    Takes two seqlets (S x 4 arrays) and computes the maximum similarity over
    all possible pairwise windows. Returns the starting indices of the best
    window for each sequence, and the resulting score.
    """
    seq_1_len, seq_2_len = seq_1.shape[0], seq_2.shape[0]
    best_window_1, best_window_2, best_score = None, None, -float("inf")
    for i in range(0, seq_1_len - window_size + 1):
        for j in range(0, seq_2_len - window_size + 1):
            window_score = window_similarities(
                seq_1[i : i + window_size], seq_2[j : j + window_size]
            )
            if window_score > best_score:
                best_window_1, best_window_2, best_score = i, j, window_score
    return best_window_1, best_window_2, best_score


def cluster_gradients(
    input_grads, input_seqs, seqlet_size=20, seqlet_thresh=0.7, window_size=16,
    sample_size=500, num_exemplars=3
):
    """
    Clusters gradients into groups of similar clusters.
    Arguments:
        `input_grads`: N x L x 4 array of input gradients (hypothetical scores)
        `input_seqs`: N x L x 4 array of one-hot sequences, parallel to
            `input_grads`
        `seqlet_size`: size of seqlet to use S, which will be the length of
            motifs
        `seqlet_thresh`: proportion of maximum height to consider for other
            seqlets in an input example
        `window_size`: window size W to use for computing similarity between
            seqlets
        `sample_size`: number of seqlets M to use for generating clusters using
            Louvain
        `num_exemplars`: number of exemplars E per cluster to use for assigning
            clusters to other seqlets
    Returns:
        - Hypothetical scores for all Q seqlets found (Q x S x 4)
        - Actual scores for all Q seqlets found (Q x S x 4)
        - One-hot sequence for all Q seqlets found (Q x S x 4)
        - Cluster assignments by cluster index of every seqlet (Q)
        - Cluster IDs (C)
    """
    print("Extracting seqlets...")
    hyp_seqlets, act_seqlets, seqlet_seqs = [], [], []
    actual = []
    for i in tqdm.trange(len(input_grads)):
        centers = find_seqlets(
            input_grads[i] * input_seqs[i], seqlet_size=seqlet_size,
            thresh=seqlet_thresh
        )
        seqlet_slices = [
            slice(center - (seqlet_size // 2), center + (seqlet_size // 2))
            for center in centers
        ]
        hyp_seqlets.extend([input_grads[i][slc] for slc in seqlet_slices])
        act_seqlets.extend(
            [input_grads[i][slc] * input_seqs[i][slc] for slc in seqlet_slices]
        )
        seqlet_seqs.extend([input_seqs[i][slc] for slc in seqlet_slices])
    hyp_seqlets = np.stack(hyp_seqlets, axis=0)
    act_seqlets = np.stack(act_seqlets, axis=0)
    seqlet_seqs = np.stack(seqlet_seqs, axis=0)
    num_seqlets = len(hyp_seqlets)
    print(
        "\t%d seqlets found over %d sequences" % (num_seqlets, len(input_grads))
    )

    print("Computing similarity matrix...")
    sample_inds = np.random.choice(num_seqlets, size=sample_size, replace=False)
    sim_matrix = np.empty((sample_size, sample_size))
    np.fill_diagonal(sim_matrix, 0)
    for i in tqdm.trange(sample_size):
        for j in range(i, sample_size):
            sim = max_seqlet_similarity(
                hyp_seqlets[sample_inds[i]], hyp_seqlets[sample_inds[j]],
                window_size=window_size
            )[2]
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim
    
    # Turn all negatives into 0, for clustering
    sim_matrix[sim_matrix < 0] = 0

    print("Running Louvain clustering...")
    # Perform Louvain clustering to identify clusters
    louvain = skn.clustering.Louvain()
    cluster_labels = louvain.fit_transform(sim_matrix)
    cluster_ids, cluster_counts = np.unique(cluster_labels, return_counts=True)
    print("\tNumber of clusters: %d" % len(cluster_ids))
    print("\tCluster sizes: %s" % " ".join([str(x) for x in cluster_counts]))
    print("\tModularity: %f" % skn.clustering.modularity(
        sim_matrix, cluster_labels
    ))

    print("Computing pairwise similarity within each cluster...")
    for cluster_id in cluster_ids:
        cluster_inds = np.where(cluster_labels == cluster_id)[0]
        sample_cluster_inds = sample_inds[cluster_inds]
        
        cluster_sims = []
        for i in tqdm.trange(len(sample_cluster_inds)):
            for j in range(i, len(sample_cluster_inds)):
                sim = max_seqlet_similarity(
                    hyp_seqlets[sample_cluster_inds[i]],
                    hyp_seqlets[sample_cluster_inds[j]], window_size=window_size
                )[2]
                cluster_sims.append(sim)
        
        print("\tCluster %d average: %f" % (cluster_id, np.mean(cluster_sims)))

    # For each cluster, select a subset of the examples to be the "exemplars" to
    # which everything else is aligned to
    # The selection of the exemplars is based on the sum of the edge weights of
    # each node, within its cluster
    cluster_exemplars = []
    for cluster_id in cluster_ids:
        cluster_inds = np.where(cluster_labels == cluster_id)[0]
        cluster_sim_matrix = sim_matrix[cluster_inds][:, cluster_inds]
        edge_sums = np.sum(cluster_sim_matrix, axis=0)
        exemplar_inds = cluster_inds[np.flip(np.argsort(edge_sums))][:num_exemplars]
        cluster_exemplars.append(exemplar_inds)

    print("Assigning all seqlets to clusters...")
    # For every example that is not the exemplars (including the examples not
    # used for clustering), compute the best possible similarity score to the
    # exemplars of each cluster to assign them to a cluster
    all_exemplar_inds = np.concatenate(cluster_exemplars)
    # Nasty trick to get the indices that are NOT exemplar indices
    inds_to_cluster = np.arange(num_seqlets)
    inds_to_cluster[all_exemplar_inds] = -1
    inds_to_cluster = inds_to_cluster[inds_to_cluster >= 0]
    
    cluster_sim_scores = np.zeros((num_seqlets, len(cluster_ids)))
    # Number of seqlets x number of clusters
    for query_ind in tqdm.tqdm(inds_to_cluster):
        for cluster_ind in range(len(cluster_ids)):
            cluster_sims = []
            for exemplar_ind in cluster_exemplars[cluster_ind]:
                cluster_sims.append(
                    max_seqlet_similarity(
                        hyp_seqlets[query_ind],
                        hyp_seqlets[sample_inds[exemplar_ind]],
                        window_size=window_size
                    )[2]
                )
            cluster_sim_scores[query_ind, cluster_ind] = np.max(cluster_sims)
    cluster_assignments = np.argmax(cluster_sim_scores, axis=1)

    # Check the accuracy of the cluster assignments; ideally, the seqlets that
    # were originally used for the clustering would all be reassigned back to
    # their proper clusters
    num_correct = 0
    for i, ind in enumerate(sample_inds):
        if cluster_assignments[ind] == cluster_labels[i]:
            num_correct += 1
    print("Cluster recovery: %5.2f%%" % (100 * num_correct / sample_size))

    print("Average similarity score to exemplars in each cluster")
    for cluster_ind, cluster_id in enumerate(cluster_ids):
        scores = cluster_sim_scores[
            cluster_assignments == cluster_ind, cluster_ind
        ]
        avg_score = np.mean(scores)
        print("\tCluster %d: %6.4f" % (cluster_id, avg_score))

    return hyp_seqlets, act_seqlets, seqlet_seqs, cluster_assignments, \
        cluster_ids


@click.command()
@click.option(
    "-r", "--ref-fasta", default="/users/amtseng/genomes/hg38.fasta",
    help="Path to reference FASTA; defaults to /users/amtseng/genomes/hg38.fasta"
)
@click.option(
    "-t", "--model-type", type=click.Choice(["profile", "binary"]),
    required=True, help="Type of model"
)
@click.option(
    "-l", "--input-length",
    help="Length of input; defaults to 1000 for binary models, 1346 for profile models"
)
@click.option(
    "-p", "--profile-length", default=1000,
    help="Length of profile output for profile models; default 1000"
)
@click.option(
    "-c", "--use-controls", is_flag=True,
    help="For profiles models, if specified, use control tracks"
)
@click.option(
    "-m", "--chrom-set", required=True,
    help="Comma-separated list of chromosomes to use for positive set"
)
@click.option(
    "-o", "--out-dir", required=True, help="Where to store outputs"
)
@click.argument("files_spec_path")
@click.argument("model_path")
def main(
    ref_fasta, model_type, input_length, profile_length, use_controls,
    chrom_set, out_dir, files_spec_path, model_path
):
    if not input_length:
        if model_type == "binary":
            input_length = 1000
        else:
            input_length = 1346

    if model_type == "binary":
        model_class = binary_models.BinaryPredictor
    elif model_type == "profile" and use_controls:
        model_class = profile_models.ProfilePredictorWithControls
    else:
        model_class = profile_models.ProfilePredictorWithoutControls

    chrom_set = chrom_set.split(",")

    print("Inputs supplied:")
    print("\tReference fasta: %s" % ref_fasta)
    print("\tModel type: %s" % model_type)
    print("\tChromosome set: %s" % chrom_set)
    print("\tFiles spec: %s" % files_spec_path)
    print("\tModel: %s" % model_path)

    # Import model
    torch.set_grad_enabled(True)
    device = torch.device("cuda") if torch.cuda.is_available() \
        else torch.device("cpu")
    model = model_util.restore_model(model_class, model_path)
    model.eval()
    model = model.to(device)

    print("Computing gradients...")
    input_grads, input_seqs = compute_gradients.get_input_grads(
        model, model_type, files_spec_path, input_length, ref_fasta,
        chrom_set=chrom_set, profile_length=profile_length,
        use_controls=use_controls
    )
    
    hyp_seqlets, act_seqlets, seqlet_seqs, cluster_assignments, cluster_ids = \
        cluster_gradients(
            input_grads, input_seqs
        )

    print("Saving output...")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "hyp_seqlets"), hyp_seqlets)
    np.save(os.path.join(out_dir, "act_seqlets"), act_seqlets)
    np.save(os.path.join(out_dir, "seqlet_seqs"), seqlet_seqs)
    np.save(os.path.join(out_dir, "cluster_assignments"), cluster_assignments)
    np.save(os.path.join(out_dir, "cluster_ids"), cluster_ids)

if __name__ == "__main__":
    main()
