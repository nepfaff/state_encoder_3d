import zarr
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.spatial import KDTree


def contains_duplicate_arrays(data: np.ndarray) -> bool:
    return np.any(
        [
            i != j and np.array_equal(arr1, arr2)
            for i, arr1 in enumerate(data)
            for j, arr2 in enumerate(data)
        ]
    )


def prevent_duplicates(data: np.ndarray) -> np.ndarray:
    for i, arr1 in enumerate(data):
        for j, arr2 in enumerate(data):
            if i != j and np.array_equal(arr1, arr2):
                # Add a bit of random noise to prevent the arrays from being identical
                arr1 += 1e-6 * np.random.uniform(low=-1, high=1, size=len(arr1))
    return data


def evaluate_kl_divergence_after_tsne(model_latent_train_tuples, num_states) -> None:
    kl_divergence_results = []
    for name, path, train in tqdm(
        model_latent_train_tuples,
        position=0,
        desc="Model loop",
    ):
        # Number of states is limited by distinct colors in color palette
        data = zarr.open(path)
        latents = np.asarray(data.latents)

        # Pick subset of states
        rndperm = np.random.permutation(len(latents))[:num_states]
        latents = latents[rndperm]

        # Collapse view dimension into state dimension
        latents = latents.reshape((-1, latents.shape[-1]))

        # Convert to pandas dataframe
        feat_cols = [f"l{i}" for i in range(latents.shape[-1])]
        df = pd.DataFrame(latents, columns=feat_cols)

        # Reduce dimensionality using PCA
        pca = PCA(n_components=50)
        pca_result_50 = pca.fit_transform(df[feat_cols].values)
        pca_result_50 = prevent_duplicates(pca_result_50)

        assert not contains_duplicate_arrays(
            pca_result_50
        ), "Need to increase the noise magnitude to prevent duplicates"

        # Collect KL-Divergence after TSNE optimization data
        for _ in tqdm(range(10), position=1, leave=False, desc=f" Evaluating {name}"):
            tsne = TSNE(n_components=2, n_iter=5000)
            _ = tsne.fit_transform(pca_result_50)
            # Append reverses sign for some reason?
            kl_div = float(tsne.kl_divergence_)
            kl_divergence_results.append([name, kl_div, "Train" if train else "Eval"])

    # Create bar plot
    df = pd.DataFrame(kl_divergence_results, columns=["name", "kl_divergence", "Data"])
    fig = plt.figure(figsize=(16, 10))
    sns.barplot(data=df, x="name", y="kl_divergence", hue="Data")
    plt.title("KL-Divergence After TSNE Optimization")
    plt.ylabel("KL-Divergence")
    plt.xlabel("Model")
    plt.savefig("kl_divergence_bar_plot1.png")
    plt.close()


def evaluate_nearest_neighbors(model_latent_train_tuples, num_states) -> None:
    mean_correct_fractions = []
    for name, path, train in tqdm(
        model_latent_train_tuples,
        position=0,
        desc="Model loop",
    ):
        # Number of states is limited by distinct colors in color palette
        data = zarr.open(path)
        latents = np.asarray(data.latents)

        # # Pick subset of states
        # rndperm = np.random.permutation(len(latents))[:num_states]
        # latents = latents[rndperm]

        # Collapse view dimension into state dimension
        labels_single = [i for i in range(len(latents))]
        num_views = latents.shape[1]
        labels = np.repeat(labels_single, num_views)
        latents = latents.reshape((-1, latents.shape[-1]))

        # Reduce dimensionality
        pca = PCA(n_components=50)
        pca_result = pca.fit_transform(latents)

        tsne = TSNE(n_components=2, n_iter=2000)
        tsne_results = tsne.fit_transform(pca_result)

        kdtree = KDTree(tsne_results)

        correct_fractions = []
        for label, latent in zip(labels, tsne_results):
            _, indices = kdtree.query(latent, k=num_views, workers=-1)
            neighbor_labels = labels[indices]
            correct_fractions.append(
                np.count_nonzero(neighbor_labels == label) / num_views
            )
        mean_correct_fraction = np.mean(correct_fractions)
        mean_correct_fractions.append(
            [name, mean_correct_fraction, "Train" if train else "Eval"]
        )

    # Create bar plot
    df = pd.DataFrame(
        mean_correct_fractions, columns=["name", "correct_fraction", "Data"]
    )
    fig = plt.figure(figsize=(16, 10))
    # sns.barplot(data=df, x="name", y="correct_fraction", hue="Data")
    sns.barplot(data=df, x="name", y="correct_fraction")
    # plt.title("Nearest Neighbor Score")
    plt.ylabel("Ratio of Correct Nearest Neighbors")
    plt.xlabel("Model")
    plt.savefig("nn_correct_fractions3.png")
    plt.close()


def main():
    num_states = 1000
    np.random.seed(13)

    model_latent_train_tuples = [
        # ("vanilla_ae", "data/checkpoints/vanilla_ae/train_latents.zarr", True),
        ("vanilla_ae", "data/checkpoints/vanilla_ae/eval_latents.zarr", False),
        # (
        #     "vanilla_ae_ct",
        #     "data/checkpoints/vanilla_ae_ct/train_latents.zarr",
        #     True,
        # ),
        (
            "vanilla_ae_ct",
            "data/checkpoints/vanilla_ae_ct/eval_latents.zarr",
            False,
        ),
        # ("nerf_ae", "data/checkpoints/nerf_ae/train_latents.zarr", True),
        ("nerf_ae", "data/checkpoints/nerf_ae/eval_latents.zarr", False),
        # ("nerf_ae_ct", "data/checkpoints/nerf_ae_ct/train_latents.zarr", True),
        ("nerf_ae_ct", "data/checkpoints/nerf_ae_ct/eval_latents.zarr", False),
        # (
        #     "nerf_ae_depth",
        #     "data/checkpoints/nerf_ae_depth/train_latents.zarr",
        #     True,
        # ),
        (
            "nerf_ae_depth",
            "data/checkpoints/nerf_ae_depth/eval_latents.zarr",
            False,
        ),
        # (
        #     "nerf_ae_ct_depth",
        #     "data/checkpoints/nerf_ae_ct_depth/train_latents.zarr",
        #     True,
        # ),
        (
            "nerf_ae_ct_depth",
            "data/checkpoints/nerf_ae_ct_depth/eval_latents.zarr",
            False,
        ),
        # (
        #     "ct_info_nce",
        #     "data/checkpoints/ct_info_nce/train_latents.zarr",
        #     True,
        # ),
        (
            "ct_info_nce",
            "data/checkpoints/ct_info_nce/eval_latents_old.zarr",
            False,
        ),
        # ("ct_triplet", "data/checkpoints/ct_triplet/train_latents.zarr", True),
        ("ct_triplet", "data/checkpoints/ct_triplet/eval_latents.zarr", False),
    ]

    # evaluate_kl_divergence_after_tsne(model_latent_train_tuples, num_states)

    evaluate_nearest_neighbors(model_latent_train_tuples, num_states)


if __name__ == "__main__":
    main()
