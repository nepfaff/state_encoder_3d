import argparse
import time

import zarr
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import wandb


def state_to_str(state: np.ndarray) -> str:
    return f"f: ({state[0]}, {state[1]}), b: ({state[2]}, {state[3]})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name for wandb logging.",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the latent space data zarr file.",
    )
    args = parser.parse_args()
    data_path = args.data
    run_name = args.name
    
    # Number of states is limited by distinct colors in color palette
    num_states = 15
    color_palette = "tab20"

    current_time = time.strftime("%Y-%b-%d-%H-%M-%S")
    wandb.init(
        project="state_encoder_3d",
        name=f"visualize_latent_space_{run_name}_{current_time}",
        mode="online",
        config=vars(args),
    )

    data = zarr.open(data_path)
    states = np.asarray(data.states)
    latents = np.asarray(data.latents)

    # Pick subset of states
    np.random.seed(13)
    rndperm = np.random.permutation(len(states))[:num_states]
    states = states[rndperm]
    latents = latents[rndperm]

    # Collapse view dimension into state dimension
    labels_single = [state_to_str(state) for state in states]
    labels = np.repeat(labels_single, latents.shape[1])
    latents = latents.reshape((-1, latents.shape[-1]))

    # Convert to pandas dataframe
    feat_cols = [f"l{i}" for i in range(latents.shape[-1])]
    df = pd.DataFrame(latents, columns=feat_cols)
    df["state"] = labels

    # Visualize state-space samples

    fig = plt.figure(figsize=(10, 5))
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), squeeze=False)
    finger_df = pd.DataFrame(states[:, :2], columns=["x", "y"])
    finger_df["state"] = labels_single
    sns.scatterplot(
        x="x",
        y="y",
        hue="state",
        palette=sns.color_palette(color_palette, num_states),
        data=finger_df,
        legend="full",
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Finger states")
    box_df = pd.DataFrame(states[:, 2:], columns=["x", "y"])
    box_df["state"] = labels_single
    sns.scatterplot(
        x="x",
        y="y",
        hue="state",
        palette=sns.color_palette(color_palette, num_states),
        data=box_df,
        legend="full",
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("Box states")
    wandb.log({"sampled_state_space": wandb.Image(fig)})
    plt.close()

    # Reduce dimension using PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df[feat_cols].values)
    print(
        f"Explained variance of remaining 2 components: {np.sum(pca.explained_variance_ratio_)}"
    )

    # Visualize PCA results
    df["pca-one"] = pca_result[:, 0]
    df["pca-two"] = pca_result[:, 1]
    fig = plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="pca-one",
        y="pca-two",
        hue="state",
        palette=sns.color_palette(color_palette, num_states),
        data=df,
        legend="full",
    )
    plt.title("Fist two PCA components")
    wandb.log({"first_two_pca_components": wandb.Image(fig)})
    plt.close()

    # TSNE on all dimensions
    tsne = TSNE(n_components=2, n_iter=5000)
    tsne_results = tsne.fit_transform(df[feat_cols].values)
    df["tsne-2d-one"] = tsne_results[:, 0]
    df["tsne-2d-two"] = tsne_results[:, 1]
    print(f"KL divergence after optimization: {tsne.kl_divergence_}")

    fig = plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one",
        y="tsne-2d-two",
        hue="state",
        palette=sns.color_palette(color_palette, num_states),
        data=df,
        legend="full",
    )
    plt.title("TSNE on all dimensions (2 components)")
    wandb.log({"tsne_all_dim_first_2_components": wandb.Image(fig)})
    plt.close()

    # TSNE on subset of dimensions
    pca = PCA(n_components=50)
    pca_result_50 = pca.fit_transform(df[feat_cols].values)
    print(
        f"Explained variance of remaining 50 components: {np.sum(pca.explained_variance_ratio_)}"
    )
    tsne = TSNE(n_components=2, n_iter=5000)
    tsne_pca_results = tsne.fit_transform(pca_result_50)
    df["tsne-pca50-one"] = tsne_pca_results[:, 0]
    df["tsne-pca50-two"] = tsne_pca_results[:, 1]
    print(f"KL divergence after optimization: {tsne.kl_divergence_}")

    fig = plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-pca50-one",
        y="tsne-pca50-two",
        hue="state",
        palette=sns.color_palette(color_palette, num_states),
        data=df,
        legend="full",
    )
    plt.title("TSNE on 50 dimensions (2 components)")
    wandb.log({"tsne_50_dim_first_2_components": wandb.Image(fig)})
    plt.close()

    # Collect KL-Divergence after TSNE optimization data
    kl_divergences = [tsne.kl_divergence_]
    for _ in range(9):
        tsne = TSNE(n_components=2, n_iter=5000)
        tsne_pca_results = tsne.fit_transform(pca_result_50)
        kl_divergences.append(tsne.kl_divergence_)
    mean_kl_divergence = np.mean(kl_divergences)
    print(f"Mean KL divergence after optimization: {mean_kl_divergence}")
    wandb.log(
        {
            "kl_divergence_after_optimization": wandb.Table(
                data=[
                    ["mean", mean_kl_divergence],
                    ["std", np.std(kl_divergences)],
                    ["min", np.min(kl_divergences)],
                    ["max", np.max(kl_divergences)],
                ],
                columns=["metric", "value"]
            )
        }
    )


if __name__ == "__main__":
    main()
