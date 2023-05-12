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
        "--data",
        type=str,
        required=True,
        help="Path to the latent space data zarr file.",
    )
    args = parser.parse_args()
    data_path = args.data

    # Number of states is limited by distinct colors in color palette
    num_states = 10
    color_palette = "tab10"

    current_time = time.strftime("%Y-%b-%d-%H-%M-%S")
    wandb.init(
        project="state_encoder_3d",
        name=f"visualize_latent_space_{current_time}",
        mode="online",
        config=vars(args),
    )

    data = zarr.open(data_path)
    states = np.asarray(data.states)
    latents = np.asarray(data.latents)

    # Pick subset of states
    np.random.seed(42)
    rndperm = np.random.permutation(len(states))[:num_states]
    states = states[rndperm]
    latents = latents[rndperm]

    # Collapse view dimension into state dimension
    labels = [state_to_str(state) for state in states]
    labels = np.repeat(labels, latents.shape[1])
    latents = latents.reshape((-1, latents.shape[-1]))

    # Convert to pandas dataframe
    feat_cols = [f"l{i}" for i in range(latents.shape[-1])]
    df = pd.DataFrame(latents, columns=feat_cols)
    df["state_idx"] = labels

    # Reduce dimension using PCA
    pca = PCA(n_components=5)
    pca_result = pca.fit_transform(df[feat_cols].values)
    print(
        f"Explained variance of remaining components: {np.sum(pca.explained_variance_ratio_)}"
    )

    # Visualize PCA results
    df["pca-one"] = pca_result[:, 0]
    df["pca-two"] = pca_result[:, 1]
    df["pca-three"] = pca_result[:, 2]
    fig = plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="pca-one",
        y="pca-two",
        hue="state_idx",
        palette=sns.color_palette(color_palette, num_states),
        data=df,
        legend="full",
        alpha=0.3,
    )
    plt.title("Fist two PCA components")
    # plt.show()
    wandb.log({"first_two_pca_components": wandb.Image(fig)})
    plt.close()

    # fig = plt.figure(figsize=(16, 10))
    # ax = fig.add_subplot(projection="3d")
    # ax.scatter(
    #     xs=df["pca-one"],
    #     ys=df["pca-two"],
    #     zs=df["pca-three"],
    #     c=df["state_idx"],
    #     cmap="tab10",
    # )
    # ax.set_xlabel("pca-one")
    # ax.set_ylabel("pca-two")
    # ax.set_zlabel("pca-three")
    # plt.title("Fist three PCA components")
    # plt.show()
    # wandb.log({"first_three_pca_components": fig})
    plt.close()

    # TSNE on all dimensions
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(df[feat_cols].values)
    df["tsne-2d-one"] = tsne_results[:, 0]
    df["tsne-2d-two"] = tsne_results[:, 1]

    fig = plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one",
        y="tsne-2d-two",
        hue="state_idx",
        palette=sns.color_palette(color_palette, 10),
        data=df,
        legend="full",
        alpha=0.3,
    )
    plt.title("TSNE on all dimensions (2 components)")
    # plt.show()
    wandb.log({"tsne_all_dim_first_2_components": wandb.Image(fig)})
    plt.close()

    # TSNE on subset of dimensions
    pca = PCA(n_components=50)
    pca_result_50 = pca.fit_transform(df[feat_cols].values)
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
    tsne_pca_results = tsne.fit_transform(pca_result_50)
    df["tsne-pca50-one"] = tsne_pca_results[:, 0]
    df["tsne-pca50-two"] = tsne_pca_results[:, 1]

    fig = plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-pca50-one",
        y="tsne-pca50-two",
        hue="state_idx",
        palette=sns.color_palette(color_palette, 10),
        data=df,
        legend="full",
        alpha=0.3,
    )
    plt.title("TSNE on 50 dimensions (2 components)")
    # plt.show()
    wandb.log({"tsne_50_dim_first_2_components": wandb.Image(fig)})
    plt.close()


if __name__ == "__main__":
    main()
