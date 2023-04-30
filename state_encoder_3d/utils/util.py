import matplotlib.pyplot as plt


def plot_output_ground_truth(img, depth, gt_img, resolution):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), squeeze=False)
    axes[0, 0].imshow(img.cpu().view(*resolution).detach().numpy())
    axes[0, 0].set_title("Trained MLP")
    axes[0, 1].imshow(gt_img.cpu().view(*resolution).detach().numpy())
    axes[0, 1].set_title("Ground Truth")

    depth = depth.cpu().view(*resolution[:2]).detach().numpy()
    axes[0, 2].imshow(depth, cmap="Greys")
    axes[0, 2].set_title("Depth")

    for i in range(3):
        axes[0, i].set_axis_off()

    return fig
