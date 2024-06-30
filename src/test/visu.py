from typing import Optional, List
import ipdb

import torch

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.pyplot import get_cmap
import matplotlib as mpl

# FIXME why do we only sometimes get the error because wrong backend?!
# mpl.use("Qt5Agg")


def array2rgb(
    im: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "gray",
) -> np.ndarray:
    """
    Convert array to color map, if given limits [vmin, vmax], the values are normalized.

    args:
    ---
    im: Numpy array of shape [H x W], [H x W x 1] or [B x H x W x 1]

    returns:
    ---
    rgb_img: RGB array of shape [H x W x 3] or [B x H x W x 3]
    """
    cmap = get_cmap(cmap)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    rgba_img = cmap(norm(im).astype(np.float32))
    rgb_img = np.delete(rgba_img, 3, -1)
    return rgb_img


def show_img(image: np.ndarray) -> None:
    fig = plt.figure()
    if image.max() > 1.0:
        image = image / 255.0
    plt.imshow(image)
    plt.axis("off")
    plt.show()


def fig2rgb_array(fig: plt.Figure) -> np.ndarray:
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    rgb = np.delete(rgba, 3, -1)
    return rgb


def create_animation(
    images: List[np.ndarray], titles: Optional[List[str]] = None, output_file: Optional[str] = None, **kwargs
) -> None:
    """
    Create an animation from a list of images with additional titles for each frame if wanted
    """
    import matplotlib.animation as animation

    fig = plt.figure()
    ims = []

    for img in images:
        if img.max() > 1.0:
            img = img / 255.0
        im = plt.imshow(img)
        plt.axis("off")
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, **kwargs)
    if output_file is not None:
        ani.save(output_file, writer="imagemagick")
    else:
        plt.show()


def plot_side_by_side(
    image_est: torch.Tensor,
    image_gt: torch.Tensor,
    depth_est: torch.Tensor,
    depth_gt: torch.Tensor,
    cmap: Optional[str] = "Spectral",
    title: Optional[str] = None,
    return_fig: bool = False,
    return_image: bool = False,
):

    fig = plt.figure()
    if title:
        fig.suptitle(title, fontsize=14)
    plt.axis("off")

    img, ref = image_est.cpu().numpy().transpose(1, 2, 0), image_gt.cpu().numpy().transpose(1, 2, 0)
    img, ref = img.squeeze(), ref.squeeze()
    img, ref = img.clip(0, 255.0), ref.clip(0, 255.0)

    depth_est, depth_gt = depth_est.cpu().numpy().squeeze(), depth_gt.cpu().numpy().squeeze()
    # Use the same color scheme for both depth maps, so we can directly compare differences
    z_max, z_min = max(depth_gt.max(), depth_est.max()), min(depth_gt.min(), depth_est.min())

    valid = depth_gt > 0

    dz = np.abs(depth_gt - depth_est)
    dz_min, dz_max = dz[valid].min(), dz[valid].max()
    dz_img = array2rgb(dz, cmap="viridis")
    dz_img[~valid] = np.array([0.0, 0.0, 0.0])
    if dz_img.max() > 1.0:
        dz_img = dz_img / 255.0

    di = np.abs(ref - img)
    di = np.linalg.norm(di, axis=-1)  # Take the norm over the color channels
    di_img = array2rgb(di, cmap="viridis")
    # di_img[~valid] = np.array([0.0, 0.0, 0.0])
    if di_img.max() > 1.0:
        di_img = di_img / 255.0

    if ref.max() > 1.0:
        ref = ref / 255.0
    if img.max() > 1.0:
        img = img / 255.0

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(ref)
    ax1.set_title("Ground Truth")
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(img)
    ax2.set_title("Estimate")
    ax2.axis("off")

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(di_img)
    norm = mpl.colors.Normalize(vmin=di.min(), vmax=di.max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    plt.colorbar(sm, ax=ax3)
    ax3.set_title("Delta I")
    ax3.axis("off")

    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(depth_gt, cmap=cmap, vmin=z_min, vmax=z_max)
    ax4.set_title("Ground Truth")
    ax4.axis("off")

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(depth_est, cmap=cmap, vmin=z_min, vmax=z_max)
    ax5.set_title("Estimate")
    ax5.axis("off")
    plt.colorbar(ax5.imshow(depth_est, cmap=cmap, vmin=z_min, vmax=z_max), ax=ax5)

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.imshow(dz_img)
    ax6.set_title("Delta Z")
    norm = mpl.colors.Normalize(vmin=dz_min, vmax=dz_max)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    plt.colorbar(sm, ax=ax6)
    ax6.axis("off")

    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    fig.subplots_adjust(top=0.9)

    if return_fig:
        return fig
    if return_image:
        image_from_plot = fig2rgb_array(fig)
        plt.close()
        return image_from_plot
    else:
        plt.show()
