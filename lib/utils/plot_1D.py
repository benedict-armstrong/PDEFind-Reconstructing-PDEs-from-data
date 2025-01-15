import numpy as np
import matplotlib.pyplot as plt


C_MAP = "viridis"


def plot_3d(
    sol: np.ndarray,
    ref_sol: np.ndarray,
    path: str = None,
):
    """
    Plot the solution in 3D
    """
    fig, axs = plt.subplots(1, 3, figsize=(20, 5), subplot_kw={"projection": "3d"})

    x = ref_sol[..., 0]
    y = ref_sol[..., 1]

    u_sol = sol[..., 0]
    u_ref = ref_sol[..., 2]

    axs[0].plot_surface(x, y, u_ref, cmap=C_MAP)
    axs[0].set_title("$u^{Reference}$")

    axs[1].plot_surface(x, y, u_sol, cmap=C_MAP)
    axs[1].set_title("$u^{Identified System}$")

    axs[2].plot_surface(x, y, np.abs(u_ref - u_sol), cmap=C_MAP)
    axs[2].set_title("$u^{Reference} - u^{Identified System}$")

    for ax in axs:
        ax.set_zlim(u_sol.min() - 0.1, u_sol.max() + 0.1)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$t$")
        ax.set_zlabel("$u$")

    if path:
        plt.savefig(path)


def plot_2d(
    sol: np.ndarray,
    ref_sol: np.ndarray,
    path: str = None,
):
    """
    Plot the solution in 2D
    """

    x = ref_sol[..., 0]
    y = ref_sol[..., 1]

    u_sol = sol[..., 0]
    u_ref = ref_sol[..., 2]

    fig, axs = plt.subplots(1, 3, figsize=(20, 5), sharey=True)
    fig.tight_layout(pad=2.0)
    fig.subplots_adjust(left=0.1)

    # plot data and reference (use the same color scale)
    axs[0].pcolormesh(x, y, u_ref, vmin=u_ref.min(), vmax=u_ref.max())
    axs[0].set_title("$u^{Reference}$")

    axs[1].pcolormesh(x, y, u_sol, vmin=u_ref.min(), vmax=u_ref.max())
    axs[1].set_title("$u^{Identified System}$")

    cm3 = axs[2].pcolormesh(
        x,
        y,
        np.abs(u_sol - u_ref),
        vmin=u_sol.min(),
        vmax=u_sol.max(),
    )
    axs[2].set_title("$|u^{Reference} - u^{Identified System}|$")

    fig.colorbar(cm3, ax=axs)

    for ax in axs:
        ax.set_xlabel("$x$")
        ax.set_ylabel("$x$")

    if path:
        plt.savefig(path)
