from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme("paper", "whitegrid")
sns.despine()


def create_gif(
    x,
    y,
    t_grid,
    u_ref,
    v_ref,
    u_pred,
    v_pred,
    path: str,
    cmap="viridis",
):
    # Create a figure with two 3D axis
    fig, axs = plt.subplots(2, 3, figsize=(24, 8))

    # Function to update the plot
    def update(i: int):
        x_grid = x[..., i]
        y_grid = y[..., i]

        time = t_grid[i]

        axs[0, 0].clear()
        axs[0, 0].pcolormesh(x_grid, y_grid, u_ref[..., i], cmap=cmap)

        axs[0, 0].set_title(f"$u^{{Reference}}$ at time = {time:.2f}")

        axs[0, 1].clear()
        axs[0, 1].pcolormesh(x_grid, y_grid, u_pred[..., i], cmap=cmap)

        axs[0, 1].set_title(f"$u^{{Identified System}}$ at time = {time:.2f}")

        axs[0, 2].clear()
        axs[0, 2].pcolormesh(
            x_grid,
            y_grid,
            np.abs(u_ref[..., i] - u_pred[..., i]),
            cmap=cmap,
        )

        axs[0, 2].set_title(
            f"$u^{{Reference}} - u^{{Identified System}}$ at time = {time:.2f}"
        )

        axs[1, 0].clear()
        axs[1, 0].pcolormesh(x_grid, y_grid, v_ref[..., i], cmap=cmap)

        axs[1, 0].set_title(f"$v^{{Reference}}$ at time = {time:.2f}")

        axs[1, 1].clear()
        axs[1, 1].pcolormesh(x_grid, y_grid, v_pred[..., i], cmap=cmap)

        axs[1, 1].set_title(f"$v^{{Identified System}}$ at time = {time:.2f}")

        axs[1, 2].clear()
        axs[1, 2].pcolormesh(
            x_grid,
            y_grid,
            np.abs(v_ref[..., i] - v_pred[..., i]),
            cmap=cmap,
        )

        axs[1, 2].set_title(
            f"$v^{{Reference}} - v^{{Identified System}}$ at time = {time:.2f}"
        )

        # set limits for both axes
        for ax in axs.flatten():
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_xlim(x.min(), x.max())
            ax.set_ylim(y.min(), y.max())

        plt.tight_layout()

    # Create the animation
    anim = FuncAnimation(fig, update, frames=len(t_grid), interval=100)

    # Save the animation as a gif
    anim.save(path)

    return anim
