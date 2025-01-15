from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt

CMAP = "viridis"


def plot(
    sol: np.ndarray,
    ref_sol: np.ndarray,
    path: str = None,
):
    # Create a figure with two 3D axis
    fig, axs = plt.subplots(2, 3, figsize=(12, 6))
    fig.subplots_adjust(left=0.1)

    x = ref_sol[..., 0]
    y = ref_sol[..., 1]

    # Function to update the plot
    def update(time):
        for ax in axs.flatten():
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_xlim(x.min(), y.max())
            ax.set_ylim(x.min(), y.max())
            ax.clear()

        for i, var in enumerate(["u", "v"]):
            u_pred = sol[..., i]
            u_ref = ref_sol[..., 3 + i]

            axs[i, 0].pcolormesh(
                x[..., time],
                y[..., time],
                u_ref[..., time],
                cmap=CMAP,
                vmin=u_ref.min(),
                vmax=u_ref.max(),
            )
            axs[i, 0].set_title(f"{var}_ref at time = {time}")

            cm = axs[i, 1].pcolormesh(
                x[..., time],
                y[..., time],
                u_pred[..., time],
                cmap=CMAP,
                vmin=u_ref.min(),
                vmax=u_ref.max(),
            )
            axs[i, 1].set_title(f"{var} at time = {time}")

            axs[i, 2].pcolormesh(
                x[..., time],
                y[..., time],
                np.abs(u_ref[..., time] - u_pred[..., time]),
                cmap=CMAP,
                vmin=0,
                vmax=u_ref.max(),
            )
            axs[0, 2].set_title(f"{var}_ref - {var} at time = {time}")

        if time == 0:
            fig.colorbar(cm, ax=axs)

    # # Create the animation
    anim = FuncAnimation(fig, update, frames=sol.shape[2] - 1, interval=100)

    # Save the animation as a gif
    if path:
        anim.save(path)
