from typing import Iterable
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from lib.pde_find2 import PDEFind
from matplotlib.animation import FuncAnimation


class Solver:
    def __init__(
        self,
        pde_find: PDEFind,
        data_shape: tuple,
        coeffs: Iterable[np.ndarray],
        terms_to_include: list,
    ):
        self.num_vars = pde_find.num_dep_vars
        self.data_shape = data_shape
        self.pde_find = pde_find
        self.terms_to_include = terms_to_include
        self.coeffs = coeffs

        self.dx = 0.078125
        self.dy = 0.078125

        print(self.data_shape)

    def pde_rhs(self, t, u1D):
        u, v = self.u1D_to_ND(u1D)

        library, _ = self.pde_find.create_library(
            u, v, include_terms=self.terms_to_include
        )

        u_t = (library.T @ self.coeffs[0]).T
        v_t = (library.T @ self.coeffs[1]).T

        # Convert the 2D array back to 1D for the ODE solver
        return self.uND_to_1D(u_t, v_t)

    def uND_to_1D(self, *data: np.ndarray):
        # x = np.concatenate([u.flatten(), v.flatten()])
        return np.concatenate(data, axis=0).reshape(-1)

    def u1D_to_ND(self, u1D: np.ndarray):
        reshaped_data = u1D.reshape((self.num_vars,) + self.data_shape)
        # y = np.moveaxis(reshaped_data, 0, -1)
        return reshaped_data

    def solve(self, t_eval, u_0):
        sol = solve_ivp(
            self.pde_rhs,
            (t_eval.min(), t_eval.max()),
            u_0,
            t_eval=t_eval,
            method="RK45",
        )

        print(sol.status)
        print(sol.message)

        return sol.y.reshape((self.num_vars,) + self.data_shape + (-1,))


if __name__ == "__main__":
    raw_data = np.load("data/3.npz")

    u = raw_data["u"]
    v = raw_data["v"]
    x = raw_data["x"]
    y = raw_data["y"]
    t = raw_data["t"]

    pdefind = PDEFind(
        var_labels=(["x", "y", "t"], ["u", "v"]),
        polynomial_degree=3,
        order=2,
        periodic=True,
    )

    u, v, x, y, t = pdefind.subsample_data(u, v, x, y, t, factors=[2, 2, 1])

    x_grid, y_grid, t_grid = pdefind.add_grid(x, y, t)

    term_regex = r"^(u|v)*(_{(x|y)+})?(v)*$"

    library, labels = pdefind.create_library(u, v, term_regex=term_regex)

    temp = np.load("data/3_coeffs.npz")

    coef_u = temp["u"]
    coef_v = temp["v"]

    u_pred, v_pred = pdefind.solve([u[..., 0], v[..., 0]], [coef_u, coef_v], labels)

    u_t = np.gradient(u, t_grid, axis=2)
    v_t = np.gradient(v, t_grid, axis=2)

    l2_err_u = np.linalg.norm(u - u_pred) / np.linalg.norm(u)
    l2_err_v = np.linalg.norm(v - v_pred) / np.linalg.norm(v)

    print("L2 error for u: ", l2_err_u)
    print("L2 error for v: ", l2_err_v)

    l2_err_u = np.linalg.norm(u[..., -1] - u_pred[..., -1]) / np.linalg.norm(u[..., -1])
    l2_err_v = np.linalg.norm(v[..., -1] - v_pred[..., -1]) / np.linalg.norm(v[..., -1])

    print("L2 error (final_t) for u: ", l2_err_u)
    print("L2 error (final_t) for v: ", l2_err_v)

    u_pred_whole_t = np.gradient(u_pred, t_grid, axis=2)

    # Create a figure with two 3D axis
    fig, axs = plt.subplots(2, 3, figsize=(12, 6))

    # Function to update the plot
    def update(time):
        x_i = x[..., time]
        y_i = y[..., time]

        axs[0, 0].clear()
        axs[0, 0].pcolormesh(x_i, y_i, u[..., time], cmap="viridis")

        # axs[0, 0].set_zlabel("u")
        axs[0, 0].set_title(f"u at time = {t_grid[time]:.2f}")

        axs[0, 1].clear()
        axs[0, 1].pcolormesh(x_i, y_i, u_pred[..., time], cmap="viridis")

        # axs[0, 1].set_zlabel("u_t")
        axs[0, 1].set_title(f"u_t at time = {t_grid[time]:.2f}")

        axs[0, 2].clear()
        axs[0, 2].pcolormesh(
            x_i,
            y_i,
            np.abs(u[..., time] - u_pred[..., time]),
            cmap="viridis",
        )

        # axs[0, 2].set_zlabel("u - u_t")
        axs[0, 2].set_title(f"u - u_t at time = {t_grid[time]:.2f}")

        axs[1, 0].clear()
        axs[1, 0].pcolormesh(x_i, y_i, v[..., time], cmap="viridis")

        # axs[1, 0].set_zlabel("u")
        axs[1, 0].set_title(f"v at time = {t_grid[time]:.2f}")

        axs[1, 1].clear()
        axs[1, 1].pcolormesh(x_i, y_i, v_pred[..., time], cmap="viridis")

        # axs[1, 1].set_zlabel("u_t")
        axs[1, 1].set_title(f"v_t at time = {t_grid[time]:.2f}")

        axs[1, 2].clear()
        axs[1, 2].pcolormesh(
            x_i,
            y_i,
            np.abs(v[..., time] - v_pred[..., time]),
            cmap="viridis",
        )

        # axs[1, 2].set_zlabel("v - v_t")
        axs[1, 2].set_title(f"v - v_t at time = {t_grid[time]:.2f}")

        # set limits for both axes
        for ax in axs.flatten():
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_xlim(x.min(), x.max())
            ax.set_ylim(y.min(), y.max())
            # ax.set_zlim(-1, 1)

    # Create the animation
    anim = FuncAnimation(fig, update, frames=len(t_grid), interval=100)

    # Save the animation as a gif
    anim.save("gifs/mine.gif")
