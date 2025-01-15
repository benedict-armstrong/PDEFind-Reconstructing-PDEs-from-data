# %%
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from lib.pde_find import PDEFind
from matplotlib.animation import FuncAnimation

# %%
raw_data = np.load("src/PDE-Find: Reconstructing PDEs from data/data/3.npz")

u_o = raw_data["u"]
v_o = raw_data["v"]
x_o = raw_data["x"]
y_o = raw_data["y"]
t_o = raw_data["t"]

data = np.stack(
    [
        x_o,
        y_o,
        t_o,
        u_o,
        v_o,
    ],
    axis=-1,
)

# downsample the data
data = data[::4, ::4, ::1]

vars = (["x", "y", "t"], ["u", "v"])

pdefind = PDEFind(data, vars, lib_size=3, order=2)

# %%
# Create grid for x and y
x = pdefind.ind_var_grids[0]
y = pdefind.ind_var_grids[1]
t = pdefind.ind_var_grids[2]

# Discretize the spatial derivatives (finite differences)
dx = x[1] - x[0]
dy = y[1] - y[0]

# %%
x.shape, y.shape, t.shape

# %%
Nx = x.shape[0]
Ny = y.shape[0]

# %%
u = u_o[::4, ::4, ::1]
v = v_o[::4, ::4, ::1]


# %%
# Example initial condition: a Gaussian
def initial_condition(x):
    return x[:, :, 0]


# %%
initial_condition(u)


# %%
# Convert 2D grid to 1D for solve_ivp
def u2D_to_1D(*data):
    # x = np.concatenate([u.flatten(), v.flatten()])
    return np.concatenate(data, axis=0).reshape(-1)


def u1D_to_2D(u1D: np.ndarray):
    u = u1D[: Nx * Ny].reshape((Nx, Ny))
    v = u1D[Nx * Ny :].reshape((Nx, Ny))

    reshaped_data = u1D.reshape((2, Nx, Ny))
    y = np.moveaxis(reshaped_data, 0, -1)
    u_1 = y[..., 0]
    v_1 = y[..., 1]
    return y


# %%
# Define the PDE as a system of ODEs
def pde_system(t, u1D):
    dataND = u1D_to_2D(u1D)

    u, v = dataND[..., 0], dataND[..., 1]

    padding = 3

    # If the grid is periodic, pad the grid and data
    u = np.pad(u, padding, mode="wrap")
    v = np.pad(v, padding, mode="wrap")

    # Compute the spatial derivatives using finite differences
    u_x = np.gradient(u, dx, axis=0)
    u_y = np.gradient(u, dy, axis=1)
    u_xx = np.gradient(u_x, dx, axis=0)
    u_yy = np.gradient(u_y, dy, axis=1)
    u_xy = np.gradient(u_x, dy, axis=1)

    v_x = np.gradient(v, dx, axis=0)
    v_y = np.gradient(v, dy, axis=1)
    v_xx = np.gradient(v_x, dx, axis=0)
    v_yy = np.gradient(v_y, dy, axis=1)
    v_xy = np.gradient(v_x, dy, axis=1)

    # Remove the padding (for every axis take [1:-1])
    grads = [u, v, u_x, u_y, u_xx, u_yy, u_xy, v_x, v_y, v_xx, v_yy, v_xy]
    reshaped_grads = []
    for grad in grads:
        grad = grad[tuple([slice(padding, -padding) for _ in range(len(grad.shape))])]
        reshaped_grads.append(grad)

    u, v, u_x, u_y, u_xx, u_yy, u_xy, v_x, v_y, v_xx, v_yy, v_xy = reshaped_grads

    # Master equation:
    # u_t = (
    #     0.40 * v
    #     + 0.8 * u
    #     + 0.5 * v**3
    #     - 0.8 * u * v**2
    #     + 0.5 * u**2 * v
    #     - 0.8 * u**3
    #     + 0.1 * u_xx
    #     + 0.1 * u_yy
    # )
    # v_t = (
    #     -0.40 * u
    #     + 0.8 * v
    #     - 0.8 * v**3
    #     - 0.5 * u * v**2
    #     - 0.8 * u**2 * v
    #     - 0.5 * u**3
    #     + 0.1 * v_xx
    #     + 0.1 * v_yy
    # )

    # Mine:
    # u_t = (
    #     +0.431 * u
    #     - 0.234 * u_x * v * v_x
    #     - 0.357 * u * u_x * u_x
    #     - 0.124 * u * u_y * u_y
    #     - 0.482 * u * u * u
    #     + 0.313 * u * u * v
    #     - 0.381 * u * v * v
    #     + 0.607 * v
    #     + 0.31 * v * v * v
    # )
    # v_t = (
    #     -0.572 * u
    #     - 0.264 * u * u_y * v_y
    #     - 0.349 * u * u * u
    #     - 0.424 * u * u * v
    #     - 0.35 * u * v * v
    #     + 0.45 * v
    #     + 0.0666 * v_xx
    #     - 0.388 * v * v_y * v_y
    #     - 0.471 * v * v * v
    # )

    # new
    # u_t = (
    #     +0.33 * u
    #     - 0.355 * u_x * v * v_x
    #     - 0.38 * u * u_x * u_x
    #     + 0.0846 * u * u_yy * u_y
    #     - 0.37 * u * u * u
    #     + 0.105 * u * u * u_y
    #     - 0.363 * u * v * v
    #     + 0.0537 * u * v * v_y
    #     + 0.891 * v
    # )

    # v_t = (
    #     -0.892 * u
    #     - 0.301 * u * u_x * v_x
    #     - 0.392 * u * u_y * v_y
    #     - 0.512 * u * u * v
    #     + 0.509 * v
    #     - 0.36 * v * v_x * v_x
    #     - 0.461 * v * v_y * v_y
    #     - 0.566 * v * v * v
    # )

    # u_t = (
    #     +0.33 * u
    #     - 0.355 * u_x * v * v_x
    #     - 0.38 * u * u_x * u_x
    #     + 0.0846 * u * u_yy * u_y
    #     - 0.37 * u * u * u
    #     + 0.105 * u * u * u_y
    #     - 0.363 * u * v * v
    #     + 0.0537 * u * v * v_y
    #     + 0.891 * v
    # )

    # v_t = (
    #     -0.892 * u
    #     - 0.301 * u * u_x * v_x
    #     - 0.392 * u * u_y * v_y
    #     - 0.512 * u * u * v
    #     + 0.509 * v
    #     - 0.36 * v * v_x * v_x
    #     - 0.461 * v * v_y * v_y
    #     - 0.566 * v * v * v
    # )

    u_t = (
        +0.33 * u
        - 0.355 * u_x * v * v_x
        - 0.38 * u * u_x * u_x
        + 0.0846 * u * u_yy * u_y
        - 0.37 * u * u * u
        + 0.105 * u * u * u_y
        - 0.363 * u * v * v
        + 0.0537 * u * v * v_y
        + 0.891 * v
    )
    v_t = (
        -0.892 * u
        - 0.301 * u * u_x * v_x
        - 0.392 * u * u_y * v_y
        - 0.512 * u * u * v
        + 0.509 * v
        - 0.36 * v * v_x * v_x
        - 0.461 * v * v_y * v_y
        - 0.566 * v * v * v
    )

    # Convert the 2D array back to 1D for the ODE solver
    return u2D_to_1D(u_t, v_t)


# %%
x[:, None].shape

# %%
# Set initial conditions for u(x, y)
u_initial = initial_condition(u)
v_initial = initial_condition(v)
initial_1D = u2D_to_1D(u_initial, v_initial)
print(initial_1D.shape)

# %%
# Solve the PDE using solve_ivp
sol = solve_ivp(
    pde_system,
    (t.min(), t.max()),
    initial_1D,
    t_eval=t,
    method="RK45",
)

print(sol.status)
print(sol.message)
u_pred, v_pred = sol.y.reshape(2, Nx, Ny, -1)


# %%
Lx = x.max()
Ly = y.max()

# %%
t.shape

# %%
# dt = t[1] - t[0] / len(t)
u_t = np.gradient(u, t, axis=2)
v_t = np.gradient(v, t, axis=2)


l2_err_u = np.linalg.norm(u - u_pred) / np.linalg.norm(u)
l2_err_v = np.linalg.norm(v - v_pred) / np.linalg.norm(v)

print("L2 error for u: ", l2_err_u)
print("L2 error for v: ", l2_err_v)

u_pred_whole_t = np.gradient(u_pred, t, axis=2)

# Create a figure with two 3D axis
fig, axs = plt.subplots(2, 3, figsize=(12, 6))


# Function to update the plot
def update(time):
    x_i = data[..., 0][..., time]
    y_i = data[..., 1][..., time]

    axs[0, 0].clear()
    axs[0, 0].pcolormesh(x_i, y_i, u[..., time], cmap="viridis")

    # axs[0, 0].set_zlabel("u")
    axs[0, 0].set_title(f"u at time = {t[time]:.2f}")

    axs[0, 1].clear()
    axs[0, 1].pcolormesh(x_i, y_i, u_pred[..., time], cmap="viridis")

    # axs[0, 1].set_zlabel("u_t")
    axs[0, 1].set_title(f"u_t at time = {t[time]:.2f}")

    axs[0, 2].clear()
    axs[0, 2].pcolormesh(
        x_i,
        y_i,
        np.abs(u[..., time] - u_pred[..., time]),
        cmap="viridis",
    )

    # axs[0, 2].set_zlabel("u - u_t")
    axs[0, 2].set_title(f"u - u_t at time = {t[time]:.2f}")

    axs[1, 0].clear()
    axs[1, 0].pcolormesh(x_i, y_i, v[..., time], cmap="viridis")

    # axs[1, 0].set_zlabel("u")
    axs[1, 0].set_title(f"v at time = {t[time]:.2f}")

    axs[1, 1].clear()
    axs[1, 1].pcolormesh(x_i, y_i, v_pred[..., time], cmap="viridis")

    # axs[1, 1].set_zlabel("u_t")
    axs[1, 1].set_title(f"v_t at time = {t[time]:.2f}")

    axs[1, 2].clear()
    axs[1, 2].pcolormesh(
        x_i,
        y_i,
        np.abs(v[..., time] - v_pred[..., time]),
        cmap="viridis",
    )

    # axs[1, 2].set_zlabel("v - v_t")
    axs[1, 2].set_title(f"v - v_t at time = {t[time]:.2f}")

    # set limits for both axes
    for ax in axs.flatten():
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(x_o.min(), x_o.max())
        ax.set_ylim(y_o.min(), y_o.max())
        # ax.set_zlim(-1, 1)


# # Create the animation
anim = FuncAnimation(fig, update, frames=len(t), interval=100)

# Save the animation as a gif
anim.save("pde_solution_2_extra_script_theirs_test.gif")

# plt.show()


# fig, axs = plt.subplots(1, 2, figsize=(12, 6))
# y_value = len(y) // 2


# # Function to update the plot
# def update(time):
#     axs[0].clear()
#     axs[0].plot(x_o[..., y_value, time], u[..., y_value, time], label="u")
#     # plot solution as comparison
#     axs[0].plot(
#         x_o[..., y_value, time], u_pred_whole[..., y_value, time], label="u_pred"
#     )
#     axs[0].set_xlabel("x")
#     axs[0].set_ylabel("y")
#     axs[0].set_title(f"u at time = {t[time]:.2f} and y = {y[y_value]}")
#     axs[0].legend()
#     # grid
#     axs[0].grid(True)

#     axs[1].clear()
#     axs[1].plot(x_o[..., y_value, time], u_t[..., y_value, time], label="u_t")
#     axs[1].plot(
#         x_o[..., y_value, time], u_pred_whole_t[..., y_value, time], label="u_pred_t"
#     )
#     axs[1].set_xlabel("x")
#     axs[1].set_ylabel("y")
#     axs[1].set_title(f"u_t at time = {t[time]:.2f} and y = {y[y_value]}")
#     axs[1].legend()
#     # grid
#     axs[1].grid(True)

#     # set limits for both axes
#     for ax in axs:
#         ax.set_xlim(x_o.min(), x_o.max())
#         ax.set_ylim(-1, 1)


# # Create the animation
# anim = FuncAnimation(fig, update, frames=len(t), interval=100)

# # Save the animation as a gif
# anim.save("pde_solution_slice.gif")

# plt.show()
