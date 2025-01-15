import numpy as np
from lib.pde_find import PDEFind
from lib.solve_pde import SolvePDE
from utils.plot_2D import plot

PDE_NUMBER = 3

raw_data = np.load(f"src/PDE-Find: Reconstructing PDEs from data/data/{PDE_NUMBER}.npz")

u = raw_data["u"]
v = raw_data["v"]
x = raw_data["x"]
y = raw_data["y"]
t = raw_data["t"]

data = np.stack([x, y, t, u, v], axis=-1)

# downsample the data
data = data[::4, ::4, ::2]

vars = (["x", "y", "t"], ["u", "v"])
pdefind = PDEFind(data, vars, lib_size=3, order=2, periodic=True)

library, labels = pdefind.create_library()

print(f"Library size: {len(labels)}")
print(f"Library terms: {labels}")

u_t = np.gradient(data[..., 3], data[..., 2][0, 0], axis=-1)
v_t = np.gradient(data[..., 4], data[..., 2][0, 0], axis=-1)

algorithm = "tlsq"
cutoff = 1e-2
iterations = 500
coef_u, alpha_u = pdefind.solve_regression(
    library,
    u_t,
    algorithm=algorithm,
    cutoff=cutoff,
    iterations=iterations,
)

coef_v, alpha_v = pdefind.solve_regression(
    library,
    v_t,
    algorithm=algorithm,
    cutoff=cutoff,
    iterations=iterations,
)

print(pdefind.latex_string(coef_u, "u"))
print(pdefind.latex_string(coef_v, "v"))

print(pdefind.python_string(coef_u, "u"))
print(pdefind.python_string(coef_v, "v"))

include_terms1 = pdefind.non_zero_terms(coef_u)
include_terms2 = pdefind.non_zero_terms(coef_v)

include_terms = tuple(set(include_terms1 + include_terms2))

u_0 = data[..., 3][:, :, 0]
v_0 = data[..., 4][:, :, 0]

solver = SolvePDE(data, 3, 2, pdefind)

t_grid = data[..., 2][0, 0]

sol, time_steps = solver.solve_pde(
    u_0, v_0, t_grid=t_grid, include_terms=include_terms, coefs=[coef_u, coef_v]
)

print(f"Time steps: {time_steps}")

plot(sol, data, "test_pde3.gif")
