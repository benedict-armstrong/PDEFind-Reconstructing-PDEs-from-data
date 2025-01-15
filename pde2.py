import numpy as np
import matplotlib.pyplot as plt
from lib.pde_find import PDEFind
from lib.solve_pde import SolvePDE
from utils.plot_1D import plot_2d

PDE_NUMBER = 2

raw_data = np.load(f"src/PDE-Find: Reconstructing PDEs from data/data/{PDE_NUMBER}.npz")

u = raw_data["u"]
x = raw_data["x"]
t = raw_data["t"]

data = np.stack([x, t, u], axis=-1)
vars = (["x", "t"], ["u"])
pdefind = PDEFind(data, vars, lib_size=2, order=2)

library, labels = pdefind.create_library()

print(f"Library size: {len(labels)}")
print(f"Library terms: {labels}")

u_t = np.gradient(data[..., 2], data[..., 2][0, 0], axis=-1)
algorithm = "lasso"
cutoff = 1e-2
iterations = 100
coef_u, alpha_u = pdefind.solve_regression(
    library,
    u_t,
    algorithm=algorithm,
    cutoff=cutoff,
    iterations=iterations,
)

print(pdefind.latex_string(coef_u, "u"))
# print(pdefind.python_string(coef_u, "u"))


include_terms = pdefind.non_zero_terms(coef_u)


u_0 = u[:, 0]

solver = SolvePDE(data, 2, 1, pdefind)

t_grid = pdefind.time_grid

sol = solver.solve_pde(u_0, t_grid=t_grid, include_terms=include_terms, coefs=[coef_u])

print(sol.shape)
