import numpy as np
from lib.pde_find import PDEFind
from lib.utils.plot_1D import plot_2d, plot_3d
from lib.solve_pde import SolvePDE

PDE_NUMBER = 1

raw_data = np.load(f"src/PDE-Find: Reconstructing PDEs from data/data/{PDE_NUMBER}.npz")

u = raw_data["u"]
x = raw_data["x"]
t = raw_data["t"]

data = np.stack([x, t, u], axis=-1)
vars = (["x", "t"], ["u"])
pdefind = PDEFind(data, vars, lib_size=2, order=2)

u_t = np.gradient(u, t[0], axis=1)

library, labels = pdefind.create_library()

coef, alpha = pdefind.solve_regression(library, u_t, algorithm="tlsq")

print(pdefind.latex_string())

solver = SolvePDE(data, 2, 1, pdefind)

t_grid = data[..., 1][0]
u_0 = data[..., 0]
include_terms = pdefind.non_zero_terms(coef)

sol, time_steps = solver.solve_pde(
    u_0, t_grid=t_grid, include_terms=include_terms, coefs=[coef]
)

sol = pdefind.solve_pde(u[:, 0], t[0])

plot_2d(sol, u, "pde_1_out.png")
