import numpy as np
from lib.pde_find2 import PDEFind
from lib.utils.plot_1D import plot_2d, plot_3d
from lib.solve_pde import SolvePDE

PDE_NUMBER = 1

raw_data = np.load("data/1.npz")

u = raw_data["u"]
x = raw_data["x"]
t = raw_data["t"]

pdefind = PDEFind(var_labels=(["x", "t"], ["u"]), polynomial_degree=2, order=2)

pdefind.add_grid(x, t)

u_t = np.gradient(u, t[0], axis=1)

library, labels = pdefind.create_library(u)

coef, alpha = pdefind.solve_regression(library, u_t, algorithm="tlsq")

print(pdefind.latex_string(coef, labels, "u"))

# solver = SolvePDE(data, 2, 1, pdefind)

# t_grid = data[..., 1][0]
# u_0 = data[..., 0]
# include_terms = pdefind.non_zero_terms(coef)

# sol, time_steps = solver.solve_pde(
#     u_0, t_grid=t_grid, include_terms=include_terms, coefs=[coef]
# )

# sol = pdefind.solve_pde(u[:, 0], t[0])

# plot_2d(sol, u, "pde_1_out.png")
