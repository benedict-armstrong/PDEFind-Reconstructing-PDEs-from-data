import numpy as np
from lib.pde_find2 import PDEFind
from lib.solve_pde import SolvePDE
from lib.utils.plot_2D import plot

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

u, v, x, y, t = pdefind.subsample_data(u, v, x, y, t, factors=[3, 3, 2])

pdefind.add_grid(x, y, t)

term_regex = r"^(u|v)*(_{(x|y)+})?(v)*$"

library, labels = pdefind.create_library(u, v, term_regex=term_regex)

print(f"Library size: {len(labels)}")
print(f"Library: {labels}")

u_t = np.gradient(u, pdefind.time_grid, axis=-1)
v_t = np.gradient(v, pdefind.time_grid, axis=-1)

algorithm = "tlsq"
cutoff = 1e-4
iterations = 500
max_terms = 10

coef_u, alpha_u = pdefind.solve_regression(
    library,
    u_t,
    algorithm=algorithm,
    cutoff=cutoff,
    iterations=iterations,
    num_term_limit=max_terms,
)

print(pdefind.latex_string(coef_u, labels, "u"))
print(pdefind.python_string(coef_u, labels, "u"))


coef_v, alpha_v = pdefind.solve_regression(
    library,
    v_t,
    algorithm=algorithm,
    cutoff=cutoff,
    iterations=iterations,
    num_term_limit=max_terms,
)

print(pdefind.latex_string(coef_v, labels, "v"))
print(pdefind.python_string(coef_v, labels, "v"))

# include_terms1 = pdefind.non_zero_terms(coef_u)
# include_terms2 = pdefind.non_zero_terms(coef_v)

# include_terms = tuple(set(include_terms1 + include_terms2))

# u_0 = data[..., 3][:, :, 0]
# v_0 = data[..., 4][:, :, 0]

# solver = SolvePDE(data, 3, 2, pdefind)

# t_grid = data[..., 2][0, 0]

# sol, time_steps = solver.solve_pde(
#     u_0, v_0, t_grid=t_grid, include_terms=include_terms, coefs=[coef_u, coef_v]
# )

# print(f"Time steps: {time_steps}")

# plot(sol, data, "test_pde3.gif")
