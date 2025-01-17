import numpy as np
from lib.pde_find2 import PDEFind

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

# save the coefficients to a file
np.savez("data/3_coefs.npz", u=coef_u, v=coef_v)
