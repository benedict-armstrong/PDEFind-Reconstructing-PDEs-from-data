from typing import Any, Dict, Iterable, List, Tuple
import numpy as np
import itertools
from sklearn.linear_model import LassoCV, RidgeCV
from scipy.integrate import solve_ivp
import re
from functools import lru_cache
import tqdm


class PDEFind:
    """
    PDE Find class to find underlying PDEs from data (based on the PySINDy paper: https://arxiv.org/pdf/1509.03580)
    """

    def __init__(
        self,
        var_labels: Tuple[List[str], List[str]],
        polynomial_degree: int = 3,
        order: int = 2,
        periodic: bool = False,
    ):
        """
        `data`: np.ndarray (..., number_of_independent_variables + number_of_dependent_variables)
        e.g.: u(x, t) on square domain => (number of x_points, number of time points, 3) and `vars` = ([x, t],[u])

        `vars`: List[str] List of independent variables
        `polynomial_degree`: int - polynomial degree of the terms use in the library
        `order`: int - order of the PDE
        """

        # set configuration
        self.polynomial_degree = polynomial_degree
        self.order = order
        self.var_labels = var_labels
        self.periodic = periodic

        self.indep_vars_labels = var_labels[0]
        self.indep_vars_labels_without_time = var_labels[0][:]
        self.indep_vars_labels_without_time.remove("t")
        self.dep_vars_labels = var_labels[1]

        # get the number of independent and dependent variables
        self.num_indep_vars = len(self.indep_vars_labels)
        self.num_dep_vars = len(self.dep_vars_labels)

        self.time_index = self.indep_vars_labels.index("t")

    def add_grid(self, *independent_vars: np.ndarray):
        """ """
        self.data_shape = independent_vars[0].shape
        self.ind_var_grids = []
        for i in range(len(independent_vars)):
            x = independent_vars[i][
                tuple(
                    [0 if j != i else slice(None) for j in range(self.num_indep_vars)]
                )
            ]
            self.ind_var_grids.append(x)

        self.time_grid = self.ind_var_grids[self.time_index]

        return self.ind_var_grids

    @lru_cache(maxsize=10)
    def gradients_to_include(self, include_terms: Tuple[str]) -> Dict[str, List[str]]:
        """
        Parse the include_terms list to extract the variables and orders to include in the library
        """
        gradients_to_include: Dict[str, List[str]] = {}
        for term in include_terms:
            # terms are of the form "u_{xyx}u_{xy}u" for example
            # use regex to extract the variables and orders
            matches = re.findall(r"([a-z])_{([a-z]+)}", term)
            for var, ds in matches:
                if var not in gradients_to_include:
                    gradients_to_include[var] = []
                gradients_to_include[var].append(ds)

        return gradients_to_include

    def calculate_gradients(
        self,
        dependent_vars: Iterable[np.ndarray],
        gradients_to_calculate: Dict[str, List[str]] = None,
    ) -> List[Tuple[np.ndarray, str]]:
        """
        Calculate the gradients up to the order of the PDE for the dependent variables

        Args:
            `independent_vars`: List of independent variables (must be in same order as labels passed to class)
            dependent_vars: List of dependent variables
            gradients_to_calculate: Dict of gradients to calculate for each dependent variable e.g. {"u": ["x", "y"]}

        Returns:
            List of tuples containing the gradient and the variable string
        """

        if not hasattr(self, "ind_var_grids"):
            raise ValueError("Grids for the independent variables must be set first")

        gradients = []
        for dep_var, dep_var_label in zip(dependent_vars, self.dep_vars_labels):
            # skip if the gradient is not in the include_terms list
            if gradients_to_calculate and dep_var_label not in gradients_to_calculate:
                continue

            # calculate the gradients for all combinations of independent variables of length `order`
            for order in range(1, self.order + 1):
                for d in itertools.combinations_with_replacement(
                    self.indep_vars_labels_without_time, order
                ):
                    d_string = "".join(sorted(d))
                    gradient_label = f"{dep_var_label}_{{{d_string}}}"
                    if (
                        gradients_to_calculate
                        and gradient_label in gradients_to_calculate[dep_var_label]
                    ):
                        # Skip if the term is not in the include_terms list
                        continue

                    data_to_diff = dep_var

                    for diff_by_var_label in d:
                        axis = self.indep_vars_labels.index(diff_by_var_label)
                        diff_grid = self.ind_var_grids[axis]

                        if self.periodic:
                            # If the grid is periodic, pad the grid and data
                            diff_grid = np.concatenate(
                                [
                                    [diff_grid[1] - diff_grid[0]],
                                    diff_grid,
                                    [diff_grid[-1] - diff_grid[-2]],
                                ]
                            )
                            data_to_diff = np.pad(data_to_diff, 1, mode="wrap")

                            data_to_diff = np.gradient(
                                data_to_diff, diff_grid, axis=axis
                            )

                            # Remove the padding (for every axis take [1:-1])
                            data_to_diff = data_to_diff[
                                tuple(
                                    [
                                        slice(1, -1)
                                        for _ in range(len(data_to_diff.shape))
                                    ]
                                )
                            ]

                        else:
                            data_to_diff = np.gradient(
                                data_to_diff, diff_grid, axis=axis
                            )

                    gradients.append((data_to_diff, gradient_label))

        return gradients

    def create_library(
        self,
        *dependent_vars: np.ndarray,
        include_terms: Tuple[str] = None,
        term_regex: str = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Create a library of candidate functions for the PDE
        """

        dep_var_shape = dependent_vars[0].shape

        elements = []

        elements.append((np.ones(dep_var_shape), "1"))

        for dep_var_label, dep_var in zip(self.dep_vars_labels, dependent_vars):
            elements.append((dep_var, dep_var_label))

        gradients_to_calculate = None
        if include_terms:
            gradients_to_calculate = self.gradients_to_include(tuple(include_terms))

        elements += self.calculate_gradients(
            dependent_vars=dependent_vars,
            gradients_to_calculate=gradients_to_calculate,
        )

        lib = {}

        for pairs in itertools.product(elements, repeat=self.polynomial_degree):
            label = "".join(sorted([p[1] for p in pairs])).replace("1", "")
            if label == "":
                label = "1"

            # Skip if the label is already in the library (duplicates)
            if label in lib:
                continue

            if term_regex and not re.match(term_regex, label):
                continue

            # terms not in include_terms are set to zero
            if include_terms and label not in include_terms:
                continue

            lib[label] = np.prod([p[0] for p in pairs], axis=0)

        labels = sorted(lib.keys())

        return np.stack([lib[label] for label in labels]), labels

    def solve_regression(
        self,
        library: np.ndarray,
        target: np.ndarray,
        cutoff: float = 1e-2,
        iterations: int = 50,
        algorithm: str = "lasso",
        num_term_limit=10,
        alphas=[1e-6, 1e-5, 1e-4, 1e-3],
    ):
        """
        Solve regression problem to find the coefficients of the PDE
        """

        n_terms = library.shape[0]

        print(f"Solving using {algorithm} regression")
        print(f"Library size: {library.shape}, Target size: {target.size}")

        if algorithm == "lasso":
            library = library.reshape((n_terms, -1)).T
            target = target.flatten()

            reg = LassoCV(
                cv=5, max_iter=int(1e9), fit_intercept=False, alphas=[1e-6, 1e-5]
            )

            reg.fit(library, target)
            self.coef = reg.coef_
            self.alpha = reg.alpha_
        elif algorithm == "ridge":
            library = library.reshape((n_terms, -1)).T
            target = target.flatten()

            reg = RidgeCV(cv=5, fit_intercept=False, alphas=alphas)

            reg.fit(library, target)

            self.coef = reg.coef_
            self.alpha = reg.alpha_

        elif algorithm == "tlsq":
            target_flat = target.flatten()

            library

            X = np.linalg.lstsq(
                library.reshape((n_terms, -1)).T, target.flatten(), rcond=None
            )[0]

            progress = tqdm.tqdm(range(iterations), desc="TLSQ Iterations")

            num_terms = []
            original_cutoff = cutoff

            for _ in progress:
                # if the number of terms is the same for 2 iterations and less than the limit, break
                if (
                    len(num_terms) > 1
                    and num_terms[-1] == num_terms[-2]
                    and num_terms[-1] < num_term_limit
                ):
                    break

                # if the number of terms is the same for 2 iterations and more than the limit, raise cutoff
                if (
                    len(num_terms) > 1
                    and num_terms[-1] == num_terms[-2]
                    and num_terms[-1] >= num_term_limit
                ):
                    cutoff *= 2

                # if the two consecutive iterations have different number of terms, reset the limit
                if len(num_terms) > 1 and num_terms[-1] != num_terms[-2]:
                    cutoff = original_cutoff

                X[np.abs(X) < cutoff] = 0
                biginds = np.abs(X) > 0
                new_lib = library[biginds].reshape((sum(biginds), -1), copy=True).T

                X[biginds] = np.linalg.lstsq(
                    new_lib,
                    target_flat,
                    rcond=None,
                )[0]

                num_terms.append(sum(np.abs(X) > 0))
                progress.set_postfix(
                    {
                        "#nzz_terms": num_terms[-1],
                        "cutoff": cutoff,
                    }
                )

            self.coef = X
            self.alpha = None
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # apply a cutoff to the coefficients
        if algorithm != "tlsq":
            self.coef[np.abs(self.coef) < cutoff] = 0

        # print the non-zero terms
        print(f"# of non-zero terms: {sum(np.abs(self.coef) > 0)}")

        return self.coef, self.alpha

    def pde_rhs(
        self,
        t: Any,
        u1D: np.ndarray,
        terms_to_include: Tuple[str],
        coeffs: np.ndarray,
    ):
        dep_vars = self.u1D_to_ND(u1D)

        library, _ = self.create_library(*dep_vars, include_terms=terms_to_include)

        out = []
        for coef in coeffs:
            out.append((library.T @ coef).T)

        # Convert the 2D array back to 1D for the ODE solver
        return self.uND_to_1D(*out)

    def uND_to_1D(self, *data: np.ndarray):
        return np.concatenate(data, axis=0).reshape(-1)

    def u1D_to_ND(self, u1D: np.ndarray):
        return u1D.reshape((self.num_dep_vars,) + self.data_shape[:-1])

    def solve(
        self,
        initial_condition: Iterable[np.ndarray],
        coeffs: Iterable[np.ndarray],
        labels: List[str],
    ):
        nnz_terms = self.non_zero_terms(coeffs, labels)
        nnz_coeffs = self.condense_coeffs(coeffs, labels)
        initial_1D = self.uND_to_1D(*initial_condition)

        sol = solve_ivp(
            self.pde_rhs,
            (self.time_grid.min(), self.time_grid.max()),
            initial_1D,
            t_eval=self.time_grid,
            method="RK45",
            args=(nnz_terms, nnz_coeffs),
        )

        print(sol.status)
        print(sol.message)

        return sol.y.reshape((self.num_dep_vars,) + self.data_shape[:-1] + (-1,))

    @staticmethod
    def subsample_data(
        *arrays: np.ndarray, factors: List[int], random: bool = False
    ) -> Tuple[np.ndarray]:
        """
        Subsample the data by a factor of `factor` in each dimension
        """

        shape = arrays[0].shape

        if len(shape) != len(factors):
            raise ValueError(
                "The number of dimensions of the data and the number of factors must be the same"
            )

        for dim, factor in enumerate(factors):
            if factor != 1:
                if random:
                    number_of_ids = shape[dim] // factor
                    idx = np.random.choice(
                        np.arange(shape[dim]), number_of_ids, replace=False
                    )
                    arrays = [array.take(idx, axis=dim) for array in arrays]
                else:
                    arrays = [
                        array[tuple([slice(None)] * dim + [slice(None, None, factor)])]
                        for array in arrays
                    ]

        return tuple(arrays)

    @staticmethod
    def latex_string(coefs: np.ndarray, labels: List[str], var: str) -> str:
        """
        Return LaTeX string version of found PDE
        """

        out_str = rf"\frac{{\partial {{{var}}}}}{{\partial {{t}}}}="
        for term, c in zip(labels, coefs):
            if c == 0:
                continue

            if c > 0:
                if out_str[-1] == "=":
                    c = f"{c:.3}"
                else:
                    c = f"+{c:.3}"
            else:
                c = f"{c:.3}"
            out_str += f"{c}{term}"

        return out_str.strip()

    @staticmethod
    def python_string(coefs: np.ndarray, labels: List[str], var: str) -> str:
        """
        Return Python string version of found PDE
        """

        out_str = f"d{var}/dt = "
        for term, c in zip(labels, coefs):
            if c == 0:
                continue

            if c > 0:
                if out_str[-1] == "=":
                    c = f"{c:.3}"
                else:
                    c = f"+{c:.3}"
            else:
                c = f"{c:.3}"

            # use regex to extract variables from the term
            matches = re.findall(r"([a-z])(_{[a-z]+})?", term)

            # join using "*"
            term = "*".join([f"{m[0]}{m[1] if m[1][2:-1] else ''}" for m in matches])

            out_str += f"{c}*{term}"

        return out_str.strip().replace("{", "").replace("}", "")

    @staticmethod
    def non_zero_terms(coefs: Iterable[np.ndarray], labels: List[str]) -> List[str]:
        """
        Return the non-zero terms of the PDE for each set of coefficients
        """
        terms = set()
        for c in coefs:
            t = [labels[i] for i in np.where(c != 0)[0]]
            terms.update(t)

        return sorted(list(terms))

    @staticmethod
    def condense_coeffs(
        coefs: Iterable[np.ndarray], labels: List[str]
    ) -> tuple[np.ndarray]:
        """
        Return the non-zero coefficients of the PDE
        """

        non_zero_terms = PDEFind.non_zero_terms(coefs, labels)

        new_coefs = []

        for c in coefs:
            condensed = []
            for c, term in zip(c, labels):
                if term in non_zero_terms:
                    condensed.append(c)

            new_coefs.append(np.array(condensed))

        return tuple(new_coefs)


if __name__ == "__main__":
    PDE_NUMBER = 1

    raw_data = np.load("data/1.npz")

    u = raw_data["u"]
    x = raw_data["x"]
    t = raw_data["t"]

    pdefind = PDEFind(var_labels=(["x", "t"], ["u"]), polynomial_degree=2, order=2)

    pdefind.add_grid(x, t)

    library, labels = pdefind.create_library(u)

    u_t = np.gradient(u, t[0], axis=1)

    coef, alpha = pdefind.solve_regression(library, u_t, algorithm="tlsq")

    print(pdefind.latex_string(coef, labels, "u"))
