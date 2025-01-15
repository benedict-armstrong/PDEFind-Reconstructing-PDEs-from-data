from typing import Dict, List, Tuple
import numpy as np
import itertools
from sklearn.linear_model import LassoCV, RidgeCV
from scipy.integrate import solve_ivp
import re
from functools import lru_cache
import tqdm
import pickle


class PDEFind:
    """
    PDE Find class to find underlying PDEs from data (based on the PySINDy paper: https://arxiv.org/pdf/1509.03580)
    """

    def __init__(
        self,
        data: np.ndarray,
        vars: Tuple[List[str], List[str]],
        lib_size: int = 3,
        order: int = 2,
        periodic: bool = False,
    ):
        """
        `data`: np.ndarray (..., number_of_independent_variables + number_of_dependent_variables)
        e.g.: u(x, t) on square domain => (number of x_points, number of time points, 3) and `vars` = ([x, t],[u])

        `vars`: Tuple(List[str], List[str]) - Tuple of independent and dependent variables
        `lib_size`: int - number of terms in the library
        `order`: int - order of the PDE
        """

        self.data = data
        self.periodic = periodic
        self.data_shape = data.shape
        self.n_vars = data.shape[-1]
        self.vars = vars
        self.ind_vars = vars[0][:]
        self.ind_vars_without_time = vars[0]
        try:
            self.ind_vars_without_time.remove("t")
        except ValueError:
            pass
        self.dep_vars = vars[1]
        self.lib_size = lib_size
        self.order = order

        if len(self.ind_vars) + len(vars[1]) != self.n_vars:
            raise ValueError(
                f"Number of variables {self.n_vars} does not match provided variables {vars}"
            )

        self.ind_var_grids = []
        for i in range(len(self.ind_vars)):
            # unsplit the data on last axis
            x = np.split(data, data.shape[-1], axis=-1)[i]
            # remove the last axis
            x = x[..., 0]
            # select 0th element of all except the ith axis
            x = x[
                tuple(
                    [0 if j != i else slice(None) for j in range(len(data.shape) - 1)]
                )
            ]
            self.ind_var_grids.append(x)

        self.time_grid = self.ind_var_grids[self.ind_vars.index("t")]

    def calculate_gradients(
        self,
        data: np.ndarray = None,
        gradients_to_calculate: Dict[str, List[str]] = None,
    ) -> List[Tuple[np.ndarray, str]]:
        """
        Calculate the gradients up to the order of the PDE for the dependent variables
        """

        if data is None:
            data = self.data

        gradients = []
        for i, dep_var in enumerate(self.dep_vars):
            if gradients_to_calculate and dep_var not in gradients_to_calculate:
                continue

            for order in range(1, self.order + 1):
                # calculate the gradients for all combinations of independent variables of lenght `order`
                for d in itertools.combinations_with_replacement(
                    self.ind_vars_without_time, order
                ):
                    d_string = "".join(sorted(d))
                    var_string = f"{dep_var}_{{{d_string}}}"
                    if gradients_to_calculate and d in gradients_to_calculate[dep_var]:
                        # Skip if the term is not in the include_terms list
                        continue

                    data_to_diff = data[
                        ..., len(self.ind_vars) + i
                    ]  # Select the dependent variable

                    for diff_by_var in d:
                        axis = self.ind_vars.index(diff_by_var)

                        diff_grid = self.ind_var_grids[axis]
                        if self.periodic:
                            padding = 3

                            # If the grid is periodic, pad the grid and data
                            diff_grid = np.pad(diff_grid, padding, mode="wrap")
                            data_to_diff = np.pad(data_to_diff, padding, mode="wrap")

                            data_to_diff = np.gradient(
                                data_to_diff, diff_grid, axis=axis
                            )

                            # Remove the padding (for every axis take [1:-1])
                            data_to_diff = data_to_diff[
                                tuple(
                                    [
                                        slice(padding, -padding)
                                        for _ in range(len(data_to_diff.shape))
                                    ]
                                )
                            ]

                        else:
                            data_to_diff = np.gradient(
                                data_to_diff, diff_grid, axis=axis
                            )

                    gradients.append((data_to_diff, var_string))

        return gradients

    @lru_cache(maxsize=10)
    def gradients_to_include(self, include_terms: Tuple[str]):
        """
        Parse the include_terms list to extract the variables and orders to include in the library
        """
        gradients_to_include = {}
        for term in include_terms:
            # terms are in the form "u_{xyx}u_{xy}u" for example
            # use regex to extract the variables and orders
            matches = re.findall(r"([a-z])_{([a-z]+)}", term)
            for var, ds in matches:
                if var not in gradients_to_include:
                    gradients_to_include[var] = []
                gradients_to_include[var].append(ds)

        return gradients_to_include

    def create_library(
        self,
        input_data: np.ndarray = None,
        include_terms: List[str] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Create a library of candidate functions for the PDE
        """

        data = input_data

        if data is None:
            data = self.data

        elements = []

        if not include_terms or "1" in include_terms:
            elements.append((np.ones(data.shape[:-1]), "1"))

        for i, var in enumerate(self.dep_vars):
            if not include_terms or var in include_terms:
                elements.append((data[..., len(self.ind_vars) + i], var))

        gradients_to_calculate = None
        if include_terms:
            gradients_to_calculate = self.gradients_to_include(include_terms)

        elements += self.calculate_gradients(
            data, gradients_to_calculate=gradients_to_calculate
        )

        lib = {}

        if hasattr(self, "labels"):
            for label in self.labels:
                lib[label] = None

        for pairs in itertools.product(elements, repeat=self.lib_size):
            label = "".join(sorted([p[1] for p in pairs])).replace("1", "")
            if label == "":
                label = "1"
            if label in lib and lib[label] is not None:
                continue

            if include_terms and label not in include_terms:
                lib[label] = np.zeros(data.shape[:-1])
                continue

            term = [p[0] for p in pairs]
            lib[label] = np.prod(term, axis=0)

        if not hasattr(self, "labels"):
            self.labels = sorted(lib.keys())

        for label in self.labels:
            if lib[label] is None:
                lib[label] = np.zeros(data.shape[:-1])

        return np.stack([lib[label] for label in self.labels]), self.labels

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
        print(f"Library size: {library.size}, Target size: {target.size}")

        if algorithm == "lasso":
            library = library.reshape((n_terms, -1)).T
            target = target.flatten()

            reg = LassoCV(cv=5, max_iter=int(1e9), fit_intercept=False, alphas=alphas)

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

            X = np.linalg.lstsq(
                library.reshape((n_terms, -1)).T, target_flat, rcond=None
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
                        "# of non-zero terms": num_terms[-1],
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

    def save(self, filename: str):
        """
        Save the PDE and coefficients and terms to a file using pickle
        """

        with open(filename, "wb") as f:
            pickle.dump(
                {
                    "coef": self.coef,
                    "labels": self.labels,
                    "ind_vars": self.ind_vars,
                    "dep_vars": self.dep_vars,
                },
                f,
            )

    def load(self, filename: str):
        """
        Load the PDE and coefficients and terms from a file using pickle
        """

        with open(filename, "rb") as f:
            data = pickle.load(f)

    def non_zero_terms(self, coef) -> Tuple[str]:
        """
        Return the non-zero terms in the PDE
        """
        return tuple([term for term, c in zip(self.labels, coef) if c != 0])

    def latex_string(self, coefs: np.ndarray = None, var: str = None) -> str:
        """
        Return LaTeX string version of found PDE
        """
        if not hasattr(self, "coef") and coefs is None:
            raise ValueError("PDE coefficients not found")

        if coefs is None:
            coefs = self.coef

        if var is None:
            var = self.dep_vars[0]

        out_str = rf"\frac{{\partial {{{var}}}}}{{\partial {{t}}}}="
        for term, c in zip(self.labels, coefs):
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

    def python_string(self, coefs: np.ndarray = None, var: str = None) -> str:
        """
        Return Python string version of found PDE
        """
        if not hasattr(self, "coef") and coefs is None:
            raise ValueError("PDE coefficients not found")

        if coefs is None:
            coefs = self.coef

        if var is None:
            var = self.dep_vars[0]

        out_str = f"d{var}/dt = "
        for term, c in zip(self.labels, coefs):
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
