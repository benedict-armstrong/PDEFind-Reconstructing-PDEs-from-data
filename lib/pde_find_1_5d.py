from typing import List, Tuple
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from scipy.integrate import solve_ivp


class PDEFind:
    """
    PDE Find class to find underlying PDEs from data (loosly based on the PySINDy paper: https://arxiv.org/pdf/1509.03580
    """

    def __init__(
        self,
        u: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        vars: List[str] = None,
        lib_size: int = 3,
        order: int = 2,
    ):
        """ """
        self.u = u
        self.x = x
        self.y = y
        self.vars = ["u", "x", "t"] if vars is None else vars
        self.lib_size = lib_size

    def create_library(self, u: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Create a library of candidate functions for the PDE
        """

        u_x = np.gradient(u, self.x[:, 0], axis=0)
        u_xx = np.gradient(u_x, self.x[:, 0], axis=0)
        u_xxx = np.gradient(u_xx, self.x[:, 0], axis=0)

        elements = [
            (np.ones(u.shape), "1"),
            (u, self.vars[0]),
            (u_x, f"{self.vars[0]}_{{{self.vars[1]}}}"),
            (u_xx, f"{self.vars[0]}_{{{self.vars[1] * 2}}}"),
            (u_xxx, f"{self.vars[0]}_{{{self.vars[1] * 3}}}"),
        ]

        library = []
        labels = []
        for pairs in itertools.product(elements, repeat=self.lib_size):
            terms = [p[0] for p in pairs]
            label = "".join(sorted([p[1] for p in pairs])).replace("1", "")
            if label == "":
                label = "1"
            if label in labels:
                continue
            library.append(np.prod(terms, axis=0))
            labels.append(label)

        self.library = library
        self.labels = labels

        return np.array(library), labels

    def solve_regression(
        self,
        library: np.ndarray,
        target: np.ndarray,
        cutoff: float = 1e-2,
        algorithm: str = "lasso",
    ):
        """ """

        n_terms = library.shape[0]

        if algorithm == "lasso":
            library = library.reshape((n_terms, -1)).T
            target = target.flatten()

            reg = LassoCV(
                cv=5, max_iter=int(1e9), fit_intercept=False, alphas=[1e-5, 1e-6]
            )

            reg.fit(library, target)
            self.coef = reg.coef_
            self.alpha = reg.alpha_
        elif algorithm == "tlsq":
            library_flat = library.reshape((n_terms, -1), copy=True).T
            target_flat = target.flatten()

            X = np.linalg.lstsq(library_flat, target_flat, rcond=None)[0]

            for _ in range(100):
                X[np.abs(X) < cutoff] = 0
                biginds = np.abs(X) > 0
                new_lib = library[biginds].reshape((sum(biginds), -1), copy=True).T

                X[biginds] = np.linalg.lstsq(
                    new_lib,
                    target_flat,
                    rcond=None,
                )[0]

            self.coef = X
            self.alpha = None
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # apply a cutoff to the coefficients
        self.coef[np.abs(self.coef) < cutoff] = 0

        return self.coef, self.alpha

    def pde_rhs(self, t, u):
        """
        Right-hand side of the PDE to be passed to the ODE solver (solve_ivp)
        """
        library, _ = self.create_library(u)
        lib = np.array([term.flatten() for term in library])

        return self.coef @ lib

    def solve_pde(self, u0, t_eval):
        """
        Solve the PDE using the coefficients found by the regression
        """
        sol = solve_ivp(
            self.pde_rhs,
            t_span=(t_eval[0], t_eval[-1]),
            y0=u0,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-6,
            atol=1e-6,
        )

        if sol.status != 0:
            print(
                f"Warning: Solver did not converge. Status: {sol.status}, Message: {sol.message}"
            )

        self.sol = sol.y

        return sol.y

    def verify_pde_rel_l2(self):
        """
        Verify the solution by comparing it to the true solution using the relative L2 norm
        """
        if not hasattr(self, "sol"):
            raise
        error = np.linalg.norm(self.u - self.sol) / np.linalg.norm(self.u)
        return error

    def latex_string(self) -> str:
        """
        Return LaTeX string version of found PDE
        """
        out_str = rf"""\frac{{
            \partial {{{self.vars[0]}}}
        }}{{
            \partial {{{self.vars[2]}}}
        }}="""
        for term, coef in zip(self.labels, self.coef):
            if coef == 0:
                continue

            if coef > 0:
                if out_str[-1] == "=":
                    coef = f"{coef:.3}"
                else:
                    coef = f"+{coef:.3}"
            else:
                coef = f"{coef:.3}"
            out_str += f"{coef}{term}"

        return out_str.strip()
