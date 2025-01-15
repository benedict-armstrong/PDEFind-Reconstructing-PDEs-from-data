from typing import List, Tuple
import numpy as np
from scipy.integrate import solve_ivp
from lib.pde_find import PDEFind


class SolvePDE:
    def __init__(
        self,
        data: np.ndarray,
        ind_vars: int,
        dep_vars: int,
        pdeFind: PDEFind,
    ):
        self.pdeFind = pdeFind
        self.data = data
        self.ind_vars = ind_vars
        self.dep_vars = dep_vars

    def to_1D(self, *data):
        """
        Convert the dependent variables to 1D
        """
        return np.concatenate(data, axis=0).reshape(-1)

    def to_ND(self, data: np.ndarray) -> Tuple[np.ndarray]:
        """
        Convert the data (1, len(data)) to orgininal N-D (..., dep_vars)
        """
        reshaped_data = data.reshape((self.dep_vars,) + self.data.shape[:-2])
        return np.moveaxis(reshaped_data, 0, -1)

    def pde_rhs(
        self,
        t: float,
        u: np.ndarray,
        include_terms: Tuple[str] = None,
        coefs: List[np.ndarray] = None,
    ):
        """
        Right-hand side of the PDE to be passed to the ODE solver (solve_ivp)
        """

        data_ND = self.to_ND(u)
        space_time_grid = self.data[..., 0, : self.ind_vars]

        # concatenate the space_time_grid and data_ND on last axis
        data_ND = np.concatenate([space_time_grid, data_ND], axis=-1)

        library, _ = self.pdeFind.create_library(
            input_data=data_ND, include_terms=include_terms
        )
        n_terms = library.shape[0]
        library = library.reshape((n_terms, -1))
        out = []
        for coef in coefs:
            out.append(coef @ library)

        return self.to_1D(*out)

    def solve_pde(
        self,
        *initial_data,
        t_grid: np.ndarray,
        coefs: List[np.ndarray],
        include_terms: Tuple[str] = None,
    ):
        """
        Solve the PDE using the coefficients found by the regression
        """

        # t_grid must be 1D
        assert t_grid.ndim == 1

        initial_data_1d = self.to_1D(*initial_data)

        sol = solve_ivp(
            self.pde_rhs,
            t_span=(t_grid.min(), t_grid.max()),
            y0=initial_data_1d,
            t_eval=t_grid,
            method="RK45",
            rtol=1e-6,
            atol=1e-6,
            args=(include_terms, coefs),
        )

        if sol.status != 0:
            print(
                f"Warning: Solver did not converge after {sol.nfev} iterations. Status: {sol.status}, Message: {sol.message}"
            )

        self.solver_time_steps = len(sol.t)
        self.sol = np.stack(
            sol.y.reshape((self.dep_vars,) + self.data.shape[:-2] + (-1,)), axis=-1
        )

        return self.sol, self.solver_time_steps
