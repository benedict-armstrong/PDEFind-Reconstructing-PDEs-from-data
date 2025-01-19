import numpy as np
import tqdm


def truncated_least_squares(
    X: np.ndarray,
    y: np.ndarray,
    cutoff: float = 1e-3,
    max_iterations: int = 100,
    num_term_limit: int = 1,
    verbose: bool = False,
):
    """
    Truncated Least Squares (TLSQ) algorithm for sparse regression

    Args:

    X: np.ndarray of shape (n_samples, n_features)
        The input data

    y: np.ndarray of shape (n_samples,)
        The target data

    cutoff: float, default=1e-3
        The cutoff value for the coefficients

    max_iterations: int, default=100
        The maximum number of iterations

    num_term_limit: int, default=10
        If the number of terms is the same for 2 iterations and more than the limit, raise the cutoff
        If the number of terms is the same for 2 iterations and less than the limit, return the coefficients
    """

    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]

    if verbose:
        progress = tqdm.tqdm(range(max_iterations), desc="TLSQ Iterations", position=1)
    else:
        progress = range(max_iterations)

    num_terms = []
    original_cutoff = cutoff

    for _ in progress:
        # if the number of terms is the same for 2 iterations and less than the limit, break
        if (
            len(num_terms) > 1
            and num_terms[-1] == num_terms[-2]
            and num_terms[-1] <= num_term_limit
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

        # set coefficients to zero if they are less than the cutoff
        biginds = np.abs(coeffs) > cutoff

        if not biginds.any():
            biginds = np.zeros_like(coeffs, dtype=bool)
            biginds[np.argsort(-np.abs(coeffs))[:num_term_limit]] = True
            # set the smallest coefficients to zero
            coeffs[~biginds] = 0
        else:
            coeffs[np.abs(coeffs) < cutoff] = 0

        coeffs[biginds] = np.linalg.lstsq(
            X.T[biginds].reshape((sum(biginds), -1)).T,
            y,
            rcond=None,
        )[0]

        num_terms.append(sum(np.abs(coeffs) > 0))
        if verbose:
            progress.set_postfix(
                {
                    "#nzz_terms": num_terms[-1],
                    "cutoff": cutoff,
                }
            )

    return coeffs
