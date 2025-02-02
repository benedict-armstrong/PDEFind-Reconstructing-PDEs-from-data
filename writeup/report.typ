#import "@preview/lovelace:0.3.0": *

#set page(
  paper: "a4",
  numbering: "1",
  columns: 2,
)

#place(
  top + center,
  float: true,
  scope: "parent",
  clearance: 3em,
)[
  // Add eth logo
  #align(center, image("figures/eth_logo.png", height: 5em))

  #align(
    center,
    text(20pt)[
      PDE-Find using Truncated Least Squares with dynamic
      thresholding
    ],
  )

  #align(
    center,
    text(14pt)[
      Benedict Armstrong \
      benedict.armstrong\@inf.ethz.ch
    ],
  )

  #align(
    center,
    text(14pt)[
      December 2024
    ],
  )

]

#set text(size: 11.5pt)
#set par(justify: true)
#show link: underline

= Introduction

As part of this year's AI in the Sciences and Engineering course, I replicated and extended PDE-Find, as described in the paper by Rudy et al. @rudy2017data. The implementation is structured as a library, capable not only of identifying partial differential equations but also of solving them using SciPy's `solve_ivp` to verify the discovered equations. The notebooks, library code, and scripts for generating all plots and figures are available in #link("https://github.com/benedict-armstrong/PDEFind-Reconstructing-PDEs-from-data","this repository").

#let body = [
  = Implementation

  The first step involved constructing a library of potential PDE terms. The main challenge was accurately estimating derivatives of the solution. Initially, I implemented a finite difference method but later transitioned to using `numpy`'s `np.gradient` for more efficient and reliable derivative estimation.

  For solving the sparse linear system, I implemented a Truncated Least Squares with Dynamic Thresholding (TLS-DT) algorithm, based on the STRidge method detailed in the PDE-FIND paper @rudy2017data.

  The implementation, structured across three Jupyter notebooks (`pde1.ipynb`, `pde2.ipynb`, and `pde3.ipynb`), handles each dataset individually. Each file also contains a more detailed analysis for each problem. The core components for the Library, including code for generating the library of terms and the TLS-DT algorithm, are encapsulated in `lib/pde_find.py` and `lib/tlsq.py`.

  == Truncated Least Squares with dynamic thresholding (TLS-DT) <tls_dt_algorithm_section>

  The TLS-DT algorithm extends STRidge @rudy2017data by dynamically adjusting the threshold for coefficient selection. The threshold is raised if the number of non-zero coefficients stays the same for a number of iterations and the coefficient contains more non-zero terms than allowed. This prevents the algorithm from becoming trapped in local minima, ensuring a more robust selection of sparse terms in the PDE. The full algorithm is outlined @tls_dt_algorithm.

  = Results

  The accuracy of the reconstructed PDEs was validated by comparing solutions generated using `scipy.solve_ivp` with the original data. The results for each PDE are summarized below.

  == PDE 1

  Initial inspection suggested a form resembling Burgers' equation. The term library included all second-order polynomial combinations of $u$ and it's 1st, 2nd and 3rd order derivatives (see @pde1_terms). The algorithm converged rapidly to @pde1_eq.

  #figure(
    $
      partial(u) / partial(t) = -0.102u_("x""x") - 0.997 u u_("x")
    $,
    caption: "Reconstructed solution for PDE 1",
    // scope: "parent",
    // placement: bottom,
    supplement: "Equation",
    kind: "math",
  ) <pde1_eq>

  The relative $L_2$ error across the domain, of the reconstructed solution, compared to the given data was $5.304e^(-3)$. The reconstructed solution closely matched the reference data, as shown in @pde1_fig.

  #figure(
    image("../figures/pde1_2d.png", width: 120%),
    caption: "Reconstructed solution for PDE 1",
    scope: "parent",
    placement: bottom,
  ) <pde1_fig>

  == PDE 2

  For dataset 2 the PDE was not immediately obvious (at least to the untrained eye). Using a similar term library (see @pde2_terms),the algorithm initially identified a four-term solution resembling the _Korteweg-de Vries_ equation. Refinement with adjusted `max_terms` value yields @pde2_eq, a two term solution with only a slight increase in the relative $L_2$ error and more desired sparsity.

  #figure(
    $
      partial(u) / partial(t) = - 1.02u_("x""x""x") -6.04"u"u_x
    $,
    caption: "Reconstructed solution for PDE 2",
    // scope: "parent",
    // placement: bottom,
    supplement: "Equation",
    kind: "math",
  ) <pde2_eq>

  The relative $L_2$ error was $6.023e^(-2)$. The reconstructed solution again closely matched the reference data, as shown in @pde2_fig.

  #figure(
    image("../figures/pde2_2d.png", width: 120%),
    caption: "Reconstructed solution for PDE 2",
    scope: "parent",
    placement: bottom,
  ) <pde2_fig>

  == PDE 3

  Dataset 3 presented a convection-diffusion equation in two dimensions. Due to dataset size, spatial and temporal dimensions were scaled by a factor of 0.5. The term library was restricted to third-order polynomial terms of $u$ and $v$ with a maximum of one derivative of any order (see @pde3_terms). The algorithm converged to @pde3_eq with a relative $L_2$ error of $4.173e^(-1)$ for both $u$ and $v$. This error is significantly higher than for the previous two PDEs, but might be due to the larger domain and higher order terms. The hyperparameter used were: `max_terms`=10 and `cutoff`=$1e^(-4)$.


  #figure(
    $
      partial(u) / partial(t) = 0.828u +0.251v -0.807"u"^3 +0.71"v"^3 +0.711"u"^2"v" -0.809"u""v"^2 +0.106u_("x""x") +0.103u_("y""y") \ partial(v) / partial(t) = -0.251u +0.828v -0.71"u"^3 -0.807"v"^3 -0.809"u"^2"v" -0.711"u""v"^2 +0.103v_("x""x") +0.106v_("y""y")
    $,
    caption: "Reconstructed solution for PDE 3",
    scope: "parent",
    placement: bottom,
    supplement: "Equation",
    kind: "math",
  ) <pde3_eq>

  A visualization (`pde3.gif`) over time is available in the `figures` folder.

  // Table with errors
  #figure(
    table(
      columns: 2,
      stroke: (x: none),
      row-gutter: (2.5pt, auto),
      table.header[PDE][Relative $L_2$ error],
      [PDE 1], [$5.304e^(-3)$],
      [PDE 2], [$6.023e^(-2)$],
      [PDE 3], [$4.173e^(-1)$],
    ),
    caption: [Relative $L_2$ error for each PDE],
  ) <pde_errors>

  = Conclusion & Improvements
  The PDE-FIND implementation successfully identified plausible PDEs for all datasets. Future improvements include:

  - Extending the library to support other differentiation methods such as polynomial interpolation.
  - Addressing boundary condition issues observed in PDE 3.
  - Improve sub-sampling: the current implementation samples the data along entire rows/columns, random sub-sampling could improve the algorithm's performance and robustness to noise.
  - Add measures to make solvers more robust to noise in the data.
  - Benchmarking against other sparse regression algorithms.
  - Extending the library to support non-polynomial terms.
  - Decoupling solver components to allow easy integration with alternative sparse solvers.

]

#body

#bibliography("refs.bib")

#pagebreak()

// Appendix
#set page(
  numbering: "A",
  columns: 1,
)
#counter(page).update(1)
#counter(heading).update(0)
#set heading(numbering: none)

= Appendix

#let appendix = [

  #show figure: set block(breakable: true)
  #figure(
    kind: "algorithm",
    supplement: [Algorithm],
    [
      #pseudocode-list(
        booktabs: true,
        hooks: .5em,
        numbered-title: [Truncated Least Squares with dynamic thresholding (TLS-DT)],
      )[

        - *X* - Matrix of shape (n_samples, n_features)
        - *y* - Vector of shape (n_samples,)
        - *cutoff* - Initial cutoff value for coefficients
        - *max_iterations* - Maximum number of iterations
        - *num_term_limit* - Limit for the number of terms
        - \

        + coeffs = #smallcaps("LeastSquares") (*X*, *y*)
        + num_terms = []
        + original_cutoff = cutoff
        - \

        + *for* iteration *in* max_iterations:
          + *if* the number of terms is consistent across two iterations and $<=$ num_term_limit:
            + *break*
          - \
          + *if* the number of terms is consistent and $ >=$ num_term_limit:
            + $"cutoff" *=2$
          - \
          + *if* the number of terms differs between iterations:
            + $"cutoff" = "original_cutoff"$
          - \

          - #text(gray)[\# Identify indices of coefficients greater than the current cutoff]
          + indices = #smallcaps("true where") ($"coeffs" > "cutoff"$)
          - \
          + *if* #smallcaps("sum") (indices) $=$ 0:
            + Set the indices of the num_term_limit smallest coefficients to #smallcaps("true")
          + *else:*
            + Set coefficients smaller than the cutoff to zero
          - \
          - #text(gray)[\# Recalculate coefficients using only the non-zero terms]
          + coeffs[:, indices] = #smallcaps("LeastSquares") (*X*[:, indices], *y*)
          - \
          - #text(gray)[\# Append the count of non-zero coefficients to num_terms]
          + num_terms append (#smallcaps("sum") (indices))

        - \
        + *return* the final coeffs vector
      ]
    ],
    caption: "Truncated Least Squares with dynamic thresholding (TLS-DT)",
  ) <tls_dt_algorithm>

  #v(30pt)
  #figure(
    $
      1, u, u_("x""x""x"), u_("x""x""x")u_("x""x""x"), u_("x""x""x")u_("x""x"), u_("x""x""x")u_(x), u_("x""x"), u_("x""x")u_("x""x"), u_("x""x")u_(x), u_(x), u_(x)u_(x), "u""u", "u""u"_("x"x"x"), "u""u"_("x""x"), "u"u_(x)
    $,
    caption: "Term library used for PDE 1",
    kind: table,
  ) <pde1_terms>
  #v(30pt)
  #figure(
    $
      1, u, u_("x""x""x"), u_("x""x""x")u_("x""x""x"), u_("x""x""x")u_("x""x"), u_("x""x""x")u_(x), u_("x""x"), u_("x""x")u_("x""x"), u_("x""x")u_(x), u_(x), u_(x)u_(x), "u""u", "u""u"_("x"x"x"), "u""u"_("x""x"), "u"u_(x)
    $,
    caption: "Term library used for PDE 2",
    kind: table,
  ) <pde2_terms>

  #v(30pt)

  #figure(
    $
      1, "u", "u"_("x""x"), "u"_("x""x")"v", "u"_("x""x")"v""v", "u"_("x""y"), "u"_("x""y")"v", "u"_("x""y")"v""v", "u"_("x"), "u"_("x")"v", "u"_("x")"v""v", \
      "u"_("y""y"), "u"_("y""y")"v", "u"_("y""y")"v""v", "u"_("y"), "u"_("y")"v", "u"_("y")"v""v", "u""u", "u""u"_("x""x"), "u""u"_("x""x")"v", "u""u"_("x""y"), \
      "u""u"_("x""y")"v", "u""u"_("x"), "u""u"_("x")"v", "u""u"_("y""y"), "u""u"_("y""y")"v", "u""u"_("y"), "u""u"_("y")"v", "u""u""u", "u""u""u"_("x""x"), "u""u""u"_("x""y"), \
      "u""u""u"_("x"), "u""u""u"_("y""y"), "u""u""u"_("y"), "u""u""v", "u""u""v"_("x""x"), "u""u""v"_("x""y"), "u""u""v"_("x"), "u""u""v"_("y""y"), "u""u""v"_("y"), \
      "u""v", "u""v"_("x""x"), "u""v"_("x""y"), "u""v"_("x"), "u""v"_("y""y"), "u""v"_("y"), "u""v""v", "u""v""v"_("x""x"), "u""v""v"_("x""y"), "u""v""v"_("x"), "u""v""v"_("y""y"),\
      "u""v""v"_("y"), "v", "v"_("x""x"), "v"_("x""y"), "v"_("x"), "v"_("y""y"), "v"_("y"), "v""v", "v""v"_("x""x"), "v""v"_("x""y"), "v""v"_("x"), "v""v"_("y""y"), "v""v"_("y"), \
      "v""v""v", "v""v""v"_("x""x"), "v""v""v"_("x""y"), "v""v""v"_("x"), "v""v""v"_("y""y"), "v""v""v"_("y")
    $,
    caption: "Term library used for PDE 3",
    kind: table,
  ) <pde3_terms>
]

#appendix
