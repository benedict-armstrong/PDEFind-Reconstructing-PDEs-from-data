# PDE-Find using Truncated Least Squares with dynamic thresholding

As part of this year's AI in the Sciences and Engineering course, I replicated and extended PDE-Find, as described in the paper by Rudy et al [1]. The implementation is structured as a library, capable not only of identifying partial differential equations but also of solving them using SciPy's `solve_ivp` to verify the discovered equations. The notebooks, library code, and scripts for generating all plots and figures are available in [this repository](https://github.com/benedict-armstrong/PDEFind-Reconstructing-PDEs-from-data).

A detailed write-up of the project is available in [this report](https://github.com/benedict-armstrong/PDEFind-Reconstructing-PDEs-from-data/blob/main/writeup/report.pdf).

[1] Rudy, S. H., Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2017). Data-driven discovery of partial differential equations. Science Advances, 3(4), e1602614.