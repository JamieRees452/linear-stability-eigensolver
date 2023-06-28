# Linear Stability Solver

The aim of this solver is to investigate tropical instability waves (TIWs) in the equatorial oceans by means of a linear stability analysis.

Our linear stability solver solves the two-dimensional linearised equations of motion for an inviscid Buossinesq fluid on an equatorial beta plane. The equations of motion are discretised using a finite difference method and the resulting generalised eigenvalue problem is solved with scipy using an implicitly restarted Arnoldi method. With this repository, one will be able to reproduce the results in the linear stability analysis section of my thesis.
 
## Table of Contents
* [Setup](#setup)
* [Procedure](#procedure)
* [To Do](#todo)
* [Citations](#citations)

## Setup

The file structure should be as follows

```bash
/
├───linear_figures
│   ├───debug
│   └───thesis
├───original_data
└───saved_data
    ├───interpolated_data
    ├───NEMO_12
    ├───NEMO_25
    ├───Proehl_1
    ├───Proehl_2
    ├───Proehl_3
    ├───Proehl_4
    ├───Proehl_5
    ├───Proehl_6
    ├───Proehl_7
    └───Proehl_8
```

## Procedure

As a complete example with the code, we investigate the linear stability of an eastward flowing Gaussian Jet centered on the equator and in depth (Proehl_1). The mean fields are visualised via

```
python linear_plot_mean_fields.py 100 50 Proehl_1 00 00
```

The mean fields are saved to /linear_figures/debug

Mean Zonal Velocity        |  Mean Density
:-------------------------:|:-------------------------:
![](/images/U_Proehl_1.png)|  ![](/images/r_Proehl_1.png)

Calculate the growth rates over a range of wavenumbers by sweeping initial guesses of the eigenvalues

```
python calculate_evals_evecs_multiple.py 100 50 0.25 Proehl_1 00 00 3 1e-8 1e-5 150
```

Plot the calculated growth rates

```
python linear_plot_growth_rates.py 100 50 Proehl_1 00 00 3 1e-8 1e-5 150
```

Output is shown below

(Figure)

Calculate the eigenvalue and eigenvector at the most unstable mode 

```
python calculate_evals_evecs_single.py 100 50 0.25 Proehl_1 00 00
```

![Figure](/images/Proehl_1_evals_single_output.png)

Plot eigenvectors

```
python linear_plot_eigenvectors.py 100 50 6.67e-6 Proehl_1 00 00
```

Calculate and plot terms in the linearised eddy energy budget

```
python linear_plot_eddy_energy_budget.py 100 50 6.67e-6 Proehl_1 00 00
```

Calculate and plot GEOMETRIC diagnostics

```
python linear_plot_geometric_diagnostics.py 100 50 6.67e-6 Proehl_1 00 00
```

## To Do

* Solve the linear stability problem using an additional independent numerical method
  - We have solved the linear stability problem using a finite difference method. However, we might have concerns over the reliability of the results obtained using only one method. To this extent, we aim to solve the linear stability problem using a Chebyshev collocation method.

* Pathlib
  - Files are specific to Unix. We should use pathlib so that the file structures also agree with Windows and Mac.
  
* Thesis Figures
  - Currently, we have only output simlpe figures which are primarily used for debugging. The neater figures which will be included in my thesis will be added to this repository eventually

## Citations

* Proehl, Jeffrey A. "Linear stability of equatorial zonal flows." Journal of physical oceanography 26.4 (1996): 601-621.
* Proehl, Jeffrey A. "The role of meridional flow asymmetry in the dynamics of tropical instability." Journal of Geophysical Research: Oceans 103.C11 (1998): 24597-24618.
