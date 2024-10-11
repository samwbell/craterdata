# craterdata
A Python3 package for statistical analysis of crater count data.

To see some of the functionality, start with three example notebooks:

1) The examples/age_calculation_example notebook shows how to calculate both age and slope PDFs (Probability Density Functions) from real data.

2) The examples/random_variable_example notebook shows how to use RandomVariable objects to do random variable math.

3) The examples/plotting_examples notebook shows how to make cumulative, differential, and R plots.

## Installation
1) Clone the repo:
`git clone https://github.com/samwbell/craterdata`

2) Navigate to its directory:
`cd craterdata`

3) If you are using a virtual environment, enter it.  Installing craterdata will update to the minimum versions of numpy, scipy, and pandas.  Otherwise, it should not affect the underlying packages.

4) Install:
`pip install -e .`

5) If your environment does not yet have Jupyter installed, to open the Jupyter notebooks, install Jupyter by following the directions here: https://jupyter.org/install

## Dependencies
The craterdata package primarily uses the standard numpy and scipy packages.  In a few cases, it uses pandas, mostly for reading and saving CSV files.  The one nonstandard package it uses is the ash package by Alexander Dittman (github.com/ajdittmann/ash), which implements the Average Shifted Histogram (ASH) method in Python.  

This method is used to produce smoothed histograms of observations that best approximate the underlying PDF as long as the distribution is roughly Gaussian.  We use it for handling synthetic modeling results.  While we are loath to rely on nonstandard packages, ASH dramatically reduces the N required to see the same quality of empirical PDF with synthetic modeling--often by more than an order of magnitude.

The ASH algorithm can be used on crater count diameters for estimating the PDF of diameter from an observed set of crater diameters.  However, it will produce distortions because the distribution is not roughly Gaussian.  (It is instead roughly Pareto.)  This effect is similar to the distortions noted by Robbins et al. (2018) from the Kernel Density Estimation (KDE) method, which has mathematical similarities to ASH.

