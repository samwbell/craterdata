from .generic_plotting_module import *


def fit_linear(sorted_ds, density_array, uncertainties=None, guess=None):
    def linear_equation(x, m, b):
        return m * x + b
    X, Y = np.log10(sorted_ds), np.log10(density_array)
    try:
        if uncertainties is not None:
            result, cov = optimize.curve_fit(
                linear_equation, X, Y, sigma=uncertainties, p0=guess
            )
            result = tuple(result)
        else:
            linregress_result = linregress(X, Y)
            result = tuple([
                linregress_result.slope, linregress_result.intercept
            ])
    except:
        result = tuple([float('nan'), float('nan')])
    return result


def fit_production_function(
    sorted_ds, density_array, uncertainties=None, guess=None,
    loglog_production_function=loglog_linear_pf(N1=0.001, slope=-2)
):
    def fit_func(x, a):
        return a + loglog_production_function(x)
    X, Y = np.log10(sorted_ds), np.log10(density_array)
    if uncertainties is not None:
        try:
            res, cov = optimize.curve_fit(
                fit_func, X, Y, sigma=uncertainties, p0=guess
            )
        except:
            res = [float('nan')]
    else:
        try:
            res, cov = optimize.curve_fit(fit_func, X, Y, p0=guess)
        except:
            res = [float('nan')]
    return 10**res[0]


