from .random_variable_module import *

from .generic_plotting_module import *
from .cumulative_unbinned_module import *
from .cumulative_binned_module import *
from .differential_module import *
from .R_module import *

from .production_function_module import *

from .plot_fitting_module import *

def plot_log_of_normal_approximation(N, color='mediumslateblue', upshift=0, linewidth=1,
                                     log_space=False, label=False):
    X = true_error_pdf(N).X
    if N > 0:
        left_error = get_lower_log_log_space(N)
        left = norm.pdf(np.log10(X), np.log10(N), left_error) / norm.pdf(np.log10(X),
                                                        np.log10(N), left_error).max()
        left += upshift
        if log_space:
            X_left = np.log10(X[(X < N) & (X > 0)])
            left = left[(X < N) & (X > 0)]
        else:
            X_left = X[X < N]
            left = left[X < N]
        plt.plot(X_left, left, color=color, linewidth=linewidth)
    right_error = get_upper_log_log_space(N)
    right = norm.pdf(np.log10(X), np.log10(N), right_error) / norm.pdf(np.log10(X),
                                                       np.log10(N), right_error).max()
    right += upshift
    if log_space:
        X_right = np.log10(X[(X >= N) & (X > 0)])
        right = right[(X >= N) & (X > 0)]
    else:
        X_right = X[X >= N]
        right = right[X >= N]
    plt.plot(X_right, right, color=color, linewidth=linewidth)

def plot_linear_approximation(N, color='mediumslateblue', X_array = None, upshift=0, linewidth=1,
                              log_space=False):
    if X_array is None:
        X = true_error_pdf(N).X
    else:
        X = X_array
    if N > 0:
        left_error = get_lower_linear(N)
        X_min = N - 5 * get_lower_linear(N)
        X_left = np.concatenate((np.linspace(X_min, X.min(), 2000), X))
        left = norm.pdf(X_left, N, left_error) / norm.pdf(X_left, N, left_error).max()
        left += upshift
        if log_space:
            left = left[(X_left > 0) & (X_left < N)]
            X_left = np.log10(X_left[(X_left > 0) & (X_left < N)])
        else:
            left = left[X_left < N]
            X_left = X_left[X_left < N]
        plt.plot(X_left, left, color=color, linewidth=linewidth)
    right_error = get_upper_linear(N)
    right = norm.pdf(X, N, right_error) / norm.pdf(X, N, right_error).max()
    right += upshift
    if log_space:
        X_right = np.log10(X[(X >= N) & (X > 0)])
        right = right[(X >= N) & (X > 0)]
    else:
        X_right = X[X >= N]
        right = right[X >= N]
    plt.plot(X_right, right, color=color, linewidth=linewidth)
    
def plot_median_approximation(N, color='mediumslateblue', upshift=0, linewidth=1, 
                              log_space=False, label=False):
    X = np.linspace(-3, true_error_pdf(N).X.max(), 10000)
    left_error, right_error = get_true_error_bars_median(N)
    median = true_error_median(N)
    left = norm.pdf(X, median, left_error) / norm.pdf(X, median, left_error).max()
    left += upshift
    if log_space:
        X_left = np.log10(X[(X < median) & (X > 0)])
        left = left[(X < median) & (X > 0)]
    else:
        X_left = X[X < median]
        left = left[X < median]
    plt.plot(X_left, left, color=color, linewidth=linewidth)
    right = norm.pdf(X, median, right_error) / norm.pdf(X, median, right_error).max()
    right += upshift
    if log_space:
        X_right = np.log10(X[(X >= median) & (X > 0)])
        right = right[(X >= median) & (X > 0)]
    else:
        X_right = X[X >= median]
        right = right[X >= median]
    plt.plot(X_right, right, color=color, linewidth=linewidth, label=label)
    
    