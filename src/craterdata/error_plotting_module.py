from .random_variable_module import *

from .generic_plotting_module import *
from .cumulative_unbinned_module import *
from .cumulative_binned_module import *
from .differential_module import *
from .R_module import *

from .production_function_module import *

from .plot_fitting_module import *

def plot_log_of_normal_approximation(
    N, color='mediumslateblue', X_array=None, upshift=0, linewidth=1, 
    log_space=False
):
        
    if X_array is None:
        X = true_error_pdf(N).X
    else:
        X = X_array
        
    left_error, right_error = get_error_bars(N, log_space=True)
    
    if N > 0:
        X_left = X[(X < N) & (X > 0)]
        left = norm.pdf(np.log10(X_left), np.log10(N), left_error)
        left = left / left.max()
        left += upshift
        if log_space:
            X_left = np.log10(X_left)
        plt.plot(X_left, left, color=color, linewidth=linewidth)

    X_right = X[(X >= N) & (X > 0)]
    right = norm.pdf(np.log10(X_right), np.log10(N), right_error)
    right = right / right.max()
    right += upshift
    if log_space:
        X_right = np.log10(X_right)
    plt.plot(X_right, right, color=color, linewidth=linewidth)


def plot_linear_approximation(
    N, color='mediumslateblue', X_array=None, upshift=0, linewidth=1, 
    log_space=False, forced_right_error=None
):
    
    if X_array is None:
        X = true_error_pdf(N).X
    else:
        X = X_array
        
    if N > 0:
        left_error = get_error_bars(N, kind='linear')[0]
        X_min = N - 5 * left_error
        X_left = np.concatenate((np.linspace(X_min, X.min(), 2000), X))
        X_left = X_left[X_left < N]
        if log_space:
            X_left = np.log10(X_left[X_left > 0])
            log_left_error = np.log10(N) - np.log10(N - left_error)
            left = norm.pdf(X_left, np.log10(N), log_left_error)
        else:
            left = norm.pdf(X_left, N, left_error)
        left = left / left.max()
        left += upshift

        plt.plot(X_left, left, color=color, linewidth=linewidth)
        
    if forced_right_error is None:
        right_error = get_error_bars(N, kind='linear')[1]
    else:
        right_error = forced_right_error
    X_right = X[X >= N]
    if log_space:
        X_right = np.log10(X_right[X_right > 0])
        log_right_error = np.log10(N + right_error) - np.log10(N)
        right = norm.pdf(X_right, np.log10(N), log_right_error)
    else:
        right = norm.pdf(X_right, N, right_error)
    right = right / right.max()
    right += upshift
    plt.plot(X_right, right, color=color, linewidth=linewidth)


def plot_median_approximation(
    N, color='mediumslateblue', X_array=None, upshift=0, linewidth=1, 
    log_space=False
):
    
    if X_array is None:
        X = np.linspace(-3, true_error_pdf(N).X.max(), 10000)
    else:
        X = X_array
        
    median, left_error, right_error = get_error_bars(
        N, kind='median', return_val=True
    )
    
    X_left = X[X < median]
    if log_space:
        X_left = np.log10(X_left[X_left > 0])
        log_left_error = np.log10(median) - np.log10(median - left_error)
        left = norm.pdf(X_left, np.log10(median), log_left_error)
    else:
        left = norm.pdf(X_left, median, left_error)
    left = left / left.max()
    left += upshift
    plt.plot(X_left, left, color=color, linewidth=linewidth)
    
    X_right = X[X >= median]
    if log_space:
        X_right = np.log10(X_right[X_right > 0])
        log_right_error = np.log10(median + right_error) - np.log10(median)
        right = norm.pdf(X_right, np.log10(median), log_right_error)
    else:
        right = norm.pdf(X_right, median, right_error)
    right = right / right.max()
    right += upshift
    plt.plot(X_right, right, color=color, linewidth=linewidth)
    

def plot_approximation(
    val, left_error, right_error, color='mediumslateblue',
    X_array=None, upshift=0, linewidth=1, log_space=False
):
    
    if X_array is None:
        X = true_error_pdf(val).X
    else:
        X = X_array
        
    if val > 0:
        X_min = val - 5 * left_error
        X_left = np.concatenate((np.linspace(X_min, X.min(), 2000), X))
        X_left = X_left[X_left < val]
        if log_space:
            X_left = np.log10(X_left[X_left > 0])
            log_left_error = np.log10(val) - np.log10(val - left_error)
            left = norm.pdf(X_left, np.log10(val), log_left_error)
        else:
            left = norm.pdf(X_left, val, left_error)
        left = left / left.max()
        left += upshift

        plt.plot(X_left, left, color=color, linewidth=linewidth)
        
    X_right = X[X >= val]
    if log_space:
        X_right = np.log10(X_right[X_right > 0])
        log_right_error = np.log10(val + right_error) - np.log10(val)
        right = norm.pdf(X_right, np.log10(val), log_right_error)
    else:
        right = norm.pdf(X_right, val, right_error)
    right = right / right.max()
    right += upshift
    plt.plot(X_right, right, color=color, linewidth=linewidth)

