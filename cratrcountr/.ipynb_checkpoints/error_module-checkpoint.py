from .fit_module import *

# Cumulative percentage cutoff equivalent to 1 sigma
p_1_sigma = 0.841345


def get_kwargs(func):
    defaults = func.__defaults__
    if not defaults:
        return {}

    n_args = func.__code__.co_argcount
    var_names = func.__code__.co_varnames
    kwarg_names = var_names[:n_args][-len(defaults):]
    kwargs = {kwarg_name: default for kwarg_name, 
                   default in zip(kwarg_names, defaults)}
    return kwargs


def true_error_pdf_XP(
    N, n_points=10000, cum_prob_edge=1E-7, log_spacing=False
):
    
    X_min = gamma.ppf(cum_prob_edge, N + 1)
    X_max = gamma.ppf(1 - cum_prob_edge, N + 1)
    
    if gamma.pdf(X_min, N + 1) > 0.001:
        X_min_search = np.logspace(-150, np.log10(X_min), 1000)
        X_min = np.interp(
            0.001, gamma.pdf(X_min_search, N + 1), X_min_search
        )
    
    if log_spacing:
        
        X = np.logspace(
            np.log10(X_min), np.log10(X_max), n_points, endpoint=True
        )
        
    else:
        
        X = np.linspace(X_min, X_max, n_points, endpoint=True)
        
    P = gamma.pdf(X, N + 1)
    
    return X, P / P.sum()


def is_equally_spaced(X, tolerance=1E-3):
    spacing = np.diff(X)
    mismatch = np.abs(spacing / spacing.mean() - 1)
    return bool(np.sum(mismatch > tolerance) < 1)


def error_bar_log(X_raw, P_raw, max_likelihood=None):
    
    X = X_raw[~np.isnan(X_raw) & ~np.isnan(P_raw)]
    P = P_raw[~np.isnan(X_raw) & ~np.isnan(P_raw)]
    P = P[X > 0]
    X = X[X > 0]
    logX = np.log10(X)

    if not is_equally_spaced(logX):
        X_es = np.logspace(logX.min(), logX.max(), logX.shape[0])
        P = np.interp(X_es, X, P)
        X = X_es
        logX = np.log10(X)
    
    if max_likelihood is None:
        max_val = X[np.argmax(P)]
    else:
        max_val = max_likelihood
    log_max = np.log10(max_val)
    
    def fit_eq(x, std):
        y = norm.pdf(x, log_max, std)
        return y / y.max()

    left_logX = logX[X < max_val]
    left_P = (P / P.max())[X < max_val]
    left_C = cumulative_trapezoid(left_P, left_logX, initial=0)
    left_C = left_C / left_C.max()
    log_low_guess = np.interp(2 - 2 * p_1_sigma, left_C, left_logX)
    log_lower_guess = np.log10(max_val) - log_low_guess
    log_lower_result, cov = optimize.curve_fit(
        fit_eq, left_logX, left_P, p0=log_lower_guess
    )

    right_logX = logX[X > max_val]
    right_P = (P / P.max())[X > max_val]
    right_C = cumulative_trapezoid(right_P, right_logX, initial=0)
    right_C = right_C / right_C.max()
    log_high_guess = np.interp(2 * p_1_sigma - 1, right_C, right_logX)
    log_upper_guess = log_high_guess - np.log10(max_val)
    log_upper_result, cov = optimize.curve_fit(
        fit_eq, right_logX, right_P, p0=log_upper_guess
    )
    
    return log_lower_result[0], log_upper_result[0]


def error_bar_linear(X_raw, P_raw, max_likelihood=None):
    
    X = X_raw[~np.isnan(X_raw) & ~np.isnan( P_raw)]
    P = P_raw[~np.isnan(X_raw) & ~np.isnan(P_raw)]

    if not is_equally_spaced(X):
        X_es = np.linspace(X.min(), X.max(), X.shape[0])
        P = np.interp(X_es, X, P)
        X = X_es
    
    if max_likelihood is None:
        max_val = X[np.argmax(P)]
    else:
        max_val = max_likelihood
    
    def fit_eq(x, std):
        y = norm.pdf(x, max_val, std)
        return y / y.max()

    left_X = X[X < max_val]
    left_P = (P / P.max())[X < max_val]
    left_C = cumulative_trapezoid(left_P, left_X, initial=0)
    
    low_guess = np.interp(
        2 - 2 * p_1_sigma, left_C, left_X
    )
    lower_guess = max_val - low_guess
    
    lower_result, cov = optimize.curve_fit(
        fit_eq, left_X, left_P, p0=lower_guess
    )

    right_X = X[X > max_val]
    right_P = (P / P.max())[X > max_val]
    right_C = cumulative_trapezoid(right_P, right_X, initial=0)
    
    high_guess = np.interp(2 * p_1_sigma - 1, right_C, right_X)
    upper_guess = high_guess - max_val 
    
    upper_result, cov = optimize.curve_fit(
        fit_eq, right_X, right_P, p0=upper_guess
    )
    
    return lower_result[0], upper_result[0]


def error_bar_log_N(N, n_points=10000, log_spacing=True):

    if N > 0:

        X, P = true_error_pdf_XP(
            N, n_points=n_points, log_spacing=log_spacing
        )
        log_left, log_right = error_bar_log(X, P, max_likelihood=N)
    
    else:
        
        log_left = None
        log_right = None
    
    return log_left, log_right


def error_bar_linear_N(N, n_points=10000, log_spacing=False):

    if N > 0:

        X, P = true_error_pdf_XP(
            N, n_points=n_points, log_spacing=log_spacing
        )
        left, right = error_bar_linear(X, P, max_likelihood=N)
    
    else:
        
        left = None
        right = None
    
    return left, right


# Load the saved error bar fits from file
_lower_PPFit = read_PPFit('saved/lower_PPFit')
_upper_PPFit = read_PPFit('saved/upper_PPFit')
_lower_PPFit_linear = read_PPFit('saved/lower_PPFit_linear')
_upper_PPFit_linear = read_PPFit('saved/upper_PPFit_linear')


_X0, _P0 = true_error_pdf_XP(0)
_C0 = cumulative_trapezoid(_P0, _X0, initial=0)
_C0 = _C0 / _C0.max()
N_0_upper = np.interp(2 * p_1_sigma - 1, _C0, _X0)


def get_error_bars(
    N_number_or_array, kind='log', log_space=False, 
    multiplication_form=False
):

    if type(N_number_or_array) in {float, int}:
        full_N = np.array([N_number_or_array])
    else:
        full_N = np.array(N_number_or_array)
    nonzero = full_N > 0
    N = full_N[nonzero]
        
    logN = np.log10(N)

    if kind in {'log', 'Log'}:
        lower = 10**_lower_PPFit.apply(logN)
        upper = 10**_upper_PPFit.apply(logN)
        if not log_space:
            if multiplication_form:
                lower = 10**lower
                upper = 10**upper
            else:
                lower = N - 10**(logN - lower)
                upper = 10**(logN + upper) - N

    if kind in {'linear', 'Linear'}:
        lower = 10**_lower_PPFit_linear.apply(logN)
        upper = 10**_upper_PPFit_linear.apply(logN)
        if log_space:
            lower = np.log10(N) - np.log10(N - lower)
            upper = np.log10(N + upper) - np.log10(N)
        elif multiplication_form:
            lower = N / (N - lower)
            upper = (N + upper) / N
        
    full_lower = np.empty(nonzero.shape, dtype=lower.dtype)
    full_lower[nonzero] = lower
    full_lower[~nonzero] = None
    full_upper = np.empty(nonzero.shape, dtype=upper.dtype)
    full_upper[nonzero] = upper
    if log_space:
        full_upper[~nonzero] = None
    else:
        full_upper[~nonzero] = N_0_upper

    if full_lower.shape[0] == 1:
        full_lower = float(full_lower[0])
        full_upper = float(full_upper[0])

    return full_lower, full_upper


def get_sqrt_N_error_bars(N_array):
    lower = -1 * np.sqrt(N_array)
    upper = np.sqrt(N_array)
    return lower, upper


def get_sqrt_N_error_bars_log_space(N_array, zero_point=0.00001):
    N_low = N_array - np.sqrt(N_array)
    N_low[N_low < zero_point] = zero_point
    low = np.log10(N_low)
    high = np.log10(N_array + np.sqrt(N_array))
    max_val = np.log10(N_array)
    lower = low - max_val
    upper = high - max_val
    return lower, upper


def true_error_median(N):
    return gamma.ppf(0.5, N + 1)


def true_error_percentile(N, p):
    return gamma.ppf(p, N + 1)


def get_true_error_bars_median(N_array):
    med = true_error_median(N_array)
    lower = med - true_error_percentile(N_array, 1 - p_1_sigma)
    upper = true_error_percentile(N_array, p_1_sigma) - med
    return lower, upper


def get_true_error_bars_median_log_space(N_array):
    med = true_error_median(N_array)
    lower = np.log10(med) - np.log10(true_error_percentile(N_array, 1 - p_1_sigma))
    upper = np.log10(true_error_percentile(N_array, p_1_sigma)) - np.log10(med)
    return lower, upper


def get_true_error_bounds(N_raw, area):
    N_array = np.atleast_1d(N_raw)
    lower, upper = get_error_bars(N_array, log_space=False)
    low_array = N_array - lower
    high_array = N_array + upper
    return low_array / area, high_array / area


def get_true_error_bounds_linear(N_raw, area):
    N_array = np.atleast_1d(N_raw)
    lower, upper = get_error_bars(N_array, kind='linear', log_space=False)
    low_array = N_array - lower
    high_array = N_array + upper
    return low_array / area, high_array / area

                                                     
def scientific_notation(n, rounding_n=2):
    e = math.floor(np.log10(n))
    n_str = str(round(n / 10**e, rounding_n))
    return n_str + f"x10$^{{{e}}}$"


