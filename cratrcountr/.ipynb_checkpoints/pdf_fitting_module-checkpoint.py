from .error_module import *


def piecewise_normal_fit_equation(x, log_max, std_left, std_right):
    left_side = norm.pdf(x, log_max, std_left) / norm.pdf(
                            x, log_max, std_left).max()
    right_side = norm.pdf(x, log_max, std_right) / norm.pdf(
                            x, log_max, std_right).max()
    return np.where(x < log_max, left_side, right_side)


def fit_log_of_normal(X_raw, P_raw):

    X_pos = X_raw[X_raw > 0]
    P_pos = P_raw[X_raw > 0]

    X = X_pos[~np.isnan(X_pos) & ~np.isnan(P_pos)]
    P = P_pos[~np.isnan(X_pos) & ~np.isnan(P_pos)]
    P = P / P.max()

    logX = np.log10(X)

    if not is_equally_spaced(logX):
        X_es = np.logspace(logX.min(), logX.max(), logX.shape[0])
        P = np.interp(X_es, X, P)
        X = X_es
        logX = np.log10(X)

    C = cumulative_trapezoid(P, logX, initial=0)
    C = C / C.max()

    max_guess = np.log10(X[np.argmax(P)])
    ps = [1 - p_1_sigma, p_1_sigma]
    low_median, high_median = tuple(np.interp(ps, C, X))

    std_guess = np.log10(high_median / low_median) / 2
    guess = [max_guess, std_guess, std_guess]

    result, cov = optimize.curve_fit(
        piecewise_normal_fit_equation, logX, P, p0=guess,
        bounds=(
            (np.log10(X.min()), 0, 0), 
            (np.log10(X.max()), np.inf, np.inf)
        )
    )

    log_max, log_lower, log_upper = tuple(result)
    return log_max, log_lower, log_upper


def plot_log_of_normal_fit(
    log_max, log_lower, log_upper, color='mediumslateblue', 
    upshift=0, linewidth=1, log_space=True, label=False,
    slope_data=False
):
    X_min = log_max - 5 * log_lower
    X_max = log_max + 5 * log_upper
    X = np.linspace(X_min, X_max, 10000)
    X_left = X[X < log_max]
    X_right = X[X >= log_max]
    left = norm.pdf(X_left, log_max, log_lower) / norm.pdf(
            X, log_max, log_lower).max()
    left += upshift
    if not log_space:
        X_left = 10**X_left
        if slope_data:
            X_left *= -1
    plt.plot(X_left, left, color=color, linewidth=linewidth)
    right = norm.pdf(X_right, log_max, log_upper) / norm.pdf(
            X, log_max, log_upper).max()
    right += upshift
    if not log_space:
        X_right = 10**X_right
        if slope_data:
            X_right *= -1
    plt.plot(X_right, right, color=color, linewidth=linewidth) 
    
    
def fit_slope_pdf(X_raw, P_raw):
    X = -1 * np.flip(X_raw)
    P = np.flip(P_raw)
    return fit_log_of_normal(X, P)


def rv_percentiles_XP(X, P):
    C = cumulative_trapezoid(P, X, initial=0)
    C = C / C.max()
    ps = [1 - p_1_sigma, 0.5, p_1_sigma]
    return tuple(np.interp(ps, C, X))


def rv_mean_XP(X, P):
    _P = P / trapezoid(P, X)
    return trapezoid(X * _P, X)


def rv_std_XP(X, P, mean=None):
    _P = P / trapezoid(P, X)
    if mean is None:
        mu = rv_mean_XP(X, _P)
    else:
        mu = mean
    return np.sqrt(trapezoid((X - mu)**2 * _P, X))


def rv_skewness_XP(X, P, mean=None, std=None):
    _P = P / trapezoid(P, X)
    if mean is None:
        mu = rv_mean_XP(X, _P)
    else:
        mu = mean
    if std is None:
        sigma = rv_std_XP(X, _P)
    else:
        sigma = std
    return trapezoid(((X - mu) / sigma)**3 * _P, X)


def error_bar_N(N, n_points=10000, log_spacing=None, kind='log'):

    if log_spacing is None:
        _log_spacing = {
            'log' : True,
            'auto log' : True,
            'linear' : False,
            'median' : False,
            'mean' : False
        }[kind.lower()]
    else:
        _log_spacing = log_spacing

    if kind == 'log':
        return error_bar_log_N(
            N, n_points=n_points, log_spacing=_log_spacing
        )
        
    elif kind == 'linear':
        return error_bar_linear_N(
            N, n_points=n_points, log_spacing=_log_spacing
        )

    else:
        if N <= 0 and kind.lower() == 'auto log':
            return None, None
        else:
            X, P = true_error_pdf_XP(
                N, n_points=n_points, log_spacing=_log_spacing
            )
            return {
                'auto log' : fit_log_of_normal(X, P),
                'median' : rv_percentiles_XP(X, P),
                'mean' : tuple([
                    rv_mean_XP(X, P), rv_std_XP(X, P), rv_skewness_XP(X, P)
                ])
            }[kind.lower()]



