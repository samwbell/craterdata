from .production_function_module import *

def pareto_P(d, alpha, dmin=1):
    D = np.atleast_1d(np.array(d))
    D = D[D > dmin]
    N = D.shape[0]
    beta = np.sum(np.log(D / dmin))
    return gamma.pdf(alpha, N + 1, scale = 1 / beta)

def pareto_pdf(d, dmin=1, alpha_min=1E-5, alpha_max=10, 
    n_points=10000, alpha=None
):
    if alpha is None:
        _alpha = np.linspace(alpha_min, alpha_max, n_points)
    else:
        _alpha = alpha
    P = pareto_P(d, _alpha, dmin=dmin)
    return RandomVariable(_alpha, P, kind='mean')

def truncated_pareto_P(d, alpha, dmin=1, dmax=1E4):
    D = np.atleast_1d(np.array(d))
    D = D[(D > dmin) & (D < dmax)]
    N = D.shape[0]
    beta = np.sum(np.log(D / dmin))
    log_P = gamma.logpdf(alpha, N + 1, scale = 1 / beta)
    log_truncator = N * np.log((1 - (dmin / dmax)**alpha))
    P_truncated = np.e**(log_P - log_truncator)
    return P_truncated / trapezoid(P_truncated, alpha)

def synth_flat_slope(N=100, slope=-2, dmin=1, n_datasets=None):
    if n_datasets is None:
        shape = N
    else:
        shape = (N, n_datasets)
    ds = dmin * (np.random.pareto(-1 * slope, shape).T + 1)
    return ds

def _combine_2(Ps, X):
    P_matrix = np.array([Pi / Pi.max() for Pi in Ps])
    P = np.prod(P_matrix, axis=0)
    return P / P.max()

def combine_Ps(Ps, X):
    while Ps.shape[0] > 2:
        Ps_list = np.array_split(Ps, round(Ps.shape[0] / 2))
        Ps = np.array([_combine_2(Psi, X) for Psi in Ps_list])
    P = _combine_2(Ps, X)
    return P / trapezoid(P, X)

def alpha_meshed(m, dmin=1, m_max=10, n_points=10000):
    left_n = round(n_points / 2)
    alpha = np.flip(np.linspace(m_max, 0, left_n, endpoint=False))
    alpha_right = m + np.logspace(-5, np.log10(m_max - m), 5000)
    left_edge = np.log10(m - 10**(np.log10(m) - np.log10(m_max - m)))
    alpha_left = np.flip(m - np.logspace(-5, left_edge, n_points - left_n))
    alpha = np.concat([alpha_left, [m], alpha_right])
    return alpha

def mle_slope(ds, dmin):
    N = np.array(ds).shape[0]
    return N / np.sum(np.log(ds / dmin))

def _truncated_solve_func(d, dmin=1.0, dmax=1E4):
    def r_func(m):
        a = dmin / dmax
        N = d.shape[0]
        sum_term = np.sum(np.log(d) - np.log(dmin))
        ratio_term = N * a**m * np.log(a) / (1 - a**m)
        return N / m + ratio_term - sum_term 
    return r_func

def truncated_mle_slope(d, dmin=1.0, dmax=1E4):
    D = np.atleast_1d(np.array(d))
    D = D[(D > dmin) & (D < dmax)]
    f = _truncated_solve_func(D, dmin=dmin, dmax=dmax)
    return -1 * optimize.root_scalar(f, bracket=[0.1, 10]).root

def pareto_mle_dist(mle, a, N):
    num = (a * N)**N * np.exp(-1 * a * N / mle)
    denom = mle**(N+1) * gf(N)
    return num / denom

def rise_over_run_pdf(ds, dmin=1.0, dmax=1E3):
    Nmin = ds[ds >= dmin].shape[0]
    Nmax = ds[ds >= dmax].shape[0]
    deltaN = Nmin - Nmax
    delta_lambda = true_error_pdf(deltaN, kind='mean')
    lambda_max = true_error_pdf(Nmax, kind='mean')
    linear_rise = delta_lambda / lambda_max + 1
    rise = linear_rise.apply(np.log10)
    run = np.log10(dmax) - np.log10(dmin)
    return rise / run

def truncated_pareto_pdf(
    ds, dmin=1.0, dmax=1E3, alpha_min=1E-5, alpha_max=10, 
    n_points=10000, alpha=None
):
    if alpha is None:
        _alpha = np.linspace(alpha_min, alpha_max, n_points)
    else:
        _alpha = alpha
    P = truncated_pareto_P(ds, _alpha, dmin=dmin, dmax=dmax)
    return RandomVariable(_alpha, P, kind='mean')

def slope_pdf(
    ds, dmin=1.0, dmax=1E3, alpha_min=1E-5, alpha_max=10, 
    n_points=10000, alpha=None
):
    if alpha is None:
        _alpha = np.linspace(alpha_min, alpha_max, n_points)
    else:
        _alpha = alpha
    rr_rv = rise_over_run_pdf(ds, dmin=dmin, dmax=dmax)
    tp_rv = truncated_pareto_pdf(
        ds, dmin=dmin, dmax=dmax, alpha_max=alpha_max, 
        n_points=n_points, alpha=alpha
    )
    return rr_rv.update(tp_rv)




