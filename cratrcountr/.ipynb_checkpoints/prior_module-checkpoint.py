from .reverse_fit_module import *

with open('saved/ReverseFit_dict.official.pkl', 'rb') as f:
    ReverseFit_dict = pkl.load(f)

    
def calculate_cumulative_priors(
    ds : np.ndarray, area, age, d_min, D, addition_n=5000, 
    full_output=False, m_pad=0.3, std_pad=0.5
):
    """
    For the cumulative function, the equation for the priors is: 
    logC = log(λ(N_total)/area) + m * log(D/D_min)
    """
    m, b = get_slope_cumulative_binned(
        ds, area, age, uncertainties='asymmetric', 
        pf=loglog_linear_pf(N1=0.001, slope=-2), 
        bin_width_exponent=neukum_bwe, x_axis_position='left',
        reference_point=1.0, skip_zero_crater_bins=False,
        start_at_reference_point=True, d_max=None
    )
    log_max_m = np.log10(-1 * m)
    log_lower_m, log_upper_m = m_pad, m_pad
    rho = np.log10(ds.shape[0] / area)
    m_max = -1 * 10**log_max_m
    m_low = -1 * 10**(log_max_m - log_lower_m)
    m_high = -1 * 10**(log_max_m + log_upper_m)
    log_max, log_high, log_low = tuple([
        rho + m * np.log10(D / d_min) for m in [m_max, m_low, m_high]
    ])
    log_lower = log_max - log_low + std_pad
    log_upper = log_high - log_max + std_pad
    priors = np.array([
        logspace_normal_pdf(lm, ll, lu)
        for lm, ll, lu in zip(log_max, log_lower, log_upper)
    ])
    if full_output:
        return priors, -1 * 10**log_max_m, b
    else:
        return priors


def fit_posteriors(D, posteriors, m_guess, b_guess):
    maxes = np.array([posterior.max for posterior in posteriors])
    lows = np.array([posterior.low for posterior in posteriors])
    highs = np.array([posterior.high for posterior in posteriors])
    lower = np.log10(maxes) - np.log10(lows)
    upper = np.log10(highs) - np.log10(maxes)
    sigma = (lower + upper) / 2
    m, b, switch_count = pick_a_side_fit(
        D, maxes, sigma, m_guess, b_guess, lower, upper
    )
    return m, b

