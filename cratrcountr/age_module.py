from .production_function_module import *

def N1_pdf(N, area, dmin, pf=npf_new, kind='log'):
    lambda_rv = true_error_pdf(N, kind=kind)
    crater_density_rv = lambda_rv / area
    N1_shift = pf(1) / pf(dmin)
    N1_rv = crater_density_rv * N1_shift
    return N1_rv


def age_pdf(N, area, dmin, pf=npf_new, cf_inv=ncf_inv, kind='log'):
    N1_rv = N1_pdf(N, area, dmin, pf=pf, kind=kind)
    age_rv = N1_rv.apply(cf_inv)
    return age_rv


def m16_age_p(N, A, dmin, pf, cf, t):
    Cdmin = pf(dmin) / pf(1) * cf(t)
    return np.exp(-1 * A * Cdmin) * cf(t)**N


def m16_age_pdf(
    N, A, dmin, pf=npf_new, cf=ncf, cf_inv=ncf_inv, kind='median'
):
    N_min = true_error_pdf(N).percentile(0.00001)
    N_max = true_error_pdf(N).percentile(0.99999)
    T_min = cf_inv(float(N_min / A * pf(1) / pf(dmin)))
    T_max = cf_inv(float(N_max / A * pf(1) / pf(dmin)))
    T = np.linspace(T_min, T_max, 10000)
    P = m16_age_p(N, A, dmin, pf, cf, T)
    return RandomVariable(T, P, kind=kind)


def age_scaled_N1_pdf(
    N, area, dmin, pf=npf_new, cf_inv=ncf_inv, kind='log'
):
    N1_rv = N1_pdf(N, area, dmin, pf=pf, kind=kind)
    return RandomVariable(cf_inv(N1_rv.X), N1_rv.P)


