from .random_variable_module import *


def linear_pf(N1=0.001, slope=-2):
    def out_f(d):
        return 10**(slope * np.log10(d) + np.log10(N1))
    return out_f


def loglog_linear_pf(N1=0.001, slope=-2):
    def out_f(logd):
        return slope * logd + np.log10(N1)
    return out_f


def polynomial_pf(D, coefficients):
    logD = np.log10(D)
    a_n = np.array(coefficients)
    n_max = a_n.shape[0]
    n = np.arange(n_max)
    logD_matrix = np.tile(logD, (n_max, 1)).T
    N1 = 10**np.sum(a_n * logD_matrix**n, axis=1)
    if N1.shape[0] == 1:
        N1 = float(N1[0])
    return N1


def polynomial_pf_dif(D, coefficients):
    logD = np.log10(D)
    a_n = np.array(coefficients)
    n_max = a_n.shape[0]
    n = np.arange(1, n_max)
    p = polynomial_pf(D, coefficients)
    logD_matrix = np.tile(logD, (n_max - 1, 1)).T
    summation = np.sum(a_n[1:] * n * logD_matrix**(n - 1), axis=1)
    return -1 * p / D * summation


def polynomial_pf_R(D, coefficients):
    return polynomial_pf_dif(D, coefficients) * D**3


npf_new_coefficients = np.array([
    -3.0876, -3.557528, 0.781027, 1.021521, -0.156012, -0.444058, 0.019977,
    0.086850, -0.005874, -0.006809, 8.25*10**-4, 5.54*10**-5
])


def npf_new(D):
    return polynomial_pf(D, npf_new_coefficients)
    

def npf_new_dif(D):
    return polynomial_pf_dif(D, npf_new_coefficients)


def npf_new_R(D):
    return polynomial_pf_R(D, npf_new_coefficients)


def npf_new_loglog(logD):
    return np.log10(npf_new(10**logD))


def npf_error(_D):
    m0 = 0.4 / (np.log10(0.8) - np.log10(0.1))
    m1 = 0.1 / (0 - np.log10(0.8))
    m2 = 0.1 / np.log10(3)
    m3 = 0.65 / (np.log10(75) - np.log10(3))
    D = np.array(_D).astype('float')
    return np.piecewise(
        D,
        [
            D <= 0.8,
            (D > 0.8) & (D <= 1.0),
            (D > 1) & (D <= 3.0),
            D > 3.0
        ],
        [
            0.1 + m0 * (np.log10(0.8) - np.log10(D)),
            m1 * (0 - np.log10(D)),
            m2 * np.log10(D),
            0.1 + m3 * (np.log10(D) - np.log10(3))
        ]
    )


npf_mars_coefficients = np.array([
    -3.384, -3.197, 1.257, 0.7915, -0.4861, -0.3630, 0.1016,
    6.756E-2, -1.181E-2, -4.753E-3, 6.233E-4, 5.805E-5
])


def npf_mars(D):
    return polynomial_pf(D, npf_mars_coefficients)


def geometric_sat(D):
    return 0.385 * (D / 2)**-2


def hartmann84_sat(D):
    return 10**(-1.83 * np.log10(D) - 1.33)


def hartmann84_sat_D(age, Ds, pf=npf_new_loglog):
    d = np.flip(Ds)
    diff = np.log10(age) + pf(np.log10(d)) - np.log10(hartmann84_sat(d))
    return np.interp(0, diff, d)


# The Neukum Chronology Function.
def ncf(t):
    return 5.44E-14 *(np.exp(6.93 * t) - 1) + 8.38E-4 * t

# This is an object used in the calculation of ncf_inv.
class ncf_model():
    def __init__(self, nseg=10000):
        self.nseg_pts = nseg
        self.T = np.linspace(0, 5, nseg)
        self.N1 = ncf(self.T)
    def inv(self, N1):
        return np.interp(N1, self.N1, self.T)
    
_ncf_model = ncf_model(nseg=1000000)
    
def ncf_inv(N1, ncf_model=_ncf_model):
    return ncf_model.inv(N1)
    

def ncf_mars(t):
    return 2.68E-14 *(np.exp(6.93 * t) - 1) + 4.13E-4 * t

class ncf_mars_model():
    def __init__(self, nseg=10000):
        self.nseg_pts = nseg
        self.T = np.linspace(0, 5, nseg)
        self.N1 = ncf_mars(self.T)
    def inv(self, N1):
        return np.interp(N1, self.N1, self.T)
    
_ncf_mars_model = ncf_mars_model(nseg=1000000)
    
def ncf_mars_inv(N1, ncf_mars_model=_ncf_mars_model):
    return ncf_mars_model.inv(N1)



def synth_pf(d_km):
    return 10**(-3 - 2*np.log10(d_km))
    

def synth_cf_inv(n1):
    return n1/synth_pf(1)
    

