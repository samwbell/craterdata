from .synth_module import *
    
    
def reverse_model(observed_slope, mean_slope_factor, 
                  mean_upper, mean_lower):
    n_X = 10000
    X_min = 0.2
    X_max = 17.0
    X = np.logspace(np.log10(X_min), np.log10(X_max), 
                    n_X, endpoint=False)
    slope_array = -1 * np.logspace(np.log10(X_min), 
                                   np.log10(X_max), 
                                   500, endpoint=False)
    prob_list = []
    for test_slope in slope_array:
        log_max = mean_slope_factor * np.log10(-1 * test_slope)
        log_upper = mean_upper
        log_lower = mean_lower
        P = piecewise_normal_fit_equation(np.log10(X), log_max, 
                                          mean_lower, mean_upper)
        P = P / P.sum()
        prob = np.interp(-1 * observed_slope, X, P)
        inc = (X_max - X_min) / n_X
        prob /= inc
        prob_list.append(prob)
    log_max, log_lower, log_upper = fit_log_of_normal(
                    -1 * slope_array, np.array(prob_list))
    return log_max, log_lower, log_upper


class SearchFit:
    
    def __init__(self, max_factor_fit, lower_fit, upper_fit):
        self.max_factor_fit = max_factor_fit
        self.lower_fit = lower_fit
        self.upper_fit = upper_fit
    
    def apply(self, N=20, slope=-2.0):
        max_factor = self.max_factor_fit.apply(np.log10(N))
        log_max = np.log10(-1 * max_factor * slope)
        log_lower = 10**self.lower_fit.apply(np.log10(N))
        log_upper = 10**self.upper_fit.apply(np.log10(N))
        return log_max, log_lower, log_upper
    
    def plot(self, N, slope=-2.0, color='mediumslateblue', alpha=0.07,
             lw=1.5):
        log_max, log_lower, log_upper = self.apply(N=N, slope=slope)
        low = -1 * 10**(log_max + log_upper)
        high = -1 * 10**(log_max - log_lower)
        plt.plot(N, -1 * 10**(log_max), color=color, lw=lw)
        plt.plot(N, low, ':', color=color, lw=lw)
        plt.plot(N, high, ':', color=color, lw=lw)
        plt.fill_between(N, low, high, alpha=alpha, color=color)
        
        
class ReverseFit(SearchFit):
    
    def slope_plot(self, slopes=np.linspace(-4, -0.5, 20, endpoint=True), 
                   N=20, color='mediumslateblue', alpha=0.07, lw=1.5):
        log_max, log_lower, log_upper = self.apply(N=N, slope=slopes)
        low = -1 * 10**(log_max + log_upper)
        high = -1 * 10**(log_max - log_lower)
        plt.plot(slopes, slopes, 'k', lw=1)
        plt.plot(slopes, -1 * 10**(log_max), color=color, lw=lw)
        plt.plot(slopes, low, ':', color=color, lw=lw)
        plt.plot(slopes, high, ':', color=color, lw=lw)
        plt.fill_between(slopes, low, high, alpha=alpha, color=color, lw=lw)
        
        
class NSearchFit(SearchFit):
        
    def apply(self, N=20, slope=-2.0):
        max_factor = self.max_factor_fit.apply(N)
        log_max = np.log10(-1 * max_factor * slope)
        log_lower = 10**self.lower_fit.apply(np.log10(N))
        log_upper = 10**self.upper_fit.apply(np.log10(N))
        return log_max, log_lower, log_upper
    
    def reverse_model(self, N=20, slope=-2.0):
        max_factor = self.max_factor_fit.apply(N)
        log_lower = 10**self.lower_fit.apply(np.log10(N))
        log_upper = 10**self.upper_fit.apply(np.log10(N))
        return reverse_model(slope, max_factor, log_lower, log_upper)
    
    def reverse_plot(self, N, slope=-2.0, color='mediumslateblue', 
                     alpha=0.07):
        reverse_matrix = np.array([self.reverse_model(N=Ni, slope=slope)
                                   for Ni in N])
        log_max, log_lower, log_upper = tuple(reverse_matrix.T)
        low = -1 * 10**(log_max + log_upper)
        high = -1 * 10**(log_max - log_lower)
        plt.plot(N, -1 * 10**(log_max), color=color)
        plt.plot(N, low, ':', color=color)
        plt.plot(N, high, ':', color=color)
        plt.fill_between(N, low, high, alpha=alpha, color=color)
        
        
def fit_eq_max_factor(x, a, b):
    return 1.0 - a * x**b

def cf_eq(x, a, b, c):
    return a * (np.exp(b * x) - 1) + c * x

def cf_eq2(x, a, b, c, d):
    return 10**a * (np.exp(b * x) - 1) + 10**c * x + 10**d * x**2

def linear_eq(x, m, b):
    return m * x + b

def power_eq(x, a, b, c, d):
    return ((x + a) / b)**c + d

power_eq_bounds = tuple([[-1, 0, -50, -5], [100, 100, 0, 5]])


def get_reverse_fit(reverse_matrix, slope, 
                    X = np.linspace(1, 4, 200, endpoint=True)):
    Y = -1 * 10**reverse_matrix[:, 0] / slope
    max_factor_fit = get_fit(power_eq, X, Y, 
                             bounds=power_eq_bounds)
    Y = np.log10(reverse_matrix[:, 1])
    lower_fit = get_fit(polynomial_degree_5, X, Y)
    Y = np.log10(reverse_matrix[:, 2])
    upper_fit = get_fit(polynomial_degree_5, X, Y)
    return ReverseFit(max_factor_fit, lower_fit, upper_fit)


class SlopeSearchParams:
    def __init__(self, plot_type='unbinned', N=None,
                 use_uncertainties=False, pick_a_side=False):
        self.plot_type = plot_type
        self.N = N
        self.use_uncertainties = use_uncertainties
        self.pick_a_side = pick_a_side
        part_1 = plot_type
        if plot_type == 'unbinned corrected':
            part_1 = 'unbinned_cor'
        if use_uncertainties:
            part_2 = '_u'
        else:
            part_2 = ''
        if pick_a_side:
            part_3 = '_p'
        else:
            part_3 = ''
        if N is not None:
            part_3 += '_' + str(N)
        self.str = part_1 + part_2 + part_3
        if use_uncertainties:
            part_2 = '.uncertainties'
        else:
            part_2 = ''
        if pick_a_side:
            part_3 = '.pick_a_side'
        else:
            part_3 = ''
        if N is not None:
            part_3 += '_' + str(N)
        self.file_str = part_1 + part_2 + part_3

        
def plot_matrix(matrix, Ns, color='black', ms=4):
    log_max, log_lower, log_upper = tuple(matrix.T)
    plt.plot(Ns, -1 * 10**log_max, '.', color=color, ms=ms)
    plt.plot(Ns, -1 * 10**(log_max - log_lower), '.', color=color, ms=ms)
    plt.plot(Ns, -1 * 10**(log_max + log_upper), '.', color=color, ms=ms)
    
    
