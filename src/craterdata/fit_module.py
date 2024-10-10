from .base_module import *

class Fit:
    
    def __init__(self, fit_eq, params):
        self.fit_eq = fit_eq
        self.params = params
    
    def apply(self, X):
        return self.fit_eq(*tuple([X] + list(self.params)))


def get_fit(eq, X, Y, p0=None, bounds=None):
    if bounds is None:
        result, cov = optimize.curve_fit(eq, X, Y, p0=p0)
    else:
        result, cov = optimize.curve_fit(eq, X, Y, p0=p0, bounds=bounds)
    return Fit(eq, result)


def polynomial_degree_3(x, a, b, c, d):
    a_n = [a, b, c, d]
    return sum([a_n[n] * x**n for n in range(4)])

def polynomial_degree_5(x, a, b, c, d, e, f):
    return a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f

def polynomial_degree_7(x, a, b, c, d, e, f, g, h):
    a_n = [a, b, c, d, e, f, g, h]
    return sum([a_n[n] * x**n for n in range(8)])

def polynomial_degree_9(x, a, b, c, d, e, f, g, h, i, j):
    a_n = [a, b, c, d, e, f, g, h, i, j]
    return sum([a_n[n] * x**n for n in range(10)])

def polynomial_degree_11(x, a, b, c, d, e, f, g, h, i, j, k, l):
    a_n = [a, b, c, d, e, f, g, h, i, j, k, l]
    return sum([a_n[n] * x**n for n in range(12)])

def polynomial_degree_13(x, a, b, c, d, e, f, g, h, i, j, k, l, m, n):
    a_n = [a, b, c, d, e, f, g, h, i, j, k, l, m, n]
    return sum([a_n[n] * x**n for n in range(14)])

def polynomial_degree_15(x, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p):
    a_n = [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p]
    return sum([a_n[n] * x**n for n in range(16)])


polynomial_eq_dict = {
    3: polynomial_degree_3,
    5: polynomial_degree_5,
    7: polynomial_degree_7,
    9: polynomial_degree_9,
    11: polynomial_degree_11,
    13: polynomial_degree_13,
    15: polynomial_degree_15,
}


class PiecewisePolynomialFit:
    
    def __init__(self, fits, edges):
        self.fits = fits
        self.edges = edges
        
    def apply(self, x):
        x0 = self.edges[0]
        x1 = self.edges[-1]
        m0 = np.polyval(np.polyder(np.flip(self.fits[0].params)), x0)
        m1 = np.polyval(np.polyder(np.flip(self.fits[-1].params)), x1)
        b0 = self.fits[0].apply(x0) - m0 * x0
        b1 = self.fits[-1].apply(x1) - m1 * x1
        line0 = lambda x: m0 * x + b0
        line1 = lambda x: m1 * x + b1
        functions = [line0] + [fit.apply for fit in self.fits] + [line1]
        conditions = [x < x0] + [
            (x >= self.edges[i]) & (x <= self.edges[i + 1])
            for i in range(self.edges.shape[0] - 1)
        ] + [x > x1]
        if type(x) is int:
            x_casted = float(x)
        else:
            x_casted = x
        return np.piecewise(x_casted, conditions, functions)
    
    
def get_PPFit(X, Y, edges, eq):
    fits = [
        get_fit(
            eq, 
            X[(X >= edges[i]) & (X < edges[i + 1])], 
            Y[(X >= edges[i]) & (X < edges[i + 1])]
        )
        for i in range(edges.shape[0] - 1)
    ]
    return PiecewisePolynomialFit(fits, edges)


def save_PPFit(ppfit, file_base):
    ppfit.edges.tofile(
        file_base + '.edges.csv', sep=','
    )
    columns = [
        'a' + str(i) for i in range(ppfit.fits[0].params.shape[0])
    ]
    edges = np.round(ppfit.edges, 3)
    index = [
        str(edges[i]) + ' to ' + str(edges[i + 1])
        for i in range(edges.shape[0] - 1)
    ]

    fit_df = pd.DataFrame(
        np.array([fit.params for fit in ppfit.fits]),
        columns=columns, index=index
    )
    fit_df.to_csv(file_base + '.params.csv')


def read_PPFit(file_base):
    with ior.path('craterdata.files', file_base + '.edges.csv') as f:
        edges = np.loadtxt(f, delimiter=',')
    with ior.path('craterdata.files', file_base + '.params.csv') as f:
        param_df = pd.read_csv(f, index_col=0)
    fit_list = [
        Fit(polynomial_eq_dict[row.shape[0] - 1], row) 
        for i, row in param_df.iterrows()
    ]
    return PiecewisePolynomialFit(fit_list, edges)


