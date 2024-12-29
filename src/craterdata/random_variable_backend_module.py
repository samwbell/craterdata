from .pdf_fitting_module import *



class CoreRandomVariable:
    
    def __init__(self, X, P, val=None, low=None, high=None):
        
        self.X = np.array(X)
        self.P = np.array(P)
        self.val = val
        self.low = low
        self.high = high

        if None in {self.val, self.low}:
            self.lower = None
        else:
            self.lower = self.val - self.low

        if None in {self.val, self.high}:
            self.upper = None
        else:
            self.upper = self.high - self.val

    def new_kwargs(self):
        crv_code = CoreRandomVariable.__init__.__code__
        class_args = set(crv_code.co_varnames[1:crv_code.co_argcount])
        self_code = self.__init__.__code__
        self_args = set(self_code.co_varnames[1:self_code.co_argcount])
        new_args = self_args - class_args
        return {k:v for k, v in self.__dict__.items() if k in new_args}

    def __getitem__(self, a):
        if isinstance(a, slice):
            return self.__class__(
                self.X[a.start : a.stop : a.step],
                self.P[a.start : a.stop : a.step],
                val=self.val, low=self.low, high=self.high, 
                **self.new_kwargs()
            )
        elif np.array(a).shape == ():
            return np.interp(a, self.X, self.P)
        else:
            return self.__class__(
                self.X[a],
                self.P[a],
                val=self.val, low=self.low, high=self.high, 
                **self.new_kwargs()
            )

    def C(self):
        non_inf = (self.X > -1 * np.inf) & (self.X < 1 * np.inf)
        C = self.P.copy()
        C[non_inf] = cumulative_trapezoid(
            self.P[non_inf], self.X[non_inf], initial=0
        )
        C[non_inf] = C[non_inf] / C[non_inf].max()
        C[self.X == -1 * np.inf] = 0
        C[self.X == np.inf] = 1
        return C

    def percentile(self, p):
        v = np.interp(p, self.C(), self.X)
        _p = np.array(p)
        is_scalar = v.shape == ()
        if is_scalar:
            v = v.reshape((1,))
        v[(_p < 0) | (_p > 1)] = np.nan
        if is_scalar:
            v = v[0]
        return v
    
    def sample(self, n_samples, n_points=10000):
        X = np.linspace(self.X.min(), self.X.max(), n_points)
        P = np.interp(X, self.X, self.P)
        P = P / P.sum()
        return np.random.choice(X, p=P, size=n_samples)

    def function(self):
        def r_func(x):
            return np.interp(x, self.X, self.P)
        return r_func

    def normalize(self):
        integral = trapezoid(self.P, self.X)
        return self.__class__(
            self.X, self.P / integral, 
            val=self.val, low=self.low, high=self.high, 
            **self.new_kwargs()
        )

    def standardize(self):
        return self.__class__(
            self.X, self.P / self.P.max(),
            val=self.val, low=self.low, high=self.high, 
            **self.new_kwargs()
        )

    def match_X_of(self, other):
        return self.__class__(
            other.X, np.interp(other.X, self.X, self.P),
            val=self.val, low=self.low, high=self.high, 
            **self.new_kwargs()
        )

    def match_X(self, X):
        return self.__class__(
            X, np.interp(X, self.X, self.P),
            val=self.val, low=self.low, high=self.high, 
            **self.new_kwargs()
        )

    def cut_below(self, c, recalculate=False):
        if recalculate:
            return self.__class__(
                self.X[self.X > c], self.P[self.X > c],
                val=None, low=None, high=None, **new_kwargs()
            )
        else:
            return self.__class__(
                self.X[self.X > c], self.P[self.X > c],
                val=self.val, low=self.low, high=self.high, 
                **self.new_kwargs()
            )

    def cut_above(self, c, recalculate=False):
        if recalculate:
            return self.__class__(
                self.X[self.X < c], self.P[self.X < c],
                val=None, low=None, high=None, **new_kwargs()
            )
        else:
            return self.__class__(
                self.X[self.X < c], self.P[self.X < c],
                val=self.val, low=self.low, high=self.high, 
                **self.new_kwargs()
            )

    def update(self, other):
        other_P = np.interp(self.X, other.X, other.P)
        kwargs = self.__dict__.copy()
        kwargs['val'] = None
        kwargs['low'] = None
        kwargs[''] = None
        return self.__class__(
            self.X, self.P * other_P, val=None, low=None, high=None,
            **self.new_kwargs()
        )

    def trim(self, precision=0.9999, recalculate=False):
        trim_max = self.percentile(precision)
        X_new = self.X[self.X < trim_max]
        P_new = self.P[self.X < trim_max]
        trim_min = self.percentile(1 - precision)
        if X_new[0] < 0:
            P_new = P_new[X_new > trim_min]
            X_new = X_new[X_new > trim_min]
        if recalculate:
            return self.__class__(
                X_new, P_new,
                val=None, low=None, high=None, **self.new_kwargs()
            )
        else:
            return self.__class__(
                X_new, P_new,
                val=self.val, low=self.low, high=self.high, 
                **self.new_kwargs()
            )


class BaseRandomVariable(CoreRandomVariable):

    def __init__(
        self, X, P, val=None, low=None, high=None, kind='log'
    ):
        super().__init__(X, P, val=val, low=low, high=high)
        self.kind=kind

        if kind is None:
            self.val = None
            self.low = None
            self.high = None
            self.lower = None
            self.upper = None

        elif type(kind) != str:
            raise ValueError(
                'kind must be a string: \'log\', \'auto log\', '
                '\'linear\', \'median\', \'mean\' or \'sqrt(N)\''
            )
            
        elif kind.lower() == 'log':
            if self.val in {None, np.nan}:
                self.val = self.X[np.argmax(self.P)]
            if low in {None, np.nan} or high in {None, np.nan}:
                log_lower, log_upper = error_bar_log(
                    self.X, self.P, max_likelihood=self.val
                )
            if low in {None, np.nan}:
                self.low = 10**(np.log10(self.val) - log_lower)
            if high in {None, np.nan}:
                self.high = 10**(np.log10(self.val) + log_upper)

        elif kind.lower() in {'auto log', 'auto_log'}:
            if {None, np.nan} & {low, val, high}:
                log_max, log_lower, log_upper = fit_log_of_normal(
                    self.X, self.P
                )
            if self.val in {None, np.nan}:
                self.val = 10**log_max
            if low in {None, np.nan}:
                self.low = 10**(log_max - log_lower)
            if high in {None, np.nan}:
                self.high = 10**(log_max + log_upper)

        elif kind.lower() == 'linear':
            if self.val is None:
                self.val = self.X[np.argmax(self.P)]
            if low in {None, np.nan} or high in {None, np.nan}:
                lower, upper = error_bar_linear(
                    self.X, self.P, max_likelihood=self.val
                )
            if low in {None, np.nan}:
                self.low = self.val - lower
            if high in {None, np.nan}:
                self.high = self.val + upper

        elif kind.lower() in {'median', 'percentile'}:
            if {None, np.nan} & {low, val, high}:
                _low, _val, _high = self.percentile([
                    1 - p_1_sigma, 0.5, p_1_sigma
                ])
            if self.val in {None, np.nan}:
                self.val = _val
            if self.low in {None, np.nan}:
                self.low = _low
            if self.high in {None, np.nan}:
                self.high = _high

        elif kind.lower() == 'mean':
            if {None, np.nan} & {low, val, high}:
                _low, _high = self.percentile([
                    1 - p_1_sigma, p_1_sigma
                ])
            if self.val in {None, np.nan}:
                self.val = self.mean()
            if self.low in {None, np.nan}:
                self.low = _low
            if self.high in {None, np.nan}:
                self.high = _high

        elif kind.lower() == 'moments':
            if self.val is None:
                self.val = rv_mean_XP(self.X, self.P)
            if low in {None, np.nan} or high in {None, np.nan}:
                self.std = rv_std_XP(self.X, self.P)
                self.skewness = rv_skewness_XP(self.X, self.P)
            if low in {None, np.nan}:
                self.low = self.val - self.std
            if high in {None, np.nan}:
                self.high = self.val + self.std

        elif kind.lower() in {'sqrt(n)', 'sqrtn', 'sqrt n'}:
            if self.val is None:
                self.val = self.X[np.argmax(self.P)]
            if low in {None, np.nan}:
                self.low = self.val - np.sqrt(self.val)
            if high in {None, np.nan}:
                self.high = self.val + np.sqrt(self.val)

        else:
            raise ValueError(
                'kind must be: \'log\', \'auto log\', '
                '\'linear\', \'median\', \'mean\', '
                '\'moments\', \'sqrt(N)\', or None'
            )

        self.lower = self.val - self.low
        self.upper = self.high - self.val

    
    def as_kind(self, kind):
        if kind == self.kind:
            return self
        else:
            return self.__class__(
                self.X, self.P, low=None, val=None, high=None, kind=kind
            )

    def mean(self):
        return rv_mean_XP(self.X, self.P)

    def std(self):
        return rv_std_XP(self.X, self.P)

    def skewnewss(self):
        return rv_skewness_XP(self.X, self.P)

    def max(self):
        return self.X[np.argmax(self.P)]

    def mode(self):
        return self.X[np.argmax(self.P)]


    
def apply2rv(rv, f, kind=None, even_out=True):
    X = rv.X
    PX = rv.P
    C = rv.C()
    Y = f(X)
    v = np.isfinite(Y) & ~np.isnan(Y)
    X, Y, C = X[v], Y[v], C[v]
    PY = np.gradient(C, Y)
    if Y[0] > Y[-1]:
        PY = -1 * PY
        Y, PY = np.flip(Y), np.flip(PY)
    if kind is None:
        _kind = rv.kind
    else:
        _kind = kind
    if even_out:
        Y_even_spacing = np.linspace(
            Y.min(), Y.max(), Y.shape[0], endpoint=True
        )
        PY_even_spacing = np.interp(Y_even_spacing, Y, PY)
        return rv.__class__(Y_even_spacing, PY_even_spacing, kind=_kind)
    else:
        return rv.__class__(Y, PY, kind=_kind)



class MathRandomVariable(BaseRandomVariable):
    
    def __add__(self, other):
        if isinstance(other, MathRandomVariable):
            X_new_min = np.min([self.X.min(), other.X.min()])
            X_new_max = np.max([self.X.max(), other.X.max()])
            X_new_n = np.max([self.X.shape[0], other.X.shape[0]])
            X_new = np.linspace(X_new_min, X_new_max, X_new_n)
            self_P = np.interp(X_new, self.X, self.P)
            other_P = np.interp(X_new, other.X, other.P)
            conv_P = np.convolve(self_P, other_P)
            conv_n = conv_P.shape[0]
            conv_X = np.linspace(2 * X_new.min(), 2 * X_new.max(), conv_n)
            return self.__class__(conv_X, conv_P, kind=self.kind)
        elif other == 0:
            return self
        else:
            return self.__class__(
                self.X + other, self.P, val=self.val + other,
                low=self.low + other, high=self.high + other, 
                kind=self.kind
            )
    
    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)
    
    def __mul__(self, other):
        if isinstance(other, MathRandomVariable):
            f1 = self.function()
            f2 = other.function()
            X1 = self.X
            X2 = other.X
            Y = np.linspace(
                X1.min() * X2.min(), 
                X1.max() * X2.max(), 
                max(X1.shape[0], X2.shape[0]), 
                endpoint=True
            )[:, np.newaxis]
            # Because Y has even spacing, we can use np.sum
            Py = (f1(X1) * f2(Y / X1) / np.abs(X1)).sum(axis=1)
            return self.__class__(Y.T[0], Py, kind=self.kind)
        elif other == 0:
            return 0
        elif other < 0:
            return self.__class__(
                np.flip(self.X * other), np.flip(self.P), 
                val=self.val * other, high=self.low * other, 
                low=self.high * other, kind=self.kind
            )
        else:
            return self.__class__(
                self.X * other, self.P, val=self.val * other,
                low=self.low * other, high=self.high * other, 
                kind=self.kind
            )
    
    def __rmul__(self, other):
        if other == 0:
            return 0
        else:
            return self.__mul__(other)

    def __sub__(self, other):
        return self + (-1 * other)
    
    def __rsub__(self, other):
        return (-1 * self) + other
        
    def __truediv__(self, other):
        if isinstance(other, MathRandomVariable):
            f1 = self.function()
            f2 = other.function()
            X1 = self.X
            X2 = other.X
            Y = np.linspace(
                X1.min() / np.percentile(other.X, 99.99), 
                X1.max() / np.percentile(other.X, 0.01), 
                max(X1.shape[0], X2.shape[0]), 
                endpoint=True
            )[:, np.newaxis]
            # Because Y has even spacing, we can use np.sum
            Py = (f1(Y * X2) * f2(X2) * np.abs(X2)).sum(axis=1)
            return self.__class__(Y.T[0], Py, kind=self.kind)
        elif other < 0:
            return self.__class__(
                self.X / other, self.P, val=self.val / other,
                high=self.low / other, low=self.high / other, 
                kind=self.kind
            )
        else:
            return self.__class__(
                self.X / other, self.P, val=self.val / other,
                low=self.low / other, high=self.high / other, 
                kind=self.kind
            )
        
    def __rtruediv__(self, other):
        if isinstance(other, MathRandomVariable):
            return other.__truediv__(self)
        elif other < 0:
            return self.__class__(
                other / self.X, self.P, val=other / self.val,
                high=other / self.low, low=other / self.high, 
                kind=self.kind
            )
        else:
            return self.__class__(
                other / self.X, self.P, val=other / self.val,
                low=other / self.low, high=other / self.high, 
                kind=self.kind
            )
    
    def __rpow__(self, other):
        if isinstance(other, MathRandomVariable):
            raise ValueError(
                'The a**X operator is not for applying exponential '
                'functions to random variables.  It is for scaling '
                'random variables back out of log space with 10**X.  '
                'As a result, it cannot be applied to two random '
                'variables.'
            )
        else:
            if self.kind == 'linear':
                _kind = 'log'
            else:
                _kind = self.kind
            return self.__class__(
                other**self.X, self.P, val=other**self.val,
                low=other**self.low, high=other**self.high, 
                kind=_kind
            )

    def apply(self, f, kind=None, even_out=True):
        return apply2rv(self, f, kind=kind)

    def ten2the(self, kind=None):
        return self.apply(lambda x: 10**x, kind=kind)

    def scale(self, f, recalculate_bounds=True):
        X, P = f(self.X), self.P
        if X[0] > X[-1]:
            X, P = np.flip(X), np.flip(P)
        if recalculate_bounds:
            return self.__class__(
                X, P, low=None, val=None, high=None, kind=self.kind
            )
        else:
            return self.__class__(
                X, P, kind=self.kind,
                val=f(self.val), low=f(self.low), high=f(self.high)
            )

    def log(self, recalculate_bounds=False):
        rv_log = self[self.X > 0].scale(
            np.log10, recalculate_bounds=recalculate_bounds
        )
        if self.kind.lower() in {'log', 'auto log'}:
            rv_log.kind = 'linear'
        return rv_log


