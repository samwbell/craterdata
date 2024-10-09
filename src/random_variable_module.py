from .random_variable_backend_module import *


def round_dec(dec, dn, rounding_n):
    if np.isnan(dec):
        dec_str = 'None'
    else:
        dec_str = str(round(float(dec / 10**dn), rounding_n))
        pre_dot_len = len(dec_str.split('.')[0]) + 1
        dec_str += '0' * (rounding_n + pre_dot_len - len(dec_str))
    return dec_str
    

def plot_label(
    rounding_n, low, high, val, X, P, xlim,
    label_shift_x, label_shift_y, upshift,
    force_label_side, label_text_size, pdf_label,
    color, label_color, unit, pdf_label_size,
    pdf_gap_shift, dn, label_x, label_y, mf
):
    ax = plt.gca()
    fig = plt.gcf()
    v0 = 0
    if dn is None:
        if val == 0:
            dn = 0
            v0 = 1
        elif np.isnan(val):
            dn = 0
        else:
            dn = int(np.floor(np.log10(np.abs(val))))
    if mf and (val != 0):
        if dn == 0:
            val_str = round_dec(val, 0, rounding_n + v0)
            upper_str = round_dec(high / val, 0, rounding_n + v0)
            lower_str = round_dec(val / low, 0, rounding_n + v0)
            exp_str = ''
        elif dn == -1:
            val_str = round_dec(val, 0, rounding_n + 1)
            upper_str = round_dec(high / val, 0, rounding_n + 1)
            lower_str = round_dec(val / low, 0, rounding_n + 1)
            exp_str = ''
        elif dn == 1:
            val_str = round_dec(val, 0, rounding_n - 1)
            upper_str = round_dec(high / val, 0, rounding_n - 1)
            lower_str = round_dec(val / low, 0, rounding_n - 1)
            exp_str = ''
        else:
            val_str = round_dec(val, dn, rounding_n)
            upper_str = round_dec(high / val, 0, rounding_n)
            lower_str = round_dec(val / low, 0, rounding_n)
            exp_str = rf'×10$^{{{dn}}}$'
        upper_str = '×' + upper_str
        lower_str = '÷' + lower_str
    else:
        if dn == 0:
            val_str = round_dec(val, 0, rounding_n + v0)
            upper_str = round_dec(high - val, 0, rounding_n + v0)
            lower_str = round_dec(val - low, 0, rounding_n + v0)
            exp_str = ''
        elif dn == -1:
            val_str = round_dec(val, 0, rounding_n + 1)
            upper_str = round_dec(high - val, 0, rounding_n + 1)
            lower_str = round_dec(val - low, 0, rounding_n + 1)
            exp_str = ''
        elif dn == 1:
            val_str = round_dec(val, 0, rounding_n - 1)
            upper_str = round_dec(high - val, 0, rounding_n - 1)
            lower_str = round_dec(val - low, 0, rounding_n - 1)
            exp_str = ''
        else:
            val_str = round_dec(val, dn, rounding_n)
            upper_str = round_dec(high - val, dn, rounding_n)
            lower_str = round_dec(val - low, dn, rounding_n)
            exp_str = rf'×10$^{{{dn}}}$'
        upper_str = '+' + upper_str
        lower_str = '-' + lower_str
    num_str = rf'${val_str}_{{{lower_str}}}^{{{upper_str}}}$'
    label_str = num_str + exp_str
    if unit is not None:
        label_str += unit
    # if pdf_label is not None:
    #     label_str = pdf_label + label_str
    
    min_X = xlim[0]
    max_X = xlim[1]
    peak_X = X[np.argmax(P)]
    if force_label_side is None:
        if (peak_X - min_X) < (max_X - peak_X):
            label_side = 'right'
            pdf_label_side = 'left'
        else:
            label_side = 'left'
            pdf_label_side = 'right'
    else:
        label_side = force_label_side
    if ax.spines[label_side].get_visible():
        buffer = 0.007
    else:
        buffer = 0
    text_x_dict = {
        'left' : min_X + buffer * (max_X - min_X),
        'right' : max_X - buffer * (max_X - min_X)
    }
    if label_x is None:
        text_x = text_x_dict[label_side] + label_shift_x
    else:
        text_x = label_x
    
    if label_color=='same':
        l_color = color
    else:
        l_color = label_color

    label_text = plt.text(
        text_x, upshift, label_str, ha=label_side, va='bottom',
        size=label_text_size, color=l_color
    )
    if label_side == 'right':
        x0 = min_X + 0.8 * (max_X - min_X)
    else:
        x0 = min_X + 0.2 * (max_X - min_X)
    y0 = np.interp(x0, X, P)
    if label_y is None:
        text_y = y0 + 0.03 * (P.max() - upshift) + label_shift_y
    else:
        text_y = label_y
    
    if pdf_label is not None:
        if pdf_label_size is None:
            _pdf_label_size = label_text_size - 1
        else:
            _pdf_label_size = pdf_label_size
        pdf_text = plt.text(
            text_x, text_y, pdf_label, ha=label_side, va='bottom',
            size=_pdf_label_size, color=l_color
        )
        text_y = y0 + 0.25 * (P.max() - upshift) + label_shift_y
        text_y += pdf_gap_shift
        
    label_text.set_position((text_x, text_y))
    fig.canvas.draw()
    
    
def fix_start(X, P, fixed_start_x, fixed_start_p):
    P = P[X > fixed_start_x]
    X = X[X > fixed_start_x]
    min_X = fixed_start_x
    X = np.insert(X, 0, fixed_start_x)
    if fixed_start_p is not None:
        P = np.insert(P, 0, fixed_start_p)
    else:
        P = np.insert(P, 0, round(P[0]))
    return X, P, min_X
    

def erase_box(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.yticks([])

    
class RandomVariable(MathRandomVariable):
    
    
    def plot(
        self, upshift=0, color='mediumslateblue', 
        fixed_start_x=None, fixed_start_p=None, label=False, 
        rounding_n=2, label_shift_x=0, label_shift_y=0, unit=None, 
        label_text_size=10, force_label_side=None, xlim=None, 
        error_bar_type='same', label_color='same', alpha=0.07,
        pdf_label=None, standardize=True, force_erase_box=None,
        pdf_label_size=None, pdf_gap_shift=0, dn=None,
        label_x=None, label_y=None, lw=2, mf=None
    ):
            
        axis_exists = any(plt.gcf().get_axes())
        if not axis_exists:
            fig = plt.figure(figsize=(5, 2))
            ax = fig.add_subplot(111)
            if force_erase_box is None:
                erase_box(ax)
        else:
            ax = plt.gca()
        if force_erase_box:
            erase_box(ax)
        fig = plt.gcf()
        
        X, P, C = self.X, self.P, self.C()
        if standardize:
            P = P / P.max()
        if fixed_start_x is not None:
            X, P, min_X = fix_start(X, P, fixed_start_x, fixed_start_p)
        P = P + upshift
        if X[0] > X[-1]:
            X = np.flip(X)
            P = np.flip(P)
        
        plt.plot(X, P, color, linewidth=lw)
        if xlim is not None:
            plt.xlim(xlim)
        xlim = ax.get_xlim()
        
        if error_bar_type.lower() not in {'same', self.kind.lower()}:
            krv = self.as_kind(error_bar_type)
            low, val, high = krv.low, krv.val, krv.high
            kind = error_bar_type
        else:
            low, val, high = self.low, self.val, self.high
            kind = self.kind
        if np.isnan(low):
            ilow = np.min(X)
        else:
            ilow = low
        if np.isnan(high):
            ihigh = np.max(X)
        else:
            ihigh = high
        interp_n = np.max([np.sum((X > ilow) & (X < ihigh)), 13000])
        X_interp = np.linspace(ilow, ihigh, interp_n)
        P_interp = np.interp(X_interp, X, P)
        
        plt.fill_between(
            X_interp, upshift, P_interp, facecolor=color, alpha=alpha
        )
        
        low_P = np.interp(low, X_interp, P_interp)
        high_P = np.interp(high, X_interp, P_interp)
        val_P = np.interp(val, X_interp, P_interp)

        plt.plot([low, low], [upshift, low_P], ':', color=color)
        plt.plot([high, high], [upshift, high_P], ':', color=color)
        plt.plot([val, val], [upshift, val_P], color=color)

        if mf is None:
            if kind.lower() in {'log', 'auto log'}:
                _mf = True
            else:
                _mf = False
        else:
            _mf = mf
        
        if label:
            plot_label(
                rounding_n, low, high, val, X, P, xlim,
                label_shift_x, label_shift_y, upshift,
                force_label_side, label_text_size, pdf_label,
                color, label_color, unit, pdf_label_size,
                pdf_gap_shift, dn, label_x, label_y, _mf
            )



true_error_dict = {}


def true_error_pdf_single(
    N, n_points=10000, cum_prob_edge=1E-7, kind='log'
):
    
    if tuple([N, n_points, cum_prob_edge, kind]) in true_error_dict:
        
        return_rv = true_error_dict[tuple([
            N, n_points, cum_prob_edge, kind
        ])]
        
    else:
    
        X, P = true_error_pdf_XP(
            N, n_points=n_points, cum_prob_edge=cum_prob_edge
        )
        
        val, lower, upper = get_error_bars(
            N, log_space=False, kind=kind, return_val=True
        )
        low = val - lower
        high = val + upper

        return_rv = RandomVariable(
            X, P, val=val, low=low, high=high, kind=kind
        )

        true_error_dict[tuple([
            N, n_points, cum_prob_edge, kind
        ])] = return_rv
    
    return return_rv


def true_error_pdf(N_raw, n_points=10000, cum_prob_edge=1E-7, kind='log'):
    '''A numerical PDF of cratering rate (lambda), producing paired 
    cratering rate and propability arrays.  It creates minor numerical 
    errors at the low end of the distribution, which are mitigated by 
    oversampling the low end.
    
    Inputs
    
    N: the number of craters observed (cannot be negative but does not
       have to be a whole number), can be array or int
    
    n_points: gives the number of points in the resulting random variable
              object (default 10000)
              
    pivot_point_n: gives the number of points in percentile space
                          where oversampling begins (default 10)
    
    Output
    
    returns the PDF as a RandomVariable object
    '''
    if type(N_raw) in {np.ndarray, list, set}:
        return np.array([true_error_pdf_single(
            N, n_points=n_points, cum_prob_edge=cum_prob_edge, kind=kind
        ) for N in N_raw])
    else:
        return true_error_pdf_single(
            N_raw, n_points=n_points, cum_prob_edge=cum_prob_edge,
            kind=kind
        )


def sqrt_N_error_pdf(N):
    sqrtn_lambda = np.linspace(-3, 5 + N + 5 * np.sqrt(N), 10000)
    sqrtn_P = norm.pdf(sqrtn_lambda, loc=N, scale=np.sqrt(N))
    low = N - np.sqrt(N)
    high = N + np.sqrt(N)
    return RandomVariable(
        sqrtn_lambda, sqrtn_P, val=N, low=low, high=high, 
        kind='sqrt(N)'
    )


def get_bin_info(sample_array, min_val, max_val, n_bins, n_bins_baseline,
                 slope_data):
    
    sample_min, sample_max = np.min(sample_array), np.max(sample_array)
    
    if min_val == 'Auto':
        min_X = sample_min - 0.1 * (sample_max - sample_min)
        if np.min(sample_array) > 0 and min_X < 0:
            min_X = 0
    else:
        min_X = min_val
        
    if max_val == 'Auto':
        if slope_data:
            max_X = np.max(sample_array[sample_array < 0])
        else:
            max_X = sample_max + 0.1 * (sample_max - sample_min)
    else:
        max_X = max_val
        
    if n_bins == 'Auto':
        X_P_min = np.percentile(sample_array, 1)
        X_P_max = np.percentile(sample_array, 99)
        range_ratio = (max_X - min_X) / (X_P_max - X_P_min)
        n_bins_used = int(round(n_bins_baseline * range_ratio))
    else:
        n_bins_used = n_bins
        
    bins = np.linspace(min_X, max_X, n_bins_used)
    
    return bins, min_X, max_X


def make_pdf_from_samples(
    samples, n_bins='Auto', n_points=100000, min_val='Auto', max_val='Auto',
    fixed_start_x=None, fixed_start_p=None, n_bins_baseline=50, 
    slope_data=False, drop_na=True, kind='auto log'
):
    
    sample_array = np.array(samples)
    if drop_na:
        sample_array = sample_array[~np.isnan(sample_array)]
    
    bins, min_X, max_X = get_bin_info(
        sample_array, min_val, max_val, n_bins, n_bins_baseline, slope_data
    )
    
    P, bin_edges = np.histogram(sample_array, bins=bins, density=True)
    P = P / max(P)
    bin_edges = np.array(bin_edges)
    X = (bin_edges[1:] + bin_edges[:-1]) / 2
    if fixed_start_x is not None:
        X, P, min_X = fix_start(X, P, fixed_start_x, fixed_start_p)
        
    X_interp = np.linspace(min_X, max_X, n_points)
    P_interp = np.interp(X_interp, X, P)

    if slope_data and kind=='auto log':
        log_max, log_low, log_high = fit_slope_pdf(X, P)
        high = -1 * 10**(log_max - log_low)
        low = -1 * 10**(log_max + log_high)
        val = -1 * 10**log_max
        return_rv = RandomVariable(
            X_interp, P_interp, low=low, high=high, val=val, kind=kind
        )
    else:
        return_rv = RandomVariable(X_interp, P_interp, kind=kind)
        if kind.lower() == 'median':
            ps = [100 - 100 * p_1_sigma, 50.0, 100 * p_1_sigma]
            low, val, high = np.percentile(sample_array, ps)
            return_rv = RandomVariable(
                X_interp, P_interp, val=val, low=low, high=high,
                kind=kind
            )
        else:
            return_rv = RandomVariable(X_interp, P_interp, kind=kind)

    return return_rv


def ash_pdf(data, nbins=25, nshifts=10, kind='mean'):
    bins, heights = ash.ash1d(data, nbins, nshifts)
    if kind == 'mean':
        val = np.mean(data)
        low = np.percentile(data, 100 * (1 - p_1_sigma))
        high = np.percentile(data, 100 * p_1_sigma)
    else:
        val, low, high = None, None, None
    return RandomVariable(
        bins, heights, kind=kind, val=val, low=low, high=high
    )


def logspace_normal_pdf(log_max, log_lower, log_upper,
                        negative=False):
    X_min = log_max - 5 * log_lower
    X_max = log_max + 5 * log_upper
    X = np.linspace(X_min, X_max, 10000)
    X_left = X[X < log_max]
    X_right = X[X >= log_max]
    left = norm.pdf(X_left, log_max, log_lower) / norm.pdf(
            X, log_max, log_lower).max()
    right = norm.pdf(X_right, log_max, log_upper) / norm.pdf(
            X, log_max, log_upper).max()
    P = np.append(left, right)
    if negative:
        rv = RandomVariable(
            -1 * 10**X, P, val=-1 * 10**log_max, 
            high=-1 * 10**(log_max - log_lower),
            low=-1 * 10**(log_max + log_upper)
        )
    else:
        rv = RandomVariable(
            10**X, P, val=10**log_max, 
            low=10**(log_max - log_lower),
            high=10**(log_max + log_upper)
        )
    return rv



def _get(param, i):
    if type(param) in {list, np.array, set}:
        return param[i]
    else:
        return param

def plot_pdfs(
    rvs, color=cs, fixed_start_x=None, fixed_start_p=None, label=False, 
    rounding_n=2, label_shift_x=0, label_shift_y=0, unit=None, 
    label_text_size=10, force_label_side=None, xlim=None, 
    error_bar_type='same', label_color='same', alpha=0.07,
    pdf_label=None, standardize=True, force_erase_box=None,
    pdf_label_size=None, pdf_gap_shift=0, dn=None
 ):
    for i in range(len(rvs)):
        rvs[i].plot(
            upshift = 1.1 * (len(rvs) - i - 1), xlim=xlim,
            color=_get(color, i),
            fixed_start_x=_get(fixed_start_x, i), 
            fixed_start_p=_get(fixed_start_p, i), 
            label=_get(label, i), 
            rounding_n=_get(rounding_n, i), 
            label_shift_x=_get(label_shift_x, i), 
            label_shift_y=_get(label_shift_y, i), 
            unit=_get(unit, i), 
            label_text_size=_get(label_text_size, i), 
            force_label_side=_get(force_label_side, i), 
            error_bar_type=_get(error_bar_type, i), 
            label_color=_get(label_color, i), 
            alpha=_get(alpha, i),
            pdf_label=_get(pdf_label, i), 
            standardize=_get(standardize, i), 
            force_erase_box=_get(force_erase_box, i), 
            pdf_label_size=_get(pdf_label_size, i), 
            pdf_gap_shift=_get(pdf_gap_shift, i), 
            dn=_get(dn, i)
        )

def combine_rvs(rv_list):
    rvs = rv_list.copy()
    while len(rvs) > 2:
        n = math.floor(len(rvs) / 2)
        last_rv = rvs[-1]
        if len(rvs) > n * 2:
            is_odd = True
        else:
            is_odd = False
        rvs = [
            (rv_list[2 * i].update(rv_list[2 * i + 1])).normalize() 
            for i in range(n)
        ]
        if is_odd:
            rvs.append(last_rv)
    rv = rvs[0].update(rvs[1])
    return rv


