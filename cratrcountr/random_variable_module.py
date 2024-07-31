from .random_variable_backend_module import *


def plot_label(
    rounding_n, low, self_max, high, mid, X, label_shift_x,
    label_shift_y, upshift, xlim, right_position_modifier,
    force_label_side, fig_size_adjustor, label_text_size, 
    color, label_color, unit
):
    ax = plt.gca()
    f_str = "{:." + str(rounding_n) + "f}"
    label_text = f_str.format(round(float(mid), rounding_n))
    low_text = '-' + f_str.format(round(float(mid - low), rounding_n))
    high_text = '+' + f_str.format(round(float(high - mid), rounding_n))
    display_text = f"$^{{{high_text}}}$" + f"\n$_{{{low_text}}}$"
    if xlim is None:
        min_X = X.min()
        max_X = X.max()
    else:
        min_X = xlim[0]
        max_X = xlim[1]
    left_postion = min_X
    right_postion = right_position_modifier * max_X
    if (self_max - min_X) < (max_X - self_max):
        text_x = right_postion
    else:
        text_x = left_postion
    if force_label_side == 'right':
        text_x = right_postion
    if force_label_side == 'left':
        text_x = left_postion
    if label_color=='same':
        l_color = color
    else:
        l_color = label_color
    fig_size = ax.figure.get_size_inches()[0] * plt.gca().get_position().width
    fig_size_adjustment = fig_size_adjustor / fig_size
    plt.text(text_x + label_shift_x, 0.8 + label_shift_y + upshift, 
             label_text, ha='left', va='center',
             size=label_text_size, color=l_color)
    label_factor = (0.02 + 0.02 * len(label_text))
    x_adjustment = fig_size_adjustment * label_factor * (max_X - min_X)
    plt.text(text_x + x_adjustment + label_shift_x, 
             0.8 + label_shift_y + upshift, display_text, 
             ha='left', va='center', multialignment='left', 
             size=label_text_size, color=l_color)
    if unit is not None:
        plt.text(
            text_x + 2 * x_adjustment + label_shift_x, 
            0.8 + label_shift_y + upshift, unit, 
            ha='left', va='center', multialignment='left', 
            size=label_text_size, color=l_color
        )
    return text_x
    
    
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
        rounding_n=2, label_shift_x=0, label_shift_y=0, 
        fig_size_adjustor=3.25, label_text_size=10, 
        return_text_x=False, force_label_side=None, xlim=None, 
        right_position_modifier=0.8, error_bar_type='same',
        label_color='same', alpha=0.07, unit=None
    ):
            
        axis_exists = any(plt.gcf().get_axes())
        if not axis_exists:
            fig = plt.figure(figsize=(5, 2))
            ax = fig.add_subplot(111)
            erase_box(ax)
        
        X, P, C = self.X, self.P, self.C()
        P = P / P.max()
        if fixed_start_x is not None:
            X, P, min_X = fix_start(X, P, fixed_start_x, fixed_start_p)
        P = P + upshift
        
        plt.plot(X, P, color, linewidth=2)

        if error_bar_type.lower() not in {'same', self.kind.lower()}:
            krv = self.as_kind(error_bar_type)
            low, val, high = krv.low, krv.val, krv.high
        else:
            low, val, high = self.low, self.val, self.high
        interp_n = np.max([np.sum((X > low) & (X < high)), 13000])
        X_interp = np.linspace(low, high, interp_n)
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
        
        if label:
            text_x = plot_label(
                rounding_n, low, self.val, high, val, X,
                label_shift_x, label_shift_y, upshift, xlim, 
                right_position_modifier, force_label_side,
                fig_size_adjustor, label_text_size, color,
                label_color, unit
            )

        if xlim is not None:
            plt.xlim(xlim)
            
        if return_text_x:
            return text_x



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
        
        lower, upper = get_error_bars(N, log_space=False, kind=kind)
        low = N - lower
        high = N + upper

        return_rv = RandomVariable(
            X, P, val=N, low=low, high=high, kind=kind
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

    ps = [100 - 100 * p_1_sigma, 50.0, 100 * p_1_sigma]
    low_median, median, high_median = np.percentile(sample_array, ps)

    if slope_data and kind=='auto log':
        log_max, log_low, log_high = fit_slope_pdf(X, P)
        high = -1 * 10**(log_max - log_low)
        low = -1 * 10**(log_max + log_high)
        val = -1 * 10**log_max
        return_rv = RandomVariable(
            X_interp, P_interp, low=low, high=high, val=val, kind=kind
        )
    else:
        return_rv = RandomVariable(
            X_interp, P_interp, val=median, low=low_median, 
            high=high_median, kind=kind
        ).as_kind(kind)

    return return_rv


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







