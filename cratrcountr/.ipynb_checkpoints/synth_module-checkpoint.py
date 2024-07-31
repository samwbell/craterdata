from .plot_fitting_module import *
from .cumulative_unbinned_module import *
from .cumulative_binned_module import *
from .differential_module import *
from .R_module import *

def randomize_synth_data(
    synth_list_raw, left_edge_lambda, right_edge_lambda, inc
):
    synth_list_raw_flattened = np.array([
        item for row in synth_list_raw for item in row
    ])
    mean_lambda_drop = np.mean(left_edge_lambda / right_edge_lambda)
    shifts = np.linspace(-0.5, 0.5, 1001)
    p_shift = mean_lambda_drop + (1 - mean_lambda_drop) * (shifts + 0.5)
    p_shift = p_shift / p_shift.sum()
    shift_array = np.random.choice(
        shifts, synth_list_raw_flattened.shape[0], p=p_shift
    )
    synth_list_flattened = 10**(
        np.log10(synth_list_raw_flattened) - shift_array * inc
    )
    splitting_indices = np.array([
        ds.shape[0] for ds in synth_list_raw
    ]).cumsum()
    return np.split(synth_list_flattened, splitting_indices)[:-1]


bin_count_dict = {}


def sample_by_Poisson(lambda_array, n_steps, saved_data_on, runtime_off):

    max_count = int(poisson.ppf(0.9999999999, lambda_array[0]))
    N_array = np.arange(0, max_count + 1)

    if saved_data_on:
        log_lambda_bins = np.round(np.log10(lambda_array), 4)
        lambda_keys = [
            tuple([max_count, n_steps, lambda_bin])
            for lambda_bin in log_lambda_bins
        ]
        lambda_saved = np.array([
            lambda_key in bin_count_dict for lambda_key in lambda_keys
        ])
        bin_count_matrix = np.zeros(
            tuple([lambda_array.shape[0], n_steps]), dtype=np.int8
        )
        if not runtime_off:
            n_bins_saved = np.where(lambda_saved)[0].shape[0]
            print(str(n_bins_saved) + ' bins taken from saved data')
        if np.where(lambda_saved)[0].shape[0] != 0:
            saved_keys = [lambda_keys[i] for i in np.where(lambda_saved)[0]]
            bin_count_matrix[np.where(lambda_saved)] = np.array([
                bin_count_dict[lambda_key] for lambda_key in saved_keys
            ])
        if np.where(~lambda_saved)[0].shape[0] != 0:
            unsaved_lambdas = lambda_array[np.where(~lambda_saved)]
            p_array = np.array([
                poisson.pmf(N, unsaved_lambdas) for N in N_array
            ]).T
            bin_count_matrix[np.where(~lambda_saved)] = np.array([
                np.random.choice(N_array, n_steps, p=p) for p in p_array
            ])
        for i in np.where(~lambda_saved)[0]:
            rounded_lambda = np.round(np.log10(lambda_array[i]), 4)
            dict_saving_key = tuple([max_count, n_steps, rounded_lambda])
            bin_count_dict[dict_saving_key] = bin_count_matrix[i]
            
    else:
        p_array = np.array([poisson.pmf(N, lambda_array) for N in N_array]).T
        bin_count_matrix = np.array([
            np.random.choice(N_array, n_steps, p=p) for p in p_array
        ])
        
    return bin_count_matrix.T


def synth_data(
    model_lambda=20, area=10000, pf=loglog_linear_pf(N1=0.001, slope=-2),
    dmin=1, dmax=1E5, dmax_tolerance=0.00001, n_steps=100000, inc=0.001, 
    runtime_off=False, saved_data_on=False, N_must_match_lambda=False
):
    
    t1 = time.time()
    
    synth_age = model_lambda / (10**pf(np.log10(dmin)) * area)
    logd_array = np.arange(
        np.log10(dmin) + inc / 2, np.log10(dmax) + inc / 2, inc
    )
    cumulative_lambda_array = 10**pf(logd_array) * area * synth_age
    logd_array = logd_array[cumulative_lambda_array > dmax_tolerance]
    if not runtime_off:
        print(str(logd_array.shape[0]) + ' lambda bins')
    left_edge_lambda = 10**pf(logd_array - inc/2)
    right_edge_lambda = 10**pf(logd_array + inc/2)
    lambda_array = (left_edge_lambda - right_edge_lambda) * area * synth_age
    slope_array = (left_edge_lambda - right_edge_lambda) / inc
    lambda_array *= differential_correction(10**inc, slope_array)
    bin_count_array = sample_by_Poisson(
        lambda_array, n_steps, saved_data_on, runtime_off
    )
    if N_must_match_lambda:
        bin_count_list = list(
            bin_count_array[bin_count_array.sum(axis=1) == model_lambda]
        )
    else:
        bin_count_list = list(bin_count_array)
    synth_list_raw = [
        10**np.repeat(logd_array, bin_count_array) 
        for bin_count_array in bin_count_list
    ]
    synth_list = randomize_synth_data(
        synth_list_raw, left_edge_lambda, right_edge_lambda, inc
    )
    
    t2 = time.time()
    
    if not runtime_off:
        print('runtime: ' + format_runtime(t2 - t1))
        
    return synth_list, synth_age


def synth_fixed_N(N=20, dmin=1, dmax=1E5, n_points=10000,
                  pf=loglog_linear_pf(N1=0.001, slope=-2),
                  n_steps=100, area=10000):
    logD = np.flip(np.linspace(np.log10(dmin), np.log10(dmax), n_points))
    D = 10**logD
    Y = 10**pf(logD)
    P_cumulative = Y / Y.max()
    synth_list = [np.interp(np.random.random(N), P_cumulative, D)
                  for i in range(n_steps)]
    synth_age = N / (10**pf(np.log10(dmin)) * area)
    return synth_list, synth_age


def pick_a_side_fit(sorted_ds, density_array, uncertainties, m_guess, 
                    b_guess, lower, upper):
    m, b = fit_linear(sorted_ds, density_array, 
                      uncertainties=uncertainties, 
                      guess=[m_guess, b_guess])
    continue_iteration = True
    iteration_count = 0
    switch_count = 5
    while continue_iteration and (iteration_count < 5):
        adjusted_uncertainties = uncertainties.copy()
        above_data = 10**(m * np.log10(sorted_ds) + b) > density_array
        adjusted_uncertainties[above_data] = upper[np.where(above_data)]
        adjusted_uncertainties[~above_data] = lower[np.where(~above_data)]
        flipQ = (np.sum(uncertainties != adjusted_uncertainties) > 0)
        if flipQ or (iteration_count == 0):
            m, b = fit_linear(sorted_ds, density_array, 
                              uncertainties = adjusted_uncertainties,
                              guess=[m_guess, b_guess])
        if not flipQ:
            continue_iteration = False
            switch_count = iteration_count
        uncertainties = adjusted_uncertainties.copy()
        iteration_count += 1
    return m, b, switch_count


def get_slope_cumulative_unbinned(
    ds, area, age, uncertainties='asymmetric', d_min=None,
    do_correction=True, warning_off=False, 
    pf=loglog_linear_pf(N1=0.001, slope=-2)
):
    D, Rho, N = fast_calc_cumulative_unbinned(ds, area, return_N=True)
    if do_correction:
        D = center_cumulative_points(D, d_min=d_min)
        if d_min is None:
            Rho = Rho[:-1]
            if not warning_off:
                print(
                    'The d_min parameter is currently set to None.  This '
                    'is not recommended.  You should choose a value.  It '
                    'describes the diameter where you began counting.  '
                    'For instance, if you counted craters larger than '
                    '1.0km, then set d_min=1.0.  To suppress this '
                    'warning, set warning_off=True.'
                )
    if uncertainties is None:
        sigma = None
    elif uncertainties in ['asymmetric', 'symmetric']:
        lower, upper = get_true_error_bars_log_space(N)
        sigma = (lower + upper) / 2
    m_guess = pf(1) - pf(0)
    b_guess = pf(0) + np.log10(age)
    if uncertainties=='asymmetric':
        m, b, switch_count = pick_a_side_fit(
            D, Rho, sigma, m_guess, b_guess, lower, upper
        )
    else:
        m, b = fit_linear(
            D, Rho, uncertainties=sigma, guess=[m_guess, b_guess]
        )
        
    return m, b


def get_slope_cumulative_binned(
    ds, area, age, uncertainties='asymmetric', 
    pf=loglog_linear_pf(N1=0.001, slope=-2), 
    bin_width_exponent=neukum_bwe, x_axis_position='left',
    reference_point=1.0, skip_zero_crater_bins=False,
    start_at_reference_point=False, d_max=None
):
    D, Rho, N = fast_calc_cumulative_binned(
        ds, area, bin_width_exponent=neukum_bwe, 
        x_axis_position=x_axis_position,
        reference_point=reference_point, 
        skip_zero_crater_bins=skip_zero_crater_bins,
        start_at_reference_point=start_at_reference_point, 
        d_max=d_max, return_N=True
    )
    if uncertainties is None:
        sigma = None
    elif uncertainties in ['asymmetric', 'symmetric']:
        lower, upper = get_true_error_bars_log_space(N)
        sigma = (lower + upper) / 2
    m_guess = pf(1) - pf(0)
    b_guess = pf(0) + np.log10(age)
    if uncertainties=='asymmetric':
        m, b, switch_count = pick_a_side_fit(
            D, Rho, sigma, m_guess, b_guess, lower, upper
        )
    else:
        m, b = fit_linear(
            D, Rho, uncertainties=sigma, guess=[m_guess, b_guess]
        )
        
    return m, b


def model_fitting_error(synth_list, synth_age, synth_area, 
                        pf=loglog_linear_pf(N1=0.001, slope=-2),
                        bin_width_exponent=neukum_bwe, 
                        use_uncertainties=False, sqrt_N=False, 
                        pick_a_side=False, plot_type='unbinned', 
                        d_min=None, skip_zero_crater_bins=False, 
                        n_pseudosteps=100, reference_point=1.0, 
                        start_at_reference_point=False, 
                        print_failures=True, skip_age=False, 
                        skip_slope=False):
    slope_list = []
    age_list = []
    failure_list = []
    failure_N_list = []
    failure_reason_list = []
    switch_list = []
    for i in range(n_pseudosteps):
        try:
            synth_ds = synth_list[i]
            
            if plot_type == 'simple N':
                age_list.append(len(synth_ds) / synth_area / 10**pf(0))
            
            else:
            
                if synth_ds.shape[0] == 0:
                    raise ValueError('There are no craters in this ' + \
                                     'synthetic observation, so ' + \
                                     'slope cannot be calculated.')

                if (plot_type == 'unbinned') or (plot_type == 
                                                 'unbinned corrected'):
                    sorted_ds, density_array = fast_calc_cumulative_unbinned(
                                                        synth_ds, synth_area)
                    if plot_type == 'unbinned corrected':
                        sorted_ds = center_cumulative_points(sorted_ds, 
                                                             d_min=d_min)
                        if d_min is None:
                            density_array = density_array[:-1]
                else:
                    sorted_ds, density_array = fast_calc_cumulative_binned(
                            synth_ds, synth_area, 
                            bin_width_exponent=neukum_bwe, 
                            x_axis_position=plot_type,
                            skip_zero_crater_bins=skip_zero_crater_bins, 
                            reference_point=reference_point, 
                            start_at_reference_point=start_at_reference_point)
                    if sorted_ds.shape[0] == 1:
                        raise ValueError('These craters fall into only ' + \
                                    'one bin, so no slope can be fit.  ' + 
                                    'Age will also not be calculated.')

                if use_uncertainties:
                    if sqrt_N:
                        lower, upper = get_sqrt_N_error_bars_log_space(
                                    density_array * synth_area)
                    else:
                        lower, upper = get_true_error_bars_log_space(np.round(
                                    density_array * synth_area, 7))
                    uncertainties = (upper + lower) / 2.0
                else:
                    uncertainties = None

                if skip_age:
                    age = None
                else:
                    age = fit_production_function(sorted_ds, density_array, 
                                uncertainties=uncertainties,
                                pf=pf,guess=np.log10(synth_age))
                
                if skip_slope:
                    m = None
                else:
                    m_guess = pf(1) - pf(0)
                    b_guess = pf(0) + np.log10(synth_age)
                    if pick_a_side and use_uncertainties:
                        m, b, switch_count = pick_a_side_fit(
                                    sorted_ds, density_array, 
                                    uncertainties, m_guess, b_guess,
                                    lower, upper)
                        switch_list.append(switch_count)
                    else:
                        m, b = fit_linear(sorted_ds, density_array, 
                                      uncertainties=uncertainties,
                                      guess=[m_guess, b_guess])
                    
                slope_list.append(m)
                age_list.append(age)
            
        except Exception as failure_reason:
            
            failure_list.append(i)
            failure_N_list.append(len(synth_ds))
            failure_reason_list.append(str(failure_reason))
            
    failure_df = pd.DataFrame({'i': failure_list, 'N': failure_N_list, 
                               'Reason': failure_reason_list})        
    if print_failures:
        if len(failure_list) > 0:
            print(failure_df)
            
    if pick_a_side and use_uncertainties:
        return slope_list, age_list, switch_list, failure_df
    else:
        return slope_list, age_list, failure_df
    
    
def plot_result_pdf(data_list, ax=None, label_text_size=10, xlim=None, 
                    right_position=0.85, custom_label_height=1.12, 
                    label_shift_x=0, reference_value=1.0, 
                    fig_size_adjustor=3.25, n_bins_baseline=50,
                    slope_data=False, upshift=0):
    if ax is None:
        fig = plt.figure(figsize=(4, 2))
        ax = fig.add_subplot(111)

    data_dist = np.array(data_list) / reference_value
    data_pdf = make_pdf_from_samples(data_dist, slope_data=slope_data,
                                     n_bins_baseline=n_bins_baseline)
    if xlim is None:
        min_d = data_pdf.X.min()
        max_d = data_pdf.X.max()
    else:
        min_d = xlim[0]
        max_d = xlim[1]
    ax, text_x = data_pdf.plot(
                               ax=ax, label='median', return_text_x=True, 
                               label_shift_x=label_shift_x,
                               label_text_size=label_text_size, 
                               force_label_side='left', xlim=xlim, 
                               fig_size_adjustor=fig_size_adjustor, 
                               label_shift_y=0.2, upshift=upshift,
                               error_bar_type='median')
    text_x = text_x + label_shift_x
    mean_text = "{:.2f}".format(round(np.mean(data_dist[~np.isnan(data_dist)]), 2))
    plt.text(min_d + right_position * (max_d - min_d), custom_label_height, 
             'mean:\n' + mean_text, ha='left', va='center', size=label_text_size)
    plt.text(text_x, custom_label_height, 'median:\n', ha='left', va='center', 
             size=label_text_size)
    
    return ax


def fit_slope_data(slope_list, n_bins_baseline=50, reference_value=1.0):
    
    data_dist = np.array(slope_list) / reference_value
    data_pdf = make_pdf_from_samples(data_dist, slope_data=True,
                                     n_bins_baseline=n_bins_baseline)
    
    return fit_slope_pdf(data_pdf.X, data_pdf.P)


def fit_age_data(age_list, n_bins_baseline=50, reference_value=1.0):
    
    data_dist = np.array(age_list) / reference_value
    data_pdf = make_pdf_from_samples(data_dist,
                                     n_bins_baseline=n_bins_baseline)
    
    return fit_log_of_normal(data_pdf.X, data_pdf.P)


def plot_log_fit(slope_list, label_text='', upshift=0, ax=None):
    
    if ax is None:
        fig = plt.figure(figsize=(4, 2))
        ax = fig.add_subplot(111)
        
    slope_pdf = make_pdf_from_samples(slope_list, slope_data=True)
    slope_pdf.flip().log().plot(color='mediumslateblue', 
                                label=True, rounding_n=3, 
                                upshift=upshift, ax=ax,
                                label_shift_y=-0.4, 
                                label_color='black')
    log_max, log_lower, log_upper = fit_slope_data(slope_list)
    plot_log_of_normal_fit(log_max, log_lower, log_upper, 
                           color='black', upshift=upshift)
    plt.text(0.7, 0.06 + 1.1, label_text, size=7.5, 
         color='black', ha='right')
    
    
