from .random_variable_module import *
from .generic_plotting_module import *


def center_cumulative_points(ds, d_min=None):
    centered_ds = np.sqrt(np.array(ds)[1:] * np.array(ds)[:-1])
    if d_min is not None:
        centered_ds = np.append(centered_ds, d_min)
    return centered_ds


def fast_calc_cumulative_unbinned(ds, area, calculate_uncertainties=False,
                                  return_N=False):
    sorted_ds = np.array(sorted(ds, reverse=True))
    N_array = np.arange(1, len(ds) + 1)
 
    if calculate_uncertainties:
        lower, upper = get_true_error_bars_log_space(N_array)
        return_tuple = sorted_ds, N_array / area, (lower + upper) / 2
    else:
        return_tuple = sorted_ds, N_array / area
    
    if return_N:
        return_tuple += (N_array,)
        
    return return_tuple


def calc_cumulative_unbinned_pdfs(ds, area, center=True, d_min=None, 
                                  sqrt_N=False):
    x_array, density_array = fast_calc_cumulative_unbinned(ds, area, 
                                    calculate_uncertainties=False)
    cumulative_counts = np.round(density_array * area, 7)
    density_pdf_list = []
    for cumulative_count in cumulative_counts:
        if sqrt_N:
            cratering_rate_pdf = sqrt_N_error_pdf(cumulative_count)
        else:
            cratering_rate_pdf = true_error_pdf(cumulative_count)
        density_pdf_list.append(cratering_rate_pdf / area)
    if center:
        x_array = center_cumulative_points(x_array, d_min=d_min)
        if d_min is None:
            density_pdf_list = density_pdf_list[:-1]
    return x_array, density_pdf_list


def get_cumulative_unbinned_lines(sorted_ds, input_density_array, area, d_min=None, 
                d_max=10000, sqrt_N=False, error_bar_type='log'):
    N_array = np.round(input_density_array * area, 7)
    if sqrt_N:
        density_array = input_density_array
        low_array = (N_array - np.sqrt(N_array)) / area
        high_array = (N_array + np.sqrt(N_array)) / area
    elif error_bar_type in {'log', 'Log'}:
        density_array = input_density_array
        low_array, high_array = get_true_error_bounds(N_array, area)
    elif error_bar_type in {'linear', 'Linear'}:
        density_array = input_density_array
        low_array, high_array = get_true_error_bounds_linear(N_array, area)
    elif error_bar_type in {'median', 'Median'}:
        density_array = true_error_median(N_array) / area
        low_array = true_error_percentile(N_array, 1 - p_1_sigma) / area
        high_array = true_error_percentile(N_array, p_1_sigma) / area
    else:
        raise ValueError(
                'error_bar_type must be \'log\', \'linear\', or \'median\'')
    
    full_ds = np.insert(sorted_ds, 0, d_max)
    if sqrt_N: 
        density_array = np.insert(density_array, 0, None)
        low_array = np.insert(low_array, 0, None)
        high_array = np.insert(high_array, 0, None)
    elif error_bar_type in {'log', 'Log', 'linear', 'Linear'}:
        density_array = np.insert(density_array, 0, None)
        low_array = np.insert(low_array, 0, None)
        high_array = np.insert(high_array, 0, N_0_upper / area)
    elif error_bar_type in {'median', 'Median'}:
        density_array = np.insert(density_array, 0, 
                                  true_error_median(0) / area)
        low_array = np.insert(low_array, 0, 
                              true_error_percentile(0, 1 - p_1_sigma) / area)
        high_array = np.insert(high_array, 0, 
                              true_error_percentile(0, p_1_sigma) / area)
    
    if d_min is None:
        density_array = density_array[:-1]
        low_array = low_array[:-1]
        high_array = high_array[:-1]
    else:
        full_ds = np.append(full_ds, d_min)
    
    return density_array, low_array, high_array, full_ds


def plot_cumulative_unbinned(ds, area, ax=None, color='black', alpha=1.0, plot_lines=True, 
                             plot_points=False, plot_point_error_bars=False, point_color='same',
                             sqrt_N=False, center=False, d_min=None, d_max=10000, 
                             error_bar_type='log', fill_alpha=0.07):
    if ax is None:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)

    plt.rcParams['lines.linewidth'] = 1.0

    if point_color == 'same':
        p_color = color
    else:
        p_color = point_color

    sorted_ds, density_array = fast_calc_cumulative_unbinned(
                                ds, area, calculate_uncertainties=False)
    density_array, low_array, high_array, full_ds = get_cumulative_unbinned_lines(
            sorted_ds, density_array, area, d_min=d_min, d_max=d_max, sqrt_N=sqrt_N,
                error_bar_type=error_bar_type)

    if plot_lines:
        plt.hlines(density_array, full_ds[:-1], full_ds[1:], color=color)
        plt.hlines(low_array, full_ds[:-1], full_ds[1:], linestyles=':', color=color)
        plt.hlines(high_array, full_ds[:-1], full_ds[1:], linestyles=':', color=color)
        ax.set_xscale('log')
        ax.set_yscale('log')
        low_fill = np.array([high_array[0] * 0.00000001] + list(low_array)[1:])
        ax.fill_between(np.repeat(full_ds, 2)[1:-1], np.repeat(low_fill, 2), 
                        np.repeat(high_array, 2), facecolor=color, alpha=fill_alpha)

    if plot_points or plot_point_error_bars:
        point_ds, pdf_list = calc_cumulative_unbinned_pdfs(ds, area, 
                               center=center, d_min=d_min, sqrt_N=sqrt_N)
        plot_pdf_list(point_ds, pdf_list, ax=ax, color=p_color, alpha=alpha, 
                      plot_error_bars=plot_point_error_bars, plot_points=plot_points,
                      error_bar_type=error_bar_type, area=area)

    plt.xticks(size=20)
    plt.yticks(size=20)

    xmax = np.max(sorted_ds)
    xmin = np.min(full_ds)
    xrange = np.log10(xmax / xmin)
    plt.xlim([xmin / (10**(0.05 * xrange)), xmax * 10**(0.5 * xrange)])

    ymax = np.nanmax(high_array)
    if not sqrt_N:
        ymin = np.nanmin(low_array[low_array > 0])
    else:
        ymin = np.nanmin(density_array) / 10
    yrange = np.log10(ymax / ymin)
    plt.ylim([ymin / (10**(0.05 * yrange)), ymax * 10**(0.05 * yrange)])

    plt.ylabel('Cumulative Crater Density', size=18)
    plt.xlabel('Crater Diameter (km)', size=18)

    plt.grid(which='major', linestyle=':', linewidth=0.5, color='black')
    plt.grid(which='minor', linestyle=':', linewidth=0.25, color='gray')
    
    return ax

