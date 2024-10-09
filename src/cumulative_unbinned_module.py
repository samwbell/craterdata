from .random_variable_module import *
from .generic_plotting_module import *


def center_cumulative_points(ds, d_min=None):
    centered_ds = np.sqrt(np.array(ds)[1:] * np.array(ds)[:-1])
    if d_min is not None:
        centered_ds = np.append(centered_ds, d_min)
    return centered_ds


def fast_calc_cumulative_unbinned(
    ds, area, calculate_uncertainties=False, return_N=False, kind='log',
    log_space=False
):
    sorted_ds = np.array(sorted(ds, reverse=True))
    N_array = np.arange(1, len(ds) + 1)
 
    val_changing_kinds = {'mean', 'median', 'auto log', 'log linear'}
    if calculate_uncertainties or kind.lower() in val_changing_kinds:
        val, lower, upper = get_error_bars(
            N_array, kind=kind, return_val=True, log_space=log_space
        )
        return_tuple = sorted_ds, val / area
        if calculate_uncertainties:
            return_tuple += (lower / area, upper / area)
        
    else:
        return_tuple = sorted_ds, N_array / area
    
    if return_N:
        return_tuple += (N_array,)
        
    return return_tuple


def calc_cumulative_unbinned_pdfs(
    ds, area, center=True, d_min=None, kind='log'
):
    d_array, density, Ns = fast_calc_cumulative_unbinned(
        ds, area, kind=kind, return_N=True
    )
    density_pdf_list = [true_error_pdf(N, kind=kind) / area for N in Ns]
    if center:
        d_array = center_cumulative_points(d_array, d_min=d_min)
        if d_min is None:
            density_pdf_list = density_pdf_list[:-1]
    return d_array, density_pdf_list


def get_cu_lines(
    sorted_ds, density, lower, upper, area, d_min=None, d_max=10000,
    kind='log'
):
    
    full_ds = np.insert(sorted_ds, 0, d_max)
    val0, lower0, upper0 = N_0_dict[kind]
    full_density = np.insert(density, 0, np.float64(val0) / area)
    full_lower = np.insert(lower, 0, np.float64(lower0) / area)
    full_upper = np.insert(upper, 0, np.float64(upper0) / area)
    full_low = full_density - full_lower
    full_high = full_density + full_upper
    full_density[full_density == 0] = np.nan
    
    if d_min is None:
        full_density = full_density[:-1]
        full_low = full_low[:-1]
        full_high = full_high[:-1]
    else:
        full_ds = np.append(full_ds, d_min)
    
    return full_density, full_low, full_high, full_ds


def plot_cumulative_unbinned(
    ds, area, ax=None, color='black', alpha=1.0, plot_lines=True, ms=4,
    plot_points=False, plot_point_error_bars=False, point_color='same',
    center=False, d_min=None, d_max=10000, kind='log', fill_alpha=0.07,
    do_formatting=True, elinewidth=0.5, point_label=None
):
    
    axis_exists = any(plt.gcf().get_axes())
    if not axis_exists:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
    else:
        ax = plt.gca()
    fig = plt.gcf()

    if point_color == 'same':
        p_color = color
    else:
        p_color = point_color

    sorted_ds, density, lower, upper = fast_calc_cumulative_unbinned(
        ds, area, calculate_uncertainties=True, kind=kind
    )
    full_density, full_low, full_high, full_ds = get_cu_lines(
        sorted_ds, density, lower, upper, area, d_min=d_min, d_max=d_max
    )
    if plot_lines:
        ys = [full_density, full_low, full_high]
        line_styles = ['', ':', ':']
        for y, ls in zip(ys, line_styles):
            plt.hlines(
                y, full_ds[:-1], full_ds[1:], ls=ls, color=color, lw=1
            )
        low_fill = np.array([full_high[0] * 0.00000001] + list(full_low)[1:])
        plt.fill_between(
            np.repeat(full_ds, 2)[1:-1], np.repeat(low_fill, 2), 
            np.repeat(full_high, 2), facecolor=color, alpha=fill_alpha
        )

    if center:
        d_points = center_cumulative_points(sorted_ds, d_min=d_min)
    else:
        d_points = sorted_ds
    
    if plot_points or plot_point_error_bars:
        plot_with_error(
            d_points, density, lower, upper, color=color, alpha=alpha, 
            plot_error_bars=plot_point_error_bars, plot_points=plot_points, 
            ylabel_type='Cumulative ', ms=ms, elinewidth=elinewidth,
            point_label=point_label
        )

    if do_formatting is None:
        format_bool = not axis_exists
    else:
        format_bool = do_formatting
    
    if format_bool:
        format_cc_plot(
            sorted_ds, full_density, full_low, full_high, full_ds=full_ds,
            ylabel_type='Cumulative ', error_bar_type=kind
        )
        

