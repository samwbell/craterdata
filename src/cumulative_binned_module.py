from .random_variable_module import *
from .generic_plotting_module import *


def fast_calc_cumulative_binned(
    ds, area, bin_width_exponent=neukum_bwe, x_axis_position='left',
    reference_point=1.0, skip_zero_crater_bins=False,
    start_at_reference_point=False, d_max=None, return_N=False,
    d_min=None
):

    if d_min is not None:
        _reference_point = d_min
        _start_at_reference_point = True
    else:
        _reference_point = reference_point
        _start_at_reference_point = start_at_reference_point
    
    bin_counts, bin_array, bin_min, bin_max = bin_craters(
        ds, bin_width_exponent=bin_width_exponent, d_max=d_max, 
        x_axis_position=x_axis_position, reference_point=_reference_point, 
        start_at_reference_point=_start_at_reference_point
    )
    
    cumulative_counts = np.flip(np.flip(bin_counts).cumsum())
    
    diameter_array, cumulative_count_array = get_bin_parameters(
        ds, cumulative_counts, bin_counts, bin_array, bin_min, bin_max,
        bin_width_exponent=bin_width_exponent, 
        x_axis_position=x_axis_position, 
        reference_point=_reference_point, 
        skip_zero_crater_bins=skip_zero_crater_bins
    )
        
    if skip_zero_crater_bins and (x_axis_position != 
                                  'Michael and Neukum (2010)'):
        diameter_array = diameter_array[bin_counts != 0]
        cumulative_count_array = cumulative_count_array[bin_counts != 0]
    
    return_tuple = diameter_array, cumulative_count_array / area
    
    if return_N:
        return_tuple += (cumulative_count_array,)
    
    return return_tuple


def calc_cumulative_binned_pdfs(
    ds, area, bin_width_exponent=neukum_bwe, x_axis_position='left',
    skip_zero_crater_bins=False, reference_point=1.0, 
    start_at_reference_point=False, d_max=1E4, d_min=None
):
    x_array, density_array = fast_calc_cumulative_binned(
        ds=ds, area=area, bin_width_exponent=bin_width_exponent, 
        x_axis_position=x_axis_position, d_max=d_max,
        skip_zero_crater_bins=skip_zero_crater_bins, 
        reference_point=reference_point, d_min=d_min,
        start_at_reference_point=start_at_reference_point
    )
    cumulative_counts = density_array * area
    density_pdf_list = []
    for cumulative_count in cumulative_counts:
        lambda_pdf = true_error_pdf(cumulative_count)
        density_pdf_list.append(lambda_pdf / area)
    return x_array, density_pdf_list


def plot_cumulative_binned(
    ds, area, bin_width_exponent=neukum_bwe, x_axis_position='left', 
    skip_zero_crater_bins=False, reference_point=1.0, d_max=1000,
    start_at_reference_point=False, color='black', 
    alpha=1.0, plot_points=True, plot_error_bars=True,
    do_formatting=True, d_min=None
):
    bin_ds, pdf_list = calc_cumulative_binned_pdfs(
        ds, area, bin_width_exponent=bin_width_exponent,
        x_axis_position=x_axis_position, d_max=d_max,
        skip_zero_crater_bins=skip_zero_crater_bins,
        reference_point=reference_point, d_min=d_min,
        start_at_reference_point=start_at_reference_point
    )
    modes = np.array([pdf.mode() for pdf in pdf_list])
    bin_ds = bin_ds[modes > 1E-100]
    pdf_list = [pdf for pdf in pdf_list if pdf.mode() > 1E-100]
    plot_pdf_list(
        bin_ds, pdf_list, color=color, alpha=alpha, 
        plot_error_bars=plot_error_bars, plot_points=plot_points,
        do_formatting=do_formatting
    )


