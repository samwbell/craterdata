from .random_variable_module import *
from .generic_plotting_module import *


def differential_correction(w, m):
    return m * (w**0.5 - w**-0.5) / (w**(m/2) - w**(-1*m/2))


def fast_calc_differential(
    ds, area, bin_width_exponent=neukum_bwe, reference_point=1.0, 
    skip_zero_crater_bins=False, d_max=None, color='black',
    start_at_reference_point=False, pf=npf_new_loglog, do_correction=True,
    full_output=False
):
    
    bin_counts, bin_array, bin_min, bin_max = bin_craters(
        ds, bin_width_exponent=bin_width_exponent, d_max=d_max, 
        x_axis_position='center', reference_point=reference_point, 
        start_at_reference_point=start_at_reference_point
    )
    
    bin_widths = bin_array[1:] - bin_array[:1]
    differential_counts = bin_counts / bin_widths

    if do_correction:
        rise = pf(np.log10(bin_array[1:])) - pf(np.log10(bin_array[:-1]))
        run = np.log10(bin_array[1:]) - np.log10(bin_array[:-1])
        m = rise / run
        w = 2**bin_width_exponent
        cfs = differential_correction(w, m)
    else:
        cfs = np.ones(bin_counts.shape[0])
        
    differential_counts *= cfs
    
    diameter_array, differential_count_array = get_bin_parameters(
        ds, differential_counts, bin_counts, bin_array, bin_min, bin_max,
        bin_width_exponent=bin_width_exponent, 
        x_axis_position='center', 
        reference_point=reference_point, 
        skip_zero_crater_bins=skip_zero_crater_bins
    )
    
    sorted_ds = diameter_array
    density_array = differential_count_array / area
    
    if full_output:
        return sorted_ds, density_array, bin_counts, bin_widths, cfs
    else:
        return sorted_ds, density_array
    

def calc_differential_pdfs(
    ds, area, bin_width_exponent=neukum_bwe, reference_point=1.0, 
    skip_zero_crater_bins=False, d_max=None,
    start_at_reference_point=False, pf=npf_new_loglog, do_correction=True
):
    sorted_ds, rhos, bin_counts, bin_widths, cfs = fast_calc_differential(
        ds=ds, area=area, bin_width_exponent=bin_width_exponent, 
        d_max=d_max, skip_zero_crater_bins=skip_zero_crater_bins, 
        reference_point=reference_point, pf=pf, do_correction=do_correction,
        start_at_reference_point=start_at_reference_point,
        full_output=True
    )
    lambda_pdfs = true_error_pdf(bin_counts)
    density_pdf_list = cfs / bin_widths / area * lambda_pdfs
    return sorted_ds, density_pdf_list


def plot_differential(
    ds, area, bin_width_exponent=neukum_bwe, reference_point=1.0, 
    skip_zero_crater_bins=False, d_max=None,
    start_at_reference_point=False, pf=npf_new_loglog, 
    do_correction=True, ax='None', color='black', 
    alpha=1.0, plot_points=True, plot_error_bars=True
):
    bin_ds, pdf_list = calc_differential_pdfs(
        ds, area, bin_width_exponent=bin_width_exponent,
        reference_point=reference_point, d_max=d_max,
        skip_zero_crater_bins=skip_zero_crater_bins,
        start_at_reference_point=start_at_reference_point,
        pf=pf, do_correction=do_correction
    )
    return_ax = plot_pdf_list(
        bin_ds, pdf_list, ax=ax, color=color, alpha=alpha, 
        plot_error_bars=plot_error_bars, plot_points=plot_points,
        ylabel_type = 'Differential'
    )
    return return_ax

