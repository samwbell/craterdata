from .random_variable_module import *
from .generic_plotting_module import *


def differential_correction(w, m):
    return m * (w**0.5 - w**-0.5) / (w**(m/2) - w**(-1*m/2))


def fast_calc_differential(
    ds, area, bin_width_exponent=neukum_bwe, reference_point=1.0, 
    skip_zero_crater_bins=False, d_max=None, d_min=None,
    start_at_reference_point=False, pf=npf_new_loglog, do_correction=True,
    full_output=False, x_axis_position='linear_center'
):

    if d_min is not None:
        _reference_point = d_min
        _start_at_reference_point = True
    else:
        _reference_point = reference_point
        _start_at_reference_point = start_at_reference_point
    
    counts, bins, bin_min, bin_max = bin_craters(
        ds, bin_width_exponent=bin_width_exponent, d_max=d_max, 
        reference_point=_reference_point, 
        start_at_reference_point=_start_at_reference_point
    )
    
    widths = bins[1:] - bins[:1]
    differential_counts = counts / widths

    if do_correction:
        rise = pf(np.log10(bins[1:])) - pf(np.log10(bins[:-1]))
        run = np.log10(bins[1:]) - np.log10(bins[:-1])
        m = rise / run
        w = 2**bin_width_exponent
        cfs = differential_correction(w, m)
    else:
        cfs = np.ones(counts.shape[0])
        
    differential_counts *= cfs
    
    diameter_array = get_bin_parameters(
        ds, counts, bins, x_axis_position=x_axis_position
    )
    
    sorted_ds = diameter_array
    densities = differential_counts / area
    
    if full_output:
        return sorted_ds, densities, counts, widths, bins, cfs
    else:
        return sorted_ds, densities
    

def calc_differential_pdfs(
    ds, area, bin_width_exponent=neukum_bwe, reference_point=1.0, 
    skip_zero_crater_bins=False, d_max=None, d_min=None,
    start_at_reference_point=False, pf=npf_new_loglog, do_correction=True,
    x_axis_position='linear_center'
):
    sorted_ds, rhos, counts, widths, bins, cfs = fast_calc_differential(
        ds=ds, area=area, bin_width_exponent=bin_width_exponent, 
        d_max=d_max, skip_zero_crater_bins=skip_zero_crater_bins, 
        reference_point=reference_point, pf=pf, do_correction=do_correction,
        start_at_reference_point=start_at_reference_point,
        full_output=True, d_min=d_min, x_axis_position=x_axis_position
    )
    lambda_pdfs = true_error_pdf(counts)
    density_pdf_list = cfs / widths / area * lambda_pdfs
    return sorted_ds, density_pdf_list


def plot_differential(
    ds, area, bin_width_exponent=neukum_bwe, reference_point=1.0, 
    skip_zero_crater_bins=False, d_max=None, d_min=None,
    start_at_reference_point=False, pf=npf_new_loglog, 
    do_correction=True, ax='None', color='black', 
    alpha=1.0, plot_points=True, plot_error_bars=True,
    x_axis_position='linear_center'
):
    bin_ds, pdf_list = calc_differential_pdfs(
        ds, area, bin_width_exponent=bin_width_exponent,
        reference_point=reference_point, d_max=d_max,
        skip_zero_crater_bins=skip_zero_crater_bins,
        start_at_reference_point=start_at_reference_point,
        pf=pf, do_correction=do_correction, d_min=d_min,
        x_axis_position=x_axis_position
    )
    plot_pdf_list(
        bin_ds, pdf_list, color=color, alpha=alpha, 
        plot_error_bars=plot_error_bars, plot_points=plot_points,
        ylabel_type = 'Differential'
    )


