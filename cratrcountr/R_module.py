from .differential_module import *
from .random_variable_module import *
from .generic_plotting_module import *

def fast_calc_R(
    ds, area, bin_width_exponent=neukum_bwe, reference_point=1.0, 
    skip_zero_crater_bins=False, d_max=None, d_min=None, do_correction=True,
    start_at_reference_point=False, pf=npf_new_loglog, full_output=False
):
    sorted_ds, densities, counts, widths, bins, cfs = fast_calc_differential(
        ds, area, bin_width_exponent=neukum_bwe, 
        reference_point=reference_point, d_max=d_max, d_min=d_min,
        skip_zero_crater_bins=skip_zero_crater_bins, 
        start_at_reference_point=start_at_reference_point, pf=pf, 
        do_correction=do_correction, full_output=True
    )
    rfs = (bins[:-1] * bins[1:])**1.5
    densities = rfs * densities

    if full_output:
        return sorted_ds, densities, counts, widths, bins, cfs, rfs
    else:
        return sorted_ds, densities

def calc_R_pdfs(
    ds, area, bin_width_exponent=neukum_bwe, reference_point=1.0, 
    skip_zero_crater_bins=False, d_max=None, d_min=None,
    start_at_reference_point=False, pf=npf_new_loglog, do_correction=True
):
    sorted_ds, densities, counts, widths, bins, cfs, rfs = fast_calc_R(
        ds=ds, area=area, bin_width_exponent=bin_width_exponent, 
        d_max=d_max, skip_zero_crater_bins=skip_zero_crater_bins, 
        reference_point=reference_point, pf=pf, do_correction=do_correction,
        start_at_reference_point=start_at_reference_point,
        full_output=True, d_min=d_min
    )
    lambda_pdfs = true_error_pdf(counts)
    density_pdf_list = rfs * cfs / widths / area * lambda_pdfs
    return sorted_ds, density_pdf_list

def plot_R(
    ds, area, bin_width_exponent=neukum_bwe, reference_point=1.0, 
    skip_zero_crater_bins=False, d_max=None, d_min=None,
    start_at_reference_point=False, pf=npf_new_loglog, 
    do_correction=True, ax='None', color='black', 
    alpha=1.0, plot_points=True, plot_error_bars=True
):
    bin_ds, pdf_list = calc_R_pdfs(
        ds, area, bin_width_exponent=bin_width_exponent,
        reference_point=reference_point, d_max=d_max,
        skip_zero_crater_bins=skip_zero_crater_bins,
        start_at_reference_point=start_at_reference_point,
        pf=pf, do_correction=do_correction, d_min=d_min
    )
    plot_pdf_list(
        bin_ds, pdf_list, color=color, alpha=alpha, 
        plot_error_bars=plot_error_bars, plot_points=plot_points,
        ylabel_type = 'R'
    )


