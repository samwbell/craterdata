from .random_variable_module import *
from .age_module import *


neukum_bwe=math.log(10, 2) / 18


def bin_craters(
    ds, bin_width_exponent=math.log(10,2)/18, x_axis_position='left', 
    reference_point=1.0, start_at_reference_point=False, d_max=None
):
    if start_at_reference_point:
        bin_min = 0
    else:
        bin_min = math.ceil(
            math.log(min(ds) / reference_point, 2) / bin_width_exponent
        )
    if d_max is not None:
        bin_max = math.ceil(
            math.log(d_max / reference_point, 2) / bin_width_exponent
        )
    else:
        bin_max = math.ceil(
            math.log(max(ds) / reference_point, 2) / bin_width_exponent
        )
    bins = [
        reference_point * 2**(bin_width_exponent * n) 
        for n in list(range(bin_min, bin_max + 1))
    ]
    bin_counts, bin_array = np.histogram(ds, bins)
    if (start_at_reference_point == False) and (ds.max() < bins[1]):
        raise ValueError(
            'The data cannot be binned because each crater is within '
            'the smallest bin.  Consider setting '
            'start_at_reference_point=True and setting a reference_point '
            'value for the minimum diameter counted.  Without a known '
            'minimum diameter, we must reject data in the smallest bin '
            'because the smallest bin is not fully sampled.'
        )
    return bin_counts, bin_array, bin_min, bin_max


def get_bin_parameters(
    ds, bin_stats, bin_counts, bin_array, bin_min, bin_max,
    bin_width_exponent=math.log(10, 2) / 18, x_axis_position='left', 
    reference_point=1.0, skip_zero_crater_bins=False
):
    
    if x_axis_position=='left':
        x_array = np.array([
            reference_point * 2.0**(bin_width_exponent * (n)) 
            for n in list(range(bin_min, bin_max))
        ])
        
    elif x_axis_position=='center':
        x_array = np.array([
            reference_point * 2.0**(bin_width_exponent * (n + 0.5)) 
            for n in list(range(bin_min, bin_max))
        ])
        
    elif x_axis_position=='gmean':
        x_array = np.zeros(len(bin_counts))
        x_array[bin_counts !=0 ] = np.array([
            gmean(ds[np.digitize(ds, bin_array) == i]) 
            for i in np.array(range(1, len(bin_counts) + 1))[bin_counts != 0]
        ])
        x_array[bin_counts == 0] = np.array([
            reference_point * 2.0**(bin_width_exponent * (n + 0.5)) 
            for n in np.array(list(range(bin_min, bin_max)))[bin_counts == 0]
        ])
        
    elif x_axis_position == 'Michael and Neukum (2010)':
        if not skip_zero_crater_bins:
            raise ValueError(
                'Michael and Neukum (2010) only used bins without zero '
                'craters in them.  To fix, set skip_zero_crater_bins=True'
            )
        x_array = np.array([
            reference_point * 2.0**(bin_width_exponent*(n+0.5)) 
            for n in list(range(bin_min, bin_max))
        ])
        mn10sr = np.where((bin_counts == 0))[0]
        if mn10sr.shape[0] > 0:
            mn10d = mn10sr[0]
            x_array = np.append(
                x_array[:mn10d], 
                np.array(ds)[np.array(ds) > bin_array[mn10d]]
            )
            
    else:
        raise ValueError(
            'x_axis_position must be one of the following: {\'left\', '
            '\'center\', \'gmean\', \'Michael and Neukum (2010)\'}'
        )
    
    if x_axis_position == 'Michael and Neukum (2010)' and mn10sr.shape[0] > 0:
        y_array = np.append(
            np.array(bin_stats)[:mn10d], 
            np.flip(np.array(
                range(len(np.array(ds)[np.array(ds) > bin_array[mn10d]]))
            ) + 1)
        )
    else:
        y_array = np.array(bin_stats)
            
    return x_array, y_array


def format_cc_plot(
    sorted_ds, full_density, full_low, full_high, 
    full_ds=None, ylabel_type='Cumulative ', error_bar_type='log'
):
    
    plt.xscale('log')
    plt.yscale('log')

    plt.xticks(size=14)
    plt.yticks(size=14)

    xmax = np.max(sorted_ds)
    if full_ds is not None:
        xmin = np.min(full_ds)
    else:
        xmin = np.min(sorted_ds) 
    xrange = np.log10(xmax / xmin)
    plt.xlim([xmin / (10**(0.05 * xrange)), xmax * 10**(0.5 * xrange)])

    ymax = np.nanmax(full_high)
    if error_bar_type.lower() == 'sqrt(n)':
        ymin = np.nanmin(full_density) / 10
    else:
        ymin = np.nanmin(full_low[full_low > 0])
    yrange = np.log10(ymax / ymin)
    plt.ylim([ymin / (10**(0.05 * yrange)), ymax * 10**(0.05 * yrange)])

    plt.ylabel(ylabel_type + rf' Crater Density (km$^{{-2}}$)', size=14)
    plt.xlabel('Crater Diameter (km)', size=14)

    plt.grid(which='major', linestyle=':', linewidth=0.5, color='black')
    plt.grid(which='minor', linestyle=':', linewidth=0.25, color='gray')


def plot_with_error(
    ds, val, lower, upper, color='black', alpha=1.0, ms=4, 
    plot_error_bars=True, plot_points=True, error_bar_type='log',  
    ylabel_type='Cumulative ', elinewidth=0.5, do_formatting=None,
    point_label=None
):

    axis_exists = any(plt.gcf().get_axes())
    if not axis_exists:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
    else:
        ax = plt.gca()
    fig = plt.gcf()
    
    if plot_points:
        plt.plot(
            ds, val, marker='s', ls='', mfc=color, 
            mec=color, mew=1.2, ms=ms, label=point_label
        )
    
    if plot_error_bars:
        plt.errorbar(
            ds, val, yerr=[lower, upper], fmt='none', 
            color=color, alpha=alpha, elinewidth=elinewidth
        )
        
    if do_formatting is None:
        format_bool = not axis_exists
    else:
        format_bool = do_formatting
    
    if format_bool:
        format_cc_plot(
            ds, val, val - lower, val + upper,
            ylabel_type=ylabel_type, error_bar_type=error_bar_type
        )
        

def plot_pdf_list(
    ds, pdf_list, color='black', alpha=1.0, plot_error_bars=True, 
    plot_points=True, error_bar_type='log', area=None, 
    ylabel_type='Cumulative ', ms=4, elinewidth=0.5,
    do_formatting=True
):

    for i in range(len(pdf_list)):
        if error_bar_type.lower() != pdf_list[i].kind.lower():
            if error_bar_type.lower() == 'sqrt(n)':
                if area is None:
                    raise ValueError(
                        'If the error_bar_type is \'sqrt(N)\', and '
                        'the pdfs are not already sqrt(N) pdfs, the '
                        'function needs to know the area to figure '
                        'out the N to find sqrt(N).  To fix, set '
                        'area=<area value>.'
                    )
                N_pdf = pdf_list[i] * area
                pdf_list[i] = N_pdf.as_kind('sqrt(N)') / area
            else:
                pdf_list[i] = pdf_list[i].as_kind(error_bar_type)
        
    val = np.array([pdf.val for pdf in pdf_list])
    lower = np.array([pdf.lower for pdf in pdf_list])
    upper = np.array([pdf.upper for pdf in pdf_list])

    plot_with_error(
        ds, val, lower, upper, color=color, alpha=alpha, ms=ms, 
        plot_error_bars=plot_error_bars, plot_points=plot_points, 
        error_bar_type=error_bar_type, ylabel_type=ylabel_type,
        elinewidth=elinewidth, do_formatting=do_formatting
    )





