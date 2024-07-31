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


def plot_pdf_list(ds, pdf_list, ax='None', color='black', alpha=1.0,
                  plot_error_bars=True, plot_points=True, error_bar_type='log',
                  area=None, ylabel_type=' Cumulative ', ms=4):
    if ax=='None':
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
    plt.rcParams['lines.linewidth'] = 1.0

    if error_bar_type in {'log', 'Log'}:
        density_array = np.array([pdf.max for pdf in pdf_list])
        lower_array = np.array([pdf.lower_max for pdf in pdf_list])
        upper_array = np.array([pdf.upper_max for pdf in pdf_list])
    elif error_bar_type in {'linear', 'Linear'}:
        density_array = np.array([pdf.max for pdf in pdf_list])
        if area is None:
            lower_array, upper_array = tuple(np.array([
                    pdf.error_bar_linear() for pdf in pdf_list]).T)
        else:
            lower_array, upper_array = tuple((np.array([
                    (pdf * area).error_bar_linear() 
                    for pdf in pdf_list]) / area).T)
    elif error_bar_type in {'median', 'Median'}:
        density_array = np.array([pdf.median for pdf in pdf_list])
        lower_array = np.array([pdf.lower_median for pdf in pdf_list])
        upper_array = np.array([pdf.upper_median for pdf in pdf_list])
    else:
        raise ValueError(
                'error_bar_type must be \'log\', \'linear\', or \'median\'')
    
    # For plotting purposes, make zeros functionally negative infinity in log space
    is_zero = lower_array >= density_array
    if lower_array[is_zero].shape[0] > 0:
        lower_array[is_zero] = 0.99999999 * density_array[is_zero]
    
    if plot_points:
        ax.loglog(ds, density_array, marker='s', ls='', mfc=color, 
                  mec=color, mew=1.2, ms=ms)
    if plot_error_bars:
        data = pd.DataFrame({'D':ds, 'lower':lower_array, 
                             'max':density_array, 'upper':upper_array})
        yerr = np.array([data[['lower','upper']].values.T])
        data.plot(x='D', y='max', yerr=yerr, logx=True, logy=True, ax=ax, 
                kind='scatter', color=color, alpha=alpha, s=0, legend=True)
    
    plt.xticks(size=20)
    plt.yticks(size=20)
    
    xmax = np.max(ds)
    xmin = np.min(ds)
    xrange = np.log10(xmax / xmin)
    plt.xlim([xmin / (10**(0.05 * xrange)), xmax * 10**(0.05 * xrange)])
    
    ymax = np.nanmax(density_array + upper_array)
    low = density_array - lower_array
    ymin = np.nanmin(low[low > 0])

    yrange = np.log10(ymax / ymin)
    plt.ylim([ymin / (10**(0.05 * yrange)), ymax * 10**(0.05 * yrange)])
    
    plt.ylabel(ylabel_type + ' Crater Density', size=18)
    plt.xlabel('Crater Diameter (km)', size=18)
    
    plt.grid(which='major', linestyle=':', linewidth=0.5, color='black')
    plt.grid(which='minor', linestyle=':', linewidth=0.25, color='gray')
    
    return ax


def format_runtime(seconds, round_to=5):
    days = math.floor(seconds / (60 * 60 * 24))
    seconds_in_final_day = seconds - days * (60 * 60 * 24)
    hours = math.floor(seconds_in_final_day / (60 * 60))
    seconds_in_final_hour = seconds_in_final_day - hours * (60 * 60)
    minutes = math.floor(seconds_in_final_hour / 60)
    seconds_in_final_minute = seconds_in_final_hour - minutes * 60
    days, hours, minutes, seconds_in_final_minute
    return_string = str(round(seconds_in_final_minute, round_to)) + ' seconds'
    if minutes != 0:
        return_string =  str(minutes) + ' minutes, ' + return_string
    if hours != 0:
        return_string =  str(hours) + ' hours, ' + return_string
    if days != 0:
        return_string =  str(days) + ' days, ' + return_string
    return return_string
