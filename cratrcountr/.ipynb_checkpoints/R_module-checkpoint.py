
from .random_variable_module import *
from .generic_plotting_module import *

def calc_R(
           ds,area,n_points=2000,bin_width_exponent=neukum_bwe,do_correction=True,production_function=npf_new\
           ,reference_point=2.0,x_axis_position='center',skip_zero_crater_bins=False\
           ,lower_bin_adjustment=0.6,upper_bin_adjustment=1.3,random_sigma=0):

    if random_sigma!=0:
        bin_min = math.ceil(math.log(lower_bin_adjustment*min(ds)/reference_point,2)/bin_width_exponent)
        bin_max = math.ceil(math.log(upper_bin_adjustment*max(ds)/reference_point,2)/bin_width_exponent)
        bins = [reference_point*2**(bin_width_exponent*n) for n in list(range(bin_min,bin_max+1))]
        bin_counts = np.array([norm(loc=d,scale=random_sigma*d).cdf(bins[1:])-\
                norm(loc=d,scale=random_sigma*d).cdf(bins[:-1]) for d in ds]).sum(axis=0)
    else:
        bin_min = math.ceil(math.log(min(ds)/reference_point,2)/bin_width_exponent)
        bin_max = math.ceil(math.log(max(ds)/reference_point,2)/bin_width_exponent)
        bins = [reference_point*2**(bin_width_exponent*n) for n in list(range(bin_min,bin_max+1))]

        bin_counts,bin_array=np.histogram(ds,bins)
        
    binned = np.digitize(ds,bins)
    bin_gmeans = np.zeros(len(bin_counts))
    bin_gmeans[bin_counts!=0] = np.array([gmean(ds[binned==i]) for i in np.array(range(1,len(bin_counts)+1))\
                                          [bin_counts!=0]])
    bin_gmeans[bin_counts==0] = np.array([reference_point*2.0**(bin_width_exponent*(n+0.5)) for n in \
                        np.array(range(bin_min,bin_max))[bin_counts==0]])
    bin_widths = np.array([bins[i+1]-bins[i] for i in list(range(len(bins)-1))])
    
    if do_correction:
        local_slope = np.array([(np.log10(production_function(bins[n+1])) - \
                np.log10(production_function(bins[n]))) / (np.log10(bins[n+1]) - np.log10(bins[n])) \
                for n in list(range(len(bins)-1))])
        correction_factors = np.array((2**(bin_width_exponent*local_slope/2) - 2**(-1*bin_width_exponent*\
                local_slope/2)) / (local_slope * (2**(bin_width_exponent/2) - 2**(-1*bin_width_exponent/2))))
    else:
        correction_factors = np.ones(len(bin_gmeans))
    bin_adjustments = np.array([(bins[i+1]*bins[i])**1.5 for i in list(range(len(bins)-1))])
    
    if x_axis_position=='left':
        x_array = np.array([reference_point*2.0**(bin_width_exponent*(n)) for n in list(range(bin_min,bin_max))])
        y_array = np.array(bin_counts)
        if skip_zero_crater_bins:
            x_array = x_array[bin_counts!=0]
            y_array = y_array[bin_counts!=0]
    elif x_axis_position=='center':
        x_array = np.array([reference_point*2.0**(bin_width_exponent*(n+0.5)) for n in list(range(bin_min,bin_max))])
        y_array = np.array(bin_counts)
        if skip_zero_crater_bins:
            x_array = x_array[bin_counts!=0]
            y_array = y_array[bin_counts!=0]
    elif x_axis_position=='gmean':
        x_array = np.zeros(len(bin_counts))
        x_array[bin_counts!=0] = np.array([gmean(ds[np.digitize(ds,np.array(bins))==i]) for i in np.array(range(1,\
                    len(bin_counts)+1))[bin_counts!=0]])
        x_array[bin_counts==0] = np.array([reference_point*2.0**(bin_width_exponent*(n+0.5)) for n in \
                    np.array(list(range(bin_min,bin_max)))[bin_counts==0]])
        y_array = np.array(bin_counts)
        if skip_zero_crater_bins:
            x_array = x_array[bin_counts!=0]
            y_array = y_array[bin_counts!=0]
    elif x_axis_position=='Michael and Neukum (2010)':
        if not skip_zero_crater_bins:
            raise ValueError('Michael and Neukum (2010) only used bins without zero craters in them.  To fix, set'\
                            +' skip_zero_crater_bins=True')
        x_array = np.array([reference_point*2.0**(bin_width_exponent*(n+0.5)) for n in list(range(bin_min,bin_max))])
        mn10sr=np.where((bin_counts==0))[0]
        if mn10sr.shape[0]>0:
            mn10d=mn10sr[0]
            x_array = np.append(x_array[:mn10d],np.array(ds)[np.array(ds)>bins[mn10d]])
            y_array = np.append(np.array(bin_counts)[:mn10d],np.flip(np.array(range(len(np.array(ds)\
                                [np.array(ds)>bins[mn10d]])))+1))
        else:
            y_array = np.array(bin_counts)
    else:
        raise ValueError('x_axis_position must be one of the following: {\'left\',\'center\',\'gmean\','+\
                         '\'Michael and Neukum (2010)\'}')
        
    if skip_zero_crater_bins:
        bin_widths = bin_widths[bin_counts!=0]
        correction_factors = correction_factors[bin_counts!=0]
        bin_adjustments = bin_adjustments[bin_counts!=0]
    
    lower_list=ig_lower(y_array)/area/bin_widths/correction_factors*bin_adjustments
    med_list=ig_50(y_array)/area/bin_widths/correction_factors*bin_adjustments
    upper_list=ig_upper(y_array)/area/bin_widths/correction_factors*bin_adjustments
    r_list=y_array/area/bin_widths/correction_factors*bin_adjustments

    density_list=[]
    P_list=[]
    λ_list=[]
    for i in list(range(len(bin_counts))):
        λ,P=igpdf(bin_counts[i],n_points=n_points)
        density_list.append(λ/area/bin_widths[i]/correction_factors[i]*bin_adjustments[i])
        P_list.append(P)
        λ_list.append(λ)
        
    return pd.DataFrame({'D':x_array,'count':y_array,'lower':lower_list,'max':r_list,\
                         'median':med_list,'upper':upper_list,'density_pdf':density_list,\
                         'P_pdf':P_list,'λ_pdf':λ_list})

def plot_R(ds,area,ax='None',bin_width_exponent=neukum_bwe,do_correction=True,production_function=npf_new,color='black'\
          ,reference_point=2.0,x_axis_position='center',skip_zero_crater_bins=False\
          ,lower_bin_adjustment=0.6,upper_bin_adjustment=1.3,random_sigma=0):
    if ax=='None':
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        
    if random_sigma!=0:
        bin_min = math.ceil(math.log(lower_bin_adjustment*min(ds)/reference_point,2)/bin_width_exponent)
        bin_max = math.ceil(math.log(upper_bin_adjustment*max(ds)/reference_point,2)/bin_width_exponent)
        bins = [reference_point*2**(bin_width_exponent*n) for n in list(range(bin_min,bin_max+1))]
        bin_counts = np.array([norm(loc=d,scale=random_sigma*d).cdf(bins[1:])-\
                norm(loc=d,scale=random_sigma*d).cdf(bins[:-1]) for d in ds]).sum(axis=0)
    else:
        bin_min = math.ceil(math.log(min(ds)/reference_point,2)/bin_width_exponent)
        bin_max = math.ceil(math.log(max(ds)/reference_point,2)/bin_width_exponent)
        bins = [reference_point*2**(bin_width_exponent*n) for n in list(range(bin_min,bin_max+1))]

        bin_counts,bin_array=np.histogram(ds,bins)
        
    binned = np.digitize(ds,bins)
    bin_gmeans = np.zeros(len(bin_counts))
    bin_gmeans[bin_counts!=0] = np.array([gmean(ds[binned==i]) for i in np.array(range(1,len(bin_counts)+1))\
                                          [bin_counts!=0]])
    bin_gmeans[bin_counts==0] = np.array([reference_point*2.0**(bin_width_exponent*(n+0.5)) for n in \
                        np.array(range(bin_min,bin_max))[bin_counts==0]])
    bin_widths = np.array([bins[i+1]-bins[i] for i in list(range(len(bins)-1))])
    
    if do_correction:
        local_slope = np.array([(np.log10(production_function(bins[n+1])) - \
                np.log10(production_function(bins[n]))) / (np.log10(bins[n+1]) - np.log10(bins[n])) \
                for n in list(range(len(bins)-1))])
        correction_factors = np.array((2**(bin_width_exponent*local_slope/2) - 2**(-1*bin_width_exponent*\
                local_slope/2)) / (local_slope * (2**(bin_width_exponent/2) - 2**(-1*bin_width_exponent/2))))
    else:
        correction_factors = np.ones(len(bin_gmeans))
    bin_adjustments = np.array([(bins[i+1]*bins[i])**1.5 for i in list(range(len(bins)-1))])
    
    if x_axis_position=='left':
        x_array = np.array([reference_point*2.0**(bin_width_exponent*(n)) for n in list(range(bin_min,bin_max))])
        y_array = np.array(bin_counts)
        if skip_zero_crater_bins:
            x_array = x_array[bin_counts!=0]
            y_array = y_array[bin_counts!=0]
    elif x_axis_position=='center':
        x_array = np.array([reference_point*2.0**(bin_width_exponent*(n+0.5)) for n in list(range(bin_min,bin_max))])
        y_array = np.array(bin_counts)
        if skip_zero_crater_bins:
            x_array = x_array[bin_counts!=0]
            y_array = y_array[bin_counts!=0]
    elif x_axis_position=='gmean':
        x_array = np.zeros(len(bin_counts))
        x_array[bin_counts!=0] = np.array([gmean(ds[np.digitize(ds,np.array(bins))==i]) for i in np.array(range(1,\
                    len(bin_counts)+1))[bin_counts!=0]])
        x_array[bin_counts==0] = np.array([reference_point*2.0**(bin_width_exponent*(n+0.5)) for n in \
                    np.array(list(range(bin_min,bin_max)))[bin_counts==0]])
        y_array = np.array(bin_counts)
        if skip_zero_crater_bins:
            x_array = x_array[bin_counts!=0]
            y_array = y_array[bin_counts!=0]
    elif x_axis_position=='Michael and Neukum (2010)':
        if not skip_zero_crater_bins:
            raise ValueError('Michael and Neukum (2010) only used bins without zero craters in them.  To fix, set'\
                            +' skip_zero_crater_bins=True')
        x_array = np.array([reference_point*2.0**(bin_width_exponent*(n+0.5)) for n in list(range(bin_min,bin_max))])
        mn10sr=np.where((bin_counts==0))[0]
        if mn10sr.shape[0]>0:
            mn10d=mn10sr[0]
            x_array = np.append(x_array[:mn10d],np.array(ds)[np.array(ds)>bins[mn10d]])
            y_array = np.append(np.array(bin_counts)[:mn10d],np.flip(np.array(range(len(np.array(ds)\
                                [np.array(ds)>bins[mn10d]])))+1))
        else:
            y_array = np.array(bin_counts)
    else:
        raise ValueError('x_axis_position must be one of the following: {\'left\',\'center\',\'gmean\','+\
                         '\'Michael and Neukum (2010)\'}')
        
    if skip_zero_crater_bins:
        bin_widths = bin_widths[bin_counts!=0]
        correction_factors = correction_factors[bin_counts!=0]
        bin_adjustments = bin_adjustments[bin_counts!=0]
    
    lower_list=ig_lower(y_array)/area/bin_widths/correction_factors*bin_adjustments
    med_list=ig_50(y_array)/area/bin_widths/correction_factors*bin_adjustments
    upper_list=ig_upper(y_array)/area/bin_widths/correction_factors*bin_adjustments
    r_list=y_array/area/bin_widths/correction_factors*bin_adjustments
    plt.rcParams['lines.linewidth'] = 1.0
    data = pd.DataFrame({'D':x_array,'lower':lower_list,'med':med_list,'upper':upper_list})
    data.plot(y='med',x='D', \
                              yerr=np.array([data[['lower','upper']].values.transpose()]), 
                              logx=True, kind='scatter', 
                              color=color, s=0,ax=ax)
    ax.loglog(x_array,med_list,marker='_',ls='',mfc='none',mec=color,mew=1.2,ms=10)
    ax.loglog(x_array,r_list,marker='o',ls='',mfc=color,mec=color,mew=1.2,ms=4)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.xlim([bins[0],bins[-1]])
    ymax=np.max(med_list+upper_list)
    ymin=np.min(med_list-lower_list)
    yrange=np.log10(ymax/ymin)
    plt.ylim([ymin/(1.15*yrange),ymax*(1.15*yrange)])
    plt.ylabel('R value',size=18)
    plt.xlabel('Crater Diameter (km)',size=18)
    plt.grid(which='major', linestyle=':', linewidth=0.5, color='black')
    plt.grid(which='minor', linestyle=':', linewidth=0.25, color='gray')
    return ax

