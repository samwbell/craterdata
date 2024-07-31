

from .random_variable_module import *
from .generic_plotting_module import *

def calc_incremental(
                     ds,area,bin_width_exponent=math.log(10,2)/18,x_axis_position='left',n_points=2000,\
                     skip_zero_crater_bins=False,reference_point=2.0,random_sigma=0.0,\
                     lower_bin_adjustment=0.6,upper_bin_adjustment=1.3):
    
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
        x_array[bin_counts!=0] = np.array([gmean(ds[np.digitize(ds,bin_array)==i]) for i in np.array(range(1,\
                    len(bin_counts)+1))[bin_counts!=0]])
        x_array[bin_counts==0] = np.array([reference_point*2.0**(bin_width_exponent*(n+0.5)) for n in \
                    np.array(list(range(bin_min,bin_max)))[bin_counts==0]])
        y_array = np.array(bin_counts)
        if skip_zero_crater_bins:
            x_array = x_array[bin_counts!=0]
            y_array = y_array[bin_counts!=0]
    else:
        raise ValueError('x_axis_position must be one of the following: {\'left\',\'center\',\'gmean\'}')
    
    lower_list=ig_lower(y_array)/area
    med_list=ig_50(y_array)/area
    upper_list=ig_upper(y_array)/area
        
    density_list=[]
    λ_list=[]
    P_list=[]
    for N in y_array:
        λ,P=igpdf(N,n_points=n_points)
        density_list.append(λ/area)
        λ_list.append(λ)
        P_list.append(P)
        
    return pd.DataFrame({'D':x_array,'count':y_array,'lower':lower_list,'max':y_array/area,\
                         'median':med_list,'upper':upper_list,'density_pdf':density_list,\
                         'P_pdf':P_list,'λ_pdf':λ_list})


def plot_incremental(ds,area,ax='None',bin_width_exponent=math.log(10,2)/18,x_axis_position='center',\
                           skip_zero_crater_bins=False,reference_point=2.0,color='black'\
                    ,random_sigma=0,lower_bin_adjustment=0.6,upper_bin_adjustment=1.3):
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
        x_array[bin_counts!=0] = np.array([gmean(ds[np.digitize(ds,bin_array)==i]) for i in np.array(range(1,\
                    len(bin_counts)+1))[bin_counts!=0]])
        x_array[bin_counts==0] = np.array([reference_point*2.0**(bin_width_exponent*(n+0.5)) for n in \
                    np.array(list(range(bin_min,bin_max)))[bin_counts==0]])
        y_array = np.array(bin_counts)
        if skip_zero_crater_bins:
            x_array = x_array[bin_counts!=0]
            y_array = y_array[bin_counts!=0]
    else:
        raise ValueError('x_axis_position must be one of the following: {\'left\',\'center\',\'gmean\'}')
    
    lower_list=ig_lower(y_array)/area
    med_list=ig_50(y_array)/area
    upper_list=ig_upper(y_array)/area
    plt.rcParams['lines.linewidth'] = 1.0
    
    data = pd.DataFrame({'D':x_array,'lower':lower_list,'med':med_list,'upper':upper_list})
    data.plot(y='med',x='D',\
                yerr=np.array([data[['lower','upper']].values.transpose()]), \
                logx=True, logy=True, kind='scatter', marker='s',color=color, \
                s=0, ax=ax)
    ax.loglog(x_array,med_list,marker='_',ls='',mfc='none',mec=color,mew=1.2,ms=10)
    ax.loglog(x_array,y_array/area,marker='o',ls='',mfc='k',mec=color,mew=1.2,ms=4)

    plt.xticks(size=20)
    plt.yticks(size=20)
    ymax=np.max(med_list+upper_list)
    ymin=np.min(med_list-lower_list)
    yrange=np.log10(ymax/ymin)
    plt.ylim([ymin/(1.15*yrange),ymax*(1.15*yrange)])
    plt.ylabel('Incremental Crater Density',size=18)
    plt.xlabel('Crater Diameter (km)',size=18)
    plt.grid(which='major', linestyle=':', linewidth=0.5, color='black')
    plt.grid(which='minor', linestyle=':', linewidth=0.25, color='gray')
    return ax
