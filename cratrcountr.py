import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math
from scipy.stats.mstats import gmean
from scipy.stats import gamma,poisson,linregress,beta,norm,lognorm
from scipy import optimize
import time
import random

# Poisson statistcs
# 
# These functions show how to calculate the cratering rate parameter and its associated distribution depending on
# how many craters you observe.  See Bell (2020) for details.

# Cumulative percentage cutoff equivalent to 1 sigma
p_1_sigma = 0.841345
# Upper bound, median, and lower bound
def ig_error(n,p=p_1_sigma):
    return gamma.ppf(1-p,n+1),gamma.ppf(0.5,n+1),gamma.ppf(p,n+1)
# Calculate percentile p for a given n
def ig_percentile(n,p=p_1_sigma):
    return gamma.ppf(p,n+1)
# Median value
def ig_50(n):
    return gamma.ppf(0.5,n+1)
# Lower bound
def ig_lower(n):
    return gamma.ppf(0.5,n+1)-gamma.ppf(1-0.841345,n+1)
# Upper bound
def ig_upper(n):
    return gamma.ppf(0.841345,n+1)-gamma.ppf(0.5,n+1)
# Full numerical PDF of λ, producing a number of points sampled from the distribution
def ig_ndist(n,n_points=1000):
    inc = 1.0/float(n_points)
    return np.array(gamma.ppf(np.linspace(inc/2,1-inc/2,n_points),n+1))
# A numerical PDF of λ, producing paired λ and propability arrays.  This version is faster and more accurate than ig_ndist.
# It creates minor numerical errors at the low end of the distribution, which are mitigated by oversampling the low end.
def igpdf(n,n_points=2000):
    inc = 1.0/float(n_points)
    P_cum=np.append(np.linspace(0,20*inc/2-inc/10,int(n_points/5)),np.linspace(20*inc/2,1-inc/2,n_points-10))
    λ_P_cum=np.array(gamma.ppf(P_cum,n+1))
    λ_linear=np.linspace(0,max(λ_P_cum),n_points)
    λ_cum=np.interp(λ_linear,λ_P_cum,P_cum)
    P=np.gradient(λ_cum)
    return λ_linear,np.gradient(λ_cum)
# A brute force numerical calculation of the PDF without using the percent-point function of the gamma distribution
def poisson_error_bars(p,n,n_points=5000):
    nrange = max(n*10.0,10.0)
    inc = nrange/float(n_points)
    X = np.arange(0.5*inc,nrange,inc)
    Y = [poisson.pmf(n,m) for m in X]
    Yc = np.cumsum(Y)/len(X)*nrange
    cdfn = pd.DataFrame({'n':X,'cp':Yc})
    return [cdfn[cdfn['cp']<cp].iloc[-1]['n'] for cp in [1.0-p,0.5,p]]

# Neukum Production and Chronology Functions
# 
# Functions associated with the "new" Neukum Production Function and the Neukum Chronology Function
# Together, these functions provide everything that you should need to calculate lunar chronology under the
# "classic" Neukum production and chronology function system.

# The coefficients of the "new" Neukum Production Function
npf_new_coefficients=[-3.0876,-3.557528,0.781027,1.021521,-0.156012,-0.444058,\
                      0.019977,0.086850,-0.005874,-0.006809,8.25*10**-4,5.54*10**-5]

# Calculates the "new" Neukum Production Function for a given diameter or numpy array of diameters, d_km.  Gives
# values for an age of 1Ga.
def npf_new(d_km):
    return 10**(npf_new_coefficients[0] + sum([npf_new_coefficients[n] * np.log10(d_km)**n\
                                          for n in list(range(1,12))]))

# Calculates the differential version of "new" Neukum Production Function for a given diameter or numpy array of 
# diameters, d_km.  WARNING: The differential production function assumes infinitessimally small differential bins.
# If you are comparing this function to binned data, you will have to apply a correction factor.  This is
# implemented in calc_differential and plot_differential.
def npf_new_differential(d_km):
    return -1 * npf_new(d_km) / d_km * sum([n * npf_new_coefficients[n] * np.log10(d_km)**(n-1)\
                                          for n in list(range(1,12))])

# Calculates the R value version of the "new" Neukum Production Function for a given diameter or numpy array of 
# diameters, d_km.  WARNING: The R production function assumes infinitessimally small differential bins.
# If you are comparing this function to binned data, you will have to apply a correction factor.  This is
# implemented in calc_R and plot_R.
def npf_new_R(d_km):
    return npf_new_differential(d_km) * d_km**3

# Calculates the incremental version of the "new" Neukum Production Function for a given diameter or numpy 
# array of diameters, d_km.  The incremental production function corresponds to the incremental plot, 
# which is a version of an incremental plot where the X-axis diameter value is the minimum (or left-hand) diameter
# of the bin, and the bin widths are equal.  This is the recommended production function for Bayesian age
# determinations.  Bins are independent of each other, and no prior knowledge of the production function is needed
# to plot data because the production function assumes finite bins.  The bin_width_exponent parameter is the
# exponent used to define the bin edges, according to the formula bin_edge[i+1]/bin_edge[i] = 2^bin_width_exponent,
# with the default value of 0.5 giving sqrt(2) bins with bin edges [..., 2^-1=1/2, 2^-0.5=1/sqrt(2), 2^0=1,
# 2^0.5=sqrt(2), 2^1=2, 2^1.5=2*sqrt(2), ...].
def npf_new_incremental(d_km,bin_width_exponent=0.5):
    return npf_new(d_km) - npf_new(d_km * 2**bin_width_exponent)

# The Neukum Chronology Function.  Calculates the cumulative crater density (in units of craters/km^2) for D=1km 
# as a function of time t (in units of Gyr).  This version has been normalized from the equation published by 
# Neukum (1983) and reprinted as Equation 5 of Neukum et al. (2001) so that it equals the values given by the
# "new" Neukum Production Function for D=1km.  There is a slight numerical discrepancy where the unnnormalized 
# chronology function gives a value of 0.0008380, and the production function gives a value of 0.0008173.  We have
# chosen to normalize to the production function value.
def ncf(t):
    norm = 5.44*10**-14*(np.exp(6.93*1.0)-1)+8.38*10**-4*1.0 / npf_new(1)
    return 5.44*10**-14*(np.exp(6.93*t)-1)+8.38*10**-4*t / norm

# This is an object used in the calculation of ncf_inv.
class num_ncf():
    def __init__(self,nseg=10000):
        self.nseg_pts = nseg
        self.norm = (5.44*10**-14*(math.exp(6.93*1.0)-1)+8.38*10**-4*1.0) / npf_new(1)
        self.t_array = np.linspace(0,5,nseg)
        self.ncf_array = (5.44*10**-14*(np.exp(6.93*self.t_array)-1)+8.38*10**-4*self.t_array) / self.norm
    def inv(self,observed_cumulative_density):
        return np.interp(observed_cumulative_density,self.ncf_array,self.t_array)
    
# Creates a numerical inverse of the Neukum Chronology Function.  Main input ncums is a numpy array of N(1)
# cumulative crater density values (in units of craters/km^2) for D=1km.  (The function will also take a single 
# value.)  The output is a numpy array of corresponding ages (in units of Ga).  The nseg parameter (set to a
# default of 10,000) gives the number of points used for the numerical approximation of the chronology function.
# The maximum age value in the numerical approximation of the function is 5Ga.
def ncf_inv(ncums,nseg=10000):
    ncf_model = num_ncf(nseg=nseg)
    if (type(ncums)!=np.ndarray) and (type(ncums)!=list):
        return ncf_model.inv(ncums)
    else:
        return np.interp(ncums,ncf_model.ncf_array,ncf_model.t_array)

# Constants
# 
# Defines the bin width exponent needed to generate "Neukum style" 18/decade bins.
neukum_bwe=math.log(10,2)/18

# Calculation Functions
#
# These functions calculate the raw data behind different plot types.  The code differs from the plotting functions
# primarily because, in this case, we do not calculate out the zero-crater bins separately.
#
# They return a PANDAS dataframe containing several different columns:
# 'D': the diameter used for plotting (here in linear scale)
# 'count': the number of craters in the bin (for the cumulative plot, the cumulative number of craters)
# 'median': the median (50th percentile) value of the PDF of the underlying cratering rate, λ, which is expressed
# in units of number of craters per square km
# 'lower' and 'upper': these columns contain the equivalent 1σ upper and lower error bars for λ
# 'density' and 'P': these columns describe the PDF of crater density (λ/area), with 'density' describing the density
# and 'P' describing the corresponding value of the PDF for each value of density
#
# They have the following standard inputs:
# ds: an array of crater diameters in km
# area: the area of the unit being counted in square km
# n_points: the number of points for the 'ndist' numerical distribution, default of 10,000
# bin_width_exponent: the exponent used to define the bin edges, according to the formula 2^bin_width_exponent,
# with a value of 0.5 giving sqrt(2) bins with bin edges [..., 2^-1=1/2, 2^-0.5=1/sqrt(2), 2^0=1,
# 2^0.5=sqrt(2), 2^1=2, 2^1.5=2*sqrt(2), ...], only used on the binned plots
# 
# For the calc_differential and calc_R functions, there are two additional parameters that describe the correction
# for finite bin widths.  See Michael (2013) for a description of the issue.  Equation used here is Equation 2 of
# Michael (2013).  (Michael, G.G. (2013), "Planetary surface dating from crater size–frequency distribution 
# measurements: Multiple resurfacing episodes and differential isochron fitting," Icarus, 226, 885-890.)  These
# parameters are:
# do_correction: boolean variable that determines if the correction will be applied, default of True means that the
# correction will be applied
# production_function: the production function to assume for making the adjustment, default here is npf_new, which
# is a function defined in this package giving the "new" version of the Neukum Production Function for the Moon

def calc_cumulative_unbinned(ds,area,n_points=2000,random_sigma=0):
    if random_sigma==0:
        sorted_ds = sorted(ds,reverse=True)
        N_list = np.array(list(range(1,len(ds)+1)))
        lower_list=ig_lower(N_list)/area
        med_list=ig_50(N_list)/area
        upper_list=ig_upper(N_list)/area
        pdf_list=[igpdf(cum_count,n_points=n_points) for cum_count in N_list]
        λ_pdf_list=[pdf_tuple[0] for pdf_tuple in pdf_list]
        density_pdf_list=[pdf_tuple[0]/area for pdf_tuple in pdf_list]
        P_pdf_list=[pdf_tuple[1] for pdf_tuple in pdf_list]
        rval=pd.DataFrame({'D':sorted_ds,'count':N_list,'max':N_list/area,'lower':lower_list,\
                    'median':med_list,'upper':upper_list,'density_pdf':density_pdf_list,\
                             'λ_pdf':λ_pdf_list,'P_pdf':P_pdf_list})
    else:
        X=np.logspace(np.log10(min(ds))-1,np.log10(max(ds))+1,1000)
        d_cdfs=[1-norm(loc=d,scale=random_sigma*d).cdf(X) for d in ds]
        N_list_random = np.array(d_cdfs).sum(axis=0)
        lower_list_random=ig_lower(N_list_random)/area
        med_list_random=ig_50(N_list_random)/area
        upper_list_random=ig_upper(N_list_random)/area
        rval=X,N_list_random/area,lower_list_random,med_list_random,upper_list_random
    return rval

def calc_cumulative_binned(ds,area,bin_width_exponent=neukum_bwe,x_axis_position='left',reference_point=2.0,\
                           random_sigma=0,n_points=2000,skip_zero_crater_bins=False):

    bin_min = math.ceil(math.log(min(ds)/reference_point,2)/bin_width_exponent)
    if random_sigma==0:
        bin_max = math.ceil(math.log(max(ds)/reference_point,2)/bin_width_exponent)
    else:
        bin_max = math.ceil(math.log(max(ds)/reference_point*(1+5*random_sigma),2)/bin_width_exponent)
    bins = [reference_point*2**(bin_width_exponent*n) for n in list(range(bin_min,bin_max+1))]
    bin_counts,bin_array=np.histogram(ds,bins)
    cum_counts = np.flip(np.flip(bin_counts).cumsum())
    
    if x_axis_position=='left':
        x_array = np.array([reference_point*2.0**(bin_width_exponent*(n)) for n in list(range(bin_min,bin_max))])
        y_array = np.array(cum_counts)
        if skip_zero_crater_bins:
            x_array = x_array[bin_counts!=0]
            y_array = y_array[bin_counts!=0]
    elif x_axis_position=='center':
        x_array = np.array([reference_point*2.0**(bin_width_exponent*(n+0.5)) for n in list(range(bin_min,bin_max))])
        y_array = np.array(cum_counts)
        if skip_zero_crater_bins:
            x_array = x_array[bin_counts!=0]
            y_array = y_array[bin_counts!=0]
    elif x_axis_position=='gmean':
        x_array = np.zeros(len(bin_counts))
        x_array[bin_counts!=0] = np.array([gmean(ds[np.digitize(ds,bin_array)==i]) for i in np.array(range(1,\
                    len(bin_counts)+1))[bin_counts!=0]])
        x_array[bin_counts==0] = np.array([reference_point*2.0**(bin_width_exponent*(n+0.5)) for n in \
                    np.array(list(range(bin_min,bin_max)))[bin_counts==0]])
        y_array = np.array(cum_counts)
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
            y_array = np.append(np.array(cum_counts)[:mn10d],np.flip(np.array(range(len(np.array(ds)\
                                [np.array(ds)>bins[mn10d]])))+1))
        else:
            y_array = np.array(cum_counts)
    else:
        raise ValueError('x_axis_position must be one of the following: {\'left\',\'center\',\'gmean\','+\
                         '\'Michael and Neukum (2010)\'}')
    
    if random_sigma==0:
        lower_list=ig_lower(y_array)/area
        med_list=ig_50(y_array)/area
        upper_list=ig_upper(y_array)/area
        N_list=y_array
    else:
        X=np.logspace(np.log10(min(ds))-1,np.log10(max(ds))+1,1000)
        d_cdfs=[1-norm(loc=d,scale=random_sigma*d).cdf(X) for d in ds]
        N_list = np.interp(x_array,X,np.array(d_cdfs).sum(axis=0))
        lower_list=ig_lower(N_list)/area
        med_list=ig_50(N_list)/area
        upper_list=ig_upper(N_list)/area
                            
    density_list=[]
    λ_list=[]
    P_list=[]
    for N in N_list:
        λ,P=igpdf(N,n_points=n_points)
        density_list.append(λ/area)
        λ_list.append(λ)
        P_list.append(P)
        
    return pd.DataFrame({'D':x_array,'count':N_list,'max':N_list/area,'lower':lower_list,\
                         'median':med_list,'upper':upper_list,'density_pdf':density_list,\
                         'P_pdf':P_list,'λ_pdf':λ_list})

def calc_incremental(ds,area,bin_width_exponent=math.log(10,2)/18,x_axis_position='left',n_points=2000,\
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

def calc_differential(ds,area,n_points=2000,bin_width_exponent=neukum_bwe,do_correction=True,\
                production_function=npf_new,reference_point=2.0,x_axis_position='center',\
                skip_zero_crater_bins=False,lower_bin_adjustment=0.6,upper_bin_adjustment=1.3,random_sigma=0):

    if random_sigma!=0:
        bin_min = math.ceil(math.log(lower_bin_adjustment*min(ds)/reference_point,2)/bin_width_exponent)
        bin_max = math.ceil(math.log(upper_bin_adjustment*max(ds)/reference_point,2)/bin_width_exponent)
        bins = [reference_point*2**(bin_width_exponent*n) for n in list(range(bin_min,bin_max+1))]
        bin_gmeans = [reference_point*2.0**(bin_width_exponent*(n+0.5)) for n in list(range(bin_min,bin_max))]
        bin_widths = np.array([bins[i+1]-bins[i] for i in list(range(len(bins)-1))])
        bin_counts = np.array([norm(loc=d,scale=random_sigma*d).cdf(bins[1:])-\
                norm(loc=d,scale=random_sigma*d).cdf(bins[:-1]) for d in ds]).sum(axis=0)
    else:
        bin_min = math.ceil(math.log(min(ds)/reference_point,2)/bin_width_exponent)
        bin_max = math.ceil(math.log(max(ds)/reference_point,2)/bin_width_exponent)
        bins = [reference_point*2**(bin_width_exponent*n) for n in list(range(bin_min,bin_max+1))]
        bin_gmeans = [reference_point*2.0**(bin_width_exponent*(n+0.5)) for n in list(range(bin_min,bin_max))]
        bin_widths = np.array([bins[i+1]-bins[i] for i in list(range(len(bins)-1))])
        bin_counts,bin_array=np.histogram(ds,bins)
    
    if do_correction:
        local_slope = np.array([(np.log10(production_function(bins[n+1])) - \
                np.log10(production_function(bins[n]))) / (np.log10(bins[n+1]) - np.log10(bins[n])) \
                for n in list(range(len(bins)-1))])
        correction_factors = np.array((2**(bin_width_exponent*local_slope/2) - 2**(-1*bin_width_exponent*\
                local_slope/2)) / (local_slope * (2**(bin_width_exponent/2) - 2**(-1*bin_width_exponent/2))))
    else:
        correction_factors = np.ones(len(bin_gmeans))
    
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
        
    lower_list=ig_lower(y_array)/area/bin_widths/correction_factors
    med_list=ig_50(y_array)/area/bin_widths/correction_factors
    upper_list=ig_upper(y_array)/area/bin_widths/correction_factors
    diff_list=y_array/area/bin_widths/correction_factors

    density_list=[]
    λ_list=[]
    P_list=[]
    for i in list(range(len(bin_counts))):
        λ,P=igpdf(y_array[i],n_points=n_points)
        density_list.append(λ/area/bin_widths[i]/correction_factors[i])
        λ_list.append(λ)
        P_list.append(P)
        
    density_list=[]
    λ_list=[]
    P_list=[]
    for N in y_array:
        λ,P=igpdf(N,n_points=n_points)
        density_list.append(λ/area)
        λ_list.append(λ)
        P_list.append(P)
        
    return pd.DataFrame({'D':x_array,'count':y_array,'lower':lower_list,'max':diff_list,\
                         'median':med_list,'upper':upper_list,'density_pdf':density_list,\
                         'P_pdf':P_list,'λ_pdf':λ_list})

def calc_R(ds,area,n_points=2000,bin_width_exponent=neukum_bwe,do_correction=True,production_function=npf_new\
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

def fast_calc_cumulative_unbinned(ds,area):
    sorted_ds = sorted(ds,reverse=True)
    N_list = np.array(list(range(1,len(ds)+1)))
    return np.array(sorted_ds),np.array(N_list)/area

def fast_calc_cumulative_binned(ds,area,bin_width_exponent=neukum_bwe,x_axis_position='left',nonzero=False,\
                                warning_off=False):
    bin_min = math.ceil(math.log(min(ds),2)/bin_width_exponent)
    bin_max = math.ceil(math.log(max(ds),2)/bin_width_exponent)
    bins = [2.0**(bin_width_exponent*n) for n in list(range(bin_min,bin_max+1))]
    bin_counts,bin_array=np.histogram(ds,bins)
    cum_counts = np.flip(np.flip(bin_counts).cumsum())
    
    if x_axis_position=='left':
        x_array = np.array([2.0**(bin_width_exponent*(n)) for n in list(range(bin_min,bin_max))]) 
    elif x_axis_position=='center':
        x_array = np.array([2.0**(bin_width_exponent*(n+0.5)) for n in list(range(bin_min,bin_max))])
    elif x_axis_position=='gmean':
        x_array = np.array([gmean(ds[np.digitize(ds,bin_array)==i]) for i in range(1,len(bin_counts)+1)])
        if not nonzero:
            if not warning_off:
                print('WARNING: For x_axis_position=\'gmean\', if nonzero=False, bins with no craters will be'\
                  +' plotted against their bin centers.  To turn off this warning, set warning_off=True.')
            x_array[np.isnan(x_array)] = np.array([2.0**(bin_width_exponent*(n+0.5)) for n in \
                                np.array(list(range(bin_min,bin_max)))[np.isnan(x_array)]])
    elif x_axis_position=='Michael and Neukum (2010)':
        if not nonzero:
            raise ValueError('Michael and Neukum (2010) only used bins without zero craters in them.  To fix, set'\
                            +' nonzero=True')
        x_array = np.array([2.0**(bin_width_exponent*(n+0.5)) for n in list(range(bin_min,bin_max))])
        mn10sr=np.where((bin_counts==0))[0]
        if mn10sr.shape[0]>0:
            mn10d=mn10sr[0]
            x_array = np.append(x_array[:mn10d],np.array(ds)[np.array(ds)>bins[mn10d]])
            density_array = np.append(np.array(cum_counts/area)[:mn10d],np.flip(np.array(range(len(np.array(ds)\
                                [np.array(ds)>bins[mn10d]])))+1)/synth_area)
        else:
            density_array = np.array(cum_counts/area)
    else:
        raise ValueError('x_axis_position must be one of the following: {\'left\',\'center\',\'gmean\',\'Michael and '+\
                         +'Neukum (2010)\'}')
        
    if x_axis_position=='Michael and Neukum (2010)':
        return_tuple = x_array,density_array
    elif nonzero:
        return_tuple = np.array(x_array)[bin_counts.nonzero()],np.array(cum_counts/area)[bin_counts.nonzero()]
    else:
        return_tuple = np.array(x_array),np.array(cum_counts/area)
    
    return return_tuple 

# Plotting Functions
#
# These functions produce different standard crater counting plots.  The code differs from the calculation 
# functions primarily because, in this case, we do calculate out the zero-crater bins separately and plot them
# in gray to visually indicate which bins are empty.
#
# They use the same inputs as the calculation functions (see above for details), with the addition of an ax
# variable for matplotlib plotting purposes.  Under the default value, 'None', the variable is plotted as a
# subplot of a newly created figure.  However, you can pass a different matplotlib axis to it if you want to
# build combined or heavily customized plots.

def plot_cumulative_unbinned(ds,area,ax='None',color='gray',alpha=1.0,random_sigma=0,\
                crater_lines=False,crater_color='same',plot_lines=True,max_point=False,\
                             med_point=False,sqrtN=False):
    if ax=='None':
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
    plt.rcParams['lines.linewidth'] = 1.0
    if crater_color=='same':
        c_color=color
    else:
        c_color=crater_color
    sorted_ds = sorted(ds,reverse=True)
    N_list = np.array(list(range(1,len(ds)+1)))
    if not sqrtN:
        lower_list=ig_lower(N_list)/area
        med_list=ig_50(N_list)/area
        upper_list=ig_upper(N_list)/area
    else:
        lower_list=np.sqrt(N_list)/area
        med_list=N_list/area
        upper_list=np.sqrt(N_list)/area
    if random_sigma==0:
        if max_point:
            ax.loglog(sorted_ds,N_list/area,marker='s',ls='',mfc=c_color,mec=c_color,mew=1.2,ms=4)
        if med_point:
            ax.loglog(sorted_ds,med_list,marker='o',ls='',mfc=c_color,mec=c_color,mew=1.2,ms=4)
        if crater_lines:
            data = pd.DataFrame({'D':sorted_ds,'N':N_list,'lower':lower_list,'med':med_list,'upper':upper_list})
            data.plot(y='med',x='D', \
                                      yerr=np.array([data[['lower','upper']].values.transpose()]), 
                                      logx=True, logy=True, kind='scatter', 
                                      color=c_color, alpha=alpha, legend=True, s=0, ax=ax)
        if plot_lines and (len(sorted_ds)>1):
            plt.hlines(med_list[:-2],sorted_ds[:-2],sorted_ds[1:],linestyles='--',color=color)
            plt.hlines((N_list/area)[:-2],sorted_ds[:-2],sorted_ds[1:],color=color)
            plt.hlines(med_list[:-2]-lower_list[:-2],sorted_ds[:-2],sorted_ds[1:],linestyles=':',color=color)
            plt.hlines(med_list[:-2]+upper_list[:-2],sorted_ds[:-2],sorted_ds[1:],linestyles=':',color=color)
            ax.set_xscale('log')
            ax.set_yscale('log')
    else:
        X=np.logspace(np.log10(min(ds))-1,np.log10(max(ds))+1,1000)
        d_cdfs=[1-norm(loc=d,scale=random_sigma*d).cdf(X) for d in ds]
        N_list_random = np.array(d_cdfs).sum(axis=0)
        lower_list_random=ig_lower(N_list_random)/area
        med_list_random=ig_50(N_list_random)/area
        upper_list_random=ig_upper(N_list_random)/area
        if crater_lines:
            data = pd.DataFrame({'D':sorted_ds,'N':N_list,'lower':lower_list,'med':med_list,'upper':upper_list})
            data.plot(y='med',x='D', \
                                      yerr=np.array([data[['lower','upper']].values.transpose()]),
                                      xerr=np.array([(random_sigma*data[['D','D']]).values.transpose()]),
                                      logx=True, logy=True, kind='scatter', 
                                      color=c_color, alpha=alpha, legend=True, s=0, ax=ax)
        if max_point:
            ax.loglog(sorted_ds,N_list/area,marker='s',ls='',mfc=c_color,mec=c_color,mew=1.2,ms=4)
        if med_point:
            ax.loglog(sorted_ds,med_list,marker='s',ls='',mfc=c_color,mec=c_color,mew=1.2,ms=4)
        plt.loglog(X,N_list_random/area,'k',color=color)
        plt.loglog(X,med_list_random-lower_list_random,'k:',color=color)
        plt.loglog(X,med_list_random,'k--',color=color)
        plt.loglog(X,med_list_random+upper_list_random,'k:',color=color)
    plt.xticks(size=20)
    plt.yticks(size=20)
    xmax=np.max(sorted_ds)
    xmin=np.min(sorted_ds)
    xrange=np.log10(xmax/xmin)
    plt.xlim([xmin/(1.15*xrange),xmax*(1.15*xrange)])
    ymax=np.max(med_list+upper_list)
    if not sqrtN:
        ymin=np.min(med_list-lower_list)
    else:
        ymin=np.min(med_list)/10
    yrange=np.log10(ymax/ymin)
    plt.ylim([ymin/(1.15*yrange),ymax*(1.15*yrange)])
    plt.ylabel('Cumulative Crater Density',size=18)
    plt.xlabel('Crater Diameter (km)',size=18)
    plt.grid(which='major', linestyle=':', linewidth=0.5, color='black')
    plt.grid(which='minor', linestyle=':', linewidth=0.25, color='gray')
    return ax

def plot_cumulative_binned(ds,area,ax='None',bin_width_exponent=math.log(10,2)/18,x_axis_position='left',\
                           reference_point=2.0,skip_zero_crater_bins=False, color='black',\
                          random_sigma=0):
    if ax=='None':
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)

    bin_min = math.ceil(math.log(min(ds)/reference_point,2)/bin_width_exponent)
    if random_sigma==0:
        bin_max = math.ceil(math.log(max(ds)/reference_point,2)/bin_width_exponent)
    else:
        bin_max = math.ceil(math.log(max(ds)/reference_point*(1+5*random_sigma),2)/bin_width_exponent)
    bins = [reference_point*2**(bin_width_exponent*n) for n in list(range(bin_min,bin_max+1))]
    bin_counts,bin_array=np.histogram(ds,bins)
    cum_counts = np.flip(np.flip(bin_counts).cumsum())
    
    if x_axis_position=='left':
        x_array = np.array([reference_point*2.0**(bin_width_exponent*(n)) for n in list(range(bin_min,bin_max))])
        y_array = np.array(cum_counts)
        if skip_zero_crater_bins:
            x_array = x_array[bin_counts!=0]
            y_array = y_array[bin_counts!=0]
    elif x_axis_position=='center':
        x_array = np.array([reference_point*2.0**(bin_width_exponent*(n+0.5)) for n in list(range(bin_min,bin_max))])
        y_array = np.array(cum_counts)
        if skip_zero_crater_bins:
            x_array = x_array[bin_counts!=0]
            y_array = y_array[bin_counts!=0]
    elif x_axis_position=='gmean':
        x_array = np.zeros(len(bin_counts))
        x_array[bin_counts!=0] = np.array([gmean(ds[np.digitize(ds,bin_array)==i]) for i in np.array(range(1,\
                    len(bin_counts)+1))[bin_counts!=0]])
        x_array[bin_counts==0] = np.array([reference_point*2.0**(bin_width_exponent*(n+0.5)) for n in \
                    np.array(list(range(bin_min,bin_max)))[bin_counts==0]])
        y_array = np.array(cum_counts)
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
            y_array = np.append(np.array(cum_counts)[:mn10d],np.flip(np.array(range(len(np.array(ds)\
                                [np.array(ds)>bins[mn10d]])))+1))
        else:
            y_array = np.array(cum_counts)
    else:
        raise ValueError('x_axis_position must be one of the following: {\'left\',\'center\',\'gmean\','+\
                         '\'Michael and Neukum (2010)\'}')
    
    if random_sigma==0:
        lower_list=ig_lower(y_array)/area
        med_list=ig_50(y_array)/area
        upper_list=ig_upper(y_array)/area
        N_list=y_array
    else:
        X=np.logspace(np.log10(min(ds))-1,np.log10(max(ds))+1,1000)
        d_cdfs=[1-norm(loc=d,scale=random_sigma*d).cdf(X) for d in ds]
        N_list = np.interp(x_array,X,np.array(d_cdfs).sum(axis=0))
        lower_list=ig_lower(N_list)/area
        med_list=ig_50(N_list)/area
        upper_list=ig_upper(N_list)/area
    
    plt.rcParams['lines.linewidth'] = 1.0
    
    data = pd.DataFrame({'D':x_array,'lower':lower_list,'med':med_list,'upper':upper_list})
    data.plot(y='med',x='D',\
                yerr=np.array([data[['lower','upper']].values.transpose()]), \
                logx=True, logy=True, kind='scatter', marker='s',color=color, \
                s=0, ax=ax)
    ax.loglog(x_array,med_list,marker='_',ls='',mfc='none',mec=color,mew=1.2,ms=10)
    ax.loglog(x_array,N_list/area,marker='o',ls='',mfc='k',mec=color,mew=1.2,ms=4)

    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.xlim([bins[0]/2**bin_width_exponent,bins[-1]])
    ymax=np.max(med_list+upper_list)
    ymin=np.min(med_list-lower_list)
    yrange=np.log10(ymax/ymin)
    plt.ylim([ymin/(1.15*yrange),ymax*(1.15*yrange)])
    plt.ylabel('Cumulative Crater Density',size=18)
    plt.xlabel('Crater Diameter (km)',size=18)
    plt.grid(which='major', linestyle=':', linewidth=0.5, color='black')
    plt.grid(which='minor', linestyle=':', linewidth=0.25, color='gray')
    return ax

def plot_incremental(ds,area,ax='None',bin_width_exponent=math.log(10,2)/18,x_axis_position='left',\
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

def plot_differential(ds,area,ax='None',bin_width_exponent=neukum_bwe,do_correction=True,production_function=npf_new,\
                     reference_point=2.0,color='black',x_axis_position='center',skip_zero_crater_bins=False\
                     ,lower_bin_adjustment=0.6,upper_bin_adjustment=1.3,random_sigma=0):
    if ax=='None':
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        
    if random_sigma!=0:
        bin_min = math.ceil(math.log(lower_bin_adjustment*min(ds)/reference_point,2)/bin_width_exponent)
        bin_max = math.ceil(math.log(upper_bin_adjustment*max(ds)/reference_point,2)/bin_width_exponent)
        bins = [reference_point*2**(bin_width_exponent*n) for n in list(range(bin_min,bin_max+1))]
        bin_gmeans = [reference_point*2.0**(bin_width_exponent*(n+0.5)) for n in list(range(bin_min,bin_max))]
        bin_widths = np.array([bins[i+1]-bins[i] for i in list(range(len(bins)-1))])
        bin_counts = np.array([norm(loc=d,scale=random_sigma*d).cdf(bins[1:])-\
                norm(loc=d,scale=random_sigma*d).cdf(bins[:-1]) for d in ds]).sum(axis=0)
    else:
        bin_min = math.ceil(math.log(min(ds)/reference_point,2)/bin_width_exponent)
        bin_max = math.ceil(math.log(max(ds)/reference_point,2)/bin_width_exponent)
        bins = [reference_point*2**(bin_width_exponent*n) for n in list(range(bin_min,bin_max+1))]
        bin_gmeans = [reference_point*2.0**(bin_width_exponent*(n+0.5)) for n in list(range(bin_min,bin_max))]
        bin_widths = np.array([bins[i+1]-bins[i] for i in list(range(len(bins)-1))])
        bin_counts,bin_array=np.histogram(ds,bins)
    
    if do_correction:
        local_slope = np.array([(np.log10(production_function(bins[n+1])) - \
                np.log10(production_function(bins[n]))) / (np.log10(bins[n+1]) - np.log10(bins[n])) \
                for n in list(range(len(bins)-1))])
        correction_factors = np.array((2**(bin_width_exponent*local_slope/2) - 2**(-1*bin_width_exponent*\
                local_slope/2)) / (local_slope * (2**(bin_width_exponent/2) - 2**(-1*bin_width_exponent/2))))
    else:
        correction_factors = np.ones(len(bin_gmeans))
    
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
        
    lower_list=ig_lower(y_array)/area/bin_widths/correction_factors
    med_list=ig_50(y_array)/area/bin_widths/correction_factors
    upper_list=ig_upper(y_array)/area/bin_widths/correction_factors
    diff_list=y_array/area/bin_widths/correction_factors
    
    plt.rcParams['lines.linewidth'] = 1.0
    data = pd.DataFrame({'D':x_array,'lower':lower_list,'med':med_list,'upper':upper_list})
    data.plot(y='med',x='D', \
                              yerr=np.array([data[['lower','upper']].values.transpose()]), 
                              logx=True, logy=True, kind='scatter', 
                              color=color, s=0, ax=ax)
    ax.loglog(x_array,med_list,marker='_',ls='',mfc='none',mec=color,mew=1.2,ms=10)
    ax.loglog(x_array,diff_list,marker='o',ls='',mfc=color,mec=color,mew=1.2,ms=4)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.xlim([bins[0],bins[-1]])
    ymax=np.max(med_list+upper_list)
    ymin=np.min(med_list-lower_list)
    yrange=np.log10(ymax/ymin)
    plt.ylim([ymin/(1.15*yrange),ymax*(1.15*yrange)])
    plt.ylabel('Differential Crater Density',size=18)
    plt.xlabel('Crater Diameter (km)',size=18)
    plt.grid(which='major', linestyle=':', linewidth=0.5, color='black')
    plt.grid(which='minor', linestyle=':', linewidth=0.25, color='gray')
    return ax

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




# Numerical distribution functions
#
# These functions are used for processing and handling numerical approximations of probability distributions.
# These numerical distributions contain a large number of points randomly selected from the distribution being
# approximated numerically.

# This function plots a PDF (Probability Density Function) of a given numerical distribution, f_ndist, defined as
# a numpy array.  The n_bins variable gives the number of bins used or the discrete bin edges used.  The default
# is 100, which may need to be adjusted.  Like in the plotting functions for crater count data, there is an ax
# variable for matplotlib plotting purposes.  Under the default value, 'None', the variable is plotted as a
# subplot of a newly created figure.  However, you can pass a different matplotlib axis to it if you want to
# build combined or heavily customized plots.  The color parameter gives the plot color, and the upshift parameter
# applies a uniform vertical upwards shift.  It is used for stacking PDFs on top of each other to make combination
# plots.  See examples in the Jupyter notebooks showing how to use the code.
def plot_dist(f_ndist,n_bins='auto',ax='None',upshift=0,color='blue',show_max=True):
    if ax=='None':
        fig = plt.figure(figsize=(10,2))
        ax = fig.add_subplot(111)
    if n_bins=='Auto' or n_bins=='auto':
        bin_count = int(round((max(f_ndist)-min(f_ndist))/(np.percentile(f_ndist,90)-np.percentile(f_ndist,10))*30))
    else:
        bin_count = n_bins
    Y,bin_edges=np.histogram(f_ndist,bins=bin_count,density=True)
    bins=np.array(bin_edges)
    ymax=max(Y)
    if bins[0] < 0:
        X = np.array([bins[0]-1000*abs(bins[-1]-bins[-0])] + list(0.5*(bins[1:]+bins[:-1])) + \
                     [1000*abs(bins[-1]-bins[0])])
        Y = [0] + list(np.array(Y)/ymax) + [0]
    elif Y[0] <= 0.25*ymax:
        X = np.array([0] + list(0.5*(bins[1:]+bins[:-1])) + [1000*bins[-1]])
        Y = [0] + list(np.array(Y)/ymax) + [0]
    else:
        X = np.array([0] + list(0.5*(bins[1:]+bins[:-1])) + [1000*bins[-1]])
        Y = [max(Y[0]-(Y[1]-Y[0])/((X[2]-X[1])/(X[1]-X[0])),0)/ymax] + list(np.array(Y)/ymax) + [0]
    Y = np.array(Y)+upshift
    ax.plot(X,Y,color,linewidth=2)
    low1,med1,high1 = tuple(np.percentile(f_ndist,[100-84.1345,50,84.1345]))
    X_interp = np.linspace(np.percentile(f_ndist,7),np.percentile(f_ndist,93),10000)
    Y_interp = np.interp(X_interp,X,Y)
    ax.fill_between(X_interp,upshift,Y_interp, where=((low1<X_interp)&(X_interp<high1)), facecolor=color, alpha=0.07)
    ax.plot([med1,med1],[upshift,Y_interp[next(x[0] for x in enumerate(X_interp) if x[1] > med1)]],'--',color=color)
    ax.plot([low1,low1],[upshift,Y_interp[next(x[0] for x in enumerate(X_interp) if x[1] > low1)]],':',color=color)
    ax.plot([high1,high1],[upshift,Y_interp[next(x[0] for x in enumerate(X_interp) if x[1] > high1)]],':',color=color)
    if show_max:
        ax.plot([X[Y.argmax()],X[Y.argmax()]],[upshift,Y_interp[next(x[0] for x in enumerate(X_interp) if x[1] >\
                        X[Y.argmax()])]],color=color)
    plt.xlim([bins[0],bins[-1]])
    return X,Y

# This function calculates a PDF (Probability Density Function) of a given numerical distribution, ndist, defined
# as a numpy array.  The n_bins variable gives the number of bins used or the discrete bin edges used.  The default
# is 100, which may need to be adjusted.  The n_points variable gives the number of points used for interpolating
# between the minimum and maximum values, min_val and max_val.  The default versions of these values are in Ga and
# are intended for an age calcualtion.  These values will probably have to adjusted.  The output values, X_interp
# and Y_interp, give the values and their respective relative probabilities between min_val and max_val.
def get_pdf(ndist,n_bins=100,n_points=100000,min_val=0,max_val=5):
    Y,bin_edges=np.histogram(ndist,bins=np.linspace(min_val,max_val,n_bins),density=True)
    bins=np.array(bin_edges)
    ymax=max(Y)
    X = [0] + list(0.5*(bins[1:]+bins[:-1]))
    if Y[0] <= 0.25*ymax:
        Y = [0] + list(np.array(Y)/ymax)
    else:
        Y = [max(Y[0]-(Y[1]-Y[0])/((X[2]-X[1])/(X[1]-X[0])),0)/ymax] + list(np.array(Y)/ymax)
    X_interp = np.linspace(min_val,max_val,n_points)
    Y_interp = np.interp(X_interp,X,Y)
    return X_interp,Y_interp

# This function a given PDF (Probability Density Function) defined from a T_pdf of values and a P_pdf of those
# values' respective relative probabilities.  These should be equivalent to the outputs of the get_pdf function,
# X_interp and Y_interp.  Like in the plotting functions for crater count data, there is an ax
# variable for matplotlib plotting purposes.  Under the default value, 'None', the variable is plotted as a
# subplot of a newly created figure.  However, you can pass a different matplotlib axis to it if you want to
# build combined or heavily customized plots.  The color parameter gives the plot color, and the upshift parameter
# applies a uniform vertical upwards shift.  It is used for stacking PDFs on top of each other to make combination
# plots.  See examples in the Jupyter notebooks showing how to use the code.
def plot_pdf(T_pdf,P_pdf,ax='None',upshift=0,color='blue'):
    if ax=='None':
        fig = plt.figure(figsize=(10,2))
        ax = fig.add_subplot(111)
    ymax=max(P_pdf)
    if P_pdf[0] <= 0.25*ymax:
        Y_pdf = np.array([0] + list(np.array(P_pdf)/ymax))
    else:
        Y_pdf = np.array([max(P_pdf[0]-(P_pdf[1]-P_pdf[0])/((T_pdf[2]-T_pdf[1])/(T_pdf[1]-T_pdf[0])),0)/ymax] \
                         + list(np.array(P_pdf)/ymax))
    X_pdf = np.array([0] + list(T_pdf))
    Y_pdf = Y_pdf+upshift
    ax.plot(X_pdf,Y_pdf,color,linewidth=2)
    ilow,low1,med1,high1,ihigh = np.interp(np.array([0.13,1-0.841345,0.5,0.841345,0.87]),P_pdf.cumsum()/P_pdf.sum(),T_pdf)
    max_val=X_pdf[Y_pdf.argmax()]
    X_interp = np.linspace(ilow,ihigh,10000)
    Y_interp = np.interp(X_interp,X_pdf,Y_pdf)
    ax.fill_between(X_interp,upshift,Y_interp, where=((low1<X_interp)&(X_interp<high1)), facecolor=color, alpha=0.07)
    ax.plot([med1,med1],[upshift,Y_interp[next(x[0] for x in enumerate(X_interp) if x[1] > med1)]],'--',color=color)
    ax.plot([low1,low1],[upshift,Y_interp[next(x[0] for x in enumerate(X_interp) if x[1] > low1)]],':',color=color)
    ax.plot([high1,high1],[upshift,Y_interp[next(x[0] for x in enumerate(X_interp) if x[1] > high1)]],':',color=color)
    ax.plot([max_val,max_val],[upshift,Y_interp[next(x[0] for x in enumerate(X_interp) \
                        if x[1] > max_val)]],color=color)
    return low1,med1,high1,max_val

# A comprehensive function to plot the age.  The bin_width_exponent and num_bins_fitted parameters must be adjusted
# by hand before running.  Plot the data first before running.
def plot_age(ds, area, bin_width_exponent=0.25, saturation_cutoff='none',n_points=2000, random_sigma=0):
    t1=time.time()
    if saturation_cutoff=='none':
        binned_data = calc_incremental(ds,area,n_points=n_points,bin_width_exponent=bin_width_exponent,\
                                      random_sigma=random_sigma)
    elif random_sigma==0:
        binned_data = calc_incremental(ds[np.array(ds)>=ds[np.array(ds)<saturation_cutoff].max()],area,\
            n_points=n_points,bin_width_exponent=bin_width_exponent,reference_point=saturation_cutoff)
    else:
        binned_data = calc_incremental(ds,area,n_points=n_points,bin_width_exponent=bin_width_exponent,\
                                       reference_point=saturation_cutoff,random_sigma=random_sigma)
        binned_data = binned_data[binned_data['D']>=saturation_cutoff]
    binned_data = binned_data.assign(n1_pdf=((binned_data['density_pdf']/(binned_data['D'].\
                apply(lambda d: npf_new_incremental(d,bin_width_exponent=bin_width_exponent))))*npf_new(1)))
    X_pdf_raw,zero_pdf=igpdf(0,n_points=5*n_points)
    X_pdf=X_pdf_raw/area/npf_new(binned_data.iloc[-1]['D']*2**bin_width_exponent)*npf_new(1)
    n1_pdf_list=[np.interp(X_pdf,row['n1_pdf'],row['P_pdf']) for i,row in binned_data.iterrows()]
    Y_combined_pdf = np.prod(n1_pdf_list,axis=0)*zero_pdf
    T_pdf=np.linspace(ncf_inv(X_pdf.min()),ncf_inv(X_pdf.max()),5*n_points)
    P_pdf=np.interp(T_pdf,ncf_inv(X_pdf),np.gradient(X_pdf,ncf_inv(X_pdf))*Y_combined_pdf)
    low,med,high,max_val=plot_pdf(T_pdf,P_pdf)
    plt.xlabel('Age (Ga)',size=16)
    plt.yticks([])
    val999 = np.percentile(P_pdf.cumsum(),99.9)
    idx=np.argmin(abs(P_pdf.cumsum()-val999))
    age_max = T_pdf[idx]
    plt.xlim([-0.01*age_max,age_max])
    plt.text(0,0.7,r'$'+str(round(max_val,3))+'_{-'+str(round(max_val-low,3))+'}^{+'+str(round(high-max_val,3))\
         +'}Ga$',size=25)
    return low,med,high,max_val

# A version of the age calculation function that addresses systematic error.  Much slower.
def plot_age_systematic_error(ds, area, bin_width_exponent=0.25, saturation_cutoff='none',n_points=2000\
                             ,systematic_sigma=0.05, n_points_systematic=1000):
    if saturation_cutoff=='none':
        binned_data = calc_incremental(ds,area,n_points=n_points,bin_width_exponent=bin_width_exponent)
    else:
        binned_data = calc_incremental(ds[np.array(ds)>=ds[np.array(ds)<saturation_cutoff].max()],area,\
            n_points=n_points,bin_width_exponent=bin_width_exponent,reference_point=saturation_cutoff)
    binned_data = binned_data.assign(n1_pdf=((binned_data['density_pdf']/(binned_data['D'].\
                apply(lambda d: npf_new_incremental(d,bin_width_exponent=bin_width_exponent))))*npf_new(1)))
    X_pdf_raw,zero_pdf=igpdf(0,n_points=5*n_points)
    X_pdf=X_pdf_raw/area/npf_new(binned_data.iloc[-1]['D']*2**bin_width_exponent)*npf_new(1)
    T_pdf=np.linspace(ncf_inv(X_pdf.min()),ncf_inv(X_pdf.max()),5*n_points)
    P_pdf_list=[]
    for shift in norm.rvs(loc=1,scale=systematic_sigma,size=n_points_systematic):
        D=ds*shift
        if saturation_cutoff=='none':
            binned_data = calc_incremental(D,area,n_points=n_points,bin_width_exponent=bin_width_exponent)
        else:
            binned_data = calc_incremental(D[np.array(D)>=D[np.array(D)<saturation_cutoff*shift].max()],area,\
                n_points=n_points,bin_width_exponent=bin_width_exponent,reference_point=saturation_cutoff*shift)
        binned_data = binned_data.assign(n1_pdf=((binned_data['density_pdf']/((binned_data['D']).\
                apply(lambda d: npf_new_incremental(d,bin_width_exponent=bin_width_exponent))))*npf_new(1)))
        X_pdf=X_pdf_raw/area/npf_new((binned_data.iloc[-1]['D'])*2**bin_width_exponent)*npf_new(1)
        n1_pdf_list=[np.interp(X_pdf,row['n1_pdf'],row['P_pdf']) for i,row in binned_data.iterrows()]
        Y_combined_pdf = np.prod(n1_pdf_list,axis=0)*zero_pdf
        P_pdf_i=np.interp(T_pdf,ncf_inv(X_pdf),np.gradient(X_pdf,ncf_inv(X_pdf))*Y_combined_pdf)
        P_pdf_list.append(P_pdf_i/P_pdf_i.max())

    P_pdf=np.sum(P_pdf_list,axis=0)
    low,med,high,max_val=plot_pdf(T_pdf,P_pdf)
    plt.xlabel('Age (Ga)',size=16)
    plt.yticks([])
    val999 = np.percentile(P_pdf.cumsum(),99.9)
    idx=np.argmin(abs(P_pdf.cumsum()-val999))
    age_max = T_pdf[idx]
    plt.xlim([-0.01*age_max,age_max])
    plt.text(0,0.7,r'$'+str(round(max_val,3))+'_{-'+str(round(max_val-low,3))+'}^{+'+str(round(high-max_val,3))\
         +'}Ga$',size=25)
    return low,med,high,max_val



