import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math
from scipy.stats.mstats import gmean
from scipy.stats import gamma,poisson,linregress,beta
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
# Full numerical PDF
def ig_ndist(n,n_points=1000):
    inc = 1.0/float(n_points)
    return np.array(gamma.ppf(np.linspace(inc/2,1-inc/2,n_points),n+1))
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

# Calculates the incremental left version of the "new" Neukum Production Function for a given diameter or numpy 
# array of diameters, d_km.  The incremental left production function corresponds to the incremental left plot, 
# which is a version of an incremental plot where the X-axis diameter value is the minimum (or left-hand) diameter
# of the bin, and the bin widths are equal.  This is the recommended production function for Bayesian age
# determinations.  Bins are independent of each other, and no prior knowledge of the production function is needed
# to plot data because the production function assumes finite bins.  The bin_width_exponent parameter is the
# exponent used to define the bin edges, according to the formula bin_edge[i+1]/bin_edge[i] = 2^bin_width_exponent,
# with the default value of 0.5 giving sqrt(2) bins with bin edges [..., 2^-1=1/2, 2^-0.5=1/sqrt(2), 2^0=1,
# 2^0.5=sqrt(2), 2^1=2, 2^1.5=2*sqrt(2), ...].
def npf_new_incremental_left(d_km,bin_width_exponent=0.5):
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
        self.norm = 5.44*10**-14*(math.exp(6.93*1.0)-1)+8.38*10**-4*1.0 / npf_new(1)
        self.t_array = np.linspace(0,5,nseg)
        self.ncf_array = 5.44*10**-14*(np.exp(6.93*self.t_array)-1)+8.38*10**-4*self.t_array / self.norm
    def inv(self,observed_cumulative_density):
        i = np.argmin(np.abs(self.ncf_array-observed_cumulative_density))
        if self.ncf_array[i]==observed_cumulative_density:
            return t_array[i]
        elif self.ncf_array[i]>observed_cumulative_density:
            i -= 1
        return self.t_array[i] + (self.t_array[i+1]-self.t_array[i])*(observed_cumulative_density-self.ncf_array[i])\
            /(self.ncf_array[i+1]-self.ncf_array[i])
    
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
# 'ndist': this contains a numerical distribution of λ, with a number of points equal to n_points
#
# They have the following standard inputs:
# ds: an array of crater diameters in km
# area: the area of the unit being counted in square km
# n_points: the number of points for the 'ndist' numerical distribution, default of 10,000
# bin_width_exponent: the exponent used to define the bin edges, according to the formula 2^bin_width_exponent,
# with the default value of 0.5 giving sqrt(2) bins with bin edges [..., 2^-1=1/2, 2^-0.5=1/sqrt(2), 2^0=1,
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

def calc_cumulative_unbinned(ds,area,n_points=10000):
    sorted_ds = sorted(ds,reverse=True)
    N_list = np.array(list(range(1,len(ds)+1)))
    lower_list=ig_lower(N_list)/area
    med_list=ig_50(N_list)/area
    upper_list=ig_upper(N_list)/area
    ndist_list=[ig_ndist(cum_count,n_points=n_points)/area for cum_count in cum_counts]
    return pd.DataFrame({'D':sorted_ds,'count':N_list,'lower':lower_list,\
                         'median':med_list,'upper':upper_list,'ndist':ndist_list})

def calc_cumulative_binned(ds,area,n_points=10000,bin_width_exponent=0.5):
    bin_min = math.ceil(math.log(min(ds),2)/bin_width_exponent)
    bin_max = math.ceil(math.log(max(ds),2)/bin_width_exponent)
    bins = [2.0**(bin_width_exponent*n) for n in list(range(bin_min,bin_max+1))]
    bin_gmeans = [2.0**(bin_width_exponent*(n+0.5)) for n in list(range(bin_min,bin_max))]
    bin_counts,bin_array=np.histogram(ds,bins)
    cum_counts = np.flip(np.flip(bin_counts).cumsum())
    lower_list=ig_lower(cum_counts)/area
    med_list=ig_50(cum_counts)/area
    upper_list=ig_upper(cum_counts)/area
    ndist_list=[ig_ndist(cum_count,n_points=n_points)/area for cum_count in cum_counts]
    return pd.DataFrame({'D':bin_gmeans,'count':cum_counts,'lower':lower_list,\
                         'median':med_list,'upper':upper_list,'ndist':ndist_list})

def calc_incremental(ds,area,n_points=10000,bin_width_exponent=0.5):
    bin_min = math.ceil(math.log(min(ds),2)/bin_width_exponent)
    bin_max = math.ceil(math.log(max(ds),2)/bin_width_exponent)
    bins = [2.0**(bin_width_exponent*n) for n in list(range(bin_min,bin_max+1))]
    bin_gmeans = [2.0**(bin_width_exponent*(n+0.5)) for n in list(range(bin_min,bin_max))]
    bin_counts,bin_array=np.histogram(ds,bins)
    lower_list=ig_lower(bin_counts)/area
    med_list=ig_50(bin_counts)/area
    upper_list=ig_upper(bin_counts)/area
    ndist_list=[ig_ndist(bin_count,n_points=n_points)/area for bin_count in bin_counts]
    return pd.DataFrame({'D':bin_gmeans,'count':bin_counts,'lower':lower_list,\
                         'median':med_list,'upper':upper_list,'ndist':ndist_list})

def calc_incremental_left(ds,area,n_points=10000,bin_width_exponent=0.5):
    bin_min = math.ceil(math.log(min(ds),2)/bin_width_exponent)
    bin_max = math.ceil(math.log(max(ds),2)/bin_width_exponent)
    bins = [2.0**(bin_width_exponent*n) for n in list(range(bin_min,bin_max+1))]
    bin_counts,bin_array=np.histogram(ds,bins)
    lower_list=ig_lower(bin_counts)/area
    med_list=ig_50(bin_counts)/area
    upper_list=ig_upper(bin_counts)/area
    ndist_list=[ig_ndist(bin_count,n_points=n_points)/area for bin_count in bin_counts]
    return pd.DataFrame({'D':bins[:-1],'count':bin_counts,'lower':lower_list,\
                         'median':med_list,'upper':upper_list,'ndist':ndist_list})

def calc_differential(ds,area,n_points=10000,bin_width_exponent=0.5,do_correction=True,\
                      production_function=npf_new):
    bin_min = math.ceil(math.log(min(ds),2)/bin_width_exponent)
    bin_max = math.ceil(math.log(max(ds),2)/bin_width_exponent)
    bins = [2.0**(bin_width_exponent*n) for n in list(range(bin_min,bin_max+1))]
    bin_gmeans = [2.0**(bin_width_exponent*(n+0.5)) for n in list(range(bin_min,bin_max))]
    bin_widths = np.array([bins[i+1]-bins[i] for i in list(range(len(bins)-1))])
    if do_correction:
        local_slope = np.array([(np.log10(production_function(bins[n+1])) - \
                np.log10(production_function(bins[n]))) / (np.log10(bins[n+1]) - np.log10(bins[n])) \
                for n in list(range(len(bins)-1))])
        correction_factors = np.array((2**(bin_width_exponent*local_slope/2) - 2**(-1*bin_width_exponent*\
                local_slope/2)) / (local_slope * (2**(bin_width_exponent/2) - 2**(-1*bin_width_exponent/2))))
    else:
        correction_factors = np.ones(len(bin_gmeans))
    bin_counts,bin_array=np.histogram(ds,bins)
    lower_list=ig_lower(bin_counts)/area/bin_widths/correction_factors
    med_list=ig_50(bin_counts)/area/bin_widths/correction_factors
    upper_list=ig_upper(bin_counts)/area/bin_widths/correction_factors
    ndist_list=[ig_ndist(bin_counts[i],n_points=n_points)/area/bin_widths[i]/correction_factors[i] \
                for i in list(range(len(bin_counts)))]
    return pd.DataFrame({'D':bin_gmeans,'count':bin_counts,'lower':lower_list,\
                         'median':med_list,'upper':upper_list,'ndist':ndist_list})

def calc_R(ds,area,n_points=10000,bin_width_exponent=0.5,do_correction=True,production_function=npf_new):
    bin_min = math.ceil(math.log(min(ds),2)/bin_width_exponent)
    bin_max = math.ceil(math.log(max(ds),2)/bin_width_exponent)
    bins = [2.0**(bin_width_exponent*n) for n in list(range(bin_min,bin_max+1))]
    bin_gmeans = pd.DataFrame({'D':ds,'bin':pd.cut(pd.Series(ds),bins,labels=list(range(len(bins)-1)))})\
            .groupby('bin').apply(gmean).apply(lambda x: x[0])
    empties=bin_gmeans.isna()
    bin_gmeans[empties]=[math.sqrt(bins[i]*bins[i+1]) for i in bin_gmeans[empties].index.tolist()]
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
    bin_counts,bin_array=np.histogram(ds,bins)
    lower_list=ig_lower(bin_counts)/area/bin_widths/correction_factors*bin_adjustments
    med_list=ig_50(bin_counts)/area/bin_widths/correction_factors*bin_adjustments
    upper_list=ig_upper(bin_counts)/area/bin_widths/correction_factors*bin_adjustments
    ndist_list=[ig_ndist(bin_counts[i],n_points=n_points)/area/bin_widths[i]/correction_factors[i]\
                *bin_adjustments[i] for i in list(range(len(bin_counts)))]
    return pd.DataFrame({'D':bin_gmeans,'count':bin_counts,'lower':lower_list,\
                         'median':med_list,'upper':upper_list,'ndist':ndist_list})

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

def plot_cumulative_unbinned(ds,area,ax='None'):
    if ax=='None':
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
    plt.rcParams['lines.linewidth'] = 1.0
    sorted_ds = sorted(ds,reverse=True)
    N_list = np.array(list(range(1,len(ds)+1)))
    lower_list=ig_lower(N_list)/area
    med_list=ig_50(N_list)/area
    upper_list=ig_upper(N_list)/area
    data = pd.DataFrame({'D':sorted_ds,'N':N_list,'lower':lower_list,'med':med_list,'upper':upper_list})
    data.plot(y='med',x='D', \
                              yerr=np.array([data[['lower','upper']].values.transpose()]), 
                              logx=True, logy=True, kind='scatter', 
                              color='gray', legend=True, s=0, ax=ax)
    ax.loglog(sorted_ds,med_list,'k')
    plt.xticks(size=20)
    plt.yticks(size=20)
    xmax=np.max(sorted_ds)
    xmin=np.min(sorted_ds)
    xrange=np.log10(xmax/xmin)
    plt.xlim([xmin/(1.15*xrange),xmax*(1.15*xrange)])
    ymax=np.max(med_list+upper_list)
    ymin=np.min(med_list-lower_list)
    yrange=np.log10(ymax/ymin)
    plt.ylim([ymin/(1.15*yrange),ymax*(1.15*yrange)])
    plt.ylabel('Cumulative Crater Density',size=18)
    plt.xlabel('Crater Diameter (km)',size=18)
    plt.grid(which='major', linestyle=':', linewidth=0.5, color='black')
    plt.grid(which='minor', linestyle=':', linewidth=0.25, color='gray')

def plot_cumulative_binned(ds,area,ax='None',bin_width_exponent=0.5):
    if ax=='None':
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
    plt.rcParams['lines.linewidth'] = 1.0
    bin_min = math.ceil(math.log(min(ds),2)/bin_width_exponent)
    bin_max = math.ceil(math.log(max(ds),2)/bin_width_exponent)
    bins = [2.0**(bin_width_exponent*n) for n in list(range(bin_min,bin_max+1))]
    bin_gmeans = [2.0**(bin_width_exponent*(n+0.5)) for n in list(range(bin_min,bin_max))]
    bin_counts,bin_array=np.histogram(ds,bins)
    cum_counts = np.flip(np.flip(bin_counts).cumsum())
    lower_list=ig_lower(cum_counts)/area
    med_list=ig_50(cum_counts)/area
    upper_list=ig_upper(cum_counts)/area
    binned=pd.DataFrame({'count':bin_counts,'D':bin_gmeans})
    zero_bins = binned.copy()
    binned.loc[binned['count']==0,'D']=None
    zero_bins.loc[zero_bins['count']!=0,'D']=None
    bins_nonzero = binned['D']
    bins_zero = zero_bins['D']
    plt.rcParams['lines.linewidth'] = 1.0
    data = pd.DataFrame({'D':bins_nonzero,'lower':lower_list,'med':med_list,'upper':upper_list})
    data.plot(y='med',x='D',\
                yerr=np.array([data[['lower','upper']].values.transpose()]), \
                logx=True, logy=True, kind='scatter', marker='s',color='gray', \
                s=0, ax=ax)
    ax.loglog(bins_nonzero,med_list,marker='s',ls='',mfc='none',mec='k',mew=1.2,ms=7)
    plt.rcParams['lines.linewidth'] = 0.5
    data = pd.DataFrame({'D':bins_zero,'lower':lower_list,'med':med_list,'upper':upper_list})
    data.plot(y='med',x='D', \
                              yerr=np.array([data[['lower','upper']].values.transpose()]), 
                              logx=True, logy=True, kind='scatter', 
                              color='gray', s=0, ax=ax)
    plt.rcParams['lines.linewidth'] = 1.0
    ax.loglog(bins_zero,med_list,marker='s',ls='',mfc='none',mec='gray',mew=1.2,ms=7)
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

def plot_incremental(ds,area,ax='None',bin_width_exponent=0.5):
    if ax=='None':
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
    plt.rcParams['lines.linewidth'] = 1.0
    bin_min = math.ceil(math.log(min(ds),2)/bin_width_exponent)
    bin_max = math.ceil(math.log(max(ds),2)/bin_width_exponent)
    bins = [2.0**(bin_width_exponent*n) for n in list(range(bin_min,bin_max+1))]
    bin_gmeans = [2.0**(bin_width_exponent*(n+0.5)) for n in list(range(bin_min,bin_max))]
    bin_counts,bin_array=np.histogram(ds,bins)
    lower_list=ig_lower(bin_counts)/area
    med_list=ig_50(bin_counts)/area
    upper_list=ig_upper(bin_counts)/area
    binned=pd.DataFrame({'count':bin_counts,'D':bin_gmeans})
    zero_bins = binned.copy()
    binned.loc[binned['count']==0,'D']=None
    zero_bins.loc[zero_bins['count']!=0,'D']=None
    bin_gmeans_nonzero = binned['D']
    bin_gmeans_zero = zero_bins['D']
    plt.rcParams['lines.linewidth'] = 1.0
    data = pd.DataFrame({'D':bin_gmeans_nonzero,'lower':lower_list,'med':med_list,'upper':upper_list})
    data.plot(y='med',x='D',\
                yerr=np.array([data[['lower','upper']].values.transpose()]), \
                logx=True, logy=True, kind='scatter', marker='s',color='gray', \
                s=0, ax=ax)
    ax.loglog(bin_gmeans_nonzero,med_list,marker='s',ls='',mfc='none',mec='k',mew=1.2,ms=7)
    plt.rcParams['lines.linewidth'] = 0.5
    data = pd.DataFrame({'D':bin_gmeans_zero,'lower':lower_list,'med':med_list,'upper':upper_list})
    data.plot(y='med',x='D', \
                              yerr=np.array([data[['lower','upper']].values.transpose()]), 
                              logx=True, logy=True, kind='scatter', 
                              color='gray', s=0, ax=ax)
    plt.rcParams['lines.linewidth'] = 1.0
    ax.loglog(bin_gmeans_zero,med_list,marker='s',ls='',mfc='none',mec='gray',mew=1.2,ms=7)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.xlim([bins[0],bins[-1]])
    ymax=np.max(med_list+upper_list)
    ymin=np.min(med_list-lower_list)
    yrange=np.log10(ymax/ymin)
    plt.ylim([ymin/(1.15*yrange),ymax*(1.15*yrange)])
    plt.ylabel('Incremental Crater Density',size=18)
    plt.xlabel('Crater Diameter (km)',size=18)
    plt.grid(which='major', linestyle=':', linewidth=0.5, color='black')
    plt.grid(which='minor', linestyle=':', linewidth=0.25, color='gray')

def plot_incremental_left(ds,area,ax='None',bin_width_exponent=0.5):
    if ax=='None':
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
    plt.rcParams['lines.linewidth'] = 1.0
    bin_min = math.ceil(math.log(min(ds),2)/bin_width_exponent)
    bin_max = math.ceil(math.log(max(ds),2)/bin_width_exponent)
    bins = [2.0**(bin_width_exponent*n) for n in list(range(bin_min,bin_max+1))]
    bin_counts,bin_array=np.histogram(ds,bins)
    lower_list=ig_lower(bin_counts)/area
    med_list=ig_50(bin_counts)/area
    upper_list=ig_upper(bin_counts)/area
    binned=pd.DataFrame({'count':bin_counts,'D':bin_array[:-1]})
    zero_bins = binned.copy()
    binned.loc[binned['count']==0,'D']=None
    zero_bins.loc[zero_bins['count']!=0,'D']=None
    bin_edge_nonzero = binned['D']
    bin_edge_zero = zero_bins['D']
    plt.rcParams['lines.linewidth'] = 1.0
    data = pd.DataFrame({'D':bin_edge_nonzero,'lower':lower_list,'med':med_list,'upper':upper_list})
    data.plot(y='med',x='D',\
                yerr=np.array([data[['lower','upper']].values.transpose()]), \
                logx=True, logy=True, kind='scatter', marker='s',color='gray', \
                s=0, ax=ax)
    ax.loglog(bin_edge_nonzero,med_list,marker='s',ls='',mfc='none',mec='k',mew=1.2,ms=7)
    plt.rcParams['lines.linewidth'] = 0.5
    data = pd.DataFrame({'D':bin_edge_zero,'lower':lower_list,'med':med_list,'upper':upper_list})
    data.plot(y='med',x='D', \
                              yerr=np.array([data[['lower','upper']].values.transpose()]), 
                              logx=True, logy=True, kind='scatter', 
                              color='gray', s=0, ax=ax)
    plt.rcParams['lines.linewidth'] = 1.0
    ax.loglog(bin_edge_zero,med_list,marker='s',ls='',mfc='none',mec='gray',mew=1.2,ms=7)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.xlim([bins[0]-(bins[1]-bins[0]),bins[-1]])
    ymax=np.max(med_list+upper_list)
    ymin=np.min(med_list-lower_list)
    yrange=np.log10(ymax/ymin)
    plt.ylim([ymin/(1.15*yrange),ymax*(1.15*yrange)])
    plt.ylabel('Incremental Crater Density',size=18)
    plt.xlabel('Crater Diameter (km)',size=18)
    plt.grid(which='major', linestyle=':', linewidth=0.5, color='black')
    plt.grid(which='minor', linestyle=':', linewidth=0.25, color='gray')

def plot_differential(ds,area,ax='None',bin_width_exponent=0.5,do_correction=True,production_function=npf_new):
    if ax=='None':
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
    bin_min = math.ceil(math.log(min(ds),2)/bin_width_exponent)
    bin_max = math.ceil(math.log(max(ds),2)/bin_width_exponent)
    bins = [2.0**(bin_width_exponent*n) for n in list(range(bin_min,bin_max+1))]
    bin_gmeans = [2.0**(bin_width_exponent*(n+0.5)) for n in list(range(bin_min,bin_max))]
    bin_widths = np.array([bins[i+1]-bins[i] for i in list(range(len(bins)-1))])
    if do_correction:
        local_slope = np.array([(np.log10(production_function(bins[n+1])) - \
                np.log10(production_function(bins[n]))) / (np.log10(bins[n+1]) - np.log10(bins[n])) \
                for n in list(range(len(bins)-1))])
        correction_factors = np.array((2**(bin_width_exponent*local_slope/2) - 2**(-1*bin_width_exponent*\
                local_slope/2)) / (local_slope * (2**(bin_width_exponent/2) - 2**(-1*bin_width_exponent/2))))
    else:
        correction_factors = np.ones(len(bin_gmeans))
    bin_counts,bin_array=np.histogram(ds,bins)
    lower_list=ig_lower(bin_counts)/area/bin_widths/correction_factors
    med_list=ig_50(bin_counts)/area/bin_widths/correction_factors
    upper_list=ig_upper(bin_counts)/area/bin_widths/correction_factors
    binned=pd.DataFrame({'count':bin_counts,'D':bin_gmeans})
    zero_bins = binned.copy()
    binned.loc[binned['count']==0,'D']=None
    zero_bins.loc[zero_bins['count']!=0,'D']=None
    bin_gmeans_nonzero = binned['D']
    bin_gmeans_zero = zero_bins['D']
    plt.rcParams['lines.linewidth'] = 1.0
    data = pd.DataFrame({'D':bin_gmeans_nonzero,'lower':lower_list,'med':med_list,'upper':upper_list})
    data.plot(y='med',x='D', \
                              yerr=np.array([data[['lower','upper']].values.transpose()]), 
                              logx=True, logy=True, kind='scatter', 
                              color='gray', s=0, ax=ax)
    ax.loglog(bin_gmeans_nonzero,med_list,marker='s',ls='',mfc='none',mec='k',mew=1.2,ms=7)
    plt.rcParams['lines.linewidth'] = 0.5
    data = pd.DataFrame({'D':bin_gmeans_zero,'lower':lower_list,'med':med_list,'upper':upper_list})
    data.plot(y='med',x='D', \
                              yerr=np.array([data[['lower','upper']].values.transpose()]), 
                              logx=True, logy=True, kind='scatter', 
                              color='gray', s=0, ax=ax)
    plt.rcParams['lines.linewidth'] = 1.0
    ax.loglog(bin_gmeans_zero,med_list,marker='s',ls='',mfc='none',mec='gray',mew=1.2,ms=7)
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

def plot_R(ds,area,ax='None',bin_width_exponent=0.5,do_correction=True,production_function=npf_new):
    if ax=='None':
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
    bin_min = math.ceil(math.log(min(ds),2)/bin_width_exponent)
    bin_max = math.ceil(math.log(max(ds),2)/bin_width_exponent)
    bins = [2.0**(bin_width_exponent*n) for n in list(range(bin_min,bin_max+1))]
    bin_gmeans = pd.DataFrame({'D':ds,'bin':pd.cut(pd.Series(ds),bins,labels=list(range(len(bins)-1)))})\
            .groupby('bin').apply(gmean).apply(lambda x: x[0])
    empties=bin_gmeans.isna()
    empty_bin_gmeans = bin_gmeans.copy()
    empty_bin_gmeans[empties]=[math.sqrt(bins[i]*bins[i+1]) for i in bin_gmeans[bin_gmeans.isna()].index.tolist()]
    empty_bin_gmeans[~empties]=None
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
    bin_counts,bin_array=np.histogram(ds,bins)
    lower_list=ig_lower(bin_counts)/area/bin_widths/correction_factors*bin_adjustments
    med_list=ig_50(bin_counts)/area/bin_widths/correction_factors*bin_adjustments
    upper_list=ig_upper(bin_counts)/area/bin_widths/correction_factors*bin_adjustments
    plt.rcParams['lines.linewidth'] = 1.0
    data = pd.DataFrame({'D':bin_gmeans,'lower':lower_list,'med':med_list,'upper':upper_list})
    data.plot(y='med',x='D', \
                              yerr=np.array([data[['lower','upper']].values.transpose()]), 
                              logx=True, kind='scatter', 
                              color='gray', s=0,ax=ax)
    ax.loglog(bin_gmeans,med_list,marker='s',ls='',mfc='none',mec='k',mew=1.2,ms=7)
    plt.rcParams['lines.linewidth'] = 0.5
    empty_data = pd.DataFrame({'D':empty_bin_gmeans,'lower':lower_list,'med':med_list,'upper':upper_list})
    ax = empty_data.plot(y='med',x='D', \
                              yerr=np.array([data[['lower','upper']].values.transpose()]), 
                              logx=True, kind='scatter',linestyle='--',
                              color='gray', s=0, ax=ax)
    plt.rcParams['lines.linewidth'] = 1.0
    ax.loglog(empty_bin_gmeans,med_list,marker='s',ls='',mfc='none',mec='gray',mew=1.2,ms=7)
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
def plot_dist(f_ndist,n_bins=100,ax='None',upshift=0,color='blue'):
    if ax=='None':
        fig = plt.figure(figsize=(10,2))
        ax = fig.add_subplot(111)
    Y,bin_edges=np.histogram(f_ndist,bins=n_bins,density=True)
    bins=np.array(bin_edges)
    ymax=max(Y)
    if Y[0] <= 0.1*ymax:
        Y = [0] + list(np.array(Y)/ymax)
    else:
        Y = list(np.array(Y)/ymax)
        Y = np.array([Y[0]] + Y)
    X = [0] + list(0.5*(bins[1:]+bins[:-1]))
    Y = list(np.array(Y)+upshift)
    ax.plot(X,Y,color,linewidth=2)
    low1,med1,high1 = tuple(np.percentile(f_ndist,[100-84.1345,50,84.1345]))
    X_interp = np.linspace(np.percentile(f_ndist,13),np.percentile(f_ndist,87),10000)
    Y_interp = np.interp(X_interp,X,Y)
    ax.fill_between(X_interp,upshift,Y_interp, where=((low1<X_interp)&(X_interp<high1)), facecolor=color, alpha=0.07)
    ax.plot([med1,med1],[upshift,Y_interp[next(x[0] for x in enumerate(X_interp) if x[1] > med1)]],color=color)
    ax.plot([low1,low1],[upshift,Y_interp[next(x[0] for x in enumerate(X_interp) if x[1] > low1)]],':',color=color)
    ax.plot([high1,high1],[upshift,Y_interp[next(x[0] for x in enumerate(X_interp) if x[1] > high1)]],':',color=color)

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
    if Y[0] <= 0.1*ymax:
        Y = np.array([0] + list(np.array(Y)/ymax))
    else:
        Y = np.array([Y[0]] + list(np.array(Y)/ymax))
    X = np.array([0] + list(0.5*(bins[1:]+bins[:-1])))
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
    if P_pdf[0] <= 0.1*ymax:
        Y_pdf = np.array([0] + list(np.array(P_pdf)/ymax))
    else:
        Y_pdf = list(np.array(P_pdf)/ymax)
        Y_pdf = np.array([Y_pdf[0]] + Y_pdf)
    X_pdf = np.array([0] + list(T_pdf))
    Y_pdf = Y_pdf+upshift
    ax.plot(X_pdf,Y_pdf,color,linewidth=2)
    ilow,low1,med1,high1,ihigh = np.interp(np.array([0.13,1-0.841345,0.5,0.841345,0.87]),P_pdf.cumsum()/P_pdf.sum(),T_pdf)
    X_interp = np.linspace(ilow,ihigh,10000)
    Y_interp = np.interp(X_interp,X_pdf,Y_pdf)
    ax.fill_between(X_interp,upshift,Y_interp, where=((low1<X_interp)&(X_interp<high1)), facecolor=color, alpha=0.07)
    ax.plot([med1,med1],[upshift,Y_interp[next(x[0] for x in enumerate(X_interp) if x[1] > med1)]],color=color)
    ax.plot([low1,low1],[upshift,Y_interp[next(x[0] for x in enumerate(X_interp) if x[1] > low1)]],':',color=color)
    ax.plot([high1,high1],[upshift,Y_interp[next(x[0] for x in enumerate(X_interp) if x[1] > high1)]],':',color=color)
    return low1,med1,high1

# A comprehensive function to plot the age.  The bin_width_exponent and num_bins_fitted parameters must be adjusted
# by hand before running.  Plot the data first before running.
def plot_age(ds, area, bin_width_exponent=0.5, num_bins_fitted=5, n_bins=400, age_dist_size=10000000):
    binned_data = calc_incremental_left(ds,area,n_points=100000,bin_width_exponent=bin_width_exponent)
    binned_data = binned_data.assign(n1_dist=((binned_data['ndist']/(binned_data['D'].\
            apply(lambda d: npf_new_incremental_left(d,bin_width_exponent=bin_width_exponent))))*npf_new(1)))
    max_val=max([np.percentile(n1_dist,99) for n1_dist in binned_data.iloc[-1*num_bins_fitted:]['n1_dist']])
    density_pdf_list = [get_pdf(density_dist,n_bins=n_bins,max_val=max_val)[1] for density_dist in \
                        binned_data['n1_dist'][(-1*num_bins_fitted):]]
    zero_dist=ig_ndist(0,n_points=10000)/area/npf_new(binned_data.iloc[-1]['D']*2**bin_width_exponent)*npf_new(1)
    X_pdf,zero_pdf=get_pdf(zero_dist,n_bins=n_bins,max_val=max_val)
    Y_combined_pdf = np.prod(density_pdf_list,axis=0)*zero_pdf
    n1_combined_dist=np.random.choice(X_pdf,p=Y_combined_pdf/sum(Y_combined_pdf),size=age_dist_size)
    age_dist=ncf_inv(n1_combined_dist)
    T_pdf,P_pdf=get_pdf(age_dist,max_val=ncf_inv(max(X_pdf)))
    low,med,high=plot_pdf(T_pdf,P_pdf)
    plt.xlabel('age (Ga)',size=16)
    plt.yticks([])
    val999 = np.percentile(P_pdf.cumsum(),99.9)
    idx=np.argmin(abs(P_pdf.cumsum()-val999))
    age_max = T_pdf[idx]
    plt.xlim([-0.01*age_max,age_max])
    plt.text(0,0.75,r'$'+str(round(med,3))+'_{-'+str(round(med-low,3))+'}^{+'+str(round(high-med,3))+'}Ga$',size=25)
    return low,med,high



