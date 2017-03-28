import numpy as np
import pandas as pd
from .mfun import *
from .update_peaks import *
from .Yakitori import *
from lmfit.models import GaussianModel
from matplotlib import pyplot as plt

# find better data range 
def index_of(arrval, value):
    if value < min(arrval): return 0
    return max(np.where(arrval<=value)[0])


## Fitting functions 
def gaussians_fit_helper(ell, cl, return_fig=False):
    '''fit power spectrum with gaussians
    parameter: 
    return_fig=True -> return figures for fitting result
    '''
    ix1 = index_of(ell,400)
    ix2 = index_of(ell,600)
    ix3 = index_of(ell,1000)
    ix4 = index_of(ell,1300)
    ix5 = index_of(ell,1600)

    # fits with gaussian functions 
    gauss1 = GaussianModel(prefix='g1_')
    gauss2 = GaussianModel(prefix='g2_')
    gauss3 = GaussianModel(prefix='g3_')
    gauss4 = GaussianModel(prefix='g4_')
    gauss5 = GaussianModel(prefix='g5_')
    gmod =  gauss1 + gauss2 + gauss3 + gauss4 + gauss5

    # initial vaule 
    pars = gauss1.guess(cl[:ix1], x=ell[:ix1])
    pars += gauss2.guess(cl[ix1:ix2], x=ell[ix1:ix2])
    pars += gauss3.guess(cl[ix2:ix3], x=ell[ix2:ix3])
    pars += gauss4.guess(cl[ix3:ix4], x=ell[ix3:ix4])
    pars += gauss5.guess(cl[ix4:ix5], x=ell[ix4:ix5])

    # fit model to data array ecl 
    result = gmod.fit(cl, pars, x=ell) ## ell=ell[0:1600], ecl=ecl[0:1600]
    if return_fig==True:
        plt.plot(ell, cl, 'k', linestyle='None', marker='.')
        plt.plot(ell, result.best_fit, lw=2)

        # Components plots
        comps = result.eval_components(x=ell)
        plt.plot(ell, comps['g1_'])
        plt.plot(ell, comps['g2_'])
        plt.plot(ell, comps['g3_'])
        plt.plot(ell, comps['g4_'])
        plt.plot(ell, comps['g5_'])

        plt.xlabel(r'$\ell$')
        plt.ylabel(r'$\ell(\ell+1) C_\ell/2\pi$')
        plt.xlim([0., ell[-1]])

    return result.params
    
def peak_df(ell, table_ecls, lmax, lmin, method='leastsq'):
    peak1 = []; peak2 = []; peak3 = []; peak4 = []; peak5 =[];
    sigma1 = []; sigma2 = []; sigma3 = []; sigma4 = []; sigma5 = [];
    amp1 = []; amp2 = []; amp3 = []; amp4 = []; amp5 = [];
    redchi  = [];
    for i in range(len(table_ecls)):
        result = gaussians_fit_helper(ell[index_of(ell, lmin):index_of(ell, lmax)], 
                                      table_ecls[i][index_of(ell, lmin):index_of(ell, lmax)],
                                      method=method)
        parms = result.params
        peak1.append(parms['g1_center'].value)
        peak2.append(parms['g2_center'].value)
        peak3.append(parms['g3_center'].value)
        peak4.append(parms['g4_center'].value)
        peak5.append(parms['g5_center'].value)
        sigma1.append(parms['g1_sigma'].value)
        sigma2.append(parms['g2_sigma'].value)
        sigma3.append(parms['g3_sigma'].value)
        sigma4.append(parms['g4_sigma'].value)
        sigma5.append(parms['g5_sigma'].value)
        amp1.append(parms['g1_amplitude'].value)
        amp2.append(parms['g2_amplitude'].value)
        amp3.append(parms['g3_amplitude'].value)
        amp4.append(parms['g4_amplitude'].value)
        amp5.append(parms['g5_amplitude'].value)
        
        redchi.append(result.redchi)
    d = {'peak1': peak1,
        'peak2' : peak2,
        'peak3' : peak3,
        'peak4' : peak4,
        'peak5' : peak5,
        'sigma1': sigma1,
        'sigma2': sigma2,
        'sigma3': sigma3,
        'sigma4': sigma4,
        'sigma5': sigma5,
        'amp1'  : amp1,
        'amp2'  : amp2,
        'amp3'  : amp3,
        'amp4'  : amp4,
        'amp5'  : amp5,
        'redchi': redchi}
    df = pd.DataFrame(d)
    del peak1, peak2, peak3, peak4, peak5, sigma1, sigma2, sigma3, sigma4, sigma5, redchi, parms, result
    
    # it is an ad-hoc function ... well I will fix the program next time
    df = updating_dataframe_peaks_with_roots(df)
    return df

def outer_outlier_help(df_test, df_train, amp_cut=True, sigma_cut=True, sort_peak=False, 
                      first_three=True):
    # naive cutoff 3 std
    print('len of dataframe before cutoff :', len(df_test))
    
    if amp_cut==True:
        # amplitude cutoff
        df_test = df_test[df_test['amp1'] < (df_train['amp1'].quantile(q=0.75) + (df_train['amp1'].quantile(q=0.75) - df_train['amp1'].quantile(q=0.25)) * 3)]
        df_test = df_test[df_test['amp1'] > (df_train['amp1'].quantile(q=0.25) - (df_train['amp1'].quantile(q=0.75) - df_train['amp1'].quantile(q=0.25)) * 3)]
        df_test = df_test[df_test['amp2'] < (df_train['amp2'].quantile(q=0.75) + (df_train['amp2'].quantile(q=0.75) - df_train['amp2'].quantile(q=0.25)) * 3)]
        df_test = df_test[df_test['amp2'] > (df_train['amp2'].quantile(q=0.25) - (df_train['amp2'].quantile(q=0.75) - df_train['amp2'].quantile(q=0.25)) * 3)]
        df_test = df_test[df_test['amp3'] < (df_train['amp3'].quantile(q=0.75) + (df_train['amp3'].quantile(q=0.75) - df_train['amp3'].quantile(q=0.25)) * 3)]
        df_test = df_test[df_test['amp3'] > (df_train['amp3'].quantile(q=0.25) - (df_train['amp3'].quantile(q=0.75) - df_train['amp3'].quantile(q=0.25)) * 3)]
        if first_three==False:
            df_test = df_test[df_test['amp4'] < (df_train['amp4'].quantile(q=0.75) + (df_train['amp4'].quantile(q=0.75) - df_train['amp4'].quantile(q=0.25)) * 3)]
            df_test = df_test[df_test['amp4'] > (df_train['amp4'].quantile(q=0.25) - (df_train['amp4'].quantile(q=0.75) - df_train['amp4'].quantile(q=0.25)) * 3)]
            df_test = df_test[df_test['amp5'] < (df_train['amp5'].quantile(q=0.75) + (df_train['amp5'].quantile(q=0.75) - df_train['amp5'].quantile(q=0.25)) * 3)]
            df_test = df_test[df_test['amp5'] > (df_train['amp5'].quantile(q=0.25) - (df_train['amp5'].quantile(q=0.75) - df_train['amp5'].quantile(q=0.25)) * 3)]
        print('len of dataframe after amplitude cutoff :', len(df_test))
    
    if sigma_cut==True:
        # sigma cutoff
        df_test = df_test[df_test['sigma1'] < (df_train['sigma1'].quantile(q=0.75) + (df_train['sigma1'].quantile(q=0.75) - df_train['sigma1'].quantile(q=0.25)) * 3)]
        df_test = df_test[df_test['sigma1'] > (df_train['sigma1'].quantile(q=0.25) - (df_train['sigma1'].quantile(q=0.75) - df_train['sigma1'].quantile(q=0.25)) * 3)]
        df_test = df_test[df_test['sigma2'] < (df_train['sigma2'].quantile(q=0.75) + (df_train['sigma2'].quantile(q=0.75) - df_train['sigma2'].quantile(q=0.25)) * 3)]
        df_test = df_test[df_test['sigma2'] > (df_train['sigma2'].quantile(q=0.25) - (df_train['sigma2'].quantile(q=0.75) - df_train['sigma2'].quantile(q=0.25)) * 3)]
        df_test = df_test[df_test['sigma3'] < (df_train['sigma3'].quantile(q=0.75) + (df_train['sigma3'].quantile(q=0.75) - df_train['sigma3'].quantile(q=0.25)) * 3)]
        df_test = df_test[df_test['sigma3'] > (df_train['sigma3'].quantile(q=0.25) - (df_train['sigma3'].quantile(q=0.75) - df_train['sigma3'].quantile(q=0.25)) * 3)]
        if first_three==False:
            df_test = df_test[df_test['sigma4'] < (df_train['sigma4'].quantile(q=0.75) + (df_train['sigma4'].quantile(q=0.75) - df_train['sigma4'].quantile(q=0.25)) * 3)]
            df_test = df_test[df_test['sigma4'] > (df_train['sigma4'].quantile(q=0.25) - (df_train['sigma4'].quantile(q=0.75) - df_train['sigma4'].quantile(q=0.25)) * 3)]
            df_test = df_test[df_test['sigma5'] < (df_train['sigma5'].quantile(q=0.75) + (df_train['sigma5'].quantile(q=0.75) - df_train['sigma5'].quantile(q=0.25)) * 3)]
            df_test = df_test[df_test['sigma5'] > (df_train['sigma5'].quantile(q=0.25) - (df_train['sigma5'].quantile(q=0.75) - df_train['sigma5'].quantile(q=0.25)) * 3)]
        print('len of dataframe after sigma cutoff :', len(df_test))
    
    if sort_peak==True:
        # sort alignment peaks
        df_test = df_test[df_test['peak1'] < df_test['peak2']]
        df_test = df_test[df_test['peak1'] < df_test['peak3']]
        df_test = df_test[df_test['peak2'] < df_test['peak3']]
        if first_three==False:
            df_test = df_test[df_test['peak1'] < df_test['peak4']]
            df_test = df_test[df_test['peak1'] < df_test['peak5']]
            df_test = df_test[df_test['peak2'] < df_test['peak4']]
            df_test = df_test[df_test['peak2'] < df_test['peak5']]
            df_test = df_test[df_test['peak3'] < df_test['peak4']]
            df_test = df_test[df_test['peak3'] < df_test['peak5']]
            df_test = df_test[df_test['peak4'] < df_test['peak5']]
        print('len of dataframe after sorting alignment peaks :', len(df_test))

    return df_test

def shift_df_concat(df, return_abs=False):
    # Call in Best-Fit
    best = np.loadtxt('../Release2/COM_PowerSpect_CMB-base-plikHM-TT-lowTEB-minimum-theory_R2.02.txt')
    L = best.T[0]
    CL = best.T[1]
    C1 = ['peak1', 'peak2', 'peak3']
    C2 = ['g1_peak', 'g2_peak', 'g3_peak']
    
    # Best fit with 5 gaussians
    best = gaussians_fit_helper(L[0:1604], CL[0:1604], return_fig=False)
    best = update_gaussuan_helper_with_roots(best)
    shift = pd.DataFrame([df.loc[:, c1] - best.best_values[c2] for c1, c2 in zip(C1, C2)]).T
    shift.columns = ['shift1','shift2','shift3']

    # concat the total shift
    if return_abs==True:
        shift = pd.concat([shift, np.sum(abs(shift), axis=1), (np.sum(shift**2, axis=1))/3], axis=1)
    else:
        shift = pd.concat([shift, np.sum(shift, axis=1), (np.sum(shift**2, axis=1))/3], axis=1)
    
    shift.columns = ['shift1','shift2','shift3', 'shift_sum', 'shift_variance']
    
    # concat df and shift
    return pd.concat([df, shift], axis=1)
    

def self_df_concat(df, return_abs=False):
    C1 = ['peak1', 'peak2', 'peak3']
    
    # Best fit with 5 gaussians
    shift = pd.DataFrame([df.loc[:, c1] - df.loc[:, c1].mean() for c1 in C1]).T
    shift.columns = ['shift1','shift2','shift3']

    # concat the total shift
    if return_abs==True:
        shift = pd.concat([shift, np.sum(abs(shift), axis=1), np.sqrt(np.sum(shift**2, axis=1))], axis=1)
    else:
        shift = pd.concat([shift, np.sum(shift, axis=1), np.sqrt(np.sum(shift**2, axis=1))], axis=1)
    
    shift.columns = ['shift1','shift2','shift3', 'shift_sum', 'shift_var_distance']
    
    # concat df and shift
    return pd.concat([df, shift], axis=1)
    
def weighted_sky_map(df, a, b, size, theta_size, canvas_size, nside, return_abs=False):
    IMP = np.zeros((canvas_size//2, canvas_size))

    # suboptimal choice: let bad-fit become zero
    df_con = shift_df_concat(df, return_abs=return_abs)
    df_bad = bad_df(df)
    bad_index = list(df_bad.index)

    # subpotimal choice: rebulid list without bad-fit
    shift_array = np.copy(df_con['shift_sum'].values)
    shift_array[bad_index] = 0
    importance_list = list_rebuild(shift_array, b)

    for (i,theta),importance in zip(enumerate(a), importance_list):
        IMP += Shift_Block_Rotater(theta * theta_size, size, canvas_size, nside, 
                                            b[i], importance)

    IMP_healpix = cart_healpix(IMP, nside)
    return IMP_healpix

def list_rebuild(index, b):
    '''
    index: list, the list you want to rebuild
    b: list, the base list you want to take it as template to rebuid a list
    return:
    rebuild_list: list, reconstructed list based on b.
    '''
    # generate aggregate index
    length = [len(B) for B in b]
    if len(index) == np.sum(length):
        sum_length = np.insert(np.add.accumulate(length), 0, 0) # accumulate is cool
        return [index[sum_length[i]:sum_length[i+1]] for i,l in enumerate(sum_length[:-1])]
    else: print ('Not the same length!')
    
def bad_df(df):
    # pandas index to list
    pre_idx = df.index.tolist()

    # outer outlier eliminating
    df_cut = outer_outlier_help(df, df, amp_cut=True, sigma_cut=True, sort_peak=True)
    aft_idx = df_cut.index.tolist()

    # fancy indexing with boolean indicing operator !=
    boolean_list = [idx not in aft_idx for idx in pre_idx]
    df_bad = df[boolean_list]
    return df_bad
