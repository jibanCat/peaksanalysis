import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import healpy as hp
from .mfun import *
from .Yakitori import *
from lmfit.models import GaussianModel
from .update_peaks import *


def index_of(arrval, value):
    if value < min(arrval): return 0
    return max(np.where(arrval<=value)[0])

def gaussians_fit_helper(ell, cl, return_fig=False, 
                         bound_peak=False, range_guess=False, method='leastsq'):
    '''fit power spectrum with gaussians
    parameter: 
    return_fig=True -> return figures for fitting result
    '''

    # find better data range 
    def index_of(arrval, value):
        if value < min(arrval): return 0
        return max(np.where(arrval<=value)[0])
    ix1 = index_of(ell,415.35107897)
    ix2 = index_of(ell,670.1145778)
    ix3 = index_of(ell,1020.37320212)
    ix4 = index_of(ell,1311.72424321)
    ix5 = index_of(ell,1604)
    
    # fits with gaussian functions 
    gauss1 = GaussianModel(prefix='g1_')
    gauss2 = GaussianModel(prefix='g2_')
    gauss3 = GaussianModel(prefix='g3_')
    gauss4 = GaussianModel(prefix='g4_')
    gauss5 = GaussianModel(prefix='g5_')
    
    # initial vaule 
    if range_guess==True:
        pars = gauss1.guess(cl[:ix1], x=ell[:ix1])
        pars += gauss2.guess(cl[ix1:ix2], x=ell[ix1:ix2])
        pars += gauss3.guess(cl[ix2:ix3], x=ell[ix2:ix3])
        pars += gauss4.guess(cl[ix3:ix4], x=ell[ix3:ix4])
        pars += gauss5.guess(cl[ix4:ix5], x=ell[ix4:ix5])
    if range_guess==False:
        pars = gauss1.guess(cl[:ix1], x=ell[:ix1])
        pars.update(gauss1.make_params())
        pars['g1_amplitude'].set(1383237.6375727942)
        pars['g1_center'].set(218.25413354426811) # 218.25413354426811
        pars['g1_sigma'].set(97.08110083797027)
        pars.update(gauss2.make_params())
        pars['g2_amplitude'].set(541007.6549703055)
        pars['g2_center'].set(532.7906394119341) # 532.7906394119341
        pars['g2_sigma'].set(86.81717396133978)
        pars.update(gauss3.make_params())
        pars['g3_amplitude'].set(692701.0866300496)
        pars['g3_center'].set(812.1818016015095) # 812.1818016015095
        pars['g3_sigma'].set(110.16506040901645)
        pars.update(gauss4.make_params())
        pars['g4_amplitude'].set(243336.67891973304)
        pars['g4_center'].set(1125.9129396789704) # 1125.9129396789704
        pars['g4_sigma'].set(92.67204801014215)
        pars.update(gauss5.make_params())
        pars['g5_amplitude'].set(382571.02244731307)
        pars['g5_center'].set(1424.7207136484815) # 1424.7207136484815
        pars['g5_sigma'].set(183.43056355023555)
    
    gmod =  gauss1 + gauss2 + gauss3 + gauss4 + gauss5
    
    # bound the parameter
    #pars['g1_sigma'].set(min=48.046687036192097, max=147.56693526257902)
    #pars['g2_sigma'].set(min=6.5892775323649744, max=183.75326766522033)
    #pars['g3_sigma'].set(min=35.193792564236674, max=183.98817016897786)

    if bound_peak==True:
        #pars['g1_center'].set(min=0, max=1700)
        #pars['g2_center'].set(min=0, max=1700)
        #pars['g3_center'].set(min=0, max=1700)
        #pars['g4_center'].set(min=0, max=1700)
        #pars['g5_center'].set(min=0, max=1700)
        pars['g1_sigma'].set(max=1000)
        pars['g2_sigma'].set(max=1000)
        pars['g3_sigma'].set(max=1000)
        pars['g4_sigma'].set(max=1000)
        pars['g5_sigma'].set(max=1000)
        pars['g1_amplitude'].set(min=0)
        pars['g2_amplitude'].set(min=0)
        pars['g3_amplitude'].set(min=0)
        pars['g4_amplitude'].set(min=0)
        pars['g5_amplitude'].set(min=0)
        
    # fit model to data array ecl 
    result = gmod.fit(cl, pars, x=ell, method=method) ## ell=ell[0:1600], ecl=ecl[0:1600]
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

    return result

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

def outer_outlier_help(df_test, df_train, amp_cut=True, sigma_cut=True, sort_peak=False):
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
        df_test = df_test[df_test['sigma4'] < (df_train['sigma4'].quantile(q=0.75) + (df_train['sigma4'].quantile(q=0.75) - df_train['sigma4'].quantile(q=0.25)) * 3)]
        df_test = df_test[df_test['sigma4'] > (df_train['sigma4'].quantile(q=0.25) - (df_train['sigma4'].quantile(q=0.75) - df_train['sigma4'].quantile(q=0.25)) * 3)]
        df_test = df_test[df_test['sigma5'] < (df_train['sigma5'].quantile(q=0.75) + (df_train['sigma5'].quantile(q=0.75) - df_train['sigma5'].quantile(q=0.25)) * 3)]
        df_test = df_test[df_test['sigma5'] > (df_train['sigma5'].quantile(q=0.25) - (df_train['sigma5'].quantile(q=0.75) - df_train['sigma5'].quantile(q=0.25)) * 3)]
        print('len of dataframe after sigma cutoff :', len(df_test))
    
    if sort_peak==True:
        # sort alignment peaks
        df_test = df_test[df_test['peak1'] < df_test['peak2']]
        df_test = df_test[df_test['peak1'] < df_test['peak3']]
        df_test = df_test[df_test['peak1'] < df_test['peak4']]
        df_test = df_test[df_test['peak1'] < df_test['peak5']]
        df_test = df_test[df_test['peak2'] < df_test['peak3']]
        df_test = df_test[df_test['peak2'] < df_test['peak4']]
        df_test = df_test[df_test['peak2'] < df_test['peak5']]
        df_test = df_test[df_test['peak3'] < df_test['peak4']]
        df_test = df_test[df_test['peak3'] < df_test['peak5']]
        df_test = df_test[df_test['peak4'] < df_test['peak5']]
        print('len of dataframe after sorting alignment peaks :', len(df_test))

    return df_test

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

def cart_healpix(cartview, nside):
    '''read in an matrix and return a healpix pixelization map'''
    # Generate a flat Healpix map and angular to pixels
    healpix = np.zeros(hp.nside2npix(nside), dtype=np.double)
    hptheta = np.linspace(0, np.pi, num=cartview.shape[0])[:, None]
    hpphi = np.linspace(-np.pi, np.pi, num=cartview.shape[1])
    pix = hp.ang2pix(nside, hptheta, hpphi)
    
    # re-pixelize
    healpix[pix] = np.fliplr(np.flipud(cartview))
    
    return healpix

def healpix_rotate(healpix_map, rot):
    # pix-vec
    ipix = np.arange(len(healpix_map))
    nside = np.sqrt(len(healpix_map) / 12)
    if int(nside) != nside: return print('invalid nside');
    nside = int(nside)
    vec = hp.pix2vec(int(nside), ipix)
    rot_vec = (hp.rotator.Rotator(rot=rot)).I(vec)
    irotpix = hp.vec2pix(nside, rot_vec[0], rot_vec[1], rot_vec[2])
    return np.copy(healpix_map[irotpix])

def lonlat_block_shorten(rotate, blocksize=(18, 18), xsize=1080, nside=64):
    lon, lat, psi = rotate
    cartview = np.zeros((xsize/2, xsize))
    
    # center block
    blocksize = (blocksize[0]/360*xsize, blocksize[0]/360*xsize)
    cartview[xsize/4 - blocksize[1]/2: xsize/4 + blocksize[1]/2, 
             xsize/2 - blocksize[0]/2: xsize/2 + blocksize[0]/2] = np.ones((blocksize[0], blocksize[1]))
    
    # re-pixelization
    healpix = cart_healpix(cartview, nside)
    del cartview

    # cartview back to origin: theta 
    healpix = healpix_rotate(healpix, (0,-lat))
    
    return healpix_rotate(healpix, (-lon,0))

def shift_df_concat(df, return_abs=False):
    # Call in Best-Fit
    best = np.loadtxt('../Release2/COM_PowerSpect_CMB-base-plikHM-TT-lowTEB-minimum-theory_R2.02.txt')
    L = best.T[0]
    CL = best.T[1]
    C1 = ['peak1', 'peak2', 'peak3']
    C2 = ['g1_center', 'g2_center', 'g3_center']
    
    # Best fit with 5 gaussians
    best = gaussians_fit_helper(L[0:1604], CL[0:1604], return_fig=False)
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

def weighted_sky_map(df, a, b, size, theta_size, nside, patch_size=240):
    IMP = np.zeros(hp.nside2npix(nside), dtype=np.double)

    # suboptimal choice: let bad-fit become zero
    df_bad = bad_df(df)
    bad_index = list(df_bad.index)

    # subpotimal choice: rebulid list without bad-fit
    shift_array = np.copy(df['shift_var_distance'].values)
    if len(bad_index) != 0:
        shift_array[bad_index] = 0
    importance_list = list_rebuild(shift_array, b)

    for (i,theta),importance in zip(enumerate(a), importance_list):
        IMP += Shift_Block_Rotater(theta * theta_size, size, nside, 
                                            b[i], importance, patch_size=patch_size)
    return IMP


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
