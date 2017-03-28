import numpy as np
import healpy as hp
import pandas as pd
from matplotlib import pyplot as plt
from .mfun import *
from itertools import compress

def scanning_program(filename, size, division, threshold, df, abs_sum=False, healpix_positions=False, nside=4):
    '''
    Scanning the whole sky positions via healpy's cartview function based on given sky map (filename), 
    return tuples of positions.
    
    Parameters:
    ----------
    filename : str or HDU or HDUList
      the fits file name
    size : int or float
      the side of the square patch, should be a factor of 360 degree.
    division : int or float
      the division degree you want to sample on the whole sky map,
      If healpix_positions=True it will not be used.
    threshold : int or float
      the lower limit threshold you want to record the peak shift positions,
      if threshod = 30, then return pos pos_small that abs(shift) larger than 30.
    df : pandas Dataframe,
      the dataframe which is used on eliminate bad-fitting result,
      this dataframe should be contructed via `peak_df` and with the same size you want to analysis.
    abs_sum : bool, optional
      If True return shift_sum = np.sum([abs(s) for s in shift]). Default: False
    healpix_positions : bool, optional
      If True use healpix pixel positions to plot `cartview`. Default: False
    nside : int, optional
      The nside will be used to generate healpix pixel positions. Default: 4
    
    Returns:
    -------
    pos, pos_small : a tuple of list contains positions in (phi, theta) corresponds to shift to large scale
      and shift to small scale.
    '''
    
    # apply healpix pixels as scanning positions
    if healpix_positions==True:
        print('healpix positions')
        ipix = np.arange(12 * nside ** 2., dtype=np.int64)
        theta, phi = hp.pix2ang(nside, ipix)
        theta, phi = [theta * 180 / np.pi - 90, phi * 180 / np.pi]
        tuples = [(p, t, 0) for p,t in zip(phi, theta)]
        bool_list = [abs(t[1]) > 30 for t in tuples]
        tuples = list(compress(tuples, bool_list)) # this itertools is cute
        
    # generate a list of tuples (lon, lat) with cos(θ) decreasing rate on high latitude
    if healpix_positions==False:
        print('ordinary positions')
        tuples = [(phi, theta, 0) for theta in np.append(np.linspace(-90,-30,60//division), 
                                                         np.linspace(30,90,60//division))
                                  for phi in np.linspace(0,360, 360//division * np.cos(theta * np.pi / 180))] 
              
    # read in healpix map 
    nilc = hp.read_map(filename)

    # position list
    pos = []
    pos_small = []
    
    for i,T in enumerate(tuples):
        # image rotation and taking
        img = hp.cartview(nilc, rot=T, xsize=1335, 
                          lonra=(-size/2,size/2), latra=(-size/2,size/2), return_projected_map=True) 
        plt.clf()
        plt.close()

        # patch2cl + deconvolution
        ell, cl = mfun.patch2cl(img * 10**6., phi_size=size) # recaling to μK
        cl = mfun.decon_beam(filename, cl ,ell, noise_range=(2200, 2450))
        dl = cl * ell * (ell + 1) / 2 / np.pi

        # gaussian fit with helper
        dl = dl[0:index_of(ell, 1604)]
        ell = ell[0:index_of(ell, 1604)]
        result = gaussians_fit_helper(ell, dl, return_fig=False)

        # peak shift evaluation
        shift_list = [(result.best_values['g1_center'] - best.best_values['g1_center']),
                     (result.best_values['g2_center'] - best.best_values['g2_center']),
                     (result.best_values['g3_center'] - best.best_values['g3_center'])]
        if abs_sum==False: shift = np.sum(shift_list)
        elif abs_sum==True: shift = np.sum([abs(s) for s in shift_list])
            
        if (shift > threshold) == True and outer_outlier_detection(result, df) == False:
            result = gaussians_fit_helper(ell, dl, return_fig=False, method='tnc')
            del shift

            shift = (result.best_values['g1_center'] - best.best_values['g1_center']) +\
                    (result.best_values['g2_center'] - best.best_values['g2_center']) +\
                    (result.best_values['g3_center'] - best.best_values['g3_center'])                
            print(T)

        if (shift < -threshold) == True and outer_outlier_detection(result, df) == False:
            result = gaussians_fit_helper(ell, dl, return_fig=False, method='tnc')
            del shift

            shift = (result.best_values['g1_center'] - best.best_values['g1_center']) +\
                    (result.best_values['g2_center'] - best.best_values['g2_center']) +\
                    (result.best_values['g3_center'] - best.best_values['g3_center'])        
            print(T)

        if (shift > threshold) and outer_outlier_detection(result, df) and all([a > 0 for a in shift_list]):
            pos_small.append(T)
            print('append shift small:', T)
            del img, result, ell, dl, cl

        if (shift < -threshold) and outer_outlier_detection(result, df) and all([a < 0 for a in shift_list]):
            pos.append(T)
            print('append shift large:', T)
            del img, result, ell, dl, cl

        elif (-threshold <= shift <= threshold) or (not all([a > 0 for a in shift_list]) and not all([a < 0 for a in shift_list])):
            del img, result, ell, dl, cl
    return pos, pos_small
    
def plot_scanning(pos, xsize, nside):
    Addition_map = np.zeros(hp.nside2npix(nside), dtype=np.double)

    for i, T in enumerate(pos):
        block = lonlat_block_shorten(rotate=T, xsize=xsize, nside=nside)
        Addition_map += block
        del block
        sys.stdout.write(".")
        sys.stdout.flush()
    return Addition_map
