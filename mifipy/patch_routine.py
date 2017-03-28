import numpy as np
import healpy as hp
from matplotlib import pyplot as plt
from .mfun import *

def cosine_window(N):
    "makes a cosine window for apodizing to avoid edges effects in the 2d FFT" 
    # make a 2d coordinate system
    ones = np.ones(N)
    inds  = (np.arange(N)+.5 - N/2.)/N *np.pi ## eg runs from -pi/2 to pi/2
    X = np.outer(ones,inds)
    Y = np.transpose(X)
                              
    # make a window map
    window_map = np.cos(X) * np.cos(Y)
                                         
     # return the window map
    return(window_map)

def Cartview_pt2cl(corner_x, img, size, rotate, phisize,
                   return_ell=False, return_apodized=False):
    ## the corner is on the right bottom of the patches
    imgdata = hp.cartview(img, fig=None,
                          xsize=size, rot=rotate,         # rotation option
                          lonra=(corner_x - phisize, corner_x),        # longitude range
                          latra=(-phisize/2., phisize/2.), # latitude range
                          return_projected_map=True)
    if return_apodized==True:
        ratio = np.sqrt(np.mean(np.ones(imgdata.shape) / cosine_window(size)))
        # I dont know actually
        ell, cl = patch2cl(imgdata * cosine_window(size), phisize)
        cl = cl * ratio
    if return_apodized==False:
        ell, cl = patch2cl(imgdata, phisize)
    plt.close(); plt.clf();
    
    if return_ell:
        return ell, cl
    else:
        return cl
        
