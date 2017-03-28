import numpy as np
import healpy as hp
from matplotlib.image import pil_to_array
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from .mfun import *

def CartView_Block_Rotater(theta, phi_range, 
                           nside, chosen_list, patch_size=30):
    ## patch_size should be even
    # Basic information of your painting
    period = 360 // phi_range    
    canvas_size = period * patch_size
    
    # Generate a flat Healpix map
    Car_proj = np.zeros((canvas_size // 2, canvas_size))
    equator = canvas_size // 4
    division_waist = np.arange(period) * patch_size

    for i in chosen_list:
        block = np.ones((patch_size, patch_size))
        
        Car_proj[equator - patch_size // 2 : equator + patch_size // 2, 
                 division_waist[i] : division_waist[i] + patch_size] \
            = Car_proj[equator - patch_size // 2 : equator + patch_size // 2, 
                       division_waist[i] : division_waist[i] + patch_size] \
            + block
                
    # cartview to healpix
    healpix_map = cart_healpix(Car_proj, nside)
    return healpix_rotate(healpix_map, (0, -theta))
    
def CartView_Block_Layer(theta, phi_range,
                         nside, patch_size=30, rot_phi=False):
    Layers = []
    period = 360 // phi_range
    canvas_size = period * patch_size
     
    # Generate a flat Healpix map
    Car_proj = np.zeros((canvas_size // 2, canvas_size))
    equator  = canvas_size // 4

    # substitude one patch
    division_waist = np.arange(period) * patch_size

    for i in range(period):
        Car_clone = Car_proj.copy()
        block = np.ones((patch_size, patch_size))
        Car_clone[equator - patch_size // 2 : equator + patch_size // 2, 
                division_waist[i] : division_waist[i] + patch_size] \
        = Car_clone[equator - patch_size // 2 : equator + patch_size // 2, 
                division_waist[i] : division_waist[i] + patch_size] \
                + block
        # cartview to healpix conversion
        healpix_map = cart_healpix(Car_clone, nside)
        healpix_map = healpix_rotate(healpix_map, (0, -theta))
                                        
        Layers.append(healpix_map)
        del healpix_map, Car_clone
    return Layers

def Yakitori2tuple(a,b, theta_size, phi_size):
    r = [A*theta_size for A in a]
    C = [(R, B) for R,B in zip(r,b)]
    D = []
    for i in range(len(C)):
        E = [( -(C[i][1][j]*phi_size - 180), C[i][0]) for j in range(len(C[i][1]))]
        [D.append(e) for e in E]
    del C, E, r 
    return D            

def Shift_Block_Rotater(theta, phi_range, 
                        nside, chosen_list, importances, number_list=None, patch_size=30):
    ## patch_size should be even
    # Basic information of your painting
    period = 360 // phi_range    
    canvas_size = period * patch_size
    
    # Generate a flat Healpix map
    Car_proj = np.zeros((canvas_size // 2, canvas_size))
    equator = canvas_size // 4
    division_waist = np.arange(period) * patch_size
    
    for i, weight in zip(chosen_list, importances):
        try:
            circle = ones_number_circle(patch_size, str(number_list[i])) * weight
        except:
            circle = ones_circle(patch_size) * weight
        
        Car_proj[equator - patch_size // 2 : equator + patch_size // 2, 
                division_waist[i] : division_waist[i] + patch_size] \
        = Car_proj[equator - patch_size // 2 : equator + patch_size // 2, 
                division_waist[i] : division_waist[i] + patch_size] \
                + circle
                
    # cartview to healpix
    healpix_map = cart_healpix(Car_proj, nside)
    return healpix_rotate(healpix_map, (0, -theta))
