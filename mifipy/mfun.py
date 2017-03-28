import numpy as np, pandas as pd
import healpy as hp
import six
import os
from healpy.sphtfunc import Alm
from healpy import cookbook as cb
import astropy.io.fits as pf
from numpy import asarray as ar

def index_of(arrval, value):
    if value < min(arrval): return 0
    return max(np.where(arrval<=value)[0])
    
def nearest_idx(array, value):
    return (np.abs(array-value)).argmin()

def decon_beam(Commap, cl, ell ,noise_range=None, add_back=True):
    '''
        deconvolution from beam transfer function and
        cancel out the noise based on the input l range
        assume-> Signal = W*Sky + Nise; W = BeamTF**2 * (pixel window function)**2
        Parameters:
        -----
        Commap: given compoenet map's fits filename
        noise: (lmin, lmax) tuple, average this range of noise
        ell: given ell you want to deconvolve
        
        reutrn cl
        '''
    # get bin width from ells
    bin = np.int(ell[1] - ell[0])
    
    # subtract the noise
    if noise_range:
        noise = cl[index_of(ell, noise_range[0]):index_of(ell, noise_range[1])].mean()
        cl = cl - noise
        
        if add_back==True:
            # artifact add back the mean power of transfered best-fit cl on noise range
            cl = artifact_addback(Commap, cl, ell, noise_range)
    
    # get beamTF from commap fits
    beam = np.float64(hp.mrdfits(Commap,hdu=2))[0]
    
    # pixel window function
    NSIDE = 2048;
    pl = hp.pixwin(NSIDE); pl = pl[0:4001]
    
    ## resample based on given ell
    lmax = ell[-1]; lmax = np.int(lmax)
    pl = pl[0:lmax+bin:bin]
    beam = beam[0:lmax+bin:bin]
    
    ## becareful about the index here,
    ## by python's convention it will ignore the last element.
    return cl/beam**2./pl**2.

def artifact_addback(Commap, cl, ell, noise_range):
    '''artificially put back the mean power of cls on the tail of noise subtraction, 
    the putting back mean power is the mean of the same range from transfer-function convolved best-fit ΛCDM model.
    Parameters:
    Commap: Component map's path
    cl: cl (not dl) you want to add back, assume to be numpy array
    ell: angular size
    noise_range: a tuple (lmin, lmax)
    
    return cl
    '''
    # beam transfer function 
    beam = np.float64(hp.mrdfits(Commap,hdu=2))[0]

    # read best-fit 
    best = np.loadtxt('/Users/cicero/Documents/Physics/PLANCK/Release2/COM_PowerSpect_CMB-base-plikHM-TT-lowTEB-minimum-theory_R2.02.txt')
    L = best.T[0]
    CL = (best.T[1] / L / (L + 1) * 2 * np.pi) * beam[2:len(L)+2]**2.
    
    # rescalling
    L = L[0:int(max(ell) + ell[1]-ell[0]):int(ell[1]-ell[0])]
    CL = CL[0:int(max(ell) + ell[1]-ell[0]):int(ell[1]-ell[0])]

    # mean power on noise range
    artifact_power = np.mean(CL[index_of(L, noise_range[0]):index_of(L, noise_range[1])])
    
    try: cl = cl + artifact_power
    except: cl = [c + artifact_power for c in cl]
    return cl 
    
def read_glesp(filename, return_cos=False):
    import pyfits
    '''
        filename: str, glesp format map
        return_cos: if True, return cos(theta(rad)), too. default is False.
        return
        gcanvas: gleasp map in matrix format (x = phi,y = -theta)
    '''
    glesp = pyfits.open(filename)
    data = glesp[1].data
    gtheta = data['COS(THETA)'][0]
    gcanvas = data['TEMPERATURE'][0]
    dim = len(gtheta)                       # get the (half) length
                                            # of pixel for one theta
    gcanvas = gcanvas.reshape((dim, dim*2))
    if return_cos:
        return gcanvas, gtheta
    else:
        return gcanvas

def read_alm(filename, return_mmax=False):
    table = pd.read_csv(filename,
                        delim_whitespace=True, header = None).values
    table = table.T;
    l = table[0].astype(np.long);
    m = table[1].astype(np.long);
    almr = table[2]; almi = table[3];
    if (m<0).any():
        raise ValueError('Negative m value encountered !')
    lmax = l.max()
    mmax = m.max()
    alm = almr*(0+0j)
    i = hp.Alm.getidx(lmax,l,m)
    alm.real[i] = almr
    alm.imag[i] = almi
    if return_mmax:
        return alm, mmax
    else:
        return alm


def radialprofile(imgdata, center=None):
    y, x = np.indices((imgdata.shape))
    if not center:
        center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = np.around(r)
    r = np.int64(r)
    tbin = np.bincount(r.ravel(), imgdata.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    radialprofile = np.nan_to_num(radialprofile)
    return radialprofile

def patch2cl(image, phi_size):
    ## Fast Fourier Transform
    ak = np.fft.ifft2(image)                    # FFT scaling convention
    akshift = np.fft.fftshift(ak)               # shift low k to the center
    akabs = abs(akshift)**2.                    # take absolute square
    
    ## average to get Ck
    ck = radialprofile(akabs)                   # make use of radial profile
    k = np.arange(len(ck))
    
    ## scaling relation
    L = 2.*np.pi/(360./phi_size);
    period = np.int64(360./phi_size);
    ell = 2*np.pi/L*k
    cl = L**2.*ck
    
    ## set limit
    ell = ell[0:(np.int(2500/period))+1]
    cl = cl[0:(np.int(2500/period))+1]
    return ell, cl
    
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

def ones_circle(length):
    circle = np.zeros((length, length))
    y, x = np.indices((circle.shape))
    center = np.array([(x.max() - x.min()) / 2.0, (x.max() - x.min()) / 2.0])
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    circle[r < length / 2] = 1
    return circle
    
def ones_number_circle(length, number):

    from PIL import Image, ImageDraw, ImageFont
    from matplotlib.image import pil_to_array
    
    # decide your font
    fnt = ImageFont.truetype('/Library/Fonts/Georgia.ttf', 150) # 135
    circle = ones_circle(length)
    txt = Image.new('L', size=(length, length), color=1)
    draw = ImageDraw.Draw(txt)
    
    w, h = draw.textsize(number, font=fnt)
    
    draw.text(((length - w) / 2, (length - h) / 4), text=number, font=fnt, align='center')
    txt = pil_to_array(txt)
    return circle * txt
    
def mask_cart(corner_x, cmbmap, phisize):
    # Healpix patch catch process
    img   = cmbmap.copy()
    NSIDE = hp.get_nside(img)
    pixtheta, pixphi = hp.pix2ang(NSIDE,
                                  np.arange(hp.nside2npix(NSIDE)))
    # rescaling rad to degree
    pixtheta *= 180 / np.pi
    pixphi   *= 180 / np.pi
    # 0 1 mask making
    img[ar(pixtheta > 90 - phisize / 2) & ar(pixtheta < 90 + phisize / 2)  & 
        ar(pixphi > corner_x)           & ar(pixphi < corner_x + phisize)] = 1
    
    return img
