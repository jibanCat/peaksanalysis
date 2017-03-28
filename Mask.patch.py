import healpy as hp
import numpy as np
from numpy import asarray as ar
from mifipy import Yakitori, mfun

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

def main():
    a = [0, 1, 2, 3, 4, 5, 6, 7]
    b = [[15], [0, 9, 17], [0, 1, 2, 6, 7, 8, 9, 10, 11, 15, 16, 17], [0, 8, 9, 17], [0, 1, 2, 6, 7, 8, 9, 10, 11, 15, 16, 17], [0, 8, 9, 17], [0, 1, 2, 6, 7, 8, 9, 10, 11, 15, 16, 17], [0, 8, 9, 17]]
    size       = 20
    theta_size = 22.5
    nside      = 2048
    mask = np.zeros(hp.nside2npix(nside), dtype=np.double)

    for i, (phi, theta) in enumerate(Yakitori.Yakitori2tuple(a, b, theta_size, size)):
        print ('phi is {}, theta is {}'.format(phi, theta))
        # convert convention from 180 <- -180 to 180 <- 0 = 360 <- 180
        if phi <= 0:
            phi = phi + 360
        img  = mask_cart(phi - size, mask, size, )
        img  = mfun.healpix_rotate(img, rot=(0, -theta))
        hp.write_map('mask' + str(i).zfill(3) + '.' + 
                     str(phi - size) + 'p' + 
                     str(theta)      + 't' + '.fits', 
                     img)
        del img

if __name__=="__main__":
    main()