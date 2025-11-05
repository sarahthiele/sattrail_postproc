import numpy as np
from astropy.table import Table
import astropy.io.fits as fits


ZPOINTREFPATH = '/home/sthiele/proj/repos/sattrail_postproc/photometry/zo_ref.fits'
z0_fits = fits.getdata(ZPOINTREFPATH)
z0_table = Table(z0_fits)
