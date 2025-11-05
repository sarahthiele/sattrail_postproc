import numpy as np
import astropy.io.fits as fits
from astropy.table import Table
from zeropoints import load_z0_coeffs, get_z0_sat


def instrumental_mag_from_ADU(x, y, BPRP, subflux, ihu, field, ap_id, z0_table):

    z0 = get_z0_sat(x, y, BPRP, ihu, field, ap_id, z0_table)

    mag = -2.5 * np.log10(subflux) + z0

    return mag


def gaia_mag_from_instrumental(x, y, BPRP, intrumental




