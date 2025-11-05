import numpy as np
import astropy.io.fits as fits
import astropy.table as Table
import os

def fit_ref_z0(x, y, refcol, z0):
    nsource = len(x)

    DM = np.column_stack([np.ones(nsource), x, y, x*y,
                          x**2, y**2, refcol])

    coeffs, residuals, rank, s = np.linalg.lstsq(DM, z0, rcond=None)

    return coeffs

def calculate_ref_z0_coeffs(refdirpath,ihu,field):
    try:
        refdir = sorted(os.listdir(refdirpath))[-1]
    except:
        print(refdirpath + ' is empty for ihu {:02d}'.format(ihu))
        return 1, 1, 1, 1
    rawphotfile = "rawphot_ihu-{:02d}".format(ihu)+"_"+field+".fits"
    rawphotpath = os.path.join(refdirpath,refdir,rawphotfile)
    try:
        rawphot = fits.getdata(os.path.join(rawphotpath))
    except:
        print(rawphotpath + ' does not exist')
        return 1, 1, 1, 1
    X = rawphot['X']
    Y = rawphot['Y']
    refcol = rawphot['RefCol']
    refmag = rawphot['RefMag']

    refflux_0 = rawphot['Flux_0']
    flag_0  = (rawphot['Flag_0']==0)&(refflux_0>0)
    z0_0 = refmag[flag_0] + 2.5 * np.log10(refflux_0[flag_0])
    coeffs_0 = fit_ref_z0(X[flag_0], Y[flag_0], refcol[flag_0], z0_0)

    refflux_1 = rawphot['Flux_1']
    flag_1 = (rawphot['Flag_1']==0)&(refflux_1>0)
    z0_1 = refmag[flag_1] + 2.5 * np.log10(refflux_1[flag_1])
    coeffs_1 = fit_ref_z0(X[flag_1], Y[flag_1], refcol[flag_1], z0_1)

    refflux_2 = rawphot['Flux_2']
    flag_2 = (rawphot['Flag_2']==0)&(refflux_2>0)
    z0_2 = refmag[flag_2] + 2.5 * np.log10(refflux_2[flag_2])
    coeffs_2 = fit_ref_z0(X[flag_2], Y[flag_2], refcol[flag_2], z0_2)

    return 0, coeffs_0, coeffs_1, coeffs_2

def load_z0_coeffs(ihu, field, ap_id, z0_table):
    zflag = (z0_table['ihu']==ihu)&(z0_table['field']==field)
    if len(z0_table[zflag])==0:
        print('ihu/field combinationo not in z0_table')
        return

    c0s = 'c0_{}'.format(ap_id)
    c1s = 'c1_{}'.format(ap_id)
    c2s = 'c2_{}'.format(ap_id)
    c3s = 'c3_{}'.format(ap_id)
    c4s = 'c4_{}'.format(ap_id)
    c5s = 'c5_{}'.format(ap_id)
    c6s = 'c6_{}'.format(ap_id)

    coeffs = z0_table[zflag][c0s,c1s,c2s,c3s,c4s,c5s,c6s].as_array()
    coeffs.dtype = float

    return coeffs

def get_z0_sat(x, y, BPRP, ihu, field, ap_id, z0_table):

    coeffs = load_z0_coeffs(ihu, field, ap_id, z0_table)


    z0_sat = (coeffs[0] +
              coeffs[1] * x +
              coeffs[2] * y +
              coeffs[3] * z * y +
              coeffs[4] * x**2 +
              coeffs[5] * y**2 +
              coeffs[6] * BPRP)

    return z0_sat
