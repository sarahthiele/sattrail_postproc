import numpy as np
import pandas as pd
from pathlib import Path
from astropy.nddata import CCDData
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord as pts
from astropy.utils.data import get_pkg_data_filename


def degrees_to_hms(degrees):
    """Convert degrees to hours, minutes, seconds."""
    hours = degrees / 15  # Convert degrees to hours (1 hour = 15 degrees)
    h = np.floor(hours).astype(int)
    remaining = (hours - h) * 60
    m = np.floor(remaining).astype(int)
    s = (remaining - m) * 60
    return h, m, s

def degrees_to_dms(degrees):
    """Convert degrees to degrees, minutes, seconds."""
    sign = np.sign(degrees).astype(int)
    deg_abs = np.abs(degrees)
    d = np.floor(deg_abs).astype(int)
    remaining = np.abs(deg_abs - d) * 60
    m = np.floor(remaining).astype(int)
    s = (remaining - m) * 60
    return sign*d, m, s

def get_coords(subfile, Cpix, Rpix):
    hdr = fits.getheader(subfile)
    shifted = hdr.get('WCSSHIFT', False)
    if shifted == False:
        print('incorrect WCS system')
        return 0, 0
    else:
        w = WCS(hdr)
        RA, DEC = w.all_pix2world(Cpix, Rpix, 1)
        return RA, DEC
        
def astfile_info(RA, DEC, subfile, reverse=False, sort=True, inframe=True, log=True):
    if sort:
        RA_argsort = np.argsort(RA)
        RA_sorted = RA[RA_argsort]
        DEC_sorted = DEC[RA_argsort]
    else:
        RA_sorted = RA
        DEC_sorted = DEC
    
    ra_h, ra_m, ra_s = degrees_to_hms(RA_sorted)
    dec_d, dec_m, dec_s = degrees_to_dms(DEC_sorted)

    if log:
        for i in range(len(RA_sorted)):
            print(f"RA: {ra_h[i]}h {ra_m[i]}m {ra_s[i]:.2f}s")
            print(f"DEC: {dec_d[i]}d {dec_m[i]}m {dec_s[i]:.2f}s")
    
    hdr = fits.getheader(subfile)
    date = hdr['DATE-OBS'].split('-')
    time = hdr['TIME-OBS'].split(':')
    exptime = float(hdr['EXPTIME'])
    if log:
        print(date, time, exptime)
    
    year = int(date[0])
    month = int(date[1])
    day = float(date[2])
    hour = float(time[0])
    minute = float(time[1])
    second = float(time[2])

    if inframe:
        if reverse:
            seconds = second + np.flip(np.linspace(0,exptime,len(RA_sorted)))
        else:
            seconds = second + np.linspace(0,exptime,len(RA_sorted))
    else:
        if reverse:
            seconds = (second + exptime/2) * np.ones(len(RA_sorted))
        else:
            seconds = second * np.ones(len(RA_sorted))
        
    day_frac = hour/24 + minute/(24*60) + seconds/(24*3600)
    day_tot = day + day_frac

    return RA_sorted, DEC_sorted, ra_h, ra_m, ra_s, dec_d, dec_m, dec_s, exptime, year, month, day_tot

def write_astrometry_file(fname, lineids, ra_h, ra_m, ra_s, dec_d, dec_m, dec_s, year, month, day_tot):
    with open('{}.ast'.format(fname), 'a') as f:
        for i in range(len(ra_h)):
            if dec_d[i] > 0:
                sign = '+'
            else:
                sign = '-'
            f.write('     L{:06d}  C{} {:02d} {:.5f} {:02d} {:02d} {:05.2f} {}{:02} {:02} {:04.1f}                      304\n'.format(lineids[i],
                                                                                                                                      year, 
                                                                                                                             month, 
                                                                                                                             day_tot[i], 
                                                                                                                             ra_h[i],
                                                                                                                             ra_m[i], 
                                                                                                                             ra_s[i], 
                                                                                                                             sign,
                                                                                                                             np.abs(dec_d[i]), 
                                                                                                                             dec_m[i], 
                                                                                                                             dec_s[i]))
    f.close()

    return