import os
import numpy as np
import pandas as pd
import astropy.io.fits as fits
from astropy.table import Table
from zeropoints import fit_ref_z0, calculate_ref_z0_coeffs


rawphotpath = "/nfs/php2/ar0/P/HP1/REDUCTION/REF/OBJECT_IHUID"

pdtable = pd.DataFrame()
columns = ['ihu','field','c0_0','c1_0','c2_0','c3_0','c4_0','c5_0','c6_0',
           'c0_1','c1_1','c2_1','c3_1','c4_1','c5_1','c6_1',
           'c0_2','c1_2','c2_2','c3_2','c4_2','c5_2','c6_2']

table = np.array([columns])
for ihu in range(1,65):
    print('ihu ', ihu)
    ihustr = "{:02d}".format(ihu)
    allrefpath = os.path.join(rawphotpath,'ihu'+ihustr)
    fields = os.listdir(allrefpath)

    for field in fields:
        ihustr = "{:02d}".format(ihu)
        refdirpath = os.path.join(rawphotpath,'ihu'+ihustr,field)
        flag, coeffs_0, coeffs_1, coeffs_2 = calculate_ref_z0_coeffs(refdirpath,ihu,field)
        if flag==1:
            continue
        row = np.concatenate([np.array([ihu]),np.array([field]),coeffs_0,coeffs_1,coeffs_2])
        table = np.append(table,[row],axis=0)

pdtable = pd.DataFrame(table[1:,:],columns=table[0,:])
for col in columns:
    if col=='field':
        continue
    pdtable[col] = pdtable[col].astype(float)

pytable = Table.from_pandas(pdtable)

fitsname = '/nfs/php2/ar0/P/PROJ/sthiele/repos/sattrail_postproc/photometry/z0_ref.fits'
pytable.write(fitsname, format='fits', overwrite=True)
