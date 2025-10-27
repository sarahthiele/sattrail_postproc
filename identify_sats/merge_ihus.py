from generate_astrometry import *

# example:
# datapath = '../../PROJDATA/1-20241115_1-430769_all_ihus/SUB/'
# subroot_no_ihu = '1-430769'

def merge_ihus(datapath, subroot_no_ihu, ihu_center, PLOT=True):
    fsub = datapath+subroot+'_{:02d}-sub.fits'.format(ihu_center)
    datcenter = pd.read_hdf('dat_{}.hdf'.format(ihu_center))
    ihu_table = pd.read_hdf('IHU_TABLE.hdf')
    
    adj = []
    if np.sum(datcenter.IOT) >= 1:
        adj.extend(ihu for ihu in ihu_table.loc[ihu_table.ihu==ihu_center].top.values)   
    if np.sum(datcenter.IOB) >= 1:
        adj.extend(ihu for ihu in ihu_table.loc[ihu_table.ihu==ihu_center].bottom.values)
    if np.sum(datcenter.IOL) >= 1:
        adj.extend(ihu for ihu in ihu_table.loc[ihu_table.ihu==ihu_center].left.values)
    if np.sum(datcenter.IOR) >= 1:
        adj.extend(ihu for ihu in ihu_table.loc[ihu_table.ihu==ihu_center].right.values)
    
    adj = [item for sublist in adj for item in sublist]
    
    DAT = datcenter
    ra, dec = get_coords(fsub, datcenter[['c1','c2']].values, datcenter[['r1','r2']].values)
    datcenter['ra1'] = ra[:,0]
    datcenter['ra2'] = ra[:,1]
    datcenter['dec1'] = dec[:,0]
    datcenter['dec2'] = dec[:,1]
    datcenter['slope'] = (datcenter.r2-datcenter.r1)/(datcenter.c2-datcenter.c1)
    datcenter['ihu'] = np.ones(len(datcenter)).astype(int)*ihu_center
    
    if PLOT:
        plt.figure()
        ra_all, dec_all = get_coords(fsub, np.concatenate(datcenter['cpix'].values), np.concatenate(datcenter['rpix'].values))
        plt.scatter(ra_all,dec_all,s=1,label='44')
    
    for ihu in adj:
        fsub = datapath+subroot+'_{:02d}-sub.fits'.format(ihu)
        try:
            dat = pd.read_hdf('dat_{}.hdf'.format(ihu))
            dat['ihu'] = np.ones(len(dat)).astype(int)*ihu
        except:
            print('dat for ihu {} does not exist'.format(ihu))
            continue
    
        sub = read_fits_file(fsub)
        dat['linenum'] = 1
        dat = find_edgetrails(sub, dat, EDGE_THRESHOLD=5)
        ra, dec = get_coords(fsub, dat[['c1','c2']].values, dat[['r1','r2']].values)
        
        dat['ra1'] = ra[:,0]
        dat['ra2'] = ra[:,1]
        dat['dec1'] = dec[:,0]
        dat['dec2'] = dec[:,1]
        dat['slope'] = (dat.r2-dat.r1)/(dat.c2-dat.c1)
    
        DAT = pd.concat([DAT,dat])
    
        if PLOT:
            ra_all, dec_all = get_coords(fsub, np.concatenate(dat['cpix'].values), np.concatenate(dat['rpix'].values))
            plt.scatter(ra_all, dec_all,s=1,label=ihu)
    
    if PLOT:
        plt.legend()
    
    ### This next part doesn't work yet but something like this maybe??
    
    IOALL = DAT.loc[DAT.IOframe==1.]
    ihu = ihu_center
    for i in range(len(IOALL)):
        D = IOALL.loc[IOALL.ihu==ihu].iloc[i]
        #print(D)
        plt.figure()
        plt.plot(D[['ra1','ra2']],D[['dec1','dec2']],label='{}: original'.format(i))
        IO = IOALL.drop(IOALL.loc[IOALL.ihu==ihu].index.values)
        if D.IOL==1.:
            if ihu_table.loc[ihu_table.ihu==ihu].left.values[0]!=[0]:
                LEFT = DAT.loc[np.isin(DAT.ihu, ihu_table.loc[ihu_table.ihu==ihu].left.values[0])]
                dL = np.sqrt((D.ra1-LEFT.ra2.values)**2+(D.dec2-LEFT.dec2.values)**2)
                idL = np.argmin(dL)
                DL = LEFT.iloc[idL]
                plt.plot(DL[['ra1','ra2']],DL[['dec1','dec2']],label='{}: left, ihu {}'.format(i,DL.ihu))
            else:
                print('ihu {} has no ihu to the left'.format(ihu))
        if D.IOR==1.:
            if ihu_table.loc[ihu_table.ihu==ihu].right.values[0]!=[0]:
                RIGHT = DAT.loc[np.isin(DAT.ihu, ihu_table.loc[ihu_table.ihu==ihu].right.values[0])]
                dR = np.sqrt((D.ra2-RIGHT.ra1.values)**2+(D.dec2-RIGHT.dec1.values)**2)
                idR = np.argmin(dR)
                DR = RIGHT.iloc[idR]
                plt.plot(DR[['ra1','ra2']],DR[['dec1','dec2']],label='{}: right, ihu {}'.format(i,DR.ihu))
            else:
                print('ihu {} has no ihu to the left'.format(ihu))
        if D.IOT==1.:
            if ihu_table.loc[ihu_table.ihu==ihu].top.values[0]!=[0]:
                TOP = DAT.loc[np.isin(DAT.ihu, ihu_table.loc[ihu_table.ihu==ihu].top.values[0])]
                if D.slope > 0:
                    dT = np.sqrt((D.ra2-TOP.ra1.values)**2+(D.dec2-TOP.dec1.values)**2)
                else:
                    dT = np.sqrt((D.ra1-TOP.ra2.values)**2+(D.dec1-TOP.dec2.values)**2)
                idT = np.argmin(dT)
                DT = TOP.iloc[idT]
                plt.plot(DT[['ra1','ra2']],DT[['dec1','dec2']],label='{}: top, ihu {}'.format(i,DT.ihu))
            else:
                print('ihu {} has no ihu to the left'.format(ihu))
        if D.IOB==1.:
            if ihu_table.loc[ihu_table.ihu==ihu].bottom.values[0]!=[0]:
                BOTTOM = DAT.loc[np.isin(DAT.ihu, ihu_table.loc[ihu_table.ihu==ihu].bottom.values[0])]
                if D.slope > 0:
                    dB = np.sqrt((D.ra1-BOTTOM.ra2.values)**2+(D.dec1-BOTTOM.dec2.values)**2)
                else:
                    dB = np.sqrt((D.ra2-BOTTOM.ra1.values)**2+(D.dec2-BOTTOM.dec1.values)**2)        
                idB = np.argmin(dB)
                DB = BOTTOM.iloc[idB]
                plt.plot(DB[['ra1','ra2']],DB[['dec1','dec2']],label='{}: bottom, ihu {}'.format(i,DB.ihu))
            else:
                print('ihu {} has no ihu to the left'.format(ihu))
    
        plt.legend()
        plt.show()

    return
