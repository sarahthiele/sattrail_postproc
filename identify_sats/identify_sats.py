from skeleton_cpu_postproc import *

def find_TLE_file(subfile):
    hdr = fits.getheader(subfile)
    date = hdr['DATE-OBS'].split('-')
    year = int(date[0])
    month = int(date[1])
    day = int(date[2])

    if (year<2025)|(year==2025)&(month<=3)|(year==2025)&(month==4)&(day<=2):
        TLELOC = 'satchecker_TLE_files'
    else:
        TLELOC = 'australian_TLE_files'

    TLEFILE = '/data/CAT/TLE/ALL_TLE/{}/{}-{:2d}-{:2d}/ALL_TLE.txt'.format(TLELOC,year,month,day)

    return TLEFILE
    

def find_satellites(subfile, FINALLIST, cpoints, rpoints, ntimes=10):
    npoints = len(cpoints[0])
    print('number of points: ', npoints)

    tlefile = find_TLE_file(subfile)
    
    names = []
    offsets = []
    pixelid = []
    times = []
    periods = []
    incs = []
    eccs = []
    
    for lnum in range(len(cpoints)):
        RA, DEC = get_coords(subfile, np.array(cpoints[lnum]), np.array(cpoints[lnum]))
        RA_s, DEC_s, ra_h, ra_m, ra_s, dec_d, dec_m, dec_s, exptime, year, month, day_tot = astfile_info(RA, DEC, file, reverse=False, 
                                                                                            sort=False, inframe=False, log=False)
        repeat = np.repeat([RA, DEC, ra_h, ra_m, ra_s, dec_d, dec_m, dec_s, day_tot],ntimes,axis=1)
        repeat[-1] += np.concatenate(np.repeat(np.linspace(0,exptime,ntimes)[np.newaxis, :],npoints,axis=0))/(24*3600)
        RA, DEC, ra_h, ra_m, ra_s, dec_d, dec_m, dec_s, day_tot = repeat
        lineids = np.arange(0,len(ra_h),1)
        trueids = np.repeat(np.arange(0,npoints,1),ntimes,axis=0)
        write_astrometry_file(subfileroot, lineids, ra_h.astype(int), ra_m.astype(int), ra_s, dec_d.astype(int), dec_m.astype(int), dec_s, 
                              year, month, day_tot)
        #!sat_id test.ast -t '/data/CAT/TLE/satchecker_TLE_files/2024-11-15/ALL_TLE.txt' -r 0.5 -m 20 -v > MEGATESTER.txt
    
        
        result = subprocess.run([
            'sat_id', 'test.ast',
            '-t', tlefile,
            '-r', '0.5',
            '-m', '20',
            '-v'
        ], capture_output=True, text=True, check=True)
    
        results = result.stdout.split('\n')
        check = False
        for i in range(len(results)):
            L=results[i]
            if L[:6] == '     L':
                times.append(float(L[23:31]))
                pixelid.append(int(L[6:12]))
                check = True
                continue
            if check == True:
                name = L[52:].split(':')[1].lstrip().strip('\n')
                names.append(name)
                periods.append(float(L.split('P=')[1].split('min')[0]))
                incs.append(float(L.split('i=')[1].split(':')[0]))
                eccs.append(float(L.split('e=')[1].split(';')[0]))
                check = False
            if 'offset' in L:
                offsets.append(float(L[-11:-5]))
    
    periods = np.array(periods)
    avals = (6.67*1e-11*5.972*1e24/(4*np.pi**2)*(periods*60)**2)**(1/3)
    alts = (avals - 6371*1000)/1000