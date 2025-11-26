import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import json
from astropy.visualization import ZScaleInterval
from skimage.transform import probabilistic_hough_line as phl
import pandas as pd
import pyarrow
from skimage.draw import line
#from scipy.ndimage import binary_dilation

# from when I implemented the ndimage label thing
from skimage.morphology import binary_dilation, binary_closing
from scipy import ndimage
from skimage.morphology import skeletonize, dilation, disk
#

from skimage.segmentation import watershed
import skimage.morphology as morphology
from skimage.morphology import binary_erosion
from scipy.optimize import curve_fit, minimize
from scipy.ndimage import map_coordinates
from scipy.ndimage import median_filter
from skimage.morphology import skeletonize, thin, medial_axis
import numba as nb
import cv2
import os

zscale = ZScaleInterval()

#import sys
#sys.path.append('/home/sthiele/proj/repos/sattrail_postproc/identify_sats')
#from generate_astrometry import *
#ihu_table = pd.read_hdf('/home/sthiele/proj/repos/sattrail_postproc/identify_sats/IHU_TABLE.hdf')


def read_fits_file(filename):
    img = fits.getdata(filename)
    return img

def tophat(x, baseline, hatmid, hatwidth, height):
    y = np.ones(len(x)) * baseline
    y[(x>=hatmid-hatwidth/2)&(x<=hatmid+hatwidth/2)] = baseline + height
    return y

def objective(params, x, y):
    return np.sum(np.abs(tophat(x, *params) - y))

def finder(arr, val):
    return np.argmin(np.abs(arr - val))

def gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def gauss_with_linear(x, a, x0, sigma, b, c):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + b * x + c

def get_line_data(lines, plot=True, plotroot='~'):
    '''
    Make a pandas DataFrame of all of the lines that result from
    applying the Hough transform to the NN detection. Rejects any
    lines that have slopes with 0.5 degrees of 90, which should
    discard any star lines that have accidentally been detected.

    Inputs:
    lines --- output from probabilistic Hough transform)
    plot  --- [boolean], whether to plot Hough lines
    '''
    c1s=np.array([])
    c2s=np.array([])
    r1s=np.array([])
    r2s=np.array([])
    slopes = np.array([])
    bs = np.array([])

    if plot:
        plt.figure()
    for i, L in enumerate(lines):
        ((c1_0,r1_0),(c2_0,r2_0))=L
        if c1_0 > c2_0:
            c1 = c2_0
            c2 = c1_0
            r1 = r2_0
            r2 = r1_0
        else:
            c1 = c1_0
            c2 = c2_0
            r1 = r1_0
            r2 = r2_0
        if np.abs(np.abs(np.arctan2(r2-r1,c2-c1)*180/np.pi)-90)>0.5:
            m = (r2-r1)/(c2-c1)
            b = r2 - m*c2
            if plot:
                plt.plot([c1,c2],[r1,r2])
            c1s = np.append(c1s,c1)
            r1s = np.append(r1s,r1)
            c2s = np.append(c2s,c2)
            r2s = np.append(r2s,r2)
            slopes = np.append(slopes,m)
            bs = np.append(bs,b)
    if plot:
        plt.savefig(plotroot+'-init_line_data.png',bbox_inches='tight')

    df = pd.DataFrame(np.array([slopes, bs, c1s, c2s, r1s, r2s]).T,
                    columns=['slope','b', 'c1', 'c2', 'r1',
                             'r2']).sort_values(by=['slope','b']).reset_index(drop=True)

    slopes = df.slope.values
    cs = df.b.values

    return df, slopes, bs

def collect_segments_old_but_newer(sub, line_data, max_R=1e5, plot=False, plotroot='~', plot_individual=False):
    dfc = line_data.copy()

    # create line index array
    dindex = np.ones((len(dfc),len(dfc)))*-1

    didx = 1

    for i in range(len(line_data)):  # line of interest
        dindex[i,i] = i

        # get line i data
        c1_i = int(dfc.loc[i].c1)
        c2_i = int(dfc.loc[i].c2)
        r1_i = int(dfc.loc[i].r1)
        r2_i = int(dfc.loc[i].r2)
        s_i = dfc.loc[i].slope
        theta_i = np.arctan(s_i)
        b_i = dfc.loc[i].b

        # get line i points
        rr_i, cc_i = line(r2_i,c2_i,r1_i,c1_i)

        for j in range(i+1,len(line_data)):  # go through rest of lines

            # get line j data
            c1_j = int(dfc.loc[j].c1)
            c2_j = int(dfc.loc[j].c2)
            r1_j = int(dfc.loc[j].r1)
            r2_j = int(dfc.loc[j].r2)
            s_j = dfc.loc[j].slope
            theta_j = np.arctan(s_j)
            b_j = dfc.loc[j].b

            # get line j points
            rr_j, cc_j = line(r2_j,c2_j,r1_j,c1_j)

            # get deviation of line j from line i
            dev_ij = np.abs(s_i * cc_j - rr_j + b_i) / np.sqrt(s_i**2 + 1)

            # if line j deviates from the line of interest by more than X pixels
            # we assume they're definitely not part of the same line
            dtheta = np.min(np.array([np.abs(theta_i-theta_j),
                                      np.abs(theta_i-180-theta_j),np.abs(theta_i-theta_j+180)]))
            if (dev_ij.max() < 30)&(dtheta*180/np.pi<50):

                # combine points into one line
                c_all = np.append(cc_i,cc_j)
                r_all = np.append(rr_i,rr_j)[np.argsort(c_all)]
                c_all = c_all[np.argsort(c_all)]
                cmin, cmax = min(c_all), max(c_all)
                r_cmin, r_cmax = r_all[np.argmin(c_all)], r_all[np.argmax(c_all)]
                cfit = np.arange(cmin,cmax+1,1)

                # fit a straight line and a parabola to the points
                m, b = np.polyfit(c_all, r_all, 1)
                poly_parabola = np.poly1d(np.polyfit(c_all, r_all, 2))
                rfit_parabola = poly_parabola(cfit)
                rfit_linear = m * cfit + b

                # calculate deviation 
                dev = np.abs(m * cfit - rfit_parabola + b) / np.sqrt(m**2 + 1)
                devmax = max(dev)

                L = np.sqrt((cmax-cmin)**2 + (r_cmax-r_cmin)**2)
                dev_allowed = min(max(L**2/(8*max_R),2),10)

                if devmax < dev_allowed:
                    dindex[i,j] = j
                    dindex[j,i] = j
                    
                    cc_i = c_all
                    rr_i = r_all
                    c1_i = int(c_all.min())
                    c2_i = int(c_all.max())
                    r1_i = int(r_all[np.argmin(c_all)])
                    r2_i = int(r_all[np.argmax(c_all)])

                    s_i, b_i = np.polyfit(c_all, r_all, 1)
                else:
                    fjfjf=0
                    # assume that the rest of the lines also don't fit since the line data is
                    # sorted by slope
                    #break
            else:
                continue

        didx += 1

    dindex = dindex.astype(int)

    if plot:
        plt.figure()
        plt.imshow(dindex)
        plt.savefig(plotroot+'-line_index_dis.png', bbox_inches='tight')

    groups = []
    indexlist = np.unique(dindex[dindex>-1])

    while len(indexlist)>0:
        in1 = dindex[:,indexlist[0]][dindex[:,indexlist[0]]>-1]
        in2 = np.unique(dindex[:,in1][dindex[:,in1]>-1])
        new = np.array(list(set(in2).difference(set(in1))))
        while len(new)>0:
            in2 = np.unique(dindex[:,new][dindex[:,new]>-1])
            in1 = np.append(in1, in2)
            new = np.array(list(set(in2).difference(set(in1))))
            
        if len(in1) > 0:
            groups.append(in1)
        else:
            print(f"Warning: empty group at indexlist[0]={indexlist[0]}")
        indexlist = np.array(list(set(indexlist).difference(set(in1))))

    dindex = np.zeros(len(line_data))

    linenum = 1
    for i in range(len(groups)):
        dindex[groups[i]] = linenum
        linenum +=1
    
    numlines = len(np.unique(dindex))
    print('numlines: {}, num lines assigned: {}, len(phl data): {}, unique lines: {}'.format(numlines, len(dindex),len(line_data), np.unique(dindex)))

    if plot:
        fig, ax = plt.subplots(1,2,figsize=(8,5))
        for i in range(1,numlines+1):
            ax[0].scatter(180/np.pi*np.arctan(line_data.slope.values[dindex==i]),line_data.b.values[dindex==i],s=2,color=plt.colormaps['tab20'](i))
            for l in range(len(line_data.c1.values[dindex==i])):
                ax[1].plot([line_data.c1.values[dindex==i][l],
                            line_data.c2.values[dindex==i][l]],
                           [line_data.r1.values[dindex==i][l],
                            line_data.r2.values[dindex==i][l]],
                          color=plt.colormaps['tab20'](i),lw=5/(l+1))
        ax[0].set_xlabel(r'$\theta$ [deg]')
        ax[0].set_ylabel('y intercept')
        ax[1].set_title('number of lines: {}'.format(numlines))
        ax[1].set_xlabel('x')
        ax[1].set_ylabel('y')
        plt.savefig(plotroot+'-collect_segments.png', bbox_inches='tight')

    return dindex, numlines

def collect_segments(sub, line_data, max_R=1e5, plot=False, plotroot='~', plot_individual=False):
    dfc = line_data.copy()
    numsegs = len(line_data)

    # create line index array
    dindex = np.ones(len(dfc))*0

    line_base_number = 1
    line_number = 1

    for i in range(numsegs):  # line of interest

        # if all lines are accounted for before end collecting:
        if len(dindex[dindex==0])==0:
            print('all lines accounted for i={}, breaking'.format(i))
            break

        if dindex[i] != 0:
            line_number = dindex[i]

        # get line i data
        c1_i = int(dfc.loc[i].c1)
        c2_i = int(dfc.loc[i].c2)
        r1_i = int(dfc.loc[i].r1)
        r2_i = int(dfc.loc[i].r2)
        s_i = dfc.loc[i].slope
        theta_i = np.arctan(s_i)
        b_i = dfc.loc[i].b

        # get line i points
        rr_i, cc_i = line(r2_i,c2_i,r1_i,c1_i)

        for j in range(i+1,numsegs):  # go through rest of lines
            # get line j data
            c1_j = int(dfc.loc[j].c1)
            c2_j = int(dfc.loc[j].c2)
            r1_j = int(dfc.loc[j].r1)
            r2_j = int(dfc.loc[j].r2)
            s_j = dfc.loc[j].slope
            theta_j = np.arctan(s_j)
            b_j = dfc.loc[j].b

            # get line j points
            rr_j, cc_j = line(r2_j,c2_j,r1_j,c1_j)

            # get deviation of line j from line i
            dev_ij = np.abs(s_i * cc_j - rr_j + b_i) / np.sqrt(s_i**2 + 1)

            # if line j deviates from the line of interest by more than X pixels or
            # the slopes are more than D degrees apart we assume they're definitely 
            # not part of the same line
            dtheta = np.min(np.array([np.abs(theta_i-theta_j),
                                      np.abs(theta_i-180-theta_j),np.abs(theta_i-theta_j+180)]))

            
            if (dev_ij.max() < 30)&(dtheta*180/np.pi<30):

                # combine points into one line
                c_all = np.append(cc_i,cc_j)
                r_all = np.append(rr_i,rr_j)[np.argsort(c_all)]
                c_all = c_all[np.argsort(c_all)]
                cmin, cmax = min(c_all), max(c_all)
                r_cmin, r_cmax = r_all[np.argmin(c_all)], r_all[np.argmax(c_all)]
                cfit = np.arange(cmin,cmax+1,1)

                # fit a straight line and a parabola to the points
                m, b = np.polyfit(c_all, r_all, 1)
                poly_parabola = np.poly1d(np.polyfit(c_all, r_all, 2))
                rfit_parabola = poly_parabola(cfit)
                rfit_linear = m * cfit + b

                # calculate deviation 
                dev = np.abs(m * cfit - rfit_parabola + b) / np.sqrt(m**2 + 1)
                devmax = max(dev)

                L = np.sqrt((cmax-cmin)**2 + (r_cmax-r_cmin)**2)
                dev_allowed = min(max(L**2/(8*max_R),2),10)
                
                if devmax < dev_allowed:
                    if dindex[j] > 0:
                        dindex[dindex==line_number] = dindex[j]
                        line_number = dindex[j]
                    else:
                        dindex[i] = line_number
                        dindex[j] = line_number
                    
                    cc_i = c_all
                    rr_i = r_all
                    c1_i = int(c_all.min())
                    c2_i = int(c_all.max())
                    r1_i = int(r_all[np.argmin(c_all)])
                    r2_i = int(r_all[np.argmax(c_all)])
                    #s_i, b_i = np.polyfit(c_all, r_all, 1)
            else:
                continue
                
        if line_number != line_base_number:
            line_number = line_base_number
        else:
            if dindex[i]==0:
                dindex[i] = line_number
            line_base_number += 1
            line_number = line_base_number
                

    print(dindex)

    dindex = dindex.astype(int)
    numlines = len(np.unique(dindex))
    
    print('numlines: {}, num assigned: {}, numsegs: {}, unique lines: {}'.format(numlines,len(dindex), numsegs, np.unique(dindex)))

    if plot:
        fig, ax = plt.subplots(1,2,figsize=(10,5))
        for i in range(1,numlines+1):
            ldata = line_data.iloc[dindex==i]
            ax[0].scatter(180/np.pi*np.arctan(ldata.slope.values),
                          ldata.b.values,s=2,
                          color=plt.colormaps['tab20'](i))
            
            for l in range(len(ldata.c1.values)):
                ax[1].plot([ldata.c1.values[l],ldata.c2.values[l]],
                           [ldata.r1.values[l],ldata.r2.values[l]],
                          color=plt.colormaps['tab20'](i))
                
        ax[0].set_xlabel(r'$\theta$ [deg]')
        ax[0].set_ylabel('y intercept')
        ax[1].set_title('number of lines: {}'.format(numlines))
        ax[1].set_xlabel('x')
        ax[1].set_ylabel('y')
        plt.savefig(plotroot+'-collect_segments.png', bbox_inches='tight')

    return dindex, numlines

    
def perpendicular_line_profile(image, start_point, end_point, perp_range, num_perp_points=None):
    """
    Generate a profile showing amplitude as a function of perpendicular distance from a line.

    Args:
        image: 2D numpy array
        start_point: (x, y) tuple for line start
        end_point: (x, y) tuple for line end
        perp_range: half-width of perpendicular sampling (e.g., 10 for -10 to +10)
        num_perp_points: number of perpendicular sampling points (default: 2*perp_range+1)

    Returns:
        perp_distances: array of perpendicular distances from line
        amplitudes: array of summed pixel values at each perpendicular distance
    """
    x1, y1 = start_point
    x2, y2 = end_point

    # Calculate line direction and perpendicular direction
    dx = x2 - x1
    dy = y2 - y1
    line_length = np.sqrt(dx**2 + dy**2)

    # Unit vectors
    line_vec = np.array([dx, dy]) / line_length  # along the line
    perp_vec = np.array([-dy, dx]) / line_length  # perpendicular to line (90° rotation)

    # Generate perpendicular distances
    if num_perp_points is None:
        num_perp_points = int(2 * perp_range) + 1

    perp_distances = np.linspace(-perp_range, perp_range, num_perp_points)

    # Number of points to sample along the line direction
    num_line_points = int(np.ceil(line_length)) + 1
    t = np.linspace(0, 1, num_line_points)

    amplitudes = []

    for perp_dist in perp_distances:
        # Create parallel line at this perpendicular distance
        offset = perp_dist * perp_vec
        parallel_start = np.array([x1, y1]) + offset
        parallel_end = np.array([x2, y2]) + offset

        # Sample points along this parallel line
        parallel_points = np.outer(t, parallel_end - parallel_start) + parallel_start

        # Extract pixel values along this parallel line
        x_coords = parallel_points[:, 0]
        y_coords = parallel_points[:, 1]

        # Check bounds
        valid_mask = ((x_coords >= 0) & (x_coords < image.shape[1]) & 
                     (y_coords >= 0) & (y_coords < image.shape[0]))

        if np.any(valid_mask):
            # Sample with bilinear interpolation
            sampled_values = map_coordinates(image, [y_coords[valid_mask], x_coords[valid_mask]], 
                                           order=1, cval=0)
            total_amplitude = np.sum(sampled_values)
        else:
            total_amplitude = 0

        amplitudes.append(total_amplitude)

    return perp_distances, np.array(amplitudes)

def total_line_coords(df, linenum, plot=False):
    plot = False
    RR = []
    CC = []

    for i in [linenum]:
        dfline = df.loc[df.linenum==i]

        for j in range(len(dfline)):
            c1 = int(dfline.iloc[j].c1)
            c2 = int(dfline.iloc[j].c2)
            r1 = int(dfline.iloc[j].r1)
            r2 = int(dfline.iloc[j].r2)

            rr, cc = line(r2,c2,r1,c1)
            RR.append(rr)
            CC.append(cc)

            if plot:
                r0 = rr.min()
                c0 = cc[np.argmin(rr)]
                dx = np.sqrt((rr-r0)**2 + (cc-c0)**2)
                fig, ax = plt.subplots(1,2)
                min_value, max_value = zscale.get_limits(sub)
                ax[0].imshow(sub, origin='lower',vmin=min_value,vmax=max_value,cmap='Greys_r')
                ax[0].plot(cc,rr,c='r',lw=1)
                ax[0].set_xlim(0,2048)
                ax[0].set_ylim(0,2048)

                ax[1].axhline(0,zorder=0,color='k')
                ax[1].scatter(dx, sub[rr,cc]+2,s=1);

                dat = pd.DataFrame(np.array([dx,sub[rr,cc]]).T, columns=['x','h']).sort_values('x')
                ax[1].scatter(dat.x, dat['h'].rolling(window=50,center=True).mean(),color='r',s=1)
                ax[1].set_ylim(-500,1000)
                plt.show()

    RR = np.concatenate(RR)
    CC = np.concatenate(CC)
    ccsort = np.argsort(CC)
    CC = CC[ccsort]
    RR = RR[ccsort]
    length = (np.sqrt((CC-CC.min())**2+(RR-RR[np.argmin(CC)])**2)).max()

    return RR, CC, length

def find_common_pix(RR, CC, pixels):

    test = np.vstack([(CC,RR)]).T
    set1 = set(map(tuple, test))
    set2 = set(map(tuple, pixels))

    # Find pairs that are in array1 and array2
    shared_pairs = set1 & set2
    shared_array = np.array(list(shared_pairs))
    CCpx = shared_array[:,0]
    RRpx = shared_array[:,1]

    # OR:

    # Find pairs in array1 that are not in array2
    diff1 = set1 - set2
    # Convert back to array if needed
    unique_to_array1 = np.array(list(diff1))

    # Find pairs in array2 that are not in array1
    diff2 = set2 - set1
    unique_to_array2 = np.array(list(diff2))

    diff = np.array(list(set2.difference(set1)))

    return RRpx, CCpx

def plot_amplitude(RR, CC, sub):
    fig, ax = plt.subplots(1,2, figsize=(7,3))
    ax[0].scatter(CC, sub[RR,CC],s=1)
    dat = pd.DataFrame(np.array([RR,CC,sub[RR,CC]]).T, columns=['r','c','h']).sort_values('c')
    H = dat.groupby(['c']).mean() 
    ax[0].scatter(H.index,H.h.values,s=1)
    ax[0].scatter(H.index,H['h'].rolling(window=50,center=True).mean(),s=1)
    ax[0].axhline(0,c='k')
    ax[0].set_ylim(-500,600)
    ax[0].set_xlabel('x')

    ax[1].scatter(RR, sub[RR,CC],s=1)
    dat = pd.DataFrame(np.array([RR,CC,sub[RR,CC]]).T, columns=['r','c','h']).sort_values('c')
    H = dat.groupby(['r']).mean()
    ax[1].scatter(H.index,H.h.values,s=1)
    ax[1].scatter(H.index,H['h'].rolling(window=50,center=True).mean(),s=1)
    ax[1].axhline(0,c='k')
    ax[1].set_xlabel('y')
    plt.show()

    return

def fit_coords(RR, CC, length, plot=False):
    if length > 200:
        coefficients = np.polyfit(CC, RR, 2)
    else:
        coefficients = np.polyfit(CC, RR, 1)

    parabola_function = np.poly1d(coefficients)
    
    x_fit = np.arange(np.min(CC), np.max(CC)+0.1,0.1)
    y_fit = parabola_function(x_fit)
    cc = x_fit[(y_fit>=0)&(y_fit<2048)]
    rr = y_fit[(y_fit>=0)&(y_fit<2048)]
    c1 = cc.min()
    c2 = cc.max()
    r1 = rr[np.argmin(cc)]
    r2 = rr[np.argmax(cc)]

    x_fit = np.arange(0,2048,0.1)
    y_fit = np.round(parabola_function(x_fit),0)
    x_fit = x_fit[(y_fit>=0)&(y_fit<2048)]
    y_fit = y_fit[(y_fit>=0)&(y_fit<2048)]
    cc = x_fit
    rr = y_fit

    if plot:
        plt.figure()
        min_value, max_value = zscale.get_limits(sub)
        plt.scatter(CC,RR,s=0.5)
        plt.plot(cc, rr,c='r',lw=1)
        plt.scatter(C0,R0,c='g',s=10)
        plt.show()

    return rr, cc, c1, c2, r1, r2, coefficients
    
def rolling_mean(rr, cc, R0, C0, sub, w=50):
    dX = np.sign(cc-C0) * np.sqrt((rr-R0)**2+(cc-C0)**2)

    dat = pd.DataFrame(np.array([cc,rr,dX,sub[rr.astype(int),cc.astype(int)]]).T, columns=['c','r','dx','h'])
    dat = dat.sort_values('dx').reset_index(drop=True)

    rollmean = dat['h'].rolling(window=w,center=True).mean()
    rollstd = dat['h'].rolling(window=w,center=True).std() / np.sqrt(w)

    return dat, rollmean, rollstd

def find_gaps(rr, cc, RR, CC, R0, C0, sub, gap=2, w=50):

    datgap = pd.DataFrame(np.array([CC,RR,np.sqrt((CC-C0)**2+(RR-R0)**2),sub[RR,CC]]).T,
                          columns=['c','r','dx','h']).sort_values('dx').reset_index(drop=True)
    datgap['sep'] = np.append(0,datgap.dx.values[1:]-datgap.dx.values[:-1])

    gapsi = np.sort(np.append(datgap.loc[datgap.sep>gap].index.values-1, 
                              datgap.loc[datgap.sep>gap].index.values))
    ngaps = len(datgap.loc[datgap.sep>gap])

    gaps = np.array([])
    for i in range(len(gapsi)):
        gaps = np.append(gaps, np.argmin(np.abs(cc-datgap.loc[gapsi].c.values[i])))

    return gaps.astype(int), ngaps

def find_bounds(rr, cc, RR, CC, R0, C0, df, linenum, length, dat, rollmean,
                rollstd, gaps, ngaps, sub, w=50, nsig=3, plot=False):
    plot=False
    M, sigma = np.median(sub), sub.std()/np.sqrt(len(sub))

    lbounds = []
    rbounds = []

    if len(gaps)==0:
        allgaps = np.array([np.argmin(np.abs(dat.c.values-C0)),
                            np.argmin(np.abs(dat.c.values-CC.max()))])
    else:
        allgaps = np.append(np.append(np.argmin(cc),gaps),np.argmax(cc))

    for i in range(0,len(gaps)+1,2):
        if length < 200:
            lbound, rbound = fit_tophat(sub, CC, RR, dat, length, rollmean)
            lbound = 0
            rbound = length
            leftC = dat.c.values[np.argmin(np.abs(dat.dx.values - lbound))]
            leftR = dat.r.values[np.argmin(np.abs(dat.dx.values - lbound))]
            rightC = dat.c.values[np.argmin(np.abs(dat.dx.values - rbound))]
            rightR = dat.r.values[np.argmin(np.abs(dat.dx.values - rbound))]
        else:
            #print(allgaps[i],allgaps[i+1],rollmean[allgaps[i]:allgaps[i+1]],rollmean[allgaps[i]:allgaps[i+1]].max())
            imaxs = np.where(rollmean==rollmean[allgaps[i]:allgaps[i+1]].max())[0]
            #print(imaxs)
            imax = imaxs[(imaxs>=allgaps[i])&(imaxs<=allgaps[i+1])][0]
            #print(allgaps[i],allgaps[i+1],imax)

            maxdx = dat.dx[imax]
            left = rollmean.iloc[:imax]
            leftdx = dat.dx[:imax]
            right = rollmean.iloc[imax:]
            rightdx = dat.dx[imax:]

            wherelbound = np.where(left<nsig*sigma)[0]
            if len(wherelbound)==0:
                lbound = leftdx.min()
                leftC = dat.c.min()
                leftR = dat.r.values[np.argmin(dat.c.values)]
            else:
                lbound = leftdx.values[wherelbound].max() + w/2
                leftC = dat.c.values[np.argmin(np.abs(dat.dx.values - lbound))]
                leftR = dat.r.values[np.argmin(np.abs(dat.dx.values - lbound))]

            whererbound = np.where(right<nsig*sigma)[0]
            if len(whererbound)==0:
                rbound = rightdx.max()
                rightC = dat.c.max()
                rightR = dat.r.values[np.argmax(dat.c.values)]
            else:
                rbound = rightdx.values[whererbound].min() - w/2
                rightC = dat.c.values[np.argmin(np.abs(dat.dx.values - rbound))]
                rightR = dat.r.values[np.argmin(np.abs(dat.dx.values - rbound))]

        lbounds.append(min(lbound,rbound))
        rbounds.append(max(lbound,rbound))
        #print('LBOUND: ', lbound, 'RBOUND: ', rbound)

    if ngaps > 0:
        dfline = df.loc[df.linenum==linenum]
        newlinenums = linenum * np.ones(len(dfline))
        newline = df.linenum.values.max() + 1
        segs = np.digitize((dfline.c1+dfline.c2)/2,bins=allgaps)
        usegs = np.unique(segs)
        stopline = False
        linestop = 1
        for i in range(0,len(rbounds)-1):
            if rbounds[i]<lbounds[i+1]:
                #print('newline')
                newlinenums[np.where(np.isin(segs,usegs[i+1:]))[0]] = newline
                newline += 1
                stopline = True
            else:
                if stopline == False:
                    linestop += 1
        #import pdb
        #pdb.set_trace()
        df.loc[dfline.index,'linenum'] = newlinenums
        lbounds = np.unique(lbounds[:linestop])
        rbounds = np.unique(rbounds[:linestop])
        #print('length of arrays: ', len(lbounds), len(rbounds))

    if plot:
        fig, ax = plt.subplots(1,2,figsize=(10,5))
        min_value, max_value = zscale.get_limits(sub)
        ax[0].imshow(sub, origin='lower',vmin=min_value,vmax=max_value, cmap='Greys_r')
        ax[0].plot(dat.c.values,dat.r.values, lw=0.1,c='r')
        ax[0].set_xlim(0,2048)
        ax[0].set_ylim(0,2048)

        ax[1].axhline(0,zorder=2,color='k',lw=1)
        ax[1].scatter(dat.dx, sub[rr,cc],s=1,color='xkcd:light grey');


        ax[1].plot(dat.dx, rollmean,color='r',zorder=10)#,s=1)
        #ax[1].fill_between(dat.dx, y1=rollmean-rollstd.values,
        #                  y2=rollmean+rollstd.values,
        #                  color='r', alpha=0.2,lw=0.2)



        ax[1].axhline(M,ls='-',color='g',lw=1)
        ax[1].axhline(M-sigma,ls='--',color='g',lw=1)
        plt.subplots_adjust(wspace=0.3)

        for j in range(len(lbounds)):
            ax[0].axvline(leftC,lw=0.5,color='g')
            ax[0].axvline(rightC,lw=0.5,color='b')
        for j in range(len(np.unique(lbounds))):
            ax[1].axvline(lbounds[0],lw=0.5,color='g')
            ax[1].axvline(rbounds[0],lw=0.5,color='b')
        plt.show()

    if len(lbounds) == 1:
        newlength = rbounds[0] - lbounds[0]
    else:
        print('there is a problem!!')
        return

    return df, lbounds, rbounds, newlength


def fit_width(dat, lbound, rbound, rr, cc, R0, C0, sub, nsig=3, nhalf=3, nclose=10, plot=False):
    plot=False
    mask_fit = np.zeros((2048,2048))
    mask_fit[rr[(dat.dx.values>lbound)&(dat.dx.values<rbound)],
             cc[(dat.dx.values>lbound)&(dat.dx.values<rbound)]] = 1

    D, H = perpendicular_line_profile(sub, (cc[finder(cc,C0)], rr[finder(cc,C0)]),
                                      (cc[finder(cc,C0+30)], rr[finder(cc,C0+30)]),
                                      10, num_perp_points=None)

    #gauss(x, a, x0, sigma, offset) or gauss_with_linear(x, a, x0, sigma, b, c)
    p0 = [100,0,0.5]
    popt, _ = curve_fit(gauss, D, H, p0)
    amp, x0, sigma = popt
    sigma = np.abs(sigma)

    halfwidth = min(max(np.round(nsig*sigma+0.5,0).astype(int),1),10)
    #print('x0: {}, sigma: {}, halfwidth: {}'.format(x0,sigma,halfwidth))

    #width = np.sqrt(8*np.log(2)) * sigma + 1.
    #print(width/2)

    Hfit = gauss(D, *popt)

    if plot:
        plt.figure()
        plt.scatter(D,H)
        plt.plot(D,Hfit,c='r',lw=1)
        plt.axvline(x0+nsig*sigma+0.5)
        plt.axvline(x0-nsig*sigma-0.5)
        plt.show()

    #print('binary closing')
    mask_new = morphology.binary_closing(morphology.binary_dilation(mask_fit.astype(np.uint8),
                                                                    morphology.disk(radius=nhalf*halfwidth)),
                                         morphology.disk(nclose))
    #print('getting rr cc new')
    rr_new, cc_new = np.where(mask_new>0)[0], np.where(mask_new>0)[1]

    return mask_fit, mask_new, rr_new, cc_new, halfwidth


def fit_tophat(sub, CC, RR, dat, length, rollmean, plot=False):
    plot=False
    guess = [0,length/2,length,np.mean(sub[RR,CC])]
    #print(guess)
    res = minimize(objective, guess, args=(dat.dx.values, dat.h.values), method='Nelder-Mead')
    #print(res.x)
    baseline, hatmid, hatwidth, height = res.x
    lbound = dat.dx.values[np.argmin(np.abs(dat.c.values-(hatmid-hatwidth/2)))]
    rbound = dat.dx.values[np.argmin(np.abs(dat.c.values-(hatmid+hatwidth/2)))]
    lbound = hatmid - hatwidth / 2
    rbound = hatmid + hatwidth / 2

    if plot:
        fig, ax = plt.subplots(1,2)
        min_value, max_value = zscale.get_limits(sub)
        ax[0].imshow(sub,origin='lower',
                     vmin=min_value,
                     vmax=max_value,
                     cmap='Greys_r')
        ax[0].set_xlim(0,2048)
        ax[0].axvline(hatmid-hatwidth/2,lw=0.5,c='r')
        ax[0].axvline(hatmid+hatwidth/2,lw=0.5,c='r')

        ax[1].plot(dat.c,rollmean)
        xfit = np.linspace(0,2048,100)
        ax[1].plot(xfit, tophat(xfit, *(res.x)))
        plt.show()

    return lbound, rbound

def find_edgetrails(sub, dat, edge_threshold=5):
    Csplit = np.where(np.diff(np.where(np.sum(sub,axis=0)==0))[0]>1)[0]
    Cmin = np.where(np.sum(sub,axis=0)==0)[0][Csplit][0]
    Cmax = np.where(np.sum(sub,axis=0)==0)[0][Csplit+1][-1]
    Rsplit = np.where(np.diff(np.where(np.sum(sub,axis=1)==0))[0]>1)[0]
    Rmin = np.where(np.sum(sub,axis=1)==0)[0][Rsplit][0]
    Rmax = np.where(np.sum(sub,axis=1)==0)[0][Rsplit+1][-1]
    #print('Cmin: {},  Cmax: {}, Rmin: {}, Rmax: {}'.format(Cmin,Cmax,Rmin,Rmax))

    T = (dat.r2>=Rmax-edge_threshold)|(dat.r1>=Rmax-edge_threshold)
    B = (dat.r2<=Rmin+edge_threshold)|(dat.r1<=Rmin+edge_threshold)
    R = (dat.c2>=Cmax-edge_threshold)
    L = (dat.c1<=Cmin+edge_threshold)

    IOL = np.zeros(len(dat)).astype(int)
    IOR = np.zeros(len(dat)).astype(int)
    IOT = np.zeros(len(dat)).astype(int)
    IOB = np.zeros(len(dat)).astype(int)
    IOframe = np.zeros(len(dat)).astype(int)
    IOframe[T|B|L|R] = 1
    IOL[L] = 1
    IOR[R] = 1
    IOB[B] = 1
    IOT[T] = 1
    dat['IOframe'] = IOframe
    dat['IOL'] = IOL
    dat['IOR'] = IOR
    dat['IOT'] = IOT
    dat['IOB'] = IOB

    return dat

def median_filter_gpu(sub0, filter_radius=10, log=False):
    import torch
    import kornia

    if log:
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Image shape: {sub0.shape}")
        print(f"Image dtype: {sub0.dtype}")

    imgf32 = sub0.astype('float32')
    img_tensor = torch.from_numpy(imgf32).float().unsqueeze(0).unsqueeze(0).cuda()

    if log:
        print(f"Tensor shape: {img_tensor.shape}")
        print(f"Tensor is on: {img_tensor.device}")
        print(f"Tensor is on: {img_tensor.device}")

    w = 2 * filter_radius + 1
    filtered = kornia.filters.median_blur(img_tensor, (w, w))
    filter = filtered.squeeze().cpu().numpy()

    return filter

@nb.jit(nopython=True, parallel=True, cache=True)
def fast_masked_median_filter(sub0, masksub_valid, R, C, radius=10):
    """
    Fast masked median filter using Numba - no boolean indexing.

    Args:
        sub0: Padded image array
        masksub_valid: Padded boolean mask (True = include in median)
        R, C: Original coordinates (before padding) where to compute median
        radius: Filter radius (10 for 21x21 window)
    """
    h, w = sub0.shape
    h_orig = h - 2*radius
    w_orig = w - 2*radius

    # Output array (original size, not padded)
    filter_out = np.zeros((h_orig, w_orig), dtype=np.float32)

    for idx in nb.prange(len(R)):
        r_orig = R[idx]  # Original coordinate
        c_orig = C[idx]  # Original coordinate

        # Skip if out of bounds
        if r_orig >= h_orig or c_orig >= w_orig or r_orig < 0 or c_orig < 0:
            print('out of bounds')
            continue

        # Adjust for padding
        r_center = r_orig + radius
        c_center = c_orig + radius

        # Manually collect valid values - NO BOOLEAN INDEXING
        values_list = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                r_curr = r_center + dr
                c_curr = c_center + dc

                # Check if this pixel is valid
                if masksub_valid[r_curr, c_curr]:
                    values_list.append(sub0[r_curr, c_curr])

        # Compute median if we have values
        if len(values_list) > 0:
            values_array = np.array(values_list, dtype=np.float32)
            filter_out[r_orig, c_orig] = np.median(values_array)

    return filter_out

def median_filter_cpu(sub0, df0, mask, filter_radius=10, inner_radius=1, outer_radius=15):
    cfit = np.arange(0,2048,0.1)
    phl_mask = np.zeros((2048,2048))
    for i in df0.linenum.unique():
        seg = df0.loc[df0.linenum==i]
        rfit = seg.slope.mean()*cfit + seg.b.mean()
        inside = (rfit>=0)&(rfit<=2047)
        c = cfit[inside]
        r = np.round(rfit[inside],0).astype(int)
        phl_mask[r,c] += 1

    phl_mask_binary = (phl_mask > 0).astype(np.uint8)
    inner = cv2.dilate(phl_mask_binary, morphology.disk(radius=inner_radius),
                       iterations=1).astype(np.float32)
    outer = cv2.dilate(phl_mask_binary, morphology.disk(radius=outer_radius),
                       iterations=1).astype(np.float32)
    R, C = np.where(outer == 1)

    # Step 1: Create validity mask
    masksub_valid = (inner!=1)&(mask!=1)

    # Step 2: Pad the arrays
    padded_sub = np.pad(sub0, filter_radius, mode='constant', constant_values=0).astype(np.float32)
    masksub_valid_padded = np.pad(masksub_valid, filter_radius, mode='constant',
                                  constant_values=True)

    # Step 3: Run the filter
    # R and C should be the ORIGINAL coordinates (not adjusted for padding)
    filter = fast_masked_median_filter(padded_sub, masksub_valid_padded,
                                       R, C, radius=filter_radius)

    return filter

def progressive_hough_transform(edges, min_line_length=10, min_threshold=10, initial_threshold=100, initial_gap=50):
    """
    Progressively detect lines, removing them from consideration to allow detection of weaker lines.
    """
    edges_copy = edges.copy()
    all_lines = []
    threshold = initial_threshold
    lgap = initial_gap

    while (threshold>min_threshold)&(len(edges_copy[edges_copy>0])>0):
        # Detect lines with current threshold
        lines = phl(
            edges_copy,
            threshold=threshold,
            line_length=min_line_length,
            line_gap=50
        )

        if len(lines) == 0:
            #print('no lines')
            # Lower threshold if no lines found
            threshold = max(min_threshold, int(threshold * 0.5))
            continue

        # Add detected lines
        all_lines.extend(lines)

        # Remove detected lines from edge image
        for L in lines:
            (p0, p1) = L
            # Draw line to remove these edges
            rr, cc = line(p0[1], p0[0], p1[1], p1[0])
            # Dilate slightly to remove nearby edges
            for r, c in zip(rr, cc):
                buffer = 1
                y_min = max(0, r - buffer)
                y_max = min(edges_copy.shape[0], r + buffer + 1)
                x_min = max(0, c - buffer)
                x_max = min(edges_copy.shape[1], c + buffer + 1)
                edges_copy[y_min:y_max, x_min:x_max] = 0

        # Gradually decrease threshold
        threshold = max(min_threshold, int(threshold * 0.5))
        #lgap = max(2,int(lgap*0.5))

    return all_lines

def postproc(subfile, detfile, outputfile, plotroot, save, plot, skeleton,
             progressive, max_R, gpu, filter_radius, nclose, nhalf, nsig, gap):

    # read in fits and detection data
    sub0 = read_fits_file(subfile)
    with open(detfile, 'r') as f:
        data = json.load(f)
    pixels = np.array(data['mask'])

    if len(pixels)==0:
        if save:
            dat = pd.DataFrame([0],columns=['numlines'])
            dat.to_hdf(outputfile,key='numlines')
        return [0], [0], [0], [0], [0]

    # make sattrail detection mask
    mask = np.zeros((2048,2048))
    mask[pixels[:,1],pixels[:,0]] += 1

    if plot:
        min_value, max_value = zscale.get_limits(sub0)
        subm = sub0.copy()
        subm[mask==1] = -700
        fig, ax = plt.subplots(1,2,figsize=(10,5))
        ax[0].imshow(sub0, vmin=min_value, vmax=max_value, origin='lower', cmap='gray')
        ax[1].imshow(subm, vmin=min_value, vmax=max_value, origin='lower', cmap='gray')
        plt.savefig(plotroot+'-sattrail_mask.png', bbox_inches='tight')

    if skeleton:
        maskphl = skeletonize(mask.astype(np.uint8)).astype('>f4')
    else:
        maskphl = mask

    # run probabilistic hough transform on mask
    if progressive:
        lines = progressive_hough_transform(maskphl, min_line_length=10, min_threshold=10,
                                    initial_threshold=200, initial_gap=50)
    else:
        lines = phl(maskphl, threshold=10, line_length=10, line_gap=50)

    # make dataframe out of initial hough transform lines
    df0, slopes, bs = get_line_data(lines=lines, plot=plot, plotroot=plotroot)

    # initial guess which hough transform lines are part of same line
    dindex, numlines = collect_segments(sub=sub0, line_data=df0, max_R=max_R, 
                                        plot=plot, plotroot=plotroot)
    STOP=True
    if STOP:
        return 0,0,0,0,0
    df0['linenum'] = dindex

    if gpu:
        filtered = median_filter_gpu(sub0, df0, filter_radius=filter_radius)
    else:
        filtered = median_filter_cpu(sub0, df0, mask, filter_radius=filter_radius)

    sub = sub0 - filtered

    mask_master = np.zeros((2048,2048))
    mask_del = np.zeros((2048,2048))
    rpoints = []
    cpoints = []
    columns = ['linenum','length','c1','c2','r1','r2','cpix','rpix']
    traillist = pd.DataFrame()
    maxi = 2
    i = 0
    df = df0.copy()
    while i < maxi:
        linenum = np.unique(df.linenum.values.astype(int))[i]

        RR, CC, length = total_line_coords(df, linenum=linenum, plot=plot)

        rr, cc, R0, C0, coefficients = fit_coords(RR, CC, length, plot=plot)
        gaps, ngaps = find_gaps(rr, cc, RR, CC, R0, C0, sub, gap=gap, w=50)
        dat, rollmean, rollstd = rolling_mean(rr, cc, R0, C0, sub)

        df, lbounds, rbounds, newlength = find_bounds(rr, cc, RR, CC, R0, C0, df,
                                                  linenum, length, dat, rollmean, rollstd,
                                                  gaps, ngaps, sub, nsig=4, plot=plot)

        if newlength > 200:
            rr_end = rr[(dat.dx.values>lbounds[0])&(dat.dx.values<rbounds[0])]
            cc_end = cc[(dat.dx.values>lbounds[0])&(dat.dx.values<rbounds[0])]

            mask_fit, mask_new, rr_new, cc_new, halfwidth = fit_width(dat, lbounds[0], rbounds[0], rr, cc, R0,
                                                                      C0, sub, nsig, nhalf, nclose, plot=plot)
        elif newlength < 200:
            RR, CC, length = total_line_coords(df, linenum=linenum, plot=plot)
            rr_end = RR
            cc_end = CC

            mask_new = np.zeros((2048,2048))
            mask_new[RR,CC] += 1
            mask_new = morphology.binary_closing(mask_new, morphology.disk(nclose))

        mask_master += mask_new

        # NEW DF: lineenum, min_length, c1, c2, r1, r2, a_coeff, b_coeff, c_coeff, width, 
        # rpix, cpix, rpix_centr, cpix_centr
        data = {'linenum':linenum,'length':newlength, 'c1':cc_end.min(),'c2':cc_end.max(),
                'r1':rr_end[np.argmin(cc_end)],'r2':rr_end[np.argmax(cc_end)],
                'cpix':[cc_end],'rpix':[rr_end]}
        newlist = pd.DataFrame(data)
        traillist = pd.concat([traillist, newlist], ignore_index=True)

        # points for sat_id:
        lpoints = np.arange(0.,1.1,0.1)*length
        mids = np.argmin(np.abs(dat.dx.values-lpoints.reshape(1,len(lpoints)).T),axis=1)
        cpoints.append(cc[mids])
        rpoints.append(rr[mids])

        i += 1
        maxi = len(np.unique(df.linenum.values))
        #print('\n')

    traillist = find_edgetrails(sub, traillist)
    
    # Convert lists to arrays before saving
    traillist['cpix'] = traillist['cpix'].apply(lambda x: np.array(x, dtype=np.int32))
    traillist['rpix'] = traillist['rpix'].apply(lambda x: np.array(x, dtype=np.int32))

    print(traillist[['c1','c2','rpix','cpix']])


    if save:
        traillist.to_hdf(outputfile, key='data', format='table')

    return df0, traillist, mask_master, cpoints, rpoints
