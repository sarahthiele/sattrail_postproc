import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import json
from astropy.visualization import ZScaleInterval
import pandas as pd
import pyarrow
from skimage.draw import line
from scipy import stats
from skimage.morphology import binary_dilation, binary_closing, binary_opening
from scipy import ndimage
from skimage.morphology import skeletonize, dilation, disk
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from scipy.signal import find_peaks, peak_prominences, periodogram
from skimage.segmentation import watershed
from scipy.optimize import curve_fit, minimize
import cv2
import os
import time
import pdb

zscale = ZScaleInterval()

from postproc_phl import read_fits_file, fit_coords, find_edgetrails, fast_masked_median_filter
from gaps import find_gaps, fill_gaps, region_gaps
from tools import new_watershed, endpoints_PCA, endpoints_skeleton

def collect_segments_ndimage(sub, line_data, max_R=1e4, plot=False, plotroot='~', plot_individual=False, save=False):
    dfc = line_data.copy()
    numsegs = len(line_data)

    # create line index array
    dindex = np.ones(len(dfc))*0

    line_number = 1

    for i in range(numsegs):  # line of interest
        # if all lines are accounted for before end collecting:
        if len(dindex[dindex==0])==0:
            #print('all lines accounted for i={}, breaking'.format(i))
            break

        if dindex[i] != 0:
            continue

        # get line i data
        c1_i = int(dfc.loc[i].c1)
        c2_i = int(dfc.loc[i].c2)
        r1_i = int(dfc.loc[i].r1)
        r2_i = int(dfc.loc[i].r2)
        s_i = dfc.loc[i].slope
        theta_i = np.arctan(s_i)
        b_i = dfc.loc[i].b

        # get line i points
        #rr_i, cc_i = line(r2_i,c2_i,r1_i,c1_i)
        
        ##### new
        if theta_i*180/np.pi%180 < 45:
            cc_i = np.arange(c1_i, c2_i+0.5, 0.5)
            rr_i = s_i * cc_i + b_i
        else:
            rr_i = np.arange(min(r1_i,r2_i), max(r1_i,r2_i)+0.5, 0.5)
            cc_i = rr_i/s_i - b_i/s_i
            #print('reversing, s_i = ', s_i, 'c_i max: ', cc_i.max())
            
        ##### end new
            
        for j in range(i+1,numsegs):  # go through rest of lines
            if dindex[j] != 0:
                continue

            # get line j data
            c1_j = int(dfc.loc[j].c1)
            c2_j = int(dfc.loc[j].c2)
            r1_j = int(dfc.loc[j].r1)
            r2_j = int(dfc.loc[j].r2)
            s_j = dfc.loc[j].slope
            theta_j = np.arctan(s_j)
            b_j = dfc.loc[j].b
            
            # get line j points
            #rr_j, cc_j = line(r2_j,c2_j,r1_j,c1_j)

            ##### new
            if np.abs(theta_j*180/np.pi%180) < 45:
                cc_j = np.arange(c1_j,c2_j+0.5,0.5)
                rr_j = s_j * cc_j + b_j
            else:
                rr_j = np.arange(min(r1_j,r2_j),max(r1_j,r2_j)+0.5,0.5)
                cc_j = rr_j/s_j - b_j/s_j
                rr_j = rr_j[np.argsort(cc_j)]
                cc_j = cc_j[np.argsort(cc_j)]
                #print('reversing, s_j = ', s_j, 'c_j max: ', cc_j.max())
            ##### end new
            
            # get deviation of line j from line i
            dev_ij = np.abs(s_i * cc_j - rr_j + b_i) / np.sqrt(s_i**2 + 1)

            # if line j deviates from the line of interest by more than X pixels or
            # the slopes are more than D degrees apart we assume they're definitely 
            # not part of the same line
            dtheta = np.min(np.array([np.abs(theta_i-theta_j),
                                      np.abs(theta_i-np.pi-theta_j),np.abs(theta_i-theta_j+np.pi)]))

            if (dtheta*180/np.pi<10)&(dev_ij.max() < 100):#&(dtheta*180/np.pi<10): # 30

                # combine points into one line
                c_all = np.append(cc_i,cc_j)
                r_all = np.append(rr_i,rr_j)[np.argsort(c_all)]
                c_all = c_all[np.argsort(c_all)]
                cmin, cmax = min(c_all), max(c_all)
                r_cmin, r_cmax = r_all[np.argmin(c_all)], r_all[np.argmax(c_all)]
                
                ##### new
                #cmin, r_cmin, cmax, r_cmax = endpoints_PCA(r_all, c_all)
                ##### end new
                #print('cmin: ', cmin, 'cmax: ', cmax, 'theta_i: ', theta_i*180/np.pi, 'theta_j: ', theta_j*180/np.pi)
                cfit = np.arange(cmin,cmax+0.5,0.5)

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
                #print('dev max: ', devmax, 'L: ', L, 'dev_allowed: ', L**2/(8*max_R))

                if plot_individual:
                    fig, ax = plt.subplots(1,3, figsize=(12,3))
                    min_value, max_value = zscale.get_limits(sub)
                    ax[0].imshow(sub, vmin=min_value, vmax=max_value, cmap='Greys_r')
                    ax[0].plot([c1_i,c2_i],[r1_i,r2_i])
                    ax[0].plot([c1_j,c2_j],[r1_j,r2_j])
                    ax[0].set_ylim(0,2048)
                    ax[0].set_xlim(0,2048)

                    ax[1].scatter(c_all, r_all,s=1)
                    ax[1].scatter(cfit, m*cfit+b,s=1)

                    ax[2].scatter(cfit,dev,s=1)
                    ax[2].axhline(dev_allowed,ls='--',color='xkcd:grey',alpha=0.7)
                    ax[2].set_title(devmax)
                if devmax < dev_allowed:
                    dindex[i] = line_number
                    dindex[j] = line_number
                    
                    cc_i = c_all
                    rr_i = r_all
                    c1_i = int(c_all.min())
                    c2_i = int(c_all.max())
                    r1_i = int(r_all[np.argmin(c_all)])
                    r2_i = int(r_all[np.argmax(c_all)])
                    s_i, b_i = np.polyfit(c_all, r_all, 1)
                    
                    ##### new
                    #c1_i, r1_i, c2_i, r2_i = endpoints_PCA(rr_i, cc_i)
                    #s_i = (r2_i-r1_i)/(c2_i-c1_i)
                    #b_i = r2_i - s_i*c2_i
                    #### end new

                    
                    if plot_individual:
                        ax[0].set_title('same line: dev_ij = {}'.format((dev_ij.max())))
                        plt.show()
                else:
                    if plot_individual:
                        ax[0].set_title('different line: dev_ij = {:.4}'.format((dev_ij.max())))
                        plt.show()
            else:
                continue

        if dindex[i]==0:
            dindex[i] = line_number
            
        line_number += 1
                

    dindex = dindex.astype(int)

    numlines = len(np.unique(dindex))
    
    #print('numlines: {}, num assigned: {}, numsegs: {}, unique lines: {}'.format(numlines,len(dindex), numsegs, np.unique(dindex)))

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
        if save:
            plt.savefig(plotroot+'-collect_segments.png', bbox_inches='tight')
        else:
            plt.show()

    return dindex, numlines
    
def fit_coords_final_line(RR, CC, c1, r1, c2, r2, max_R):
    # Determine orientation
    dc = CC.max() - CC.min()
    dr = RR.max() - RR.min()
    
    # Actual data endpoints (for return)
    #c1_data = CC.min()
    #c2_data = CC.max()
    #r1_data = RR[np.argmin(CC)]
    #r2_data = RR[np.argmax(CC)]
    
    if dc > dr:  # More horizontal - fit r = f(c)
        fit_direction = 'r_of_c'
        
        # Parabolic fit
        coeff_p = np.polyfit(CC, RR, 2)
        parabola_function = np.poly1d(coeff_p)
        
        # Sample along c
        c_fit = np.arange(0., 2047.5, 0.5)
        r_fit = parabola_function(c_fit)
        
        # For trail bounds
        flag_trail = (c_fit >= CC.min()) & (c_fit <= CC.max()) & \
                     (r_fit >= RR.min()) & (r_fit <= RR.max())
        c_fit_p = c_fit[flag_trail]
        r_fit_p = r_fit[flag_trail]
        
        # For image bounds
        flag_image = (r_fit >= 0) & (r_fit < 2047.5)
        cc_fit = c_fit[flag_image]
        rr_fit = r_fit[flag_image]
        
        # Linear fit
        coeff_l = np.polyfit(CC, RR, 1)
        m, b = coeff_l
        
        # Deviation between parabolic and linear fit
        dfits = np.abs(m * c_fit_p - r_fit_p + b) / np.sqrt(m**2 + 1)
        dX_fits = dfits.max()
        
        # Deviation from straight
        c1_p, c2_p = c_fit_p.min(), c_fit_p.max()
        r1_p = r_fit_p[np.argmin(c_fit_p)]
        r2_p = r_fit_p[np.argmax(c_fit_p)]
        m_p = (r2_p - r1_p) / (c2_p - c1_p)
        b_p = r2_p - m_p * c2_p
        dev_p = np.abs(m_p * c_fit_p - r_fit_p + b_p) / np.sqrt(m_p**2 + 1)
        
    else:  # More vertical - fit c = f(r)
        fit_direction = 'c_of_r'
        
        # Parabolic fit
        coeff_p = np.polyfit(RR, CC, 2)
        parabola_function = np.poly1d(coeff_p)
        
        # Sample along r
        r_fit = np.arange(0., 2047.5, 0.5)
        c_fit = parabola_function(r_fit)
        
        # For trail bounds
        flag_trail = (r_fit >= RR.min()) & (r_fit <= RR.max()) & \
                     (c_fit >= CC.min()) & (c_fit <= CC.max())
        r_fit_p = r_fit[flag_trail]
        c_fit_p = c_fit[flag_trail]
        
        # For image bounds
        flag_image = (c_fit >= 0) & (c_fit < 2047.5)
        rr_fit = r_fit[flag_image]
        cc_fit = c_fit[flag_image]
        
        # Linear fit
        coeff_l = np.polyfit(RR, CC, 1)
        m, b = coeff_l
        
        # Deviation between parabolic and linear fit
        # For c = f(r), line is -c + m*r + b = 0, so distance is |m*r - c + b| / sqrt(m² + 1)
        dfits = np.abs(m * r_fit_p - c_fit_p + b) / np.sqrt(m**2 + 1)
        dX_fits = dfits.max()
        
        # Deviation from straight (using parabola endpoints)
        r1_p, r2_p = r_fit_p.min(), r_fit_p.max()
        c1_p = c_fit_p[np.argmin(r_fit_p)]
        c2_p = c_fit_p[np.argmax(r_fit_p)]
        m_p = (c2_p - c1_p) / (r2_p - r1_p)
        b_p = c2_p - m_p * r2_p
        dev_p = np.abs(m_p * r_fit_p - c_fit_p + b_p) / np.sqrt(m_p**2 + 1)
    
    # Common calculations (orientation-independent)
    dev_p_max = dev_p.max()
    L = np.sqrt((r2 - r1)**2 + (c2 - c1)**2)
    dev_allowed = L**2 / (8 * max_R)
    R_approx = L**2 / (8 * dev_p_max) if dev_p_max > 0 else np.inf
    
    curv_info = [dX_fits, dev_p_max, dev_allowed, R_approx]
    endpoints = [c1, r1, c2, r2]
    
    return cc_fit, rr_fit, coeff_p, coeff_l, curv_info, L, fit_direction
    
def get_bg_stats(sub, mask):
    background = sub[mask==0]
    N_bg = len(background)
    
    mean_bg = np.mean(background)
    std_bg = np.std(background)
    
    sigma_bg = std_bg / np.sqrt(N_bg)

    return N_bg, mean_bg, std_bg, sigma_bg

def check_periodicity(dat_for_bounds, plot=False):
    rollmean = dat_for_bounds.rollmean.values
    peaks = find_peaks(rollmean, prominence=1000, distance=100)[0]
    
    if len(peaks)>=3:
        peak_w, _, _ = peak_prominences(rollmean, peaks, wlen=None)
    
        # calculate coefficient of variability:
        peak_spacings = np.diff(peaks)
        CV = np.std(peak_spacings) / np.mean(peak_spacings)
        
        if plot:
            fig, ax = plt.subplots(figsize=(5,3))
            plt.plot(dat_for_bounds.c.values, rollmean)
            plt.scatter(dat_for_bounds.c.values[peaks], 
                        rollmean[peaks],s=50,c='r',label='CV = {}'.format(CV))
        if CV <= 0.5:
            return CV, True
        else:
            return CV, False
    else:
        return -1, False   

    
def rolling_mean(rr_fit, cc_fit, r1, c1, sub, exclude_zeros=False, w=50):
    dX = np.sign(cc_fit-c1) * np.sqrt((rr_fit-r1)**2+(cc_fit-c1)**2)

    if exclude_zeros:
        subz = sub.copy()
        subz[sub==0.] = np.nan

        dat = pd.DataFrame(np.array([cc_fit,rr_fit,dX,sub[rr_fit.astype(int),
                                     cc_fit.astype(int)]]).T,
                           columns=['c','r','dx','h'])
    else:
        dat = pd.DataFrame(np.array([cc_fit,rr_fit,dX,sub[rr_fit.astype(int),
                                     cc_fit.astype(int)]]).T,
                           columns=['c','r','dx','h'])
        
    dat = dat.sort_values('dx').reset_index(drop=True)

    rollmean = dat['h'].rolling(window=w,center=True).mean()
    rollstd = dat['h'].rolling(window=w,center=True).std() / np.sqrt(w)

    dat['rollmean'] = rollmean
    dat['rollstd'] = rollstd
    
    return dat
    
def find_bounds(rr_fit, cc_fit, r1, c1, r2, c2, length, dat,
                sub, mean_bg, sigma_bg, w=50, nsig=5, plot=True):

    lbounds = []
    rbounds = []

    #w /= cc_fit[1]-cc_fit[0]
    
    rollmean = dat.rollmean.values
    rollstd = dat.rollstd.values

    imax_left = np.argmin(np.abs(dat.dx.values-0))+50
    imax_right = np.argmin(np.abs(dat.dx.values-length))-50

    left = rollmean[:imax_left]
    leftdx = dat.dx[:imax_left]
    right = rollmean[imax_right:]
    rightdx = dat.dx[imax_right:]

    wherelbound = np.where(left<mean_bg+nsig*sigma_bg)[0]
    if len(wherelbound)==0:
        lbound = leftdx.min()
        c1_new = dat.c.min()
        r1_new = dat.r.values[np.argmin(dat.c.values)]
    else:
        lbound = leftdx.values[wherelbound].max() #+ w/2
        c1_new = c1 #dat.c.values[np.argmin(np.abs(dat.dx.values - lbound))]
        r1_new = r1 #dat.r.values[np.argmin(np.abs(dat.dx.values - lbound))]

    whererbound = np.where(right<mean_bg+nsig*sigma_bg)[0]
    if len(whererbound)==0:
        rbound = rightdx.max()
        c2_new = dat.c.max()
        r2_new = dat.r.values[np.argmax(dat.c.values)]
    else:
        rbound = rightdx.values[whererbound].min() #- w/2
        c2_new = c2 #dat.c.values[np.argmin(np.abs(dat.dx.values - rbound))]
        r2_new = r2 #dat.r.values[np.argmin(np.abs(dat.dx.values - rbound))]

    if (lbound > 0):
        #print('fixing left')
        lbound = 0
        c1_new = c1
        r1_new = r1
    if (rbound < length):
        #print('fixing right')
        rbound = length
        c2_new = c2
        r2_new = r2
        
    lbounds.append(min(lbound,rbound))
    rbounds.append(max(lbound,rbound))

    if plot:
        fig, ax = plt.subplots(1,2,figsize=(20,10))
        min_value, max_value = zscale.get_limits(sub)
        ax[0].imshow(sub, origin='lower',vmin=min_value,vmax=max_value, cmap='Greys_r')
        ax[0].set_xlim(0,2048)
        ax[0].set_ylim(0,2048)

        ax[1].axhline(0,zorder=2,color='k',lw=1)
        ax[1].scatter(dat.dx, sub[rr_fit.astype(int),cc_fit.astype(int)],s=1,color='xkcd:light grey');
        ax[1].plot(dat.dx, rollmean,color='r',zorder=10)
        ax[1].axhline(mean_bg,ls='-',color='g',lw=1)
        ax[1].axhline(mean_bg-sigma_bg,ls='--',color='g',lw=1)
        ax[1].set_ylim(-500,2000)
        plt.subplots_adjust(wspace=0.3)

        for j in range(len(lbounds)):
            ax[0].axvline(c1_new,lw=0.5,color='g')
            ax[0].axvline(c2_new,lw=0.5,color='b')
            ax[0].axvline(c1,lw=0.5,ls='--',color='g')
            ax[0].axvline(c2,lw=0.5,ls='--',color='b')
        for j in range(len(np.unique(lbounds))):
            ax[1].axvline(lbounds[0],lw=0.5,color='g')
            ax[1].axvline(rbounds[0],lw=0.5,color='b')
        plt.show()

    return c1_new, r1_new, c2_new, r2_new

def median_filter_cpu_skmask(sub0, skel_mask, mask, filter_radius=10, inner_radius=1, outer_radius=15):
    phl_mask_binary = (skel_mask > 0).astype(np.uint8)
    inner = cv2.dilate(phl_mask_binary, disk(radius=inner_radius),
                       iterations=1).astype(np.float32)
    outer = cv2.dilate(phl_mask_binary, disk(radius=outer_radius),
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

def median_filter_cpu_bg_stats(sub0, df0, mask, filter_radius=10, inner_radius=1, outer_radius=15):
    cfit = np.arange(0,2047.5,0.5)
    phl_mask = np.zeros((2048,2048))
    for i in df0.line_id.unique():
        seg = df0.loc[df0.line_id==i]
        rfit = seg.slope.mean()*cfit + seg.b.mean()
        inside = (rfit>=0)&(rfit<=2047)
        c = np.round(cfit[inside],0).astype(int)
        r = np.round(rfit[inside],0).astype(int)
        phl_mask[r,c] += 1

    phl_mask_binary = (phl_mask > 0).astype(np.uint8)
    inner = cv2.dilate(phl_mask_binary, disk(radius=inner_radius),
                       iterations=1).astype(np.float32)
    outer = cv2.dilate(phl_mask_binary, disk(radius=outer_radius),
                       iterations=1).astype(np.float32)
    R, C = np.where(outer == 1)

    # Step 1: Create validity mask
    masksub_valid = (inner!=1)&(mask!=1)

    # Step 2: Pad the arrays
    padded_sub = np.pad(sub0, filter_radius, mode='constant', 
                        constant_values=0).astype(np.float32)
    masksub_valid_padded = np.pad(masksub_valid, filter_radius, mode='constant',
                                  constant_values=True)

    # Step 3: Run the filter
    # R and C should be the ORIGINAL coordinates (not adjusted for padding)
    filter = fast_masked_median_filter(padded_sub, masksub_valid_padded,
                                       R, C, radius=filter_radius)

    sub = sub0 - filter

    N_bg, mean_bg, std_bg, sigma_bg = get_bg_stats(sub, 
                                                   np.invert((outer==1)&(inner!=1)&(mask!=1)))
    

    return sub, N_bg, mean_bg, std_bg, sigma_bg

def test_significance(traillist, sub, mean_bg, std_bg): 
    is_significant = []
    ratios = []
    SNRs = []
    for line_id in traillist.line_id.unique():
        l = traillist.loc[traillist.line_id==line_id]
        mask_c = l.mask_sk_c.values[0]
        mask_r = l.mask_sk_r.values[0]

        # calculate SNR
        N_trail = len(mask_r)
        mean_trail = np.mean(sub[mask_r,mask_c])
        sigma_trail = std_bg / np.sqrt(N_trail)
        SNR_trail = (mean_trail - mean_bg) / sigma_trail
        SNRs.append(SNR_trail)
        
        threshold = mean_bg + 2*std_bg
        significant_pixels = np.sum(sub[mask_r,mask_c] > threshold)
        
        # Expected by chance
        total_pixels_near_line = len(mask_c)
        prob_random = 0.0027  # ~3-sigma false positive rate
        expected_random = total_pixels_near_line * prob_random

        ratios.append(significant_pixels/expected_random)
        
        # Significance
        if significant_pixels > expected_random * 3:  # 5x more than random
            is_significant.append(True)
            #print(f"Line is significant: {significant_pixels} vs {expected_random:.3f} expected")
        else:
            is_significant.append(False)
            #print(f"Line is not significant: {significant_pixels} vs {expected_random:.3f} expected")
    return is_significant, ratios, SNRs

def t_test(traillist, sub, mean_bg):
    is_significant = []
    for line_id in traillist.line_id.unique():
        l = traillist.loc[traillist.line_id==line_id]
        mask_c = l.mask_all_c.values[0]
        mask_r = l.mask_all_r.values[0]

        t_statistic, p_value = stats.ttest_1samp(sub[mask_r,mask_c], mean_bg)
        
        # Significance
        if p_value < 0.5:  # 5x more than random
            is_significant.append(True)
        else:
            is_significant.append(False)
            
    return is_significant

def postproc_fitgaps(subfile, detfile, outputfile, plotroot, save=True, skeleton=True, plot_final=False, 
             plot_verbose=False, max_R=1e4, filter_radius=10, nsig=5, gap=2, min_line_size=10):

    # read in fits and detection data
    try:
        sub0 = read_fits_file(subfile)
    except:
        print('subfile {} does not exist'.format(subfile))
        return

    try:
        with open(detfile, 'r') as f:
            data = json.load(f)
    except:
        print('detfile {} does not exist'.format(detfile))
        return

    pixels = np.array(data['mask'])

    if len(pixels)==0:
        if save:
            dat = pd.DataFrame([0],columns=['line_id'])
            dat.to_parquet(outputfile, engine='pyarrow', compression='snappy') 
        return

    mask = np.zeros((2048,2048))
    mask[pixels[:,1],pixels[:,0]] += 1

    if skeleton:
        mask_skeleton = skeletonize(binary_closing(mask.astype(np.uint8), footprint=np.ones((2,2)))).astype('>f4')
        skeleton_dilated = binary_closing(binary_dilation(mask_skeleton, footprint=np.ones((1, 1))), footprint=np.ones((3,3)))#, footprint=np.ones((3, 3))), footprint=np.ones((1,1)))
        skeleton_reconnected = skeletonize(skeleton_dilated)
        
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
        neighbor_count = ndimage.convolve(skeleton_reconnected.astype(np.uint8), kernel, mode='constant')

        branch_points = (neighbor_count > 2) & (skeleton_reconnected > 0)
        closed_mask = np.where(branch_points > 0, 0, skeleton_reconnected)
        
    else:
        closed_mask = binary_closing(binary_dilation(mask.astype(np.uint8), 
                                                     footprint=np.ones((3, 3))), footprint=np.ones((6,6)))
        
    # Then label
    labeled_mask, num_labels = ndimage.label(closed_mask, structure=np.ones((3,3)))

    # Count pixels in each label (excluding background label 0)
    drop_id = []
    kept_labels = []
    for label_id in range(1, num_labels+1):
        size = np.sum(labeled_mask == label_id)
        if size < min_line_size:
            drop_id.append(label_id)
            labeled_mask[labeled_mask==label_id] = 0
        else:
            kept_labels.append(label_id)

    if len(kept_labels)!=num_labels:
        new_labels = np.arange(1,len(kept_labels)+1,1)
        for i, label_id in enumerate(kept_labels):
            labeled_mask[labeled_mask==label_id] = new_labels[i]
        num_labels = len(kept_labels)
    else:
        new_labels = kept_labels
    #print('number of segments: ', num_labels)
    
    if num_labels==0:
        if save:
            dat = pd.DataFrame([0],columns=['line_id'])
            dat.to_parquet(outputfile, engine='pyarrow', compression='snappy') 
        return
    
    ms = []
    bs = []
    c1s = []
    c2s = []
    r1s = []
    r2s = []
    labels = []
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = ndimage.convolve(closed_mask.astype(np.uint8), kernel, mode='constant')
    kept_labels = []
    
    if plot_verbose:
        plt.figure()
        plt.scatter(pixels[:,0],pixels[:,1], color='xkcd:light grey', s=1)
        
    for label_id in new_labels:
        RR, CC = np.where(labeled_mask==label_id)[0], np.where(labeled_mask==label_id)[1]
        c1, r1, c2, r2 = endpoints_skeleton(label_id, labeled_mask, neighbor_count)

        if plot_verbose:
            plt.scatter(CC, RR,s=1, label='line {}'.format(label_id))
            plt.scatter([c1,c2],[r1,r2],s=2,color='r')

        if (c1==c2):
            labeled_mask[labeled_mask==label_id] = 0
            continue

        if abs(np.degrees(np.arctan2(r2-r1,c2-c1))%180 - 90)>1:
            labels.append(label_id)
            c1s.append(c1)
            c2s.append(c2)
            r1s.append(r1)
            r2s.append(r2)
            length = np.sqrt((c2-c1)**2+(r2-r1)**2)
            coefficients = np.polyfit(CC, RR, 1)
            ms.append(coefficients[0])
            bs.append(coefficients[1])
        else:
            labeled_mask[labeled_mask==label_id] = 0
            
    if plot_verbose:
        plt.legend()
        plt.show()

    new_labels = labels

    df0 = pd.DataFrame(np.array([labels, ms, bs, c1s, c2s, r1s, r2s]).T,
                                 columns=['label','slope', 'b', 'c1', 'c2', 'r1', 'r2'])
    df0 = df0.sort_values('slope').reset_index(drop=True)

    dindex, numlines = collect_segments_ndimage(sub=sub0, line_data=df0, max_R=max_R, 
                                            plot=plot_verbose, plotroot=plotroot, save=save)
    df0['line_id'] = dindex

    if numlines==0:  # if lines are vertical e.g., they might have been discarded
        if save:
            dat = pd.DataFrame([0],columns=['line_id'])
            dat.to_parquet(outputfile, engine='pyarrow', compression='snappy') 
        if plot_final:
            min_value, max_value = zscale.get_limits(sub0)
            subm = sub0.copy()
            subm[mask==1] = -700
            fig, ax = plt.subplots(1,3,figsize=(15,5))
            ax[0].imshow(sub0, vmin=min_value, vmax=max_value, origin='lower', cmap='gray')
            ax[1].imshow(subm, vmin=min_value, vmax=max_value, origin='lower', cmap='gray')
            ax[2].scatter(pixels[:,0],pixels[:,1],s=1,color='xkcd:light grey')
            ax[0].set_title('image {}'.format(subfile.split('/')[-1].split('-sub')[0]))
            ax[2].set_title('number of trails: 0')
            ax[2].set_xlim(0,2048)
            ax[2].set_ylim(0,2048)
            ax[2].set_aspect('equal')
            if save:
                plt.savefig(plotroot+'-trails.png', bbox_inches='tight',dpi=300)
            plt.close()
        return

    # make line-labeled_skeletonized_mask
    labeled_line_mask = labeled_mask.copy()
    for i, label_id in enumerate(new_labels):
        if label_id in df0.label.values:
            labeled_line_mask[labeled_mask==label_id] = df0.loc[df0.label==label_id].line_id

    labeled_full_line_mask = new_watershed(labeled_line_mask, mask)
    
    plt.figure()
    for line_id in np.unique(df0.line_id):
        plt.scatter(np.where(labeled_full_line_mask==line_id)[1],
                    np.where(labeled_full_line_mask==line_id)[0],s=1,label='line {}'.format(line_id))
    plt.legend()
    plt.show()

    # make line-labeled full mask
    #labeled_full_line_mask = new_watershed(labeled_mask, mask)
    
    #for i, label_id in enumerate(new_labels):
    #    if label_id in df0.label.values:
    #        labeled_full_line_mask[labeled_full_line_mask==label_id] = df0.loc[df0.label==label_id].line_id
    
    #  Now time to find out if the lines with gaps are all one line or separate lines:
    sub, N_bg, mean_bg, std_bg, sigma_bg = median_filter_cpu_bg_stats(sub0, df0, mask, 
                                                                      filter_radius=filter_radius)
    
    traillist = pd.DataFrame()

    line_numbers = np.unique(dindex)
    line_numbers_to_process = list(line_numbers)
    new_line_id = np.max(line_numbers_to_process) + 1

    
    while len(line_numbers_to_process) > 0:
        line_id = line_numbers_to_process.pop(0)
            
        indices_all = np.where(labeled_full_line_mask==line_id)
        RR_all, CC_all = indices_all[0], indices_all[1]
        sort_idx_all = np.argsort(CC_all)
        RR_all = RR_all[sort_idx_all]
        CC_all = CC_all[sort_idx_all]
        
        indices_sk = np.where(labeled_line_mask==line_id)
        RR_sk, CC_sk = indices_sk[0], indices_sk[1]
        sort_idx_sk = np.argsort(CC_sk)
        RR_sk = RR_sk[sort_idx_sk]
        CC_sk = CC_sk[sort_idx_sk]

        if plot_verbose:
            plt.figure()
            plt.scatter(pixels[:,0],pixels[:,1], s=1,color='xkcd:light grey')
            plt.scatter(CC_all,RR_all, s=1)
            plt.title(line_id)
            
        length_approx = np.sqrt((RR_sk.max()-RR_sk.min())**2 + (CC_sk.max()-CC_sk.min())**2)
        rr_fit, cc_fit, c1, c2, r1, r2, coefficients = fit_coords(RR_sk, CC_sk, length_approx)
        dat_for_bounds = rolling_mean(rr_fit, cc_fit, r1, c1, sub, w=50)
        CV, periodic = check_periodicity(dat_for_bounds)
        gaps_loc_cc_fit, ngaps = find_gaps(rr_fit=rr_fit, cc_fit=cc_fit, RR=RR_sk, 
                                           CC=CC_sk, r1=r1, c1=c1, sub=sub, gap=gap, w=50)
        
        if ngaps>0:
            if plot_verbose:
                plt.scatter(dat_for_bounds.c.values[gaps_loc_cc_fit], 
                            dat_for_bounds.r.values[gaps_loc_cc_fit],s=3,c='r')
                plt.title('gap locations')
                plt.show()
            if periodic:
                is_gap_real = np.zeros(ngaps, dtype=int)
            else:
                is_gap_real = fill_gaps(sub, line_id, mask, RR_all, CC_all, RR_sk, CC_sk, ngaps, 
                      gaps_loc_cc_fit, dat_for_bounds, N_bg, mean_bg, std_bg, sigma_bg, plot=plot_verbose)
            
            dx_gaps = list(dat_for_bounds.iloc[gaps_loc_cc_fit].dx.values)
            dx_all = np.sqrt((CC_all-c1)**2+(RR_all-r1)**2)
            dx_sk = np.sqrt((CC_sk-c1)**2+(RR_sk-r1)**2)
            
            new_edges = [dx_all.min()]
            
            if ngaps==1:
                if is_gap_real[0] == 1:
                    new_edges.extend(dx_gaps)
            else:
                for i in range(ngaps):
                    if is_gap_real[i] == 1:
                        new_edges.extend(dx_gaps[2*i:2*i+2])
            
            new_edges.extend([dx_all.max()])

            num_new_lines = len(is_gap_real[is_gap_real==1])

            if plot_verbose:
                plt.figure()
                plt.scatter(CC_all, RR_all,s=1,color='xkcd:light grey')

            if num_new_lines>0:
                labeled_full_line_mask[labeled_full_line_mask==line_id] = -1
                labeled_line_mask[labeled_line_mask==line_id] = -1
                
                for i in range(num_new_lines+1):
                    flag_all = (dx_all>=new_edges[2*i])*(dx_all<=new_edges[2*i+1])
                    flag_sk = (dx_sk>=new_edges[2*i])*(dx_sk<=new_edges[2*i+1])
                    
                    if plot_verbose:
                        plt.scatter(CC_all[flag_all], RR_all[flag_all],s=1)
                        
                    if i==0:
                        labeled_full_line_mask[RR_all[flag_all],CC_all[flag_all]] = line_id
                        labeled_line_mask[RR_sk[flag_sk],CC_sk[flag_sk]] = line_id
                    else:
                        labeled_full_line_mask[RR_all[flag_all],CC_all[flag_all]] = new_line_id
                        labeled_line_mask[RR_sk[flag_sk],CC_sk[flag_sk]] = new_line_id
                        line_numbers = np.append(line_numbers, new_line_id)
                        line_numbers_to_process.append(new_line_id)
                        new_line_id += 1

                labeled_full_line_mask[labeled_full_line_mask==-1] = 0
                labeled_line_mask[labeled_line_mask==-1] = 0
                
                indices_all = np.where(labeled_full_line_mask==line_id)
                RR_all, CC_all = indices_all[0], indices_all[1]
                sort_idx_all = np.argsort(CC_all)
                RR_all = RR_all[sort_idx_all]
                CC_all = CC_all[sort_idx_all]
                
                indices_sk = np.where(labeled_line_mask==line_id)
                RR_sk, CC_sk = indices_sk[0], indices_sk[1]
                sort_idx_sk = np.argsort(CC_sk)
                RR_sk = RR_sk[sort_idx_sk]
                CC_sk = CC_sk[sort_idx_sk]

    
            #RR_sk = RR_sk[(dx_sk>=new_edges[0])&(dx_sk<=new_edges[1])]
            #CC_sk = CC_sk[(dx_sk>=new_edges[0])&(dx_sk<=new_edges[1])]
            #length_approx = np.sqrt((CC_sk.max()-CC_sk.min())**2+(RR_sk.max()-RR_sk.min())**2)
            #rr_fit, cc_fit, c1, c2, r1, r2, coefficients = fit_coords(RR_sk, CC_sk, length_approx)
            #dat_for_bounds = rolling_mean(rr_fit, cc_fit, r1, c1, sub, w=50)
            
            if plot_verbose:
                plt.scatter(CC_sk, RR_sk, color='k', s=1)
                #plt.scatter([c1,c2],[r1,r2],color='r')
                plt.title('after gap fitting')
                plt.show()
                
        c1, r1, c2, r2 = endpoints_skeleton(line_id, labeled_line_mask, neighbor_count)
        cc_fit, rr_fit, coeff_p, coeff_l, curv_info, length_approx, fit_direction = fit_coords_final_line(RR_all,CC_all,c1, r1, c2, r2, max_R)

        # get fraction of mask and skeletonized mask pixels that are 0 so we can check
        # for false positives from stray starlight
        zero_frac_all = len(sub[RR_all,CC_all]==0.)/len(sub[RR_all,CC_all])
        zero_frac_sk = len(sub[RR_sk,CC_sk]==0.)/len(sub[RR_sk,CC_sk])
        
        coeff_a, coeff_b, coeff_c = coeff_p
        slope, y_int = coeff_l
        dX_fits, dev_p_max, dev_p_allowed, R_approx = curv_info
    
        if length_approx > 200:
            dat_for_bounds = rolling_mean(rr_fit, cc_fit, r1, c1, sub, w=50)
            c1_new, r1_new, c2_new, r2_new = find_bounds(rr_fit, cc_fit, r1, c1, r2, c2,
                                                         length_approx, dat_for_bounds, sub, 
                                                         mean_bg, sigma_bg, w=50, nsig=5, 
                                                         plot=False)
            line_data = {'line_id':line_id,'length':np.sqrt((r2_new-r1_new)**2+(c2_new-c1_new)**2),
                         'c1':c1_new,'c2':c2_new,'r1':r1_new,'r2':r2_new,'coeff_a':coeff_a,
                         'coeff_b':coeff_b, 'coeff_c':coeff_c,'slope': slope,'y_int':y_int,
                         'fit_direction':fit_direction, 'dX_fits':dX_fits,'dev_p_max':dev_p_max,
                         'dev_p_allowed':dev_p_allowed, 'R_approx':R_approx,
                         'mask_all_c':CC_all.tolist(), 'mask_all_r':RR_all.tolist(),
                         'mask_sk_c':CC_sk.tolist(), 'mask_sk_r':RR_sk.tolist(),
                        'CV':CV, 'zero_frac_all':zero_frac_all, 'zero_frac_sk':zero_frac_sk}
        else:
            line_data = {'line_id':line_id,'length':length_approx,
                         'c1':c1,'c2':c2,'r1':r1,'r2':r2,'coeff_a':coeff_a,
                         'coeff_b':coeff_b, 'coeff_c':coeff_c,'slope': slope,'y_int':y_int,
                         'fit_direction':fit_direction, 'dX_fits':dX_fits,'dev_p_max':dev_p_max,
                         'dev_p_allowed':dev_p_allowed, 'R_approx':R_approx,
                         'mask_all_c':CC_all.tolist(), 'mask_all_r':RR_all.tolist(),
                         'mask_sk_c':CC_sk.tolist(), 'mask_sk_r':RR_sk.tolist(),
                        'CV':CV, 'zero_frac_all':zero_frac_all, 'zero_frac_sk':zero_frac_sk}
            
        line_df = pd.DataFrame([line_data])
        traillist = pd.concat([traillist, line_df], ignore_index=True)
        
    is_significant, ratios, SNRs = test_significance(traillist, sub, mean_bg, std_bg)

    traillist['SNR'] = SNRs
    traillist['significance'] = is_significant
    traillist['significance_ratio']  = ratios

    traillist = find_edgetrails(sub, traillist, edge_threshold=5)

    if plot_final:
        min_value, max_value = zscale.get_limits(sub0)
        subm = sub0.copy()
        subm[mask==1] = -700
        fig, ax = plt.subplots(1,3,figsize=(15,5))
        ax[0].imshow(sub0, vmin=min_value, vmax=max_value, origin='lower', cmap='gray')
        ax[1].imshow(subm, vmin=min_value, vmax=max_value, origin='lower', cmap='gray')
        ax[2].scatter(pixels[:,0],pixels[:,1],s=2,color='xkcd:light grey')
        for i, line_id in enumerate(line_numbers):
            ax[2].scatter(traillist.loc[traillist.line_id==line_id][['c1','c2']],
                          traillist.loc[traillist.line_id==line_id][['r1','r2']],
                          s=5, color='black')
            cm = ax[2].scatter(traillist.loc[traillist.line_id==line_id]['mask_all_c'].values[0],
                          traillist.loc[traillist.line_id==line_id]['mask_all_r'].values[0],s=1,
                         color=plt.colormaps['tab20'](line_id%20),
                              label='{}:{},{:.1f},{:.1f}'.format(line_id,is_significant[i],
                                                            ratios[i],SNRs[i]))
        plt.legend(markerscale=5,bbox_to_anchor=(1.6, 1), loc='upper right')
        ax[0].set_title('image {}'.format(subfile.split('/')[-1].split('-sub')[0]))
        ax[2].set_title('number of trails: {}'.format(len(traillist)))
        ax[2].set_xlim(0,2048)
        ax[2].set_ylim(0,2048)
        ax[2].set_aspect('equal')
        if save:
            plt.savefig(plotroot+'-trails.png', bbox_inches='tight',dpi=300)
            plt.close()
        else:
            plt.show()

    if save:
        traillist.to_parquet(outputfile, engine='pyarrow', compression='snappy')

    return
    
def postproc_regiongaps(subfile, detfile, outputfile, plotroot, save=True, skeleton=True, plot_final=False, 
             plot_verbose=False, max_R=1e4, filter_radius=10, nsig=5, gap=2, min_line_size=10):

    # read in fits and detection data
    try:
        sub0 = read_fits_file(subfile)
    except:
        print('subfile {} does not exist'.format(subfile))
        return

    try:
        with open(detfile, 'r') as f:
            data = json.load(f)
    except:
        print('detfile {} does not exist'.format(detfile))
        return

    pixels = np.array(data['mask'])

    if len(pixels)==0:
        if save:
            dat = pd.DataFrame([0],columns=['line_id'])
            dat.to_parquet(outputfile, engine='pyarrow', compression='snappy') 
        return

    mask = np.zeros((2048,2048))
    mask[pixels[:,1],pixels[:,0]] += 1

    if skeleton:
        mask_skeleton = skeletonize(binary_closing(mask.astype(np.uint8), footprint=np.ones((2,2)))).astype('>f4')
        skeleton_dilated = binary_closing(binary_dilation(mask_skeleton, footprint=np.ones((1, 1))), footprint=np.ones((3,3)))# footprint=np.ones((3, 3))), footprint=np.ones((3,3)))
        skeleton_reconnected = skeletonize(skeleton_dilated)
        
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
        neighbor_count = ndimage.convolve(skeleton_reconnected.astype(np.uint8), kernel, mode='constant')

        branch_points = (neighbor_count > 2) & (skeleton_reconnected > 0)
        closed_mask = np.where(branch_points > 0, 0, skeleton_reconnected)
        
    else:
        closed_mask = binary_closing(binary_dilation(mask.astype(np.uint8), 
                                                     footprint=np.ones((3, 3))), footprint=np.ones((6,6)))
        
    # Then label
    labeled_mask, num_labels = ndimage.label(closed_mask, structure=np.ones((3,3)))

    # Count pixels in each label (excluding background label 0)
    drop_id = []
    kept_labels = []
    for label_id in range(1, num_labels+1):
        size = np.sum(labeled_mask == label_id)
        if size < min_line_size:
            drop_id.append(label_id)
            labeled_mask[labeled_mask==label_id] = 0
        else:
            kept_labels.append(label_id)

    if len(kept_labels)!=num_labels:
        new_labels = np.arange(1,len(kept_labels)+1,1)
        for i, label_id in enumerate(kept_labels):
            labeled_mask[labeled_mask==label_id] = new_labels[i]
        num_labels = len(kept_labels)
    else:
        new_labels = kept_labels
    #print('number of segments: ', num_labels)
    
    if num_labels==0:
        if save:
            dat = pd.DataFrame([0],columns=['line_id'])
            dat.to_parquet(outputfile, engine='pyarrow', compression='snappy') 
        return
    
    ms = []
    bs = []
    c1s = []
    c2s = []
    r1s = []
    r2s = []
    labels = []
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = ndimage.convolve(closed_mask.astype(np.uint8), kernel, mode='constant')
    kept_labels = []
    
    if plot_verbose:
        plt.figure()
        plt.scatter(pixels[:,0],pixels[:,1], color='xkcd:light grey', s=1)
    
    for label_id in new_labels:
        RR, CC = np.where(labeled_mask==label_id)[0], np.where(labeled_mask==label_id)[1]
        c1, r1, c2, r2 = endpoints_skeleton(label_id, labeled_mask, neighbor_count)
            
        if plot_verbose:
            plt.scatter(CC, RR,s=1, label='line {}'.format(label_id))
            plt.scatter([c1,c2],[r1,r2],s=2,color='r')

        if (c1==c2):
            labeled_mask[labeled_mask==label_id] = 0
            continue
            
        if abs(np.degrees(np.arctan2(r2-r1,c2-c1))%180 - 90)>0.5:
            labels.append(label_id)
            c1s.append(c1)
            c2s.append(c2)
            r1s.append(r1)
            r2s.append(r2)
            length = np.sqrt((c2-c1)**2+(r2-r1)**2)
            dc = np.abs(c2-c1)
            dr = np.abs(r2-r1)
            if dc>dr:
                coefficients = np.polyfit(CC, RR, 1)
                ms.append(coefficients[0])
                bs.append(coefficients[1])
            else:
                coefficients = np.polyfit(RR,CC,1)
                ms.append(1/coefficients[0])
                bs.append(-1*coefficients[1]/coefficients[0])
        else:
            labeled_mask[labeled_mask==label_id] = 0
         
    if plot_verbose:
        plt.legend()
        plt.show()
            
    new_labels = labels
            
    df0 = pd.DataFrame(np.array([labels, ms, bs, c1s, c2s, r1s, r2s]).T,
                                 columns=['label','slope', 'b', 'c1', 'c2', 'r1', 'r2'])
    df0 = df0.sort_values('slope').reset_index(drop=True)
    

    dindex, numlines = collect_segments_ndimage(sub=sub0, line_data=df0, max_R=max_R, 
                                            plot=plot_verbose, plotroot=plotroot,save=save)
    df0['line_id'] = dindex

    if numlines==0:  # if lines are vertical e.g., they might have been discarded
        if save:
            dat = pd.DataFrame([0],columns=['line_id'])
            dat.to_parquet(outputfile, engine='pyarrow', compression='snappy') 
        if plot_final:
            min_value, max_value = zscale.get_limits(sub0)
            subm = sub0.copy()
            subm[mask==1] = -700
            fig, ax = plt.subplots(1,3,figsize=(15,5))
            ax[0].imshow(sub0, vmin=min_value, vmax=max_value, origin='lower', cmap='gray')
            ax[1].imshow(subm, vmin=min_value, vmax=max_value, origin='lower', cmap='gray')
            ax[2].scatter(pixels[:,0],pixels[:,1],s=1,color='xkcd:light grey')
            ax[0].set_title('image {}'.format(subfile.split('/')[-1].split('-sub')[0]))
            ax[2].set_title('number of trails: 0')
            ax[2].set_xlim(0,2048)
            ax[2].set_ylim(0,2048)
            ax[2].set_aspect('equal')
            if save:
                plt.savefig(plotroot+'-trails.png', bbox_inches='tight',dpi=300)
            plt.close()
        return

    # make line-labeled_skeletonized_mask
    labeled_line_mask = labeled_mask.copy()
    for i, label_id in enumerate(new_labels):
        if label_id in df0.label.values:
            labeled_line_mask[labeled_mask==label_id] = df0.loc[df0.label==label_id].line_id

    labeled_full_line_mask = new_watershed(labeled_line_mask, mask)
    
    #plt.figure()
    #for line_id in np.unique(df0.line_id):
    #    plt.scatter(np.where(labeled_full_line_mask==line_id)[1],
    #                np.where(labeled_full_line_mask==line_id)[0],s=1,label='line {}'.format(line_id))
    #plt.legend()
    #plt.show()

    # make line-labeled full mask
    #labeled_full_line_mask = new_watershed(labeled_mask, mask)

    #old_labeled_full_line_mask = watershed(-mask.astype(float), markers=labeled_mask, mask=mask)
    
    #for i, label_id in enumerate(new_labels):
    #    if label_id in df0.label.values:
    #        labeled_full_line_mask[labeled_full_line_mask==label_id] = df0.loc[df0.label==label_id].line_id
    
    #  Now time to find out if the lines with gaps are all one line or separate lines:
    sub, N_bg, mean_bg, std_bg, sigma_bg = median_filter_cpu_bg_stats(sub0, df0, mask, 
                                                                      filter_radius=filter_radius)
    
    traillist = pd.DataFrame()

    line_numbers = np.unique(dindex)
    line_numbers_to_process = list(line_numbers)
    new_line_id = np.max(line_numbers_to_process) + 1

    
    while len(line_numbers_to_process) > 0:
        line_id = line_numbers_to_process.pop(0)
        
        #print('\nline_id: ', line_id, 'line_numbers_to_process: ', line_numbers_to_process)
        indices_all = np.where(labeled_full_line_mask==line_id)
        RR_all, CC_all = indices_all[0], indices_all[1]
        sort_idx_all = np.argsort(CC_all)
        RR_all = RR_all[sort_idx_all]
        CC_all = CC_all[sort_idx_all]
        
        indices_sk = np.where(labeled_line_mask==line_id)
        RR_sk, CC_sk = indices_sk[0], indices_sk[1]
        sort_idx_sk = np.argsort(CC_sk)
        RR_sk = RR_sk[sort_idx_sk]
        CC_sk = CC_sk[sort_idx_sk]

        if plot_verbose:
            plt.figure()
            plt.scatter(pixels[:,0],pixels[:,1], s=1,color='xkcd:light grey')
            plt.scatter(CC_all,RR_all, s=1)
            plt.title(line_id)
    
        ngaps = len(df0.loc[df0.line_id==line_id])-1

        if (ngaps > 0):
            #print('ngaps = ', ngaps)

            # basic fit to get periodicity
            length_approx = np.sqrt((RR_sk.max()-RR_sk.min())**2 + (CC_sk.max()-CC_sk.min())**2)
            rr_fit, cc_fit, c1, c2, r1, r2, coefficients = fit_coords(RR_all, CC_all, length_approx)
            dat_for_bounds = rolling_mean(rr_fit, cc_fit, r1, c1, sub, w=50)
            CV, periodic = check_periodicity(dat_for_bounds)

            # if not periodic, go through gaps
            if not periodic:
                if plot_verbose:
                    plt.figure()
                    plt.scatter(pixels[:,0],pixels[:,1], s=1,color='xkcd:light grey')
                    plt.scatter(CC_all,RR_all, s=1)
                    plt.title(line_id)
                    
                gap_results = region_gaps(rr_fit, cc_fit, line_id, sub, labeled_line_mask, labeled_full_line_mask,mean_bg, std_bg,
                                        new_line_id, line_numbers, line_numbers_to_process,gap=gap,plot=plot_verbose)
                labeled_full_line_mask, labeled_line_mask, line_numbers, line_numbers_to_process, new_line_id, num_new_lines = gap_results

                # if some of the gaps were real, refine the pixels associated with this line_id
                if num_new_lines>0:
                    indices_all = np.where(labeled_full_line_mask==line_id)
                    RR_all, CC_all = indices_all[0], indices_all[1]
                    sort_idx_all = np.argsort(CC_all)
                    RR_all = RR_all[sort_idx_all]
                    CC_all = CC_all[sort_idx_all]
                    
                    indices_sk = np.where(labeled_line_mask==line_id)
                    RR_sk, CC_sk = indices_sk[0], indices_sk[1]
                    sort_idx_sk = np.argsort(CC_sk)
                    RR_sk = RR_sk[sort_idx_sk]
                    CC_sk = CC_sk[sort_idx_sk]
            #else:
            #    print('periodic')

            # fit final coordinates of the line
            c1, r1, c2, r2 = endpoints_skeleton(line_id, labeled_line_mask, neighbor_count)
            cc_fit, rr_fit, coeff_p, coeff_l, curv_info, length_approx, fit_direction = fit_coords_final_line(RR_all,CC_all,c1, r1, c2, r2, max_R)
            
            if length_approx > 200:
                dat_for_bounds = rolling_mean(rr_fit, cc_fit, r1, c1, sub, w=50) #####r1_p, c1_p, sub, w=50)
                    
            if plot_verbose:
                plt.scatter(CC_sk, RR_sk, color='k', s=1)
                plt.scatter([c1,c2],[r1,r2],color='r')
                plt.show()
                
        else:  # if no gaps, fit all coordinates as one line and check periodicity from there
            c1, r1, c2, r2 = endpoints_skeleton(line_id, labeled_line_mask, neighbor_count)
            cc_fit, rr_fit, coeff_p, coeff_l, curv_info, length_approx, fit_direction = fit_coords_final_line(RR_all,CC_all,c1, r1, c2, r2, max_R)
            
            dat_for_bounds = rolling_mean(rr_fit, cc_fit, r1, c1, sub, w=50)
            CV, periodic = check_periodicity(dat_for_bounds)

        # get fraction of mask and skeletonized mask pixels that are 0 so we can check  
        # for false positives from stray starlight
        zero_frac_all = len(sub[RR_all,CC_all]==0.)/len(sub[RR_all,CC_all])
        zero_frac_sk = len(sub[RR_sk,CC_sk]==0.)/len(sub[RR_sk,CC_sk])

        coeff_a, coeff_b, coeff_c = coeff_p
        slope, y_int = coeff_l
        dX_fits, dev_p_max, dev_p_allowed, R_approx = curv_info

        if length_approx > 200:
            c1_new, r1_new, c2_new, r2_new = find_bounds(rr_fit, cc_fit, r1, c1, r2, c2,
                                                         length_approx, dat_for_bounds, sub,
                                                         mean_bg, sigma_bg, w=50, nsig=5, 
                                                         plot=False)

            line_data = {'line_id':line_id,'length':np.sqrt((r2_new-r1_new)**2+(c2_new-c1_new)**2),
                         'c1':c1_new,'c2':c2_new,'r1':r1_new,'r2':r2_new,'coeff_a':coeff_a,
                         'coeff_b':coeff_b, 'coeff_c':coeff_c,'slope': slope,'y_int':y_int,
                         'fit_direction':fit_direction, 'dX_fits':dX_fits,'dev_p_max':dev_p_max,
                         'dev_p_allowed':dev_p_allowed, 'R_approx':R_approx,
                         'mask_all_c':CC_all.tolist(), 'mask_all_r':RR_all.tolist(),
                         'mask_sk_c':CC_sk.tolist(), 'mask_sk_r':RR_sk.tolist(),
                        'CV':CV, 'zero_frac_all':zero_frac_all, 'zero_frac_sk':zero_frac_sk}

        else:
            line_data = {'line_id':line_id,'length':length_approx,
                         'c1':c1,'c2':c2,'r1':r1,'r2':r2,'coeff_a':coeff_a,
                         'coeff_b':coeff_b, 'coeff_c':coeff_c,'slope': slope,'y_int':y_int,
                         'fit_direction':fit_direction, 'dX_fits':dX_fits,'dev_p_max':dev_p_max,
                         'dev_p_allowed':dev_p_allowed, 'R_approx':R_approx,
                         'mask_all_c':CC_all.tolist(), 'mask_all_r':RR_all.tolist(),
                         'mask_sk_c':CC_sk.tolist(), 'mask_sk_r':RR_sk.tolist(),
                        'CV':CV, 'zero_frac_all':zero_frac_all, 'zero_frac_sk':zero_frac_sk}

            
        line_df = pd.DataFrame([line_data])
        traillist = pd.concat([traillist, line_df], ignore_index=True)
        
    is_significant, ratios, SNRs = test_significance(traillist, sub, mean_bg, std_bg)

    traillist['SNR'] = SNRs
    traillist['significance'] = is_significant
    traillist['significance_ratio']  = ratios

    traillist = find_edgetrails(sub, traillist, edge_threshold=5)

    if plot_final:
        min_value, max_value = zscale.get_limits(sub0)
        subm = sub0.copy()
        subm[mask==1] = -700
        fig, ax = plt.subplots(1,3,figsize=(15,5))
        ax[0].imshow(sub0, vmin=min_value, vmax=max_value, origin='lower', cmap='gray')
        ax[1].imshow(subm, vmin=min_value, vmax=max_value, origin='lower', cmap='gray')
        ax[2].scatter(pixels[:,0],pixels[:,1],s=2,color='xkcd:light grey')
        for i, line_id in enumerate(line_numbers):
            ax[2].scatter(traillist.loc[traillist.line_id==line_id][['c1','c2']],
                          traillist.loc[traillist.line_id==line_id][['r1','r2']],
                          s=5, color='black')
            cm = ax[2].scatter(traillist.loc[traillist.line_id==line_id]['mask_all_c'].values[0],
                          traillist.loc[traillist.line_id==line_id]['mask_all_r'].values[0],s=1,
                         color=plt.colormaps['tab20'](line_id%20),
                              label='{}:{},{:.1f},{:.1f}'.format(line_id,is_significant[i],
                                                            ratios[i],SNRs[i]))
        plt.legend(markerscale=5,bbox_to_anchor=(1.6, 1), loc='upper right')
        ax[0].set_title('image {}'.format(subfile.split('/')[-1].split('-sub')[0]))
        ax[2].set_title('number of trails: {}'.format(len(traillist)))
        ax[2].set_xlim(0,2048)
        ax[2].set_ylim(0,2048)
        ax[2].set_aspect('equal')
        if save:
            plt.savefig(plotroot+'-trails.png', bbox_inches='tight',dpi=200)
            plt.close()
        else:
            plt.show()

    if save:
        traillist.to_parquet(outputfile, engine='pyarrow', compression='snappy') 

    return df0
    
