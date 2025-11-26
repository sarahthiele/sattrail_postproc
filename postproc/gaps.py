##########################################################################################
# Different methods of dealing with gaps in satellite trails. Which one you use depends
# on the postproc approach (phl vs region labeling etc.).
##########################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.spatial import cKDTree
from tools import new_watershed
from skimage.draw import line
from tools import get_neighbors, endpoints_skeleton



def find_gaps(rr_fit, cc_fit, RR, CC, r1, c1, sub, gap=2, w=50):
    datgap = pd.DataFrame(np.array([CC,RR,np.sqrt((CC-c1)**2+(RR-r1)**2),sub[RR,CC]]).T,
                    columns=['c','r','dx','h']).sort_values('dx').reset_index(drop=True)
    datgap['sep'] = np.append(0,datgap.dx.values[1:]-datgap.dx.values[:-1])
    dx_all = np.sign(cc_fit-c1)*np.sqrt((cc_fit-c1)**2+(rr_fit-r1)**2)
    
    gapsi = np.sort(np.append(datgap.loc[datgap.sep>gap].index.values-1,
                              datgap.loc[datgap.sep>gap].index.values))
    ngaps = len(datgap.loc[datgap.sep>gap])

    gaps_loc_cc_fit = np.array([])
    for i in range(len(gapsi)):
        gaps_loc_cc_fit = np.append(gaps_loc_cc_fit,
                                np.argmin(np.abs(dx_all-datgap.iloc[gapsi].dx.values[i])))

    if ngaps>0:
        for i in range(0,ngaps+1,2):
            left_of_gap = gaps_loc_cc_fit[i]
            right_of_gap = gaps_loc_cc_fit[i+1]
            if left_of_gap==right_of_gap:
                ngaps -= 1
                gaps_loc_cc_fit = np.delete(gaps_loc_cc_fit, [i,i+1])
                
    return gaps_loc_cc_fit.astype(int), ngaps

def fill_gaps(sub, line_id, mask, RR_all, CC_all, RR_sk, CC_sk, ngaps,
              gaps_loc_cc_fit, dat_for_bounds, N_bg, mean_bg, std_bg, sigma_bg,
              plot=False):
    mask_indices = np.where(mask == 1)
    maskr, maskc = mask_indices[0], mask_indices[1]
    
    # Create sets - exclude the trail coordinates
    trail_coords = set(zip(CC_all, RR_all))  # Coords to exclude
    mask_coords_all = set(zip(maskc, maskr))  # All mask coords
    mask_coords = mask_coords_all - trail_coords  # Remove trail coords
    
    # Filter dat_for_gaps using set membership (much faster than merge + drop loop)
    dat_coords = set(zip(dat_for_bounds['c'].astype(int),
                         dat_for_bounds['r'].astype(int)))
    coords_to_keep = dat_coords - mask_coords
    
    # Create boolean mask for filtering
    keep_mask = np.array([
        (c, r) in coords_to_keep
        for c, r in zip(dat_for_bounds['c'].astype(int),
                       dat_for_bounds['r'].astype(int))
    ])
    
    dat_for_gaps = dat_for_bounds[keep_mask]

    N_trail = len(RR_sk)
    mean_trail = np.mean(sub[RR_sk,CC_sk])
    std_trail = np.std(sub[RR_sk,CC_sk])
    sigma_trail = std_bg / np.sqrt(N_trail)

    is_gap_real = []
    for i in range(0,2*ngaps,2):
        left_of_gap = gaps_loc_cc_fit[i]
        right_of_gap = gaps_loc_cc_fit[i+1]
        
        gap_data = pd.merge(dat_for_gaps[['c','r']],
                            dat_for_bounds.iloc[left_of_gap:right_of_gap],
                            on=['c','r'],how='inner')

        #print(dat_for_bounds.iloc[left_of_gap].c, dat_for_bounds.iloc[right_of_gap].c)

        if len(gap_data)==0:
            #print('no gap data')
            is_gap_real.append(0)
            continue

        if plot:
            plt.figure()
            plt.plot(dat_for_bounds.c.values, dat_for_bounds.rollmean.values)
            plt.plot(gap_data.c.values, gap_data.rollmean.values)
            plt.show()

        zero_frac = len(gap_data.loc[gap_data.h==0])/len(gap_data)
        #print('zero_frac: ', zero_frac)
        if zero_frac > 0.5:
            is_gap_real.append(0)
            continue
        
        N_gap = len(gap_data)
        mean_gap = np.mean(gap_data.h)
        std_gap = np.std(gap_data.h)
        sigma_gap = std_bg / np.sqrt(N_gap)
        
        SNR_trail = (mean_trail - mean_bg) / sigma_trail
        SNR_gap = (mean_gap - mean_bg) / sigma_gap
        
        SNR_gap_vs_trail = np.abs((mean_trail - mean_gap) / std_bg) #np.sqrt(sigma_trail**2 + sigma_gap**2))

        #print('N_trail: ', N_trail, 'N_gap: ', N_gap,)
        #print('zero_frac: ', zero_frac, 'mean_bg: ', mean_bg, 'std_gap: ', std_gap, 'mean_gap: ', mean_gap)
        #print('trail: ', SNR_trail, 'gap: ', SNR_gap, 'diff: ', SNR_gap_vs_trail)

        if (SNR_gap > 8)|((SNR_gap_vs_trail<1.5)):
            is_gap_real.append(0)
        else:
            is_gap_real.append(1)
    
    return np.array(is_gap_real)
    
def closest_endpoints(c1_i, r1_i, c2_i, r2_i, c1_j, r1_j, c2_j, r2_j):
    d_1111 = np.sqrt((c1_i-c1_j)**2+(r1_i-r1_j)**2)
    d_1212 = np.sqrt((c1_i-c2_j)**2+(r1_i-r2_j)**2)
    d_2121 = np.sqrt((c2_i-c1_j)**2+(r2_i-r1_j)**2)
    d_2222 = np.sqrt((c2_i-c2_j)**2+(r2_i-r2_j)**2)
    
    darr = np.array([d_1111,d_1212,d_2121,d_2222])
    idx = np.argmin(darr)
    
    if idx==0:
        if c1_i < c1_j:
            return darr[idx], c1_i, r1_i, c1_j, r1_j
        else:
            return darr[idx], c1_j, r1_j, c1_i, r1_i
    if idx==1:
        if c1_i < c2_j:
            return darr[idx], c1_i, r1_i, c2_j, r2_j
        else:
            return darr[idx], c2_j, r2_j, c1_i, r1_i
    if idx==2:
        if c2_i < c1_j:
            return darr[idx], c2_i, r2_i, c1_j, r1_j
        else:
            return darr[idx], c1_j, r1_j, c2_i, r2_i
    if idx==3:
        if c2_i < c2_j:
            return darr[idx], c2_i, r2_i, c2_j, r2_j
        else:
            return darr[idx], c2_j, r2_j, c2_i, r2_i
    
def region_gaps(rr_fit, cc_fit, line_id, sub, labeled_line_mask, labeled_full_line_mask,mean_bg,
                std_bg, new_line_id, line_numbers, line_numbers_to_process, gap, plot=False):

    calculate_trail_data = True

    mtrail = np.zeros((2048,2048))
    #mtrail[labeled_full_line_mask==line_id] = 1
    mtrail[labeled_line_mask==line_id] = 1
    neighbor_count = get_neighbors(mtrail)
    
    import pdb
    #pdb.set_trace()
    
    rtrail, ctrail = np.where(mtrail==1)[0], np.where(mtrail==1)[1]

    labeled_trail_mask, num_labels = ndimage.label(mtrail, structure=np.ones((3,3)))

    new_line_ids = [line_id]

    num_new_lines = 0
    for i in range(num_labels-1):
        region1_coords = np.argwhere(labeled_trail_mask == i+1)
        region2_coords = np.argwhere(labeled_trail_mask == i+2)
        
        ##### new
        c1_i, r1_i, c2_i, r2_i = endpoints_skeleton(i+1, labeled_trail_mask, neighbor_count)
        c1_j, r1_j, c2_j, r2_j = endpoints_skeleton(i+2, labeled_trail_mask, neighbor_count)
        dmin, c1, r1, c2, r2 = closest_endpoints(c1_i, r1_i, c2_i, r2_i, c1_j, r1_j, c2_j, r2_j)
                
        dx_fit = np.sign(cc_fit-c1)*np.sqrt((cc_fit-c1)**2+(rr_fit-r1)**2)
        cgap_fit = cc_fit[(dx_fit>0)&(dx_fit<dmin)]
        rgap_fit = rr_fit[(dx_fit>0)&(dx_fit<dmin)]
        if len(rgap_fit) == 0:
            test_fit = False
        else:
            test_fit = True
            dr = np.abs(rgap_fit.max()-rgap_fit.min())
            dc = np.abs(cgap_fit.max()-cgap_fit.min())
            if dc > dr:
                # y = mx + b
                slope_fit, b_fit = np.polyfit(cgap_fit, rgap_fit, 1)
            else:
                # x = my + b
                # y = x/m - b/m
                m, b = np.polyfit(rgap_fit, cgap_fit, 1)
                slope_fit = 1/m
        ##### end new
        
        # Find closest points
        tree = cKDTree(region2_coords)
        distances, indices = tree.query(region1_coords, k=1)
        min_distance = distances.min()

        # If distance > threshold, it's a real gap (not just a crossing removal)
        if min_distance < gap:  # Adjust threshold - crossings should be ~1-2 pixels
            #print('line {} has min_distance: {}'.format(line_id, min_distance))
            new_line_ids.append(line_id)
            if plot:
                plt.figure()
                plt.scatter(np.where(labeled_trail_mask==i+1)[1], np.where(labeled_trail_mask==i+1)[0],zorder=-10)
                plt.scatter(np.where(labeled_trail_mask==i+2)[1], np.where(labeled_trail_mask==i+2)[0],zorder=-10)
                plt.scatter(c1p, r1p, c='r', marker='D', s=50,zorder=10)
                plt.scatter(c2p, r2p, c='r', marker='D', s=50,zorder=10)
                plt.title(line_id)
                plt.show()
            continue
            
        min_idx_1 = np.argmin(distances)
        min_idx_2 = indices[min_idx_1]
        point1 = tuple(region1_coords[min_idx_1])
        point2 = tuple(region2_coords[min_idx_2])

        r1p = point1[0]
        c1p = point1[1]
        r2p = point2[0]
        c2p = point2[1]
        
        ##### new
        if test_fit:
            if c2p!=c1p:
                slope_gap = (r2p-r1p)/(c2p-c1p)
                theta = np.degrees(np.arctan(np.abs((slope_gap-slope_fit)/(1+slope_gap*slope_fit))))
            else:
                theta = 90
            if theta < 45:
                cgap = cgap_fit.astype(int)
                rgap = rgap_fit.astype(int)
                coords_gap = np.unique(np.vstack([rgap,cgap]).T,axis=0)
                cgap = coords_gap[:,1]
                rgap = coords_gap[:,0]
            else:
                coords_gap = line(r2p,c2p,r1p,c1p)
                cgap, rgap = coords_gap[1], coords_gap[0]
        else:
            coords_gap = line(r2p,c2p,r1p,c1p)
            cgap, rgap = coords_gap[1], coords_gap[0]
        ##### end new

        if plot:
            plt.figure()
            plt.scatter(np.where(labeled_trail_mask==i+1)[1], np.where(labeled_trail_mask==i+1)[0],zorder=-10)
            plt.scatter(np.where(labeled_trail_mask==i+2)[1], np.where(labeled_trail_mask==i+2)[0],zorder=-10)
            plt.scatter(cgap, rgap,s=1)
            plt.scatter(c1p, r1p, c='r', marker='D', s=50,zorder=10)
            plt.scatter(c2p, r2p, c='r', marker='D', s=50,zorder=10)
            plt.title(line_id)
            plt.show()
        
        #####coords_gap = line(r2p,c2p,r1p,c1p)
        #####cgap, rgap = coords_gap[1], coords_gap[0]

        in_mask = (labeled_full_line_mask[rgap, cgap] > 0)
        not_in_mask = (labeled_full_line_mask[rgap, cgap] == 0)
        #print("frac of gap that's in another line: ", np.sum(in_mask)/(np.sum(in_mask)+np.sum(not_in_mask)))
        rgap_clean = rgap[not_in_mask]
        cgap_clean = cgap[not_in_mask]
        gap_data = sub[rgap_clean,cgap_clean]
        gap_data = gap_data[gap_data!=0.0]

        if len(gap_data)<=gap:
            print('no gap data')
            new_line_ids.append(line_id)
            continue
        elif np.sum(in_mask)/(np.sum(in_mask)+np.sum(not_in_mask)) > 0.5:
            print('line intersection')
            new_line_ids.append(line_id)
            continue

        zero_frac = len(gap_data[gap_data==0])/len(gap_data)
        if zero_frac > 0.25:
            print('zero_frac: ', zero_frac)
            new_line_ids.append(line_id)
            continue

        if calculate_trail_data:
            #trail = sub[labeled_full_line_mask==line_id]
            trail = sub[labeled_line_mask==line_id]
            N_trail = len(trail)
            mean_trail = np.mean(trail)
            sigma_trail = std_bg / np.sqrt(N_trail)
            SNR_trail = (mean_trail - mean_bg) / sigma_trail
            calculate_trail_data = False
            
        if len(gap_data)/len(trail)<=0.01:
            print('gap is less than 1% of trail')
            new_line_ids.append(line_id)
            continue
        
        #print('gap data: ', gap_data)
        N_gap = len(gap_data)
        mean_gap = np.mean(gap_data)
        std_gap = np.std(gap_data)
        sigma_gap = std_bg / np.sqrt(N_gap)
        SNR_gap = (mean_gap - mean_bg) / sigma_gap
        
        SNR_gap_vs_trail = np.abs((mean_trail - mean_gap) / std_bg)
        print('Ntrail: ', N_trail, 'N_gap: ', N_gap)
        print('trail: ', SNR_trail, 'gap: ', SNR_gap, 'diff: ', SNR_gap_vs_trail)

        if (SNR_gap > 7)|((SNR_gap_vs_trail<1.4)):
            new_line_ids.append(line_id)
            print('no gap')
        else:
            print('gap')
            num_new_lines += 1
            new_line_ids.append(new_line_id)
            line_numbers = np.append(line_numbers, new_line_id)
            line_numbers_to_process.append(new_line_id)
            new_line_id += 1

    if num_new_lines > 0:
    
        # Get the full mask for this line
        full_mask_this_line = (labeled_full_line_mask == line_id)
        
        # Use watershed to propagate skeleton labels to full mask
        # labeled_trail_mask acts as markers
        labeled_full_segments = new_watershed(labeled_trail_mask, full_mask_this_line)
        
        # Now relabel both masks
        for i in range(2, num_labels+1):
            if new_line_ids[i-1]!=line_id:
                # Update full mask using watershed result
                labeled_full_line_mask[labeled_full_segments==i] = new_line_ids[i-1]
                # Update skeleton mask
                labeled_line_mask[labeled_trail_mask==i] = new_line_ids[i-1]
                
    return labeled_full_line_mask, labeled_line_mask, line_numbers, line_numbers_to_process, new_line_id, num_new_lines

