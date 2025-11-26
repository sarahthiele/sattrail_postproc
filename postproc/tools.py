import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import ZScaleInterval
from astropy.io import fits
from skimage.segmentation import watershed
from sklearn.decomposition import PCA
from scipy import ndimage

zscale = ZScaleInterval()

def plot_fits(fitsfile, fitsdata, plotpath='', save=False, data=True):
    if not data:
        fitsdata = fits.getdata(fitsfile)

    vmin, vmax = zscale.get_limits(fitsdata)

    plt.imshow(fitsdata, vmin=vmin, vmax=vmax, origin='lower', cmap='gray')

    if save:
        plt.savefig(plotpath+".png", dpi=200)
    plt.show()

    return

def new_watershed(labeled_mask, mask):
    rows, cols = np.where(mask > 0)
    if len(rows) > 0:
        r_min, r_max = rows.min(), rows.max() + 1
        c_min, c_max = cols.min(), cols.max() + 1
        
        # Add small padding
        pad = 5
        r_min = max(0, r_min - pad)
        c_min = max(0, c_min - pad)
        r_max = min(2048, r_max + pad)
        c_max = min(2048, c_max + pad)
        
        # Run watershed only on cropped region
        mask_crop = mask[r_min:r_max, c_min:c_max]
        markers_crop = labeled_mask[r_min:r_max, c_min:c_max]
        labeled_crop = watershed(-mask_crop.astype(float),
                                 markers=markers_crop,
                                 mask=mask_crop)
        
        # Place back in full-size array
        labeled_full_mask = np.zeros_like(mask, dtype=np.int32)
        labeled_full_mask[r_min:r_max, c_min:c_max] = labeled_crop
    else:
        labeled_full_mask = np.zeros_like(mask, dtype=np.int32)

    return labeled_full_mask

def endpoints_PCA(r, c):
    # Get principal direction
    coords = np.column_stack([c, r])
    pca = PCA(n_components=1)
    pca.fit(coords)

    # Project all points onto principal axis
    projections = pca.transform(coords).flatten()

    # Find endpoints by min/max projection
    idx_start = np.argmin(projections)
    idx_end = np.argmax(projections)

    #c1, r1 = c[idx_start], r[idx_start]
    #c2, r2 = c[idx_end], r[idx_end]
    
    # Take points near each extreme and find their centroid
    threshold = 10  # pixels from extreme

    # Start end
    min_proj = projections.min()
    near_start = projections < (min_proj + threshold)
    c1 = np.median(c[near_start])
    r1 = np.median(r[near_start])

    # End end
    max_proj = projections.max()
    near_end = projections > (max_proj - threshold)
    c2 = np.median(c[near_end])
    r2 = np.median(r[near_end])
    
    if c2 < c1:
        return c2, r2, c1, r1
    
    else:
        return c1, r1, c2, r2
        
def endpoints_skeleton(line_id, labeled_line_mask, neighbor_count):
    endpoints = np.argwhere((labeled_line_mask == line_id) & (neighbor_count == 1))
    c1 = np.min(endpoints[:,1])
    r1 = endpoints[:,0][np.argmin(endpoints[:,1])]
    c2 = np.max(endpoints[:,1])
    r2 = endpoints[:,0][np.argmax(endpoints[:,1])]
    
    if c2 < c1:
        return c2, r2, c1, r1
    else:
        return c1, r1, c2, r2
        
def get_neighbors(mask):
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = ndimage.convolve(mask.astype(np.uint8), kernel, mode='constant')
    return neighbor_count
