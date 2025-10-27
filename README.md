# sattrail_postproc
Postprocessing for sattrails detections

## Example usage:

```
from new_postproc import *
subroot = '1-557199_63'
datapath = '/nfs/php2/ar3/P/HP1/REDUCTION/SUB/1-20250601/ihu63/'
detpath = '/data/USERS/sthiele/DETSAT/1-20250601/ihu63/'
df0, trails, newmask, cpoints, rpoints = postproc(datapath, subroot, detpath, skeleton=False, SAVE=True, PLOT=False, progressive=True, gpu=False)
```

then you can use the dateframe `df0` to fiddle with the collect_segments() and find_bounds() functions, or here is a sample from the first bit of new_postproc.postproc() you could run on its own, I'm still working out the skeletonize thing because it messes up the width and bounds calculation, but it does help make the phl much cleaner:

```
skeleton = <True/False>
progressive = <True/False>

subfile = '{}{}-sub.fits'.format(datapath, subroot)
detectionfile = '{}{}-detection.json'.format(detpath, subroot)

sub0 = read_fits_file(subfile)

with open(detectionfile, 'r') as f:
    data = json.load(f)
pixels = np.array(data['mask'])
mask = np.zeros((2048,2048))
mask[pixels[:,1],pixels[:,0]] += 1

if skeleton:
    maskphl = skeletonize(mask.astype(np.uint8)).astype('>f4')
else:
    maskphl = mask

if progressive:
    lines = progressive_hough_transform(maskphl, min_line_length=10, min_threshold=10,
                                initial_threshold=200, initial_gap=50)
else:
    lines = phl(maskphl, threshold=10, line_length=10, line_gap=50)
    
df0, slopes, bs = get_line_data(lines, PLOT=True)
dindex, numlines = collect_segments(sub0, df0, PLOT=True)
df0['linenum'] = dindex
```

## Oct 26 2025 update:
__Changes needed (see further notes below):__
1. sometimes line 494 in new_postproc.py (imax = imaxs[(imaxs>=allgaps[i])&(imaxs<=allgaps[i+1])][0] in the find_bounds() function) throws an error and then I rerun the script and its fine and I can't figure out why. In general speed up find_bounds() and make sure it doesn't give bounds that are smaller than the original mask bounds. Also I just don't love how this function works in general and would love for it to be doing things in a smarter way
2. speed up collect_segments() and make it more elegant
3. modify the find_width calculation that could work if the mask has been skeletonized (basically change the binary inflation and fitting)

__To add:__
for identifying sats and photometry:

1. working on merge_ihus.py to append all the output files together that hypothetically share lines that go off their edges. Right now the postproc() function results have in/out ("IO") columns that have a "1" if they have lines going off that side. IOframe=1 if lines go off any edges, and then IOL means a line goes off the left side, IOR off the right, IOT for top, IOB for bottom. I have the ihu table in identify_sats directory that allows you to find which ihus are adjacent ([0] means no adjacent ihu), so you should be able to just read that in and append the output files, then find lines that are within some tolerance distance from each other.
2. Better managing the sat_id output and what to do once I have that information

Working on __new_postproc.py__, which has option to use a skeletonized mask for the probabilistic hough lines (phl) instead of the entire detection mask. This might lead to more fine-tuned slope detection, but for lines that are almost horizontal (like GEO sats) it might create a false slope by thinning the line wonkily. In this case scikit-learn's thin function seems work better, but it's a lot slower than skeletonize, so this might be futile. TBD.

new_postproc.py also has a cpu-enabled median filter subtraction that is much faster (~100x) than the old cpu one, and also faster than the current gpu implementation (although the cpu one only does the rolling median in bubbles centered on the mask lines to minimize computation time, so this could be done for the gpu one also. But it looks like postprocessing will have to be done on a cpu since our current computing cluster has long wait times so we need to optimize resource management. 

Right now the new_postproc script takes about 10s to run, which needs to be cut down further, since we still need to add on the satellite identification part using sat_id.

Two functions that are both slow and don't work fantastically within new_postproc are __collect_segments()__ and __find_bounds()__. The former takes all of the lines identified by the phl and determines which segments belong to the same line. This has the problem where you need to dictate that disjoint segments might be part of the same line, which also might have some curvature, but they might also just be a bunch of short trails (like for geo sats) in a row. In the second case we then use find_gaps() and find_bounds() to determine where the gaps are and if there is significant brightness between them that they're likely part of the same trail. find_bounds() needs work because it is difficult to make an algorithm that works universally. 

Sidenote: I modified my local version of sattrails such that it works on .fz files so no need to funpack everything now woohoo




