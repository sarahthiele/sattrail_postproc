# sattrail_postproc
Postprocessing for sattrails detections

## Oct 26 2025 update:
Working on new_postproc.py, which has option to use a skeletonized mask for the probabilistic hough lines (phl) instead of the entire detection mask. This might lead to more fine-tuned slope detection, but for lines that are almost horizontal (like GEO sats) it might create a false slope by thinning the line wonkily. In this case scikit-learn's thin function seems work better, but it's a lot slower than skeletonize, so this might be futile. TBD.

new_postproc.py also has a cpu-enabled median filter subtraction that is much faster (~100x) than the old cpu one, and also faster than the current gpu implementation (although the cpu one only does the rolling median in bubbles centered on the mask lines to minimize computation time, so this could be done for the gpu one also. But it looks like postprocessing will have to be done on a cpu since our current computing cluster has long wait times so we need to optimize resource management. 

Right now the new_postproc script takes about 10s to run, which needs to be cut down further, since we still need to add on the satellite identification part using sat_id.

Two functions that are both slow and don't work fantastically within new_postproc are collect_segments() and find_bounds(). The former takes all of the lines identified by the phl and determines which segments belong to the same line. This has the problem where you need to dictate that disjoint segments might be part of the same line, which also might have some curvature, but they might also just be a bunch of short trails (like for geo sats) in a row. In the second case we then use find_gaps() and find_bounds() to determine where the gaps are and if there is significant brightness between them that they're likely part of the same trail. find_bounds() needs work because it is difficult to make an algorithm that works universally. 

Sidenote: I modified my local version of sattrails such that it works on .fz files so no need to funpack everything now woohoo

__Changes needed:__
1. speed up collect_segments() and make it more elegant
2. speed up find_bounds() and make sure it doesn't give bounds that are smaller than the original mask bounds
3. modify the find_width calculation that could work if the mask has been skeletonized (basically change the binary inflation and fitting)
