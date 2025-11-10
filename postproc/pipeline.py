from postproc import postproc
import numpy as np
import argparse
import os
print('starting!')

parser = argparse.ArgumentParser()
parser.add_argument("subroot", type=str, help="observation night in format 1-YYYYMMDD")
parser.add_argument("ihu", type=int, help="ihu number")
parser.add_argument("--subpath", type=str, help="directory of subtracted data", default="/nfs/php2/ar5/P/HP1/REDUCTION/SUB")
parser.add_argument("--detpath", type=str, help="path to sattrails detection files", default="/nfs/php2/ar0/P/PROJ/sthiele/PROJDATA/DETSAT")
parser.add_argument("-o", "--output", type=str, help="directory for output file", default="/nfs/php2/ar0/P/PROJ/sthiele/PROJDATA/DETSAT")
parser.add_argument("--plotdir", type=str, help="directory for plots", default="/nfs/php2/ar0/P/PROJ/sthiele/PROJDATA/PLOTS")
parser.add_argument("-p", "--plot", action="store_true", help="save plots")
parser.add_argument("--nosave", action="store_true", help="don't save output file")
parser.add_argument("--skeleton", action="store_true", help="use skeletonize function for phl")
parser.add_argument("--progressive", action="store_true", help="use progressive phl rather than default")
parser.add_argument("--maxr", type=int, default=int(1e5), help="max radius of curvature")
parser.add_argument("--gpu", action="store_true", help="use gpu for median filter calculation")
parser.add_argument("--filtr", type=int, default=10, help="radius for median filter window")
parser.add_argument("--nclose", type=int, default=10, help="closing radius for mask dilation")
parser.add_argument("--nhalf", type=int, default=1, help="number of half-widths for mask dilation")
parser.add_argument("--nsig", type=int, default=3, help="number of sigma for bound calculation")
parser.add_argument("--gap", type=int, default=2, help="number of pixel separation between segments to be considered a gap")

args = parser.parse_args()

# parameters:
SKELETON = args.skeleton
PROG = args.progressive
MAXR = args.maxr
GPU = args.gpu
FILTR = args.filtr
NCLOSE = args.nclose
NHALF = args.nhalf
NSIG = args.nsig
GAP = args.gap

print(f"skeleton: {SKELETON}")
print(f"progressive: {PROG}")
print(f"max_R: {MAXR}")
print(f"gpu: {GPU}")
print(f"filter_radius: {FILTR}")
print(f"nclose: {NCLOSE}")
print(f"nhalf: {NHALF}")
print(f"gap: {GAP}")

# data paths:
subroot = os.path.join(args.subroot, 'ihu{:02d}'.format(args.ihu))
subpath = os.path.join(args.subpath, subroot)
detpath = os.path.join(args.detpath, subroot)
outpath = os.path.join(args.output, subroot)
outpath = args.output
print('obs night is ', args.subroot, ', ihu is {}'.format(args.ihu))
print('sub data is in ', args.subpath)
print('sattrail detections is in ', detpath)
print('postproc files are in ', outpath)

# get data:
files = os.listdir(subpath)
subfiles = [os.path.join(subpath, item) for item in files if '-sub.fits' in item]
#detfiles = [os.path.join(detpath, subfile.split('/')[-1].split('-sub.fits')[0]+'-detection.json') for subfile in subfiles]
detfiles = [os.path.join(args.detpath, 'detection_'+subfile.split('/')[-1].split('-sub.fits')[0]+'.json') for subfile in subfiles]
outputfiles = [os.path.join(outpath, subfile.split('/')[-1].split('-sub.fits')[0]+'-sattrail_{}.hdf'.format(MAXR/1e5)) for subfile in subfiles]

PLOT = args.plot
if PLOT:
    #plotfiles = [os.path.join(args.plotdir, subroot, subfile.split('/')[-1].split('-sub.fits')[0]) for subfile in subfiles]
    if SKELETON:
        plotfiles = [os.path.join(args.plotdir, 'testing/skeleton_maxR_{}'.format(MAXR/1e5), subfile.split('/')[-1].split('-sub.fits')[0]) for subfile in subfiles]
    else:
        plotfiles = [os.path.join(args.plotdir, 'testing/maxR_{}'.format(MAXR/1e5), subfile.split('/')[-1].split('-sub.fits')[0]) for subfile in subfiles]
else:
    plotfiles = np.ones(len(subfiles))

if args.nosave:
    SAVE = False
else:
    SAVE = True

for i, subfile in enumerate(subfiles[:2]):
    print(subfile)
    print(detfiles[i])
    print(plotfiles[i])
    print(outputfiles[i])
    print('\n')

    df0, traillist, mask_master, cpoints, rpoints = postproc(subfiles[i], detfiles[i], outputfiles[i],
                                                                plotfiles[i], save=SAVE, plot=PLOT, skeleton=SKELETON,
                                                                progressive=PROG, gpu=GPU, max_R=MAXR, filter_radius=FILTR,
                                                                nclose=NCLOSE, nhalf=NHALF, nsig=NSIG, gap=GAP)
