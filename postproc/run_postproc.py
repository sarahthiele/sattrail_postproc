import postproc as pp
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("subroot", type=str, help="observation night in format 1-YYYYMMDD")
parser.add_argument("ihu", type=int, help="ihu number")
parser.add_argument("--subpath", type=str, help="directory of subtracted data", default="/nfs/php2/ar5/P/HP1/REDUCTION/SUB")
parser.add_argument("--detpath", type=str, help="path to sattrails detection files", default="/nfs/php2/ar0/P/PROJ/sthiele/PROJDATA/DETSAT")
parser.add_argument("-o", "--output", type=str, help="directory for output file", default="/nfs/php2/ar0/P/PROJ/sthiele/PROJDATA/DETSAT")
parser.add_argument("--plotdir", type=str, help="directory for plots", default="/nfs/php2/ar0/P/PROJ/sthiele/PROJDATA/PLOTS")
parser.add_argument("-p", "--plot", action="store_true", help="save plots")
parser.add_argument("--skeleton", action="store_true", help="use skeletonize function for phl")
parser.add_argument("--progressive", action="store_true", help="use progressive phl rather than default")
parser.add_argument("--gpu", action="store_true", help="use gpu for median filter calculation")
parser.add_argument("--filtr", type=float, default=10., help="radius for median filter window")
parser.add_argument("--nclose", type=int, default=10, help="closing radius for mask dilation")
parser.add_argument("--nhalf", type=int, default=1, help="number of half-widths for mask dilation")
parser.add_argument("--nsig", type=int, default=3, help="number of sigma for bound calculation")
parser.add_argument("--gap", type=int, default=2, help="number of pixel separation between segments to be considered a gap")

args = parser.parse_args()

# parameters:
skeleton = args.skeleton
progressive = args.progressive
gpu = args.gpu
filtr = args.filtr
nclose = args.nclose
nhalf = args.nhalf
nsig = args.nsig
gap = args.gap

print(f"skeleton: {skeleton}")
print(f"progressive: {progressive}")
print(f"gpu: {gpu}")
print(f"filter_radius: {filtr}")
print(f"nclose: {nclose}")
print(f"nhalf: {nhalf}")
print(f"gap: {gap}")

# data paths:
subroot = args.subroot + '/ihu{:02d}'.format(args.ihu)
subpath = args.subpath + '/' + subroot
detpath = args.detpath + '/' + subroot
outpath = args.output + '/' + subroot
print('obs night is ', subroot, ', ihu is {}'.format(args.ihu))
print('sub data is in ', args.subpath)
print('sattrail detections is in ', detpath)
print('postproc files are in ', outpath)

# get data:
files = os.listdir(subpath)
subfiles = [subpath + '/' + item for item in files if '-sub.fits' in item]
detfiles = [detpath + '/' + subfile.split('/')[-1].split('-sub.fits')[0] + '-detection.json' for subfile in subfiles]
outputfiles = [outpath + '/' + subfile.split('/')[-1].split('-sub.fits')[0] + '-sattrail.hdf' for subfile in subfiles]

plot = args.plot
if plot:
    plotfiles = [args.plotdir + '/' + subroot + '/' + subfile.split('/')[-1].split('-sub.fits')[0] +'/' for subfile in subfiles]

for i, subfile in enumerate(subfiles[:2]):
    print(subfile)
    print(detfiles[i])
    print(plotfiles[i])
    print(outputfiles[i])
    print('\n')

    df0, FINALLIST, mask_master, cpoints, rpoints = pp.postproc(subfiles[i], detfiles[i], outputfile, plotroot, SAVE=True, PLOT=False,
             skeleton=False, progressive=True, gpu=False, filter_radius=10,
             nclose=10, nhalf=1, nsig=3, gap=2)
