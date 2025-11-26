from postproc_optimized import postproc_regiongaps as pp
import numpy as np
import argparse
import os
import sys

print('This is postproc with the new gap filling mechanism!')

parser = argparse.ArgumentParser()
parser.add_argument("subroot", type=str, help="observation night in format 1-YYYYMMDD")
parser.add_argument("ihu", type=int, help="ihu number")
parser.add_argument("--subpath", type=str, help="directory of subtracted data", default="/nfs/php2/ar5/P/HP1/REDUCTION/SUB")
parser.add_argument("--detpath", type=str, help="path to sattrails detection files", default="/nfs/php2/ar0/P/PROJ/sthiele/PROJDATA/DETSAT")
parser.add_argument("-o", "--output", type=str, help="directory for output file", default="/nfs/php2/ar0/P/PROJ/sthiele/PROJDATA/DETSAT")
parser.add_argument("--plotdir", type=str, help="directory for plots", default="/nfs/php2/ar0/P/PROJ/sthiele/PROJDATA/PLOTS")
parser.add_argument("--maxr", type=int, default=10000, help="radius of curvature for segment grouping")
#parser.add_argument("--plot", action="store_true", help="plot final trail summary")

args = parser.parse_args()

# data paths:
subroot = os.path.join(args.subroot, 'ihu{:02d}'.format(args.ihu))
subpath = os.path.join(args.subpath, subroot)
detpath = os.path.join(args.detpath, subroot)
outpath = os.path.join(args.output, subroot)
print('obs night is ', args.subroot, ', ihu is {}'.format(args.ihu))
print('sub data is in ', args.subpath)
print('sattrail detections is in ', detpath)
print('postproc files are in ', outpath)

# plot final trail summary?
#PLOT = args.plot
#print('plot_final = ', PLOT)
maxr = args.maxr
print('max_R = ', maxr)

# get data:
try:
    files = os.listdir(subpath)
except:
    print(subpath+' does not exist')
    sys.exit(0)

subfiles = [os.path.join(subpath, item) for item in files if '-sub.fits' in item]
detfiles = [os.path.join(detpath, subfile.split('/')[-1].split('-sub.fits')[0]+'-detection.json') for subfile in subfiles]
outputfiles = [os.path.join(outpath, subfile.split('/')[-1].split('-sub.fits')[0]+'-trails.parquet') for subfile in subfiles]
plotfiles = [os.path.join(args.plotdir, subroot, subfile.split('/')[-1].split('-sub.fits')[0]) for subfile in subfiles]

for i, subfile in enumerate(subfiles[:5]):
    print('\n')
    print(subfile)
#    print(detfiles[i])
#    print(plotfiles[i])
#    print(outputfiles[i])

    pp(subfiles[i], detfiles[i], outputfiles[i], plotfiles[i], plot_final=True, max_R=maxr)
