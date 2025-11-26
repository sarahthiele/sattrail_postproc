#!/bin/bash

# subdir in form 1-YYYYMMDD

subdirs=(1-20240403 1-20240410 1-20240417 1-20240423 1-20241030 1-20241106 1-20241113 1-20241120 1-20241127 1-20241204 1-20241211 1-20241218 1-20241225 1-20250101 1-20250108 1-20250117 1-20250122 1-20250129 1-20250205 1-20250213 1-20250219 1-20250226)

for subdir in "${subdirs[@]}"; do
    echo "starting subdir $subdir"
    logdir="/nfs/php2/ar0/P/PROJ/sthiele/PROJDATA/LOGS/$subdir"
    if [ ! -d "$logdir" ]; then
        mkdir "$logdir"
    fi
    for ihu in {1..64}; do
        datadir=$(printf "/nfs/php2/ar2/P/HP1/REDUCTION/SUB/$subdir/ihu%02d" "$ihu")
        if [-d "$datadir" ]; then
            # Start the Python script with the parameters in the background
            logfile=$(printf "/nfs/php2/ar0/P/PROJ/sthiele/PROJDATA/LOGS/$subdir/log_ihu%02d" "$ihu")
            python /nfs/php2/ar0/P/PROJ/sthiele/repos/sattrail_postproc/postproc/pipeline.py --subpath /nfs/php2/ar2/P/HP1/REDUCTION/SUB $subdir $ihu > $logfile 2>&1 &
        fi
    done
    wait
    echo "subdir $subdir is finished"
done

# Wait for all background processes to finish
wait

echo "All processes have finished."
