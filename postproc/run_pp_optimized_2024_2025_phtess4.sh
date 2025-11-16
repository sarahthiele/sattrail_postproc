#!/bin/bash

# subdir in form 1-YYYYMMDD

#subdirs=(1-20250117 1-20250226 1-20250416 1-20250521 1-20250626 1-20250716 1-20250813 1-20250917)

subdirs=(1-20240423 1-20241023 1-20241120 1-20241225 1-20250117 1-20250226)

for subdir in "${subdirs[@]}"; do
    echo "starting subdir $subdir"
    logdir="/nfs/php2/ar0/P/PROJ/sthiele/PROJDATA/LOGS/$subdir"
    if [ ! -d "$logdir" ]; then
        mkdir "$logdir"
    fi
    for ihu in {1..64}; do
        # Start the Python script with the parameters in the background
        logfile=$(printf "/nfs/php2/ar0/P/PROJ/sthiele/PROJDATA/LOGS/$subdir/log_ihu%02d" "$ihu")
        python /nfs/php2/ar0/P/PROJ/sthiele/repos/sattrail_postproc/postproc/pipeline_pp_optimized.py --subpath /nfs/php2/ar2/P/HP1/REDUCTION/SUB $subdir $ihu > $logfile 2>&1 &
    done
    wait
    echo "subdir $subdir is finished"
done

# Wait for all background processes to finish
wait

echo "All processes have finished."
