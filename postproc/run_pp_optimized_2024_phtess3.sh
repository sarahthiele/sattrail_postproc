#!/bin/bash

# subdir in form 1-YYYYMMDD

subdirs=(1-20240124 1-20240214 1-20240321 1-20250416 1-20250521 1-20250626 1-20250716 1-20250813 1-20250917)

for subdir in "${subdirs[@]}"; do
    echo "starting subdir $subdir"
    logdir="/nfs/php2/ar0/P/PROJ/sthiele/PROJDATA/LOGS/$subdir"
    if [ ! -d "$logdir" ]; then
        mkdir "$logdir"
    fi
    for ihu in {1..64}; do
        # Start the Python script with the parameters in the background
        logfile=$(printf "/nfs/php2/ar0/P/PROJ/sthiele/PROJDATA/LOGS/$subdir/log_ihu%02d" "$ihu")
        python /nfs/php2/ar0/P/PROJ/sthiele/repos/sattrail_postproc/postproc/pipeline_pp_optimized.py $subdir $ihu > $logfile 2>&1 &
    done
    wait
    echo "subdir $subdir is finished"
done

# Wait for all background processes to finish
wait

echo "All processes have finished."
