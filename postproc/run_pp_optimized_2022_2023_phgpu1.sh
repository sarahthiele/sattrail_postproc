#!/bin/bash

# subdir in form 1-YYYYMMDD

subdirs=(1-20221026 1-20221122 1-20221222 1-20230122 1-20230222 1-20230322 1-20230426 1-20230524 1-20230622 1-20230704 1-20230818 1-20230920 1-20231019 1-20231122 1-20231220)

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
