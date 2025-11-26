#!/bin/bash

# subdir in form 1-YYYYMMDD

subdirs=(1-20221026 1-20221102 1-20221116 1-20221122 1-20221128 1-20221207 1-20221215 1-20221222 1-20230117 1-20230122 1-20230131 1-20230208 1-20230215 1-20230222 1-20230302 1-20230308 1-20230315 1-20230322 1-20230329 1-20230406 1-20230413 1-20230418 1-20230426 1-20230503 1-20230511 1-20230524 1-20230531 1-20230607 1-20230615 1-20230625 1-20230622 1-20230704 1-20230818 1-20230913 1-20230920 1-20230927 1-20231004 1-20231011 1-20231019 1-20231026 1-20231102 1-20231107 1-20231116 1-20231122 1-20231129 1-20231206 1-20231214 1-20231220)

for subdir in "${subdirs[@]}"; do
    echo "starting subdir $subdir"
    logdir="/nfs/php2/ar0/P/PROJ/sthiele/PROJDATA/LOGS/$subdir"
    if [ ! -d "$logdir" ]; then
        mkdir "$logdir"
    fi
    for ihu in {1..64}; do
        datadir=$(printf "/nfs/php2/ar5/P/HP1/REDUCTION/SUB/$subdir/ihu%02d" "$ihu")
        if [ -d "$datadir" ]; then
            # Start the Python script with the parameters in the background
            logfile=$(printf "/nfs/php2/ar0/P/PROJ/sthiele/PROJDATA/LOGS/$subdir/log_ihu%02d" "$ihu")
            python /nfs/php2/ar0/P/PROJ/sthiele/repos/sattrail_postproc/postproc/pipeline.py $subdir $ihu > $logfile 2>&1 &
        fi
    done
    wait
    echo "subdir $subdir is finished"
done

# Wait for all background processes to finish
wait

echo "All processes have finished."
