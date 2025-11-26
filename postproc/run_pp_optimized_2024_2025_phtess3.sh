#!/bin/bash

#conda init
#conda activate satvenv_phtess3
# subdir in form 1-YYYYMMDD

echo "IS YOUR ENVIRONMENT ACTIVATED?"

subdirs=(1-20240103 1-20240110 1-20240117 1-20240124 1-20240131 1-20240210 1-20240214 1-20240219 1-20240228 1-20240306 1-20240313 1-20240321 1-20240327 1-20240501 1-20240508 1-20240517 1-20240519 1-20240529 1-20240606 1-20240611 1-20240619 1-20240625 1-20240701 1-20240710 1-20240717 1-20240725 1-20240729 1-20240807 1-20240814 1-20240821 1-20240830 1-20240904 1-20240911 1-20240918 1-20240925 1-20241002 1-20241016 1-20241023 1-20250402 1-20250409 1-20250416 1-20250423 1-20250501 1-20250508 1-20250516 1-20250521 1-20250528 1-20250608 1-20250619 1-20250626 1-20250702 1-20250709 1-20250716 1-20250723 1-20250806 1-20250813 1-20250823 1-20250825 1-20250910 1-20250917 1-20250924 1-20251001)


for subdir in "${subdirs[@]}"; do
    echo "starting subdir $subdir"
    logdir="/nfs/php2/ar0/P/PROJ/sthiele/PROJDATA/LOGS/$subdir"
    plotdir="/nfs/php2/ar0/P/PROJ/sthiele/PROJDATA/PLOTS/$subdir"
    if [ ! -d "$logdir" ]; then
        mkdir "$logdir"
    fi
    if [ ! -d "$plotdir" ]; then
        mkdir "$plotdir"
    fi
    for ihu in {1..64}; do
        datadir=$(printf "/nfs/php2/ar5/P/HP1/REDUCTION/SUB/$subdir/ihu%02d" "$ihu")
        if [ -d "$datadir" ]; then
            plotdir_ihu=$(printf "/nfs/php2/ar0/P/PROJ/sthiele/PROJDATA/PLOTS/$subdir/ihu%02d" "$ihu")
            if [ ! -d "$plotdir_ihu" ]; then
                mkdir "$plotdir_ihu"
            fi
            # Start the Python script with the parameters in the background
            logfile=$(printf "/nfs/php2/ar0/P/PROJ/sthiele/PROJDATA/LOGS/$subdir/log_ihu%02d" "$ihu")
            python /nfs/php2/ar0/P/PROJ/sthiele/repos/sattrail_postproc/postproc/pipeline.py --maxr 9000 $subdir $ihu > $logfile 2>&1 &
        fi
    done
    wait
    echo "subdir $subdir is finished"
done

# Wait for all background processes to finish
wait

echo "All processes have finished."
