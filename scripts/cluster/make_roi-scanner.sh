#!/bin/bash

curr_dir=`pwd`
roi_list=('visual' 'motor' 'nacc')
scanner_list=('philips' 'ge' 'siemens')

n=0
for roi in ${roi_list[@]} ; do
    for scanner in ${scanner_list[@]} ; do
      sed -e "s|SCANNER|${scanner}|g; s|ROILABEL|${roi}|g;" ./templates/roi_batch.txt > ./batch_jobs/roitr${n}
      n=$((n+1))
    done
done

chmod +x ./batch_jobs/roitr*
