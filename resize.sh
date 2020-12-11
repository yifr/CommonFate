#!/bin/bash

root_dir=$1
i=0
total_start=$SECONDS
for scene in $root_dir/* ; do
    echo "Processing $scene"

    start=$SECONDS
    for inputfile in $scene/images/* ; do
        pref='_sm'
        fstart='img'
        if [[ "$inputfile" != *"$pref"* ]] && [[ "$inputfile" == *"$fstart"* ]]; then
            outputfile="${inputfile%.*}_sm.png"
            echo "    $inputfile --> $outputfile"
            convert "$inputfile" -gravity Center -extent 1024x1024 -resize 128x128 "$outputfile"
            # convert "$outputfile" -resize 64x64 "$outputfile"
            # convert "$outputfile" -adaptive-resize 64x64 "$outputfile"
        fi
    done
    duration=$(($SECONDS - $start))
    echo "Scene time taken: $duration seconds"
done

duration=$(($SECONDS - $total_start))
echo "Total time taken: $duration seconds"

echo "Making videos"
./make_videos.sh $1
