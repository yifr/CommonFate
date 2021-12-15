#!/bin/bash

for i in {0..15}
do
    START_SCENE=$((2500 * (($i % 4))))
    IDX=$(($i / 4 + 1))
    echo $i, $IDX, $START_SCENE
done
