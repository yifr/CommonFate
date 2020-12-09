#!/bin/bash
#SBATCH -t 00:05:00          # walltime = 1 hours and 30 minutes
#SBATCH -N 1                 # one node
#SBATCH -n 1                 # two CPU (hyperthreaded) cores
# Execute commands to run your program here. Here is an example of python.
Blender/blender -b --python blender_qrotation.py
