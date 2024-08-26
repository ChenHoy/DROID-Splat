#!/bin/bash

MODE=$1
EXPNAME=$2

# TODO write the commands manually and no loop since we have to switch configs
scenes='fr1_desk fr1_desk2 fr1_room fr2_xyz fr3_office'

echo "Start evaluating on TUM dataset..."
for sc in ${scenes};
do
  echo Running on $sc ...
  # overwrite the data yaml file based on the scene we are in, since all have different intrinsics
  if [[ "$sc" =~ "fr1" ]]; then
    python run.py data=TUM_RGBD/fr1 stride=1 data.input_folder=/media/data/tum_rgbd/benchmark_rgbd/${sc} tracking=tum mode=$MODE evaluate=True hydra.job.name=${EXPNAME}
  elif [[ "$sc" =~ "fr2" ]]; then
    python run.py data=TUM_RGBD/fr2 stride=1 data.input_folder=/media/data/tum_rgbd/benchmark_rgbd/${sc} tracking=tum mode=$MODE evaluate=True hydra.job.name=${EXPNAME}
  elif [[ "$sc" =~ "fr3" ]]; then
    python run.py data=TUM_RGBD/fr3 stride=1 data.input_folder=/media/data/tum_rgbd/benchmark_rgbd/${sc} tracking=tum mode=$MODE evaluate=True hydra.job.name=${EXPNAME}
  fi

  echo $sc done!
done