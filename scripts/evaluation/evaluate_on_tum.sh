#!/bin/bash

MODE=$1
EXPNAME=$2

scenes='desk desk2 room xyz long_office_household'

echo "Start evaluating on TUM dataset..."
for sc in ${scenes};
do
  echo Running on $sc ...
  python run.py slam.input_folder=/media/data/tum_rgbd/benchmark_rgbd/${sc} slam.dataset=tumrgbd configs/data/TUM_RGBD/base.yaml mode=$MODE slam.evaluate=True hydra.job.name=${EXPNAME}
  # if [[ $MODE == "mono" ]]
  # then
    # python run.py slam.input_folder=/media/data/tum_rgbd/benchmark_rgbd/${sc} slam.dataset=tumrgbd configs/data/TUM_RGBD/base.yaml mode=$MODE slam.evaluate=True hydra.job.name=${EXPNAME}
  # else
    # python run.py slam.input_folder=/media/data/tum_rgbd/benchmark_rgbd/${sc} slam.dataset=tumrgbd configs/data/TUM_RGBD/base.yaml mode=$MODE slam.evaluate=True hydra.job.name=${EXPNAME}
  # fi
  echo $sc done!
done

echo Results for all scenes are:

for sc in ${scenes}
do
  echo
  echo For ${sc}:
  cat ${OUT_DIR}/${sc}/${EXPNAME}/metrics_traj.txt
  echo
  # cat ${OUT_DIR}/${sc}/${EXPNAME}/metrics_mesh.txt
done

echo All Done!
