#!/bin/bash

MODE=$1
EXPNAME=$2

scenes="office0 office1 office2 office3 office4 room0 room1 room2"

echo "Start evaluating on Replica dataset..."
for sc in ${scenes}
do
  echo Running on $sc ...
  python run.py slam.input_folder=/media/data/Replica/${sc} slam.dataset=replica configs/data/Replica/base.yaml mode=$MODE slam.evaluate=True hydra.job.name=${EXPNAME}
  # TODO change configs dependent on mode if wanted
  # if [[ $MODE == "mono" ]]
  # then
    # python run.py slam.input_folder=/media/data/Replica/${sc} slam.dataset=replica configs/data/Replica/base.yaml mode=$MODE slam.evaluate=True hydra.job.name=${EXPNAME}
  # else
    # python run.py slam.input_folder=/media/data/Replica/${sc} slam.dataset=replica configs/data/Replica/base.yaml mode=$MODE slam.evaluate=True hydra.job.name=${EXPNAME}
  # fi
  echo $sc done!
done
echo All Done!
