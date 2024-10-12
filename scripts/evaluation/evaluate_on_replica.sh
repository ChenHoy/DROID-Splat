#!/bin/bash

MODE=$1
EXPNAME=$2

scenes="office0 office1 office2 office3 office4 room0 room1 room2"

echo "Start evaluating on Replica dataset..."
for sc in ${scenes}
do
  echo Running on $sc ...
  python run.py data=Replica/base data.input_folder=/media/data/Replica/${sc} mode=$MODE evaluate=True backend_every=8 mapper_every=20 hydra.job.name=${EXPNAME}
  echo $sc done!
done
echo All Done!
