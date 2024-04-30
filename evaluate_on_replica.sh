#!/bin/bash

MODE=$1
# EXPNAME=$2

OUT_DIR=/home/andrei/Go-Droid-SLAM-full/evaluation_results/replica

scenes="office0 office1 office2 office3 office4 room0 room1 room2"
# scenes="office0"

echo "Start evaluating on Replica dataset..."

for sc in ${scenes}
do
  echo Running on $sc ...
  if [[ $MODE == "mono" ]]
  then
    python run.py slam.input_folder=/media/data/Replica/${sc} slam.dataset=replica configs/data/Replica/replica_mono.yaml slam.evaluate=True slam.output_folder=${OUT_DIR}/${sc}_mono 
  else
    python run.py slam.input_folder=/media/data/Replica/${sc} slam.dataset=replica configs/data/Replica/replica.yaml slam.evaluate=True slam.output_folder=${OUT_DIR}/${sc} 
  fi
  echo $sc done!
done

echo Results for all scenes are:

SUMMARY=${OUT_DIR}/summary.txt

rm -f $SUMMARY

for sc in ${scenes}
do
  echo
  echo Trajectory results for ${sc}: >> $SUMMARY
  if [[ $MODE == "mono" ]]
  then
    cat ${OUT_DIR}/${sc}_mono/trajectory_results.csv >> $SUMMARY
  else
    cat ${OUT_DIR}/${sc}/trajectory_results.csv >> $SUMMARY
  fi
  echo >> $SUMMARY
  # cat ${OUT_DIR}/${sc}/${EXPNAME}/metrics_mesh.txt

  ## TODO: for rendering results
  echo >> $SUMMARY
  echo Rendering results for ${sc}: >> $SUMMARY
  # cat ${OUT_DIR}/${sc}/${EXPNAME}/combined_results.csv >> $SUMMARY
  echo >> SUMMARY
  # cat ${OUT_DIR}/${sc}/${EXPNAME}/metrics_mesh.txt >> $SUMMARY
done

echo All Done!

echo All Done! >> $SUMMARY
