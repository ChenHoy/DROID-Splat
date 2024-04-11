#!/bin/bash

MODE=$1
OUT_DIR=/home/andrei/Go-Droid-SLAM-full/evaluation_results/

GLOBAL_SUMMARY=${OUT_DIR}/global_summary.txt

if [ -e $GLOBAL_SUMMARY ]
then
  rm -f $GLOBAL_SUMMARY
fi

echo "Start evaluating on Replica dataset..."
./evaluate_on_replica.sh $MODE 

if [ -e $OUT_DIR/replica/summary.txt ]
then
  echo "Results for Replica:" >> $GLOBAL_SUMMARY
  echo >> $GLOBAL_SUMMARY
  cat $OUT_DIR/replica/summary.txt >> $GLOBAL_SUMMARY
fi

echo "Start evaluating on eth3d dataset..."
./evaluate_on_eth3d.sh $MODE

if [ -e $OUT_DIR/eth3d/summary.txt ]
then
  echo "Results for Eth3d:" >> $GLOBAL_SUMMARY
  echo >> $GLOBAL_SUMMARY
  cat $OUT_DIR/eth3d/summary.txt >> $GLOBAL_SUMMARY
fi

echo "Start evaluating on TUM dataset..."
./evaluate_on_tum.sh $MODE

if [ -e $OUT_DIR/tum/summary.txt ]
then
  echo "Results for TUM:" >> $GLOBAL_SUMMARY
  echo >> $GLOBAL_SUMMARY
  cat $OUT_DIR/tum/summary.txt >> $GLOBAL_SUMMARY
fi

echo "Start evaluating on Scannet dataset..."
./evaluate_on_scannet.sh $MODE

if [ -e $OUT_DIR/scannet/summary.txt ]
then
  echo "Results for Scannet:" >> $GLOBAL_SUMMARY
  echo >> $GLOBAL_SUMMARY
  cat $OUT_DIR/scannet/summary.txt >> $GLOBAL_SUMMARY
fi

echo "Start evaluating on euroc dataset..."
./evaluate_on_replica.sh $MODE

if [ -e $OUT_DIR/euroc/summary.txt ]
then
  echo "Results for euroc:" >> $GLOBAL_SUMMARY
  echo >> $GLOBAL_SUMMARY
  cat $OUT_DIR/euroc/summary.txt >> $GLOBAL_SUMMARY
fi

echo "All Done! Results for all datasets:"
cat $GLOBAL_SUMMARY
echo "####################################"
