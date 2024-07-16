#!/bin/bash

MODE=$1
EXPNAME=$2
scenes='MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult V1_01_easy V1_02_medium V1_03_difficult V2_01_easy V2_02_medium V2_03_difficult'

echo "Start evaluating on EuRoC dataset..."
for sc in ${scenes};
do
  echo Running on $sc ...
  echo $sc done!
done