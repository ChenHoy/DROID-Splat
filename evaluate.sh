#!/bin/bash

## clear output folder
rm -rf outputs/

## Scene: Replica/office0 | Loop Closure: False | Frontend + Backend + Rendering | Mode: mono

echo "Running evaluation on: Scene: Replica/office0 | Loop Closure: False | Frontend + Backend + Rendering | Mode: mono"
python run.py run_mapping_gui=False evaluate=True tracking.backend.use_loop_closure=False

echo "Running evaluation on: Scene: Replica/office0 | Loop Closure: True | Frontend + Backend + Rendering | Mode: mono"
python run.py run_mapping_gui=False evaluate=True tracking.backend.use_loop_closure=True




csvstack outputs/mono_slam/*/evaluation/evaluation_results.csv > joint_results.csv