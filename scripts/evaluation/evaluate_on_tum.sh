#!/bin/bash

MODE=$1
EXPNAME=$2

# scenes='fr1_desk fr1_desk2 fr1_room fr2_xyz fr3_office'
# targets="60000 70000 250000 25000 90000" # Number of Gaussians to match from the other experiments

sc='fr1_desk'
python run.py data=TUM_RGBD/fr1 stride=1 data.input_folder=/media/data/tum_rgbd/benchmark_rgbd/${sc} tracking=tum mapping=tum mapping.mcmc.cap_max=60000 mode=$MODE evaluate=True backend_every=8 mapper_every=20 hydra.job.name=${sc}_${EXPNAME}
sc='fr1_desk2'
python run.py data=TUM_RGBD/fr1 stride=1 data.input_folder=/media/data/tum_rgbd/benchmark_rgbd/${sc} tracking=tum mapping=tum mapping.mcmc.cap_max=70000 mode=$MODE evaluate=True backend_every=8 mapper_every=20 hydra.job.name=${sc}_${EXPNAME}
sc='fr1_room'
python run.py data=TUM_RGBD/fr1 stride=1 data.input_folder=/media/data/tum_rgbd/benchmark_rgbd/${sc} tracking=tum mapping=tum mapping.mcmc.cap_max=250000 mode=$MODE evaluate=True backend_every=8 mapper_every=20 hydra.job.name=${sc}_${EXPNAME}

sc='fr2_xyz'
python run.py data=TUM_RGBD/fr2 stride=1 data.input_folder=/media/data/tum_rgbd/benchmark_rgbd/${sc} tracking=tum mapping=tum mapping.mcmc.cap_max=25000 mode=$MODE evaluate=True backend_every=8 mapper_every=20 hydra.job.name=${sc}_${EXPNAME}

sc='fr3_office'
python run.py data=TUM_RGBD/fr3 stride=1 data.input_folder=/media/data/tum_rgbd/benchmark_rgbd/${sc} tracking=tum mapping=tum mapping.mcmc.cap_max=90000 mode=$MODE evaluate=True backend_every=8 mapper_every=20 hydra.job.name=${sc}_${EXPNAME}