#!/bin/bash

MODE=$1
EXPNAME=$2

# scenes="office0 office1 office2 office3 office4 room0 room1 room2"
# NOTE we have to substract some, because the number only limits the growdth during densification, but the final number also can surpassed as new frames come in.
# FIXME we still get many more Gaussians than the vanilla densification strategy, how to change this?

echo "Start evaluating on Replica dataset..."

sc='office0'
targets=200000
echo Running on $sc ...
python run.py data=Replica/base data.input_folder=/media/data/Replica/${sc} mode=$MODE mapping.mcmc.cap_max=${targets} evaluate=True mapping=replica backend_every=8 mapper_every=20 hydra.job.name=${sc}_${EXPNAME}

sc='office1'
targets=200000
echo Running on $sc ...
python run.py data=Replica/base data.input_folder=/media/data/Replica/${sc} mode=$MODE mapping.mcmc.cap_max=${targets} evaluate=True mapping=replica backend_every=8 mapper_every=20 hydra.job.name=${sc}_${EXPNAME}

sc='office2'
targets=200000
echo Running on $sc ...
python run.py data=Replica/base data.input_folder=/media/data/Replica/${sc} mode=$MODE mapping.mcmc.cap_max=${targets} evaluate=True mapping=replica backend_every=8 mapper_every=20 hydra.job.name=${sc}_${EXPNAME}

sc='office3'
targets=120000
echo Running on $sc ...
python run.py data=Replica/base data.input_folder=/media/data/Replica/${sc} mode=$MODE sleep_delay=0.35 mapping.mcmc.cap_max=${targets} evaluate=True mapping=replica backend_every=8 mapper_every=20 hydra.job.name=${sc}_${EXPNAME}

sc='office4'
targets=200000
echo Running on $sc ...
python run.py data=Replica/base data.input_folder=/media/data/Replica/${sc} sleep_delay=0.35 mode=$MODE mapping.mcmc.cap_max=${targets} evaluate=True mapping=replica backend_every=8 mapper_every=20 hydra.job.name=${sc}_${EXPNAME}

sc='room0'
targets=100000
echo Running on $sc ...
python run.py data=Replica/base data.input_folder=/media/data/Replica/${sc} sleep_delay=0.35 mode=$MODE mapping.mcmc.cap_max=${targets} evaluate=True mapping=replica backend_every=8 mapper_every=20 hydra.job.name=${sc}_${EXPNAME}

sc='room1'
targets=220000
echo Running on $sc ...
python run.py data=Replica/base data.input_folder=/media/data/Replica/${sc} mode=$MODE mapping.mcmc.cap_max=${targets} evaluate=True mapping=replica backend_every=8 mapper_every=20 hydra.job.name=${sc}_${EXPNAME}

sc='room2'
targets=200000
echo Running on $sc ...
python run.py data=Replica/base data.input_folder=/media/data/Replica/${sc} mode=$MODE mapping.mcmc.cap_max=${targets} evaluate=True mapping=replica backend_every=8 mapper_every=20 hydra.job.name=${sc}_${EXPNAME}

echo All Done!