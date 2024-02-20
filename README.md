

<h1 align="center"> Our Fork of: GO-SLAM together with DROID-Calib </h1> 


This repository borrows from the following work  
- "**GO-SLAM: Global Optimization for Consistent 3D Instant Reconstruction, Zhang et al**",  [ICCV 2023](https://iccv2023.thecvf.com/)
- "**Deep geometry-aware camera self-calibration from video, Hagemann et al**",  [ICCV 2023](https://iccv2023.thecvf.com/)


## :clapper: Introduction

This is a deep-learning-based dense visual SLAM framework that achieves **real-time global optimization of poses and 3D reconstruction**. This is achieved by adding a loop closure mechanism using the Co-Visibility matrix to the DROID-SLAM framework. On top of this, we add the full optimization kernel from DROID-Calib, which supports arbitrary camera models and optimizes the camera intrinsics on top of the map and pose graph.

# References
```bibtex
@inproceedings{zhang2023goslam,
    author    = {Zhang, Youmin and Tosi, Fabio and Mattoccia, Stefano and Poggi, Matteo},
    title     = {GO-SLAM: Global Optimization for Consistent 3D Instant Reconstruction},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
}
```

```bibtex
@inproceedings{hagemann2023deep,
  title={Deep geometry-aware camera self-calibration from video},
  author={Hagemann, Annika and Knorr, Moritz and Stiller, Christoph},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={3438--3448},
  year={2023}
}
```


## :memo: Code

You can create an anaconda environment called `go-slam`. For linux, you need to install **libopenexr-dev** before creating the environment.
```bash

git clone --recursive https://github.com/ChenHoy/Go-Droid-SLAM-full.git

sudo apt-get install libopenexr-dev
    
conda env create -f environment.yaml
conda activate go-slam

pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install evo --upgrade --no-binary evo

python setup.py install

```

### Replica

Download the data from [Google Drive](https://drive.google.com/drive/folders/1RJr38jvmuIV717PCEcBkzV2qkqUua-Fx?usp=sharing), and then you can run:

```bash
# please modify the OUT_DIR firstly in the script, and also DATA_ROOT in the config file
# MODE can be [rgbd, mono], EXP_NAME is the experimental name you want

./evaluate_on_replica.sh MODE EXP_NAME

# for example

./evaluate_on_replica.sh rgbd first_try

```

**Mesh and corresponding evaluated metrics are available in OUT_DIR.**

We also upload our predicted mesh on [Google Drive](https://drive.google.com/drive/folders/1RJr38jvmuIV717PCEcBkzV2qkqUua-Fx?usp=sharing). Enjoy!


### ScanNet
Please follow the data downloading procedure on [ScanNet](http://www.scan-net.org/) website, and extract color/depth frames from the `.sens` file using this [code](https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/reader.py).

<details>
  <summary>[Directory structure of ScanNet (click to expand)]</summary>
  
  DATAROOT is `./Datasets` by default. If a sequence (`sceneXXXX_XX`) is stored in other places, please change the `input_folder` path in the config file or in the command line.

```
  DATAROOT
  └── ScanNet
      └── scans
          └── scene0000_00
              └── frames
                  ├── color
                  │   ├── 0.jpg
                  │   ├── 1.jpg
                  │   ├── ...
                  │   └── ...
                  ├── depth
                  │   ├── 0.png
                  │   ├── 1.png
                  │   ├── ...
                  │   └── ...
                  ├── intrinsic
                  └── pose
                      ├── 0.txt
                      ├── 1.txt
                      ├── ...
                      └── ...

```
</details>

Once the data is downloaded and set up properly, you can run:
```bash
# please modify the OUT_DIR firstly in the script, and also DATA_ROOT in the config file
# MODE can be [rgbd, mono], EXP_NAME is the experimental name you want

./evaluate_on_scannet.sh MODE EXP_NAME

# for example

./evaluate_on_scannet.sh rgbd first_try

# besides, you can generate video as shown in our project page by:

./generate_video_on_scannet.sh rgbd first_try_on_video
```

We also upload our predicted mesh on [Google Drive](https://drive.google.com/drive/folders/1RJr38jvmuIV717PCEcBkzV2qkqUua-Fx?usp=sharing). Enjoy!

### EuRoC

Please use the following [script](https://github.com/youmi-zym/GO-SLAM/blob/main/scripts/download_euroc.sh) to download the EuRoC dataset. The GT trajectory can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1RJr38jvmuIV717PCEcBkzV2qkqUua-Fx?usp=sharing). 

Please put the GT trajectory of each scene to the corresponding folder, as shown below:


<details>
  <summary>[Directory structure of EuRoC (click to expand)]</summary>

DATAROOT is `./Datasets` by default. If a sequence (e.g., `MH_01_easy`) is stored in other places, please change the `input_folder` path in the config file or in the command line.

```
  DATAROOT
  └── EuRoC
     └── MH_01_easy
         └── mav0
             ├── cam0
             ├── cam1
             ├── imu0
             ├── leica0
             ├── state_groundtruth_estimate0
             └── body.yaml
         └── MH_01_easy.txt

```
</details>

Then you can run:

```bash
# for data downloading:

DATA_ROOT=path/to/folder
mkdir $DATA_ROOT
./scripts/download_euroc.sh $DATA_ROOT

# please modify the OUT_DIR firstly in the script, and also DATA_ROOT in the config file
# MODE can be [stereo, mono], EXP_NAME is the experimental name you want

./evaluate_on_euroc.sh MODE EXP_NAME

# for example

./evaluate_on_euroc.sh stereo first_try
```

# Acknowledgment
We adapted some codes from some awesome repositories including [NICE-SLAM](https://github.com/cvg/nice-slam), [NeuS](https://github.com/Totoro97/NeuS), [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM) and [DROID-Calib](https://github.com/boschresearch/droidcalib)
