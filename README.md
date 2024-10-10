<p align="center">
  <a href="">
    <img src="droidsplat.png" width="60%">
  </a>
</p>
<h1 align="center"> DROID-Splat </h1> 
Combining dense end-to-end SLAM with 3D Gaussian Splatting.

## :clapper: Introduction
This is a deep-learning-based dense visual SLAM framework that achieves **real-time global optimization of poses and 3D reconstruction**. This is achieved by the following: 
- SotA Tracking from DROID-SLAM
- Integration of monocular depth estimation priors 
- Dense differentiable Rendering with 3D Gaussian Splatting

- We also support the optimization kernel from DROID-Calib, which supports arbitrary camera models and optimizes the camera intrinsics on top of the map and pose graph.

# Acknowledgments
- **DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras** [Neurips 2021](https://github.com/princeton-vl/DROID-SLAM)
- "**GO-SLAM: Global Optimization for Consistent 3D Instant Reconstruction, Zhang et al**",  [ICCV 2023](https://iccv2023.thecvf.com/)
- "**Deep geometry-aware camera self-calibration from video, Hagemann et al**",  [ICCV 2023](https://iccv2023.thecvf.com/)
- "**Gaussian Splatting SLAM, Matsuki et al**",  [CVPR 2024](https://cvpr.thecvf.com/)
- **GLORIE-SLAM: Globally Optimized RGB-only Implicit Encoding Point Cloud SLAM**, [Glorie-SLAM](https://github.com/zhangganlin/GlORIE-SLAM)

# Concurrent Work
We would like to acknowledge other works, who had the same idea and apparently blindsided us. Concurrent work [Splat-SLAM](https://github.com/eriksandstroem/Splat-SLAM) is a similar system, that combines DROID-SLAM and Gaussian Splatting. We would like to note, that we released this code earlier with its entire history to proof that we did not intend to copy their work. 

# References
```bibtex
@misc{teed2021droid,
  title={Droid-slam: Deep visual slam for monocular, stereo, and rgb-d cameras},
  author={Teed, Zachary and Deng, Jia},
  journal={Advances in neural information processing systems},
  volume={34},
  pages={16558--16569},
  year={2021}
}
```

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

```bibtex
@misc{matsuki2024gaussian,
      title={Gaussian Splatting SLAM}, 
      author={Hidenobu Matsuki and Riku Murai and Paul H. J. Kelly and Andrew J. Davison},
      year={2024},
      eprint={2312.06741},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```bibtex
@misc{zhang2024glorie,
  title={Glorie-slam: Globally optimized rgb-only implicit encoding point cloud slam},
  author={Zhang, Ganlin and Sandstr{\"o}m, Erik and Zhang, Youmin and Patel, Manthan and Van Gool, Luc and Oswald, Martin R},
  journal={arXiv preprint arXiv:2403.19549},
  year={2024}

}
```

```bibtex
@misc{sandstrom2024splat,
  title={Splat-SLAM: Globally Optimized RGB-only SLAM with 3D Gaussians},
  author={Sandstr{\"o}m, Erik and Tateno, Keisuke and Oechsle, Michael and Niemeyer, Michael and Van Gool, Luc and Oswald, Martin R and Tombari, Federico},
  journal={arXiv preprint arXiv:2405.16544},
  year={2024}
}
```

## :memo: Code

You can create an anaconda environment called `droidsplat`. For linux, you need to install **libopenexr-dev** before creating the environment.
```bash

git clone --recursive https://github.com/ChenHoy/DROID-Splat.git

sudo apt-get install libopenexr-dev
    
conda env create -f environment.yaml
conda activate droidsplat

pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install evo --upgrade --no-binary evo

python setup.py install
```

### Replica
Download the data from [Google Drive](https://drive.google.com/drive/folders/1RJr38jvmuIV717PCEcBkzV2qkqUua-Fx?usp=sharing)

### TUM-RGBD

# Acknowledgment
We adapted some code from other awesome repositories including [GO-SLAM](https://github.com/youmi-zym/GO-SLAM), [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM), [DROID-Calib](https://github.com/boschresearch/droidcalib), [MonoGS](https://github.com/muskie82/MonoGS) and [Glorie-SLAM](https://github.com/zhangganlin/GlORIE-SLAM)
