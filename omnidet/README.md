## [OmniDet: Surround View Cameras based Multi-task Visual Perception Network for Autonomous Driving](https://sites.google.com/view/omnidet/home)

[Install](#install) | [Training](#training) | [Models](#models) | [License](#license) | [References](#references)

<a href="https://www.youtube.com/watch?v=b62iDkLgGSI" target="_blank">
<img width="100%, text-align:center" src="/omnidet/gif/omnidet.gif"/>
</a>

Official [PyTorch](https://pytorch.org/) boilerplate implementation of multi task learning of distance estimation, pose estimation, semantic segmentation, motion segmentation and 2D object detection methods invented by the Valeo Team, in particular for

[**OmniDet: Surround View Cameras based Multi-task Visual Perception Network for Autonomous Driving (RA-l + ICRA 2021 oral)**](https://arxiv.org/abs/2102.07448),
*Varun Ravi Kumar, Senthil Kumar Yogamani, Hazem Rashed, Ganesh Sistu, Christian Witt, Isabelle Leang, Stefan Milz, Patrick Mäder*.

The quantitative results are not evaluated and compared to the baseline results from the [**OmniDet**](https://arxiv.org/abs/2102.07448) paper using the released weights due to the following reasons:

* Distance estimation is trained using source frames with `t-1` and `t` on 8k images compared to the internal images with a temporal sequence of `t-1`, `t` and `t+1`.
* No hyperparameter tuning or NAS is performed.
* Less training data (8k vs. internal dataset).
* The novel contributions are held back due to IP reasons.
* Velodyne LiDAR GT is not released yet for distance estimation.

This code serves as a boilerplate on which researchers can leverage the MTL framework and build upon it using our [References](#references). We have released the onnx model export scripts, which can be used to export and run these models on NVIDIA's Jetson AGX device.

Although self-supervised (i.e., trained only on monocular videos), OmniDet outperforms other-self, semi, and fully supervised methods on the KITTI dataset at the time of publishing. Furthermore, the MTL model can run in real-time. See [References](#references) for more info on the different approaches.

## Install

```bash
git clone https://github.com/valeoai/WoodScape.git
cd WoodScape
```

## Requirements

`pip3 install -r requirements.txt`

## Training

Training can be fired with any one the following commands:

`python3 main.py --config data/params.yaml`

`./main.py`

For training the distance estimation kindly generate the look up tables using

`./generate_luts.py --config params.yaml`

For the code related to evaluation of the perception tasks check the `eval` folder scripts.

## Models

### WoodScape Boilerplate Weights

[ResNet18, 544x288](https://drive.google.com/drive/folders/11NSTT4qygIgGRT8dit5E7x3XCC79Q0m-?usp=sharing)

[ResNet50, 544x288](https://drive.google.com/drive/folders/11jM1FmI0TBVYB-0Y9pRHHrlru4AWhbRd?usp=sharing)

## License

This code is released under the [Apache 2.0 license](LICENSE).

## References

[**OmniDet**](#icra-omnidet) Surround View fisheye cameras are commonly deployed in automated driving for 360° near-field sensing around the vehicle. This work presents a multi-task visual perception network on unrectified fisheye images to enable the vehicle to sense its surrounding environment. It consists of six primary tasks necessary for an autonomous driving system: depth estimation, visual odometry, semantic segmentation, motion segmentation, object detection, and lens soiling detection. The output from the network is scale aware based on our [FisheyeDistanceNet (ICRA 2020)](#icra-fisheyedistancenet).

Please use the following citations when referencing our work:

<a id="icra-omnidet"> </a>
**OmniDet: Surround View Cameras based Multi-task Visual Perception Network for Autonomous Driving (RA-L + ICRA 2021 oral)** \
*Varun Ravi Kumar, Senthil Yogamani, Hazem Rashed, Ganesh Sistu, Christian Witt, Isabelle Leang, Stefan Milz and Patrick Mäder*, [**[paper]**](https://arxiv.org/abs/2102.07448), [**[video]**](https://youtu.be/xbSjZ5OfPes), [**[oral_talk]**](https://youtu.be/1qVyD-wqZO4), [**[site]**](https://sites.google.com/view/omnidet/home)

```
@inproceedings{omnidet,
  author    = {Varun Ravi Kumar and Senthil Kumar Yogamani and Hazem Rashed and Ganesh Sistu and Christian Witt and Isabelle Leang and Stefan Milz and Patrick Mäder},
  title     = {OmniDet: Surround View Cameras Based Multi-Task Visual Perception
               Network for Autonomous Driving},
  journal   = {{IEEE} Robotics Automation Letter},
  volume    = {6},
  number    = {2},
  pages     = {2830--2837},
  year      = {2021},
  url       = {https://doi.org/10.1109/LRA.2021.3062324},
  doi       = {10.1109/LRA.2021.3062324}
}
```

<a id="icra-fisheyedistancenet"> </a>
**FisheyeDistanceNet: Self-Supervised Scale-Aware Distance Estimation using Monocular Fisheye Camera for Autonomous Driving (ICRA 2020 oral)** \
*Varun Ravi Kumar, Sandesh Athni Hiremath, Markus Bach, Stefan Milz, Christian Witt, Clément Pinard, Senthil Yogamani and Patrick Mäder*, [**[paper]**](https://arxiv.org/abs/1910.04076), [**[video]**](https://youtu.be/Sgq1WzoOmXg), [**[oral_talk]**](https://youtu.be/qAsdpHP5e8c), [**[site]**](https://sites.google.com/view/fisheyedistancenet/home)

```
@inproceedings{fisheyedistancenet,
  author    = {Varun Ravi Kumar and Sandesh Athni Hiremath and Markus Bach and Stefan Milz and Christian Witt and Cl{\'{e}}ment Pinard and Senthil Kumar Yogamani and Patrick Mäder},
  title     = {FisheyeDistanceNet: Self-Supervised Scale-Aware Distance Estimation
               using Monocular Fisheye Camera for Autonomous Driving},
  booktitle = {2020 {IEEE} International Conference on Robotics and Automation, {ICRA} 2020, Paris, France, May 31 - August 31, 2020},
  pages     = {574--581},
  publisher = {{IEEE}},
  year      = {2020},
  url       = {https://doi.org/10.1109/ICRA40945.2020.9197319},
  doi       = {10.1109/ICRA40945.2020.9197319},
}
```

<a id="wacv-syndistnet"> </a>
**SynDistNet: Self-Supervised Monocular Fisheye Camera Distance Estimation Synergized with Semantic Segmentation for Autonomous Driving (WACV 2021 oral)** \
*Varun Ravi Kumar, Marvin Klingner, Senthil Yogamani, Stefan Milz, Tim Fingscheidt and Patrick Mäder*, [**[paper]**](https://arxiv.org/abs/2008.04017), [**[oral_talk]**](https://youtu.be/zL6zvtUy4cc), [**[site]**](https://sites.google.com/view/syndistnet/home)

```
@inproceedings{syndistnet,
  author    = {Varun Ravi Kumar and Marvin Klingner and Senthil Stefan Milz and
               Tim Fingscheidt and Patrick Mäder},
  title     = {SynDistNet: Self-Supervised Monocular Fisheye Camera Distance Estimation Synergized with Semantic Segmentation for Autonomous Driving},
  booktitle = {{IEEE} Winter Conference on Applications of Computer Vision, {WACV}
               2021, Waikoloa, HI, USA, January 3-8, 2021},
  pages     = {61--71},
  publisher = {{IEEE}},
  year      = {2021},
  url       = {https://doi.org/10.1109/WACV48630.2021.00011},
}
```

<a id="iros-unrectdepthnet"> </a>
**UnRectDepthNet: Self-Supervised Monocular Depth Estimation using a Generic Framework for Handling Common Camera Distortion Models (IROS 2020 oral)** \
*Varun Ravi Kumar, Senthil Yogamani, Markus Bach, Christian Witt, Stefan Milz, Patrick Mäder*, [**[paper]**](https://arxiv.org/abs/2007.06676), [**[video]**](https://youtu.be/K6pbx3bU4Ss), [**[oral_talk]**](https://youtu.be/3Br2KSWZRrY), [**[site]**](https://sites.google.com/view/unretdepthnet/home)

```
@inproceedings{unrectdepthnet,
  author    = {Varun Ravi Kumar and Senthil Kumar Yogamani and Markus Bach and Christian Witt and Stefan Milz and Patrick Mäder},
  title     = {UnRectDepthNet: Self-Supervised Monocular Depth Estimation using a
               Generic Framework for Handling Common Camera Distortion Models},
  booktitle = {{IEEE/RSJ} International Conference on Intelligent Robots and Systems, {IROS} 2020, Las Vegas, NV, USA, October 24, 2020 - January 24, 2021},
  pages     = {8177--8183},
  publisher = {{IEEE}},
  year      = {2020},
  url       = {https://doi.org/10.1109/IROS45743.2020.9340732},
}
```

<a id="iros-unrectdepthnet"> </a>
**SVDistNet: Self-Supervised Near-Field Distance Estimation on Surround View Fisheye Cameras (Journal- T-ITS 2021)** \
*Varun Ravi Kumar, Senthil Yogamani, Markus Bach, Christian Witt, Stefan Milz, Patrick Mäder*, [**[paper]**](https://arxiv.org/abs/2104.04420), [**[video]**](https://youtu.be/bmX0UcU9wtA), [**[site]**](https://sites.google.com/view/svdistnet/home)

```
@inproceedings{svdistnet,
  author    = {Varun Ravi Kumar and Marvin Klingner and Senthil Kumar Yogamani and Markus Bach and Stefan Milz and Tim Fingscheidt and Patrick Mäder},
  journal   = {IEEE Transactions on Intelligent Transportation Systems}, 
  title     = {SVDistNet: Self-Supervised Near-Field Distance Estimation on Surround View Fisheye Cameras}, 
  year      = {2021},
  volume    = {},
  number    = {},
  pages     = {1-10},
  doi       = {10.1109/TITS.2021.3088950}}
}
```

<a id="wacv-fisheyeyolo"> </a>
**FisheyeYOLO: Generalized Object Detection on Fisheye Cameras for Autonomous Driving: Dataset, Representations and Baseline (WACV 2021 oral)** \
*Hazem Rashed, Eslam Mohamed, Ganesh Sistu, Varun Ravi Kumar, Ciaran Eising, Ahmad El-Sallab, Senthil Yogamani*, [**[paper]**](https://arxiv.org/abs/2012.02124), [**[video]**](https://youtu.be/iLkOzvJpL-A), [**[site]**](https://sites.google.com/view/fisheyeyolo/home)

```
@inproceedings{fisheyeyolo,
  author    = {Rashed, Hazem and Mohamed, Eslam and Sistu, Ganesh and Kumar, Varun Ravi and Eising, Ciaran and El-Sallab, Ahmad and Yogamani, Senthil},
  title     = {Generalized object detection on fisheye cameras for autonomous driving: Dataset, representations and baseline},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages     = {2272--2280},
  year      = {2021}
}
```
