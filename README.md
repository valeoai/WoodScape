# WoodScape: A multi-task, multi-camera fisheye dataset for autonomous driving
The repository containing tools and information about the WoodScape dataset.

**Update (March 5th, 2021):**
WoodScape paper was published at ICCV in November 2019 and we announced that the dataset was planned to be released in Q1 2020. Unfortunately, there were unexpected data protection policies required in order to comply with requirements for EU GDPR and Chinese data laws. Specifically, we had to remove one third of our dataset which was recorded in China and also employ a third party anonymization company for the remaining data. It was exacerbated by COVID situation and the subsequent economic downturn impacting the automotive sector. We apologize for the delay in the release by more than a year.

Finally, we have released the first set of tasks in our Google Drive ([link](https://drive.google.com/drive/folders/1X5JOMEfVlaXfdNy24P8VA-jMs0yzf_HR?usp=sharing)). It has 8.2K images along with their corresponding 8.2K previous images needed for geometric tasks. The remaining 1.8K test samples are held out for a benchmark. It currently has annotations for semantic segmentation, instance segmentation, motion segmentation and 2D bounding boxes. Soiling Detection and end-to-end driving prediction tasks will be released by March 15th, 2021. Sample scripts to use the data will be updated in the github shortly as well. Once this first set of tasks is complete and tested, additional tasks will be gradually added. The upcoming website will include an overview about the status of the additional tasks.

Despite the delay we still believe the dataset is unique in the field. Therefore we understand that this dataset has been long awaited by many researchers. We hope that an eco-system of research in multitask fisheye camera development will thrive based on this dataset. We will continue to bugfix, support and develop the dataset and therefore any feedback will be taken onboard.


## Paper
[WoodScape: A multi-task, multi-camera fisheye dataset for autonomous driving](https://arxiv.org/abs/1905.01489)  
Senthil Yogamani, Ciaran Hughes, Jonathan Horgan, Ganesh Sistu, Padraig Varley, Derek O'Dea, Michal Uricar, Stefan Milz, Martin Simon, Karl Amende, Christian Witt, Hazem Rashed, Sumanth Chennupati, Sanjaya Nayak, Saquib Mansoor, Xavier Perroton, Patrick Perez  
Valeo  
IEEE International Conference on Computer Vision (ICCV), 2019 (**Oral**)

If you find our dataset useful, please cite our [paper](https://arxiv.org/abs/1905.01489):

```
@article{yogamani2019woodscape,
  title={WoodScape: A multi-task, multi-camera fisheye dataset for autonomous driving},
  author={Yogamani, Senthil and Hughes, Ciar{\'a}n and Horgan, Jonathan and Sistu, Ganesh and Varley, Padraig and O'Dea, Derek and Uric{\'a}r, Michal and Milz, Stefan and Simon, Martin and Amende, Karl and others},
  journal={arXiv preprint arXiv:1905.01489},
  year={2019}
}
```

## Abstract
Fisheye cameras are commonly employed for obtaining a large field of view in surveillance, augmented reality and in particular automotive applications. In spite of its prevalence, there are few public datasets for detailed evaluation of computer vision algorithms on fisheye images. We release the first extensive fisheye automotive dataset, *WoodScape*, named after Robert Wood who invented the fisheye camera in 1906. *WoodScape* comprises of four surround view cameras and nine tasks including segmentation, depth estimation, 3D bounding box detection and soiling detection. Semantic annotation of 40 classes at the instance level is provided for over 10,000 images and annotation for other tasks are provided for over 100,000 images. We would like to encourage the community to adapt computer vision models for fisheye camera instead of naive rectification.

## Demo
Please click on the image below for a teaser video showing annotated examples and sample results.

[![](./teaser.png)](https://streamable.com/aiefb "")

## Dataset Release
The dataset and code for baseline experiments will be provided in stages. First release was planned for ICCV 2019 (end of October). But the release got delayed because of licensing agreement challenges. 
