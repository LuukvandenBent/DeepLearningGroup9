# Reproducibility Project: "Restoring Extremely Dark Images In Real Time", Mohit Lamba, Kaushik Mitra, 2021
Group 9: Sahánd, Luuk, Annabel, Ethem
31/03/2022

This blog post is about the reproducibility of the Deep Learning paper “Restoring Extremely Dark Images In Real Time”[insert link] by Mohit Lamba and Kaushik Mitra. We have investigated alternative loss functions to the existing problem, as well as the effect of using different architectures of the RDB modules. This research is conducted as part of the CS4240 Deep Learning course at Delft University of Technology. 

## Introduction

The original idea of the authors of the paper is that it is possible to train a network in such a way that it is capable of restoring extremely dark images in real time. They have studied low light enhancement solutions from the Computer Vision community and identified a need for a more ‘practical’ solution. This solution should ideally feature low network latency, less memory footprint, fewer model parameters, smaller operations count while maintaining high quality results. However, many current solutions compromise speed and thus are unable to work in real time. Additionally they involve high computational power, which would result in an expensive setup making it less suitable for mid to low range devices.


For modifying the number of RDB blocks in the HSE the network.py file needs to be edited again, but this time at lines 141 and line 143, where in the first line the third RDB initialisation is removed, and in the second line the settings of the convolutional layer are adjusted to accommodate for two sets of 64 input channels, rather than three.

```
        self.RDB1 = RDB(nChannels=64, nDenselayer=4, growthRate=32)
        self.RDB2 = RDB(nChannels=64, nDenselayer=5, growthRate=32)
```

```
self.rdball = conv_layer(int(64*2), 64, kernel_size=1, groups=1, bias=False, negative_slope=1, bn=False, init_type='kaiming', fan_type='fan_in', activation=False, pixelshuffle_init=False, upscale=False, num_classes=False, weight_normalization = True)
```
