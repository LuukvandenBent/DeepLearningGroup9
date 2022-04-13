# Reproducibility Project: "Restoring Extremely Dark Images In Real Time", Mohit Lamba, Kaushik Mitra, 2021
Group 9: Sahánd, Luuk, Annabel, Ethem
31/03/2022

This blog post is about the reproducibility of the Deep Learning paper “Restoring Extremely Dark Images In Real Time”[insert link] by Mohit Lamba and Kaushik Mitra. We have investigated alternative loss functions to the existing problem, as well as the effect of using different architectures of the RDB modules. This research is conducted as part of the CS4240 Deep Learning course at Delft University of Technology. 

## Introduction

The original idea of the authors of the paper is that it is possible to train a network in such a way that it is capable of restoring extremely dark images in real time. They have studied low light enhancement solutions from the Computer Vision community and identified a need for a more ‘practical’ solution. This solution should ideally feature low network latency, less memory footprint, fewer model parameters, smaller operations count while maintaining high quality results. However, many current solutions compromise speed and thus are unable to work in real time. Additionally they involve high computational power, which would result in an expensive setup making it less suitable for mid to low range devices.


```
    def __init__(self, nChannels=64, growthRate=32, pos=False):
        super(make_dense, self).__init__()
        
        kernel_size=3
        if pos=='first':
            self.conv = conv_layer(nChannels, growthRate, kernel_size=kernel_size, groups=1, bias=False, 
                                   negative_slope=0.2, bn=False, init_type='kaiming', fan_type='fan_in', 
                                   activation=False, pixelshuffle_init=False, upscale=False, num_classes=False, 
                                   weight_normalization = True)
        elif pos=='middle':
            self.conv = conv_layer(nChannels, growthRate, kernel_size=kernel_size, groups=1, bias=False, 
                                   negative_slope=0.2, bn=False, init_type='kaiming', fan_type='fan_in', 
                                   activation='before', pixelshuffle_init=False, upscale=False, num_classes=False, 
                                   weight_normalization = True)
        elif pos=='last':
            self.conv = conv_layer(nChannels, growthRate, kernel_size=kernel_size, groups=1, bias=False, 
                                   negative_slope=1, bn=False, init_type='kaiming', fan_type='fan_in', 
                                   activation='before', pixelshuffle_init=False, upscale=False, num_classes=False, 
                                   weight_normalization = True)
        else:
            raise NotImplementedError('ReLU position error in make_dense')

```
