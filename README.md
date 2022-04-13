# Reproducibility Project: "Restoring Extremely Dark Images In Real Time", Mohit Lamba, Kaushik Mitra, 2021
Group 9: Sahánd, Luuk, Annabel, Ethem
31/03/2022

This blog post is about the reproducibility of the Deep Learning paper “Restoring Extremely Dark Images In Real Time”[insert link] by Mohit Lamba and Kaushik Mitra. We have investigated alternative loss functions to the existing problem, as well as the effect of the (amount of) proposed RDB* modules, compared to the original RDB modules. This research is conducted as part of the CS4240 Deep Learning course at Delft University of Technology. 

## Introduction

The original idea of the authors of the paper is that it is possible to train a network in such a way that it is capable of restoring extremely dark images in real time. They have studied low light enhancement solutions from the Computer Vision community and identified a need for a more ‘practical’ solution. This solution should ideally feature low network latency, less memory footprint, fewer model parameters, smaller operations count while maintaining high quality results. However, many current solutions compromise speed and thus are unable to work in real time. Additionally they involve high computational power, which would result in an expensive setup making it less suitable for mid to low range devices.

The authors propose a new deep learning architecture for extreme low-light single image restoration, which despite its fast & lightweight inference, produces a restoration that is perceptually at par with state-of-the-art computationally intense models.  The authors state that this architecture is 5 - 100x faster, 6 - 20x computationally cheaper and uses 3 - 11 fewer model parameters compared to state-of-the art [10, 18, 65 ##reference fixen]. 

This was achieved by processing in higher scale -spaces allowing the intermediate-scales to be skipped. As can be shown in figure 1, most current restoration networks use U-net style encoder-decoder wherein processing at lower scales causes significant latency and computational overhead. Therefore the authors proposed an architecture that jumps over these intermediate scales and operates at just three scales: Lower Scale Encoder (LSE) at 1/2 resolution, Medium Scale Encoder (MSE) at 1/8 resolution, and Higher Scale Encoder (HSE) at 1/32 resolution.  

Another unique feature to the model is the ability to process all the scale-spaces concurrently as all the encoder scales operate directly on the input image and do not have any inter-dependencies resulting in high inference speeds. This is because the different encoder layers can be calculated in parallel. The architectural details of the model are shown on the left of figure 2. Five different blocks exist, the three encoders, LSE, MSE, HSE and the two Fuse Blocks 1 and 2. The output of the MSE and HSE is concatenated in the Fuse Block 1, while the output of this fuse block and the LSE is then concatenated in Fuse Block 2, resulting in the output of the entire model.

A crucial part of the HSE is the widely used Residual Dense Block (RDB), which is implemented three times in series, in order to extract abundant local features via dense connected convolutional layers, while enabling a contiguous memory (CM) mechanism [72]. The typical architecture of such an RDB block can be seen on the top right of figure 2. Three convolutional layers exist, and three ReLu blocks are being used after every layer. For the ReLU block the authors have decided to use the LeakyReLU non-linearity block with a negative slope of 0.2, as suggested by the paper “Seeing in the Dark” [10]. More uniquely, however, the authors have decided to modify the entire RDB block’s structure (into RDB*) in order to receive more desirable results. According to the authors, non-linear rectification after each convolutional layer unnecessarily clips the negative values of feature maps, losing valuable information. Nevertheless, the rectifiers are necessary to infuse the model with sufficient non-linearity. Therefore, as in the RDB, each convolutional layer in RDB* passes a rectified output to subsequent convolutional layer, guaranteeing sufficient non-linearity in RDB*, yet different from RDB, not the rectified, but the non-rectified output of all layers is concatenated for the final layer. This architecture allows simultaneous processing of both rectified and non-rectified outputs, unlike the original RDB, avoiding losing information due to non-linear rectification.



## Reproduction

We found this network particularly interesting as there are many applications that real time restoration of extremely dark images can aid. We found the visual results rather pleasing and pursued understanding the methodology of this paper and network and its significance and relation to the results.
We decided to pursue investigating different loss functions with the same network, as well as architectural changes of the final HSE and the RDB* block itself.

The evaluation of such a deep learning network involves its training for which a dataset is prepared, consisting of exclusive train and test sets. Once the training is completed, the network yields weights that can be used to restore extremely dark images. The dataset that was used in our project is the Sony dataset from the SID [10].

In order to proceed with the project, first the runtime environment and hardware needed to be chosen and set up. Local personal computers were not equipped with sufficiently powerful processors, some even not with GPUs, and would result in training sessions lasting days, given the large dataset. Therefore, to allow progress and practical reproducibility, the training part of the deep learning network either needed to be done with a significantly reduced size of the dataset, potentially compromising the quality of the training or on dedicated hardware. Luckily, Google Cloud Platform service provides such dedicated powerful hardware for educational purposes. Before making use of this hardware, however, the code was analyzed on local runtime environments and Google Collab, which allows group work. where alterations to the code were tested and troubleshooted. Once there were no errors, this code was then uploaded to a remote virtual machine on Google Cloud Platform via the Git protocol and Github as our remote repository host.

The code itself is written in Python, while extensive use of PyTorch packages is made in order to program this deep learning network. Also, although a CPU could be used for running this network, it is advised to use the GPU’s graphic processing capabilities, and therefore the CUDA platform is used to access the GPU. Based on these requirements, hardware for the virtual machine (VM) that was selected had an Intel Skylake CPU and a single core NVIDA Tesla T4 GPU with 30 GB Memory (made possible by 8 virtual processors). The software bundle that was selected was a Linux/Debian 10 optimized for use with PyTorch (version 1.10) and the CPU/GPU with CUDA 11.0. Finally a 100GB boot disk is used.

Using the full dataset from the SID paper [10] would have resulted in more than 24 hours training, even with the powerful remote virtual machine, therefore the dataset needed to be reduced as well. It was ultimately reduced to 21.2GB, meaning that 557 training files and 294 validation files were used. As a result, a single training lasted around 12 hours.

Further information on the modifications to the code and the motivations behind them are provided in their respective sections.

## Loss Function Analysis
For training the network, the authors used the L1 Regularization loss function and the multiscale structural similarity index (MS-SSIM) loss function with a weightage of 0.8 and 0.2, respectively. In order to verify the performance of this combination these loss functions were tested individually., The structural similarity index (SSIM) loss function was also tested for additional validation.

Beetje uitleg over de loss functions(vooral SSIM uitleg) Annbel 
The L1 loss is defined as follows: 

The MS-SSIM is defined 

The SSIM loss function is defined as follows: 


Alinea over wat we hebben aangepast voor loss function steeds - Sahánd
The code provided in the repository alongside the paper was mostly stable. However, several changes were important to make to get the code fully working. A simple but necessary change in the train file used is to adapt the file location used to retrieve the data files: The locations provided in the github repository were incorrect for the repository itself. We set up the dataset location to be outside of the github folder to save space.
Figure X shows an example of how the locations in glob.glob were modified.


Figure X: The adapted glob.glob() locations

The other change needed to make the system work was in common_classes.py. In the latest version of the original github repository, a number of lines were commented out. It was unclear as to why this was the case, as the original code contains very little comments. However, simply uncommenting the lines shown in figure X fixed the issue.

Figure X:the lines which needed to be uncommented.

The loss function used can simply be adapted in line that defines the loss within the while-loop for the training setup. Right before this while-loop, the loss functions can be defined. The code can be found in figure X.


Figure X: The code adapted for the loss function adaptation

 


## Results(Loss function)

Alinea over resultaten, welke resultaten zijn bekeken(grafieken van hoe de loss functions over iteraties gaan), uitleggen dat we het netwerk qua testen hebben gebruikt dat al beschikbaar was met de aangepaste weights, dan images naast elkaar zetten(GT, baseline, results per loss function) 

Loss grafiek, metrics tabel - Annabel

Table X: Resulting network outputs based on the loss functions used










Ground Truth








Baseline








L_1 loss



SSIM loss










Image vergelijkingen - Sahánd
To test the network weights on a set of images, the provided script demo.py can be used. Simply add the test input images to the Demo_imgs folder, along with the weights for the network. Subsequently running demo.py then outputs the generated .jpg files, which can be seen in table X. Do note that the weights file should be called weights.

Korte alinea over analyse, welke is beter, maak thet verschil? Reden van verschil uitleggen. 



## RDB block

The authors have modified the Residual Dense Block (RDB) that is widely used for non-linear rectification and reasoned that it is more effective. They have stated that their ablation study, that that caused the PSNR/SSIM to drop from 28.66dB/0.79 to 27.96dB/0.77 We went ahead and tested this. Additionally, we investigated how effectively a change in the number of RDB blocks within the High Scale Encoder impacts the results. Therefore we ran a total number of three trainings for the study of the RDB block.

For the first training the network.py file has been adjusted by changing the activation type in the convolution layers. In the RDB* module, the first, middle and last convolutional layer’s activation functions are “false” (meaning disabled), “before” and “before” respectively, while reverting it to the original canonical RDB they all become “after”. This can be seen in the network.py file at line 85 to 105.

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

For modifying the number of RDB blocks in the HSE the network.py file needs to be edited again, but this time at lines 141 and line 143, where in the first line the third RDB initialisation is removed, and in the second line the settings of the convolutional layer are adjusted to accommodate for two sets of 64 input channels, rather than three.

```
        self.RDB1 = RDB(nChannels=64, nDenselayer=4, growthRate=32)
        self.RDB2 = RDB(nChannels=64, nDenselayer=5, growthRate=32)

        self.rdball = conv_layer(int(64*2), 64, kernel_size=1, groups=1, bias=False, negative_slope=1, 
                                 bn=False, init_type='kaiming', fan_type='fan_in', activation=False, 
                                 pixelshuffle_init=False, upscale=False, num_classes=False, weight_normalization = True)
```

Finally, at line 176, the third RDB block itself is removed, and the concatenation is adjusted to only include the first and second RDB blocks.

```
        rdb1 = self.RDB1(low32x_beforeRDB)
        rdb2 = self.RDB2(rdb1)
        rdb8x = torch.cat((rdb1,rdb2),dim=1)
```
Finally, similar to the second training, for the third training the same blocks of codes are edited, but instead of removing RDB3, an RDB4 is added, and the concatenation is adjusted to 3 sets of 64 channels.

```
        self.RDB1 = RDB(nChannels=64, nDenselayer=4, growthRate=32)
        self.RDB2 = RDB(nChannels=64, nDenselayer=5, growthRate=32)
        self.RDB3 = RDB(nChannels=64, nDenselayer=5, growthRate=32)
        self.RDB4 = RDB(nChannels=64, nDenselayer=5, growthRate=32)

        self.rdball = conv_layer(int(64*4), 64, kernel_size=1, groups=1, bias=False, negative_slope=1, 
                                 bn=False, init_type='kaiming', fan_type='fan_in', activation=False, 
                                 pixelshuffle_init=False, upscale=False, num_classes=False, weight_normalization = True)
```
```
        rdb1 = self.RDB1(low32x_beforeRDB)
        rdb2 = self.RDB2(rdb1)
        rdb3 = self.RDB3(rdb2)
        rdb4 = self.RDB4(rdb3)
        rdb8x = torch.cat((rdb1,rdb2, rdb3, rdb4),dim=1)
```


## Results (RDB block)


## Discussion

alinea over potentiele andere dingen die we hadden kunnen testen(Loss function + RDB)







Original RAW
Un-
edited



Classic





RDB* 2x



RDB* 4x



RDB





