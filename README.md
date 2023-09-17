# Advanced Machine Learning Class Repository

This repository contains the slides and related materials for the talks we presented in the Advanced Machine Learning class which is designed as a journal-club where students present and discuss ML papers. 

## Talks

### Talk 1: Score-Based Generative Modeling through Stochastic Differential Equations

- Paper Title: Score-Based Generative Modeling through Stochastic Differential Equations
- Authors: Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole

#### Abstract
The paper introduces diffusion models under the SDE framework and thus allows a better understanding of them relying on the SDE theory and Ito's calculus.  
They make the following contributions :  
* Unified framework of several diffusion methods
* Controllable generation of the samples
* Flexible sampling and likelihood computation

#### Talk Slides
- PDF version of the slides is available [here](https://github.com/yass43/AML/blob/main/Talk_1/Presentation_aml_pdf)
- Marp markdown version of the slides is available [here](https://github.com/yass43/AML/blob/main/Talk_1/Presentation_aml.md)

### Talk 2: Unsupervised Representation Learning from Pretrained Diffusion Probabilistic 

- Paper Title: Unsupervised Representation Learning from Pretrained Diffusion Probabilistic 
- Authors: Zijian Zhang, Zhou Zhao and Zhijie Lin

#### Abstract
The Pre-trained DPM AutoEncoding (PDAE) approach enhances decoder efficiency and performance by addressing information loss in pre-trained DPMs during image reconstruction. This novel technique involves predicting a mean shift based on encoded representations, effectively filling the information gap and compelling the encoder to learn more from images, ultimately resulting in meaningful and efficient representation learning.
They make the following contributions :  
* Observed that when diffusing an image, the reconstruction gap  is smaller in the conditionnal case than in the conditionnal case 
* Designed a VAE that will learn a representation that allows the latent reconstruction gap as small as possible
* Found a critical time zone in which the mean shift has a large impact
  
#### Talk Slides
- PDF version of the slides is available [here](https://github.com/yass43/AML/blob/main/Talk_2/Talk_2_presentation.pdf)
- Marp markdown version of the slides is available [here](https://github.com/yass43/AML/blob/main/Talk_2/Talk_2_presentation.pdf)


