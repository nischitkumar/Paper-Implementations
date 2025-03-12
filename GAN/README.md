# GAN Implementation  

## Overview  

This repository contains an implementation of a **Generative Adversarial Network (GAN)** from scratch in Python using PyTorch. 
The implementation follows the original GAN architecture proposed by **Ian Goodfellow et al.** in *Generative Adversarial Nets (2014)*.  

## Implementation Details  

The implementation consists of two neural networks:  
- **Generator:** Learns to generate realistic samples from random noise.  
- **Discriminator:** Distinguishes between real and generated samples.  

Both networks are trained in an adversarial manner, where the generator improves by trying to fool the discriminator, and the discriminator improves by correctly classifying real and fake samples.  

### Key Features  
- Implemented using TensorFlow/PyTorch.  
- Uses the standard adversarial loss function.  
- Trains on a chosen dataset (MNIST is used in this implementation).  
- Supports configurable hyperparameters such as learning rate, batch size, and number of epochs.  

## References  
- [*Goodfellow et al., "Generative Adversarial Nets", 2014.*](https://arxiv.org/abs/1406.2661)  

