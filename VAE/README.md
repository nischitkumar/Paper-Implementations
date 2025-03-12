# VAE Implementation  

## Overview  

This repository contains an implementation of a **Variational Autoencoder (VAE)** from scratch in Python using PyTorch.
The implementation follows the original VAE architecture proposed by **Kingma & Welling** in *Auto-Encoding Variational Bayes (2013)*.  

## Implementation Details  

The VAE consists of two main components:  
- **Encoder:** Maps input data to a latent distribution.  
- **Decoder:** Reconstructs data from latent space representations.  

The model is trained by optimizing the **Evidence Lower Bound (ELBO)**, which consists of a reconstruction loss and a KL divergence term to regularize the latent space.  

### Key Features  
- Implemented using PyTorch.  
- Uses the **reparameterization trick** to enable gradient-based optimization.  
- Trains on a chosen dataset (MNIST is used in this implementation).  
- Supports configurable hyperparameters such as learning rate, batch size, and latent dimension.  

## References  
- [*Kingma & Welling, "Auto-Encoding Variational Bayes", 2013.*](https://arxiv.org/abs/1312.6114)  