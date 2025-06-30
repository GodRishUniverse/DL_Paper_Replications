# Deep Learning Model Paper-Replications

> 📝  **For Educational Purposes only**

In this repository I will be replicating ML/DL/AI Papers and Model Architecture from Scratch using PyTorch (I may use Jax sometimes)

I will keep this file updated on how many models implemented so far.

I will be following a simple pipeline - `Implement Block -> Test Block -> Repeat till model is implented -> Test Train`

> TODO: I will be creating script files that will be accomodated in a final script so that the models can be accessed from a single file
>
> So a package will be created

## Activation Functions and Embedding:

* [X] Maxout
* [X] RMSNorm (Jax implementation as well)
* [X] RoPE

## Vision Models:

* [ ] [DDPM] - implementation of DDIM and DDPM is the same however only the noising process of the image is different. DDIM is faster.
  * [ ] Classifier free guidance
* [X] [cGAN]
* [X] CLiP ViT
* [X] [Vision Transfomer]

## Text Models and LLMs:

* [X] [Transformer] - **TODO: fix the error in the decoder cross attention mask**
* [X] [Sparsely Gated MoE]
* [ ] [xLSTM]
* [ ] [Mamba] - State Space Model
* [ ] [Titans]

## Models Trained and Tested So far:

* [X] cGAN
* [X] ViT

## Authors:

- [@GodRishUniverse](https://github.com/GodRishUniverse)
