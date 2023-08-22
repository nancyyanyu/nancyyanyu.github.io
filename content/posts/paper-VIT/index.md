---
title: "Paper Note: ViT"
categories: Machine Learning
math: true
tags:
  - Paper
  - NLP
comments: true
date: 2023-08-15 22:08:51
---


> ViT ***applies a standard Transformer directly to images***

<!--more-->

- Split an image into **patches;**
- Provide the sequence of linear embeddings of these patches as an input to a Transformer.

Image patches are treated the same way as tokens (words) in an NLP application. We train the model on image classification in supervised fashion.
{{< imgproc Untitled Resize "800x" />}}


**Methods**

1. Reshape the image $x ∈ R^{H×W×C }$ into a sequence of flattened 2D patches
    1. $N =HW/P^2=224 \times 224/16^2=196$ is the resulting number of patches
2. Flatten the patches and map to $D=768$ dimensions with a trainable **linear projection**
    1. the output of this projection as the **patch embeddings**.

3. Prepend a learnable embedding to the sequence of embedded patches $\mathbf{z}^0_0$ = $\mathbf{x}_{class}$  (1x768), whose state at the output of the Transformer encoder $\mathbf{z}^L_0$ serves as the **image representation** $\mathbf{y}$.  
4. **Position embeddings** ($E_{pos} ∈ \mathbb{R}^{(N+1)×D}$  or $\mathbb{R}^{197 \times 768}$ ) are added to the patch embeddings to retain positional information.

5. A classification head is attached to $\mathbf{z}_0^L$. The classification head is implemented by a MLP with one hidden layer at pre-training time and by a single linear layer at fine-tuning time.

{{< imgproc Untitled1 Resize "700x" />}}
{{< imgproc Untitled2 Resize "400x" />}}

**Reference:**

- https://github.com/google-research/vision_transformer
- https://arxiv.org/abs/2010.11929
