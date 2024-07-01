# Adv-KD: Adversarial Knowledge Distillation for Faster Diffusion Sampling

Diffusion Probabilistic Models (DPMs) have emerged as a powerful class of deep generative models, achieving remarkable performance in image synthesis tasks. However, these models face challenges in terms of widespread adoption due to their reliance on sequential denoising steps during sample generation. This dependence leads to substantial computational requirements, making them unsuitable for resource-constrained or real-time processing systems. To address these challenges, we propose a novel method that integrates denoising phases directly into the model's architecture, thereby reducing the need for resource-intensive computations. 

Our approach combines diffusion models with generative adversarial networks (GANs) through knowledge distillation, enabling more efficient training and evaluation. By utilizing a pre-trained diffusion model as a teacher model, we train a student model through adversarial learning, employing layerwise transformations for denoising and submodules for predicting the teacher model's output at various points in time. This integration significantly reduces the number of parameters and denoising steps required, leading to improved sampling speed at test time.

We validate our method with extensive experiments, demonstrating comparable performance with reduced computational requirements compared to existing approaches. By enabling the deployment of diffusion models on resource-constrained devices, our research mitigates their computational burden and paves the way for wider accessibility and practical use across the research community and end-users.

# Citation
If you find this repo useful, please consider citing our work:
```
@article{mekonnen2024adv,
  title={Adv-KD: Adversarial Knowledge Distillation for Faster Diffusion Sampling},
  author={Mekonnen, Kidist Amde and Dall'Asen, Nicola and Rota, Paolo},
  journal={arXiv preprint arXiv:2405.20675},
  year={2024}
}
```
