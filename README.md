# DPF-Net: Dual-Path Frequency-Domain Fusion Network for Underwater Image Enhancement

This repository provides the PyTorch implementation of DPF-Net, a dual-path frequency-domain fusion network for underwater image enhancement.

> Note  
> The repository is currently being organized. The core model files will be released first, and the remaining code, including training scripts, testing scripts, and detailed reproduction instructions, will be uploaded soon.

## Overview

Underwater images often suffer from color distortion, low contrast, haze-like degradation, and detail loss caused by light absorption and scattering. DPF-Net enhances underwater images by explicitly exploiting frequency-domain priors from wavelet and Fourier transforms.

The network first decomposes features into low-frequency and high-frequency components using DWT. The low-frequency path focuses on global color and illumination restoration, while the high-frequency path enhances structural details and textures. During decoding, WaveUp performs cross-frequency interaction and fusion to reconstruct high-quality enhanced images.



@article{ma2026dpfnet,

  title={Underwater Image Enhancement Via Dual-Path Frequency-Domain Fusion Network},
  
  author={Ma, Yupeng and Yang, Qiuling and Zhu, Rongxin and Li, Yifei and Abdel Wahab, Magd M.},
  
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  
  year={2026},
  
  doi={10.1109/TCSVT.2026.3676253}
  
}

