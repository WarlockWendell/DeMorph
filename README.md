### DeMorph

Official PyTorch implementation of "DeMorph: A Decoupled Deformation Model with Anatomical Guidance for Predicting Longitudinal Brain Atrophy".

#### Environment
We test our code with the following environment:

- Python 3.10
- PyTorch 2.5.1+cu121
- monai 1.5.0
- monai-generative 0.2.3

#### Usage

- Download the pretrained [weight]() to `./weights/`.
- Inference with the following command:
    ```bash
    cd src
    python inference.py --config ./configs/demo.yaml
    ```

#### TODO

- [] Release training code.

#### Acknowledgement

Below we show our appreciation for the exceptional work and generous contribution to open source.

- [MONAI](https://github.com/Project-MONAI/MONAI): A PyTorch-based, open-source framework for deep learning in healthcare imaging.

- [WAN2.1](https://github.com/Wan-Video/Wan2.1): A comprehensive and open suite of video foundation models that pushes the boundaries of video generation.

- [VoxelMorph](https://github.com/voxelmorph/voxelmorph): A general purpose library for learning-based tools for alignment/registration, and more generally modelling with deformations.

- [BrLP](https://github.com/LemuelPuglisi/BrLP): A wanderful work on modeling longitudinal brain MRI progression.


#### Citation

If you find this work useful in your research, please consider citing:

```
```