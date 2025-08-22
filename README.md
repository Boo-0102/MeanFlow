# MeanFlow: Unofficial Implementation on CIFAR-10

This repository provides an unofficial PyTorch implementation of the paper **[Mean Flows for One-step Generative Modeling](https://arxiv.org/abs/2505.13447)** on the CIFAR-10 dataset.

## Installation

```bash
# Clone this repository
git clone https://github.com/Boo-0102/MeanFlow.git
cd MeanFlow

# Create and activate a conda environment:
conda create -n meanflow python==3.10
conda activate meanflow

# Install dependencies
pip install -r requirements.txt
```



## Training 

**Unconditional**
- To train an unconditional model, run the following script:
```bash
cd MeanFlow
sh scripts/train.sh
```
**Conditional**
- To train a class-conditional model with CFG, run the following script:
```bash
cd MeanFlow
sh scripts/train_cfg.sh
```
## Sampling
**Unconditional**
```bash
cd MeanFlow
sh scripts/sample.sh
```
**Conditional**
```bash
cd MeanFlow
sh scripts/sample_cfg.sh
```

## Evaluate
- To generate samples and calculate the FID score for a trained unconditional model, run:
```bash
cd MeanFlow
sh scripts/evaluate.sh
```

## Acknowledgements

This implementation is based on or inspired by the code from the following repositories:
- [SongUnet](https://github.com/NVlabs/edm/blob/main/training/networks.py)
- [REPA](https://github.com/sihyun-yu/REPA/tree/main)
- [MeanFlow](https://github.com/zhuyu-cs/MeanFlow)

## Citation
If you find this implementation useful, please cite the original paper:
```
@article{geng2025mean,
  title={Mean Flows for One-step Generative Modeling},
  author={Geng, Zhengyang and Deng, Mingyang and Bai, Xingjian and Kolter, J Zico and He, Kaiming},
  journal={arXiv preprint arXiv:2505.13447},
  year={2025}
}
```
## License

[MIT License](LICENSE)
