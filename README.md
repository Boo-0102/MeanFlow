# MeanFlow: Unofficial Implementation on CIFAR-10

This repository contains an unofficial implementation for MeanFlow  on CIFAR-10.

## Installation

```bash
# Clone this repository
git clone xxx
cd MeanFlow

# Install dependencies
conda create -n meanflow python==3.10
pip install -r requirements.txt
```



## Training 

**Unconditional**
```bash
cd MeanFlow
sh scripts/train.sh
```
**Conditional**
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
- Run sampling and evaluation for unconditional CIFAR-10
```bash
cd MeanFlow
sh scripts/evaluate.sh
```

## Acknowledgements

This implementation builds upon:
- [SongUnet](https://github.com/NVlabs/edm/blob/main/training/networks.py) (model)
- [REPA](https://github.com/sihyun-yu/REPA/tree/main) (training pipeline)

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
