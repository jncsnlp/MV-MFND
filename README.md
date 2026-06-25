# Don't Judge From A Single Perspective: LVLM-Based Multi-View Multimodal Fake News Detection
![architecture](图片链接)

This repository contains the official PyTorch implementation code for Don't Judge From a Single Perspective: LVLM-Based Multiview Multimodal Fake News Detection: <a href="https://ieeexplore.ieee.org/abstract/document/11543470">MV-MFND</a>.


## Installation
First, clone the repository locally:
```
git clone git链接
cd MV-MFND
```

## Requirements

Seeing in requirement.txt

For Qwen, you should cd Qwen2-VL and use `pip install -r requirement.txt` to install the required packages.
For LLaVa, you should cd llava and use `pip install -r requirement.txt` to install the required packages.

 ## Usage

baseline LEMMA:

```
cd LEMMA-main
python lemma.py
```

other baselines are in Qwen2-VL and llava folder:

```
python + folder name.py
```

The method proposed in our paper:

```
cd Qwen2-VL
python dataset name + main.py
```

## Acknowledgement
We refer to the code of LEMMA. Thanks for their great contributions!

## Cite

```
@article{lu2026don,
  title={Don’t Judge From a Single Perspective: LVLM-Based Multiview Multimodal Fake News Detection},
  author={Lu, Heng-yang and Tang, Bin and Liu, Xinnan and Zhan, Qianyi and Fan, Chenyou and Fang, Wei},
  journal={IEEE Transactions on Computational Social Systems},
  year={2026},
  publisher={IEEE}
}
```
Contact: luhengyang@jiangnan.edu.cn

If you encounter any difficulties or problems while using our dataset, please feel free to contact us. If you find our paper or code helpful, please give us a like. ❤️

