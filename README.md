# Transformer-Based Architectures in Deep Learning

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This repository provides implementations of various Transformer-based architectures for deep learning. Additionally, this README outlines the current status of the project and suggests future work to be done.

## Table of Contents

- [Introduction](#introduction)
- [Implemented Architectures](#implemented-architectures)
  - [Vision Transformer (ViT)](#vision-transformer-vit)
  - [Detection Transformer (DETR)](#detection-transformer-detr)
- [Citation](#Citation)

## Introduction

Transformers have revolutionized various fields in deep learning, including natural language processing and computer vision. These models are known for their parallelizable and scalable nature, making them highly effective for various tasks. This repository explores the implementation of Transformer-based architectures in deep learning, primarily focusing on Vision Transformers (ViT) and Detection Transformers (Detr).

## Implemented Architectures

### Vision Transformer (ViT)

The Vision Transformer (ViT) is a novel architecture that applies the Transformer model to computer vision tasks, such as image classification. In this repository, you can find an implementation of the ViT model.

### Detection Transformer (DETR)

The Detection Transformer (Detr) is a Transformer-based architecture designed for object detection tasks. It combines the power of Transformers with object detection, making it efficient and accurate. This repository includes an implementation of Detr, complete with pre-trained models and utilities for object detection tasks.

## Citation

If you find this repository useful in your research or project, please consider citing the original Vision Transformer (ViT) paper:

```markdown
@article{dosovitskiy2020image,
  title={An image is worth 16x16 words: Transformers for image recognition at scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and others},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}
