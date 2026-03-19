# Deep-dsRNAPred 🪲🧬

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch 2.8.0](https://img.shields.io/badge/PyTorch-2.8.0%2Bcu128-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Deep-dsRNAPred** is a high-precision deep learning framework designed for screening efficient double-stranded RNA (dsRNA) sequences targeting *Tribolium castaneum* (red flour beetle) for RNA interference (RNAi) based pest control.


## 📖 Introduction

<div align="justify">

RNA interference (RNAi) is a revolutionary technology in pest management, but its real-world efficacy is heavily bottlenecked by the unpredictable silencing efficiency of designed dsRNA sequences. 

**Deep-dsRNAPred** solves this by offering a highly accurate computational pre-screening tool. Our model utilizes the **RNAErnie** pre-trained foundation model for sequence feature extraction and overcomes traditional length limitations using a novel linear interpolation algorithm for positional embeddings. The extracted features are then processed through a synergistic deep learning architecture comprising **Multi-scale Selective Kernel Attention (SKAttention)**, **Convolutional Block Attention Module (CBAM)**, and **Bidirectional LSTM (BiLSTM)** to capture features ranging from local nucleotide motifs to global spatial structures.

</div>

## ⚙️ Architecture
![图片1](https://github.com/user-attachments/assets/2aef0278-d6cb-4e58-80ca-66551800a24a)




## 💻 Installation

We recommend using [Anaconda](https://www.anaconda.com/) or Miniconda to manage your environment. 

**1. Clone the repository:**
```bash
git clone [https://github.com/LarryLjc/Deep-dsRNAPred.git](https://github.com/LarryLjc/Deep-dsRNAPred.git)
cd Deep-dsRNAPred
