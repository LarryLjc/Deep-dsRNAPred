# Deep-dsRNAPred 🪲🧬

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch 2.8.0](https://img.shields.io/badge/PyTorch-2.8.0%2Bcu128-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<div align="justify">
  
**Deep-dsRNAPred** is a high-precision deep learning framework designed for screening efficient double-stranded RNA (dsRNA) sequences targeting *Tribolium castaneum* (red flour beetle) for RNA interference (RNAi) based pest control.

</div>

## 📖 Introduction

<div align="justify">

RNA interference (RNAi) is a revolutionary technology in pest management, but its real-world efficacy is heavily bottlenecked by the unpredictable silencing efficiency of designed dsRNA sequences. 

Developed **Deep-dsRNAPred**, a deep learning framework designed to screen highly efficient dsRNA sequences for pest control targeting Tribolium castaneum. Leveraging the RiNALMo-mega pre-trained large language model for feature extraction, the model innovatively integrates Multi-scale Selective Kernel Attention (SKAttention), CBAM, and BiLSTM mechanisms to hierarchically capture sequence features from local motifs to global structures. Experiments demonstrate that Deep-dsRNAPred achieved an Accuracy of 0.9535 and an AUC of 0.9891, improving upon the existing state-of-the-art model by 9.83% and 6.53%, respectively.

</div>

## ⚙️ Architecture


<img width="4984" height="3227" alt="图片2" src="https://github.com/user-attachments/assets/c9f0eef1-87e3-4b05-8ad3-f3586e430ae1" />




## 💻 Installation

We recommend using [Anaconda](https://www.anaconda.com/) or Miniconda to manage your environment. 

**1. Clone the repository:**
```bash
git clone [https://github.com/LarryLjc/Deep-dsRNAPred.git](https://github.com/LarryLjc/Deep-dsRNAPred.git)
cd Deep-dsRNAPred
