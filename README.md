
# PEANUT

This repository is an official PyTorch implementation of the paper PEANUT: Fast Inference of Visual Autoregressive Model with Adjacency-Adaptive Dynamical Draft Trees.

All main code refers to the project [LANTERN](https://github.com/jadohu/LANTERN)

Thank the LANTERN team for their contributions to the open-source community

---

## 📰 News

- **[2025-11-28] TODO: Change the eagle tree**
- **[2025-11-20] 🎉🎉🎉 PEANUT is released! 🎉🎉🎉**

---

## Performance

Below is a comparison of the effects of different methods

![Performance](data/picture/Performance.png)

---

## ⚙️ Installation

1. **Install Required Packages**
    **Requirements**
    - Python >= 3.10
    - PyTorch >= 2.4.0
    
    Install the dependencies listed in `requirements.txt`.
    ```bash
    git clone https://github.com/Haodong-Lei-Ray/PEANUT.git
    cd PEANUT
    conda create -n PEANUT python=3.10 -y
    conda activate PEANUT
    pip install -r requirements.txt
    ```

2. **Additional Setup**
    1. **Lumina-mGPT**
        For [Lumina-mGPT](https://github.com/Alpha-VLLM/Lumina-mGPT), we need to install `flash_attention` and `xllmx` packages.
        ```bash
        pip install flash-attn --no-build-isolation
        cd models/base_models/lumina_mgpt
        pip install -e .
        ```

3. **Checkpoints**
    All model weights and other required data should be stored in `ckpts/`.
    1. **Lumina-mGPT**
        For Lumina-mGPT, since currently the Chameleon implementation in transformers does not contain the VQ-VAE decoder, please manually download the original VQ-VAE weights [provided by Meta](https://github.com/facebookresearch/chameleon) and put them to the following directory:
        ```
        ckpts
        └── lumina_mgpt
            └── chameleon
                └── tokenizer
                    ├── text_tokenizer.json
                    ├── vqgan.yaml
                    └── vqgan.ckpt
        ```

        Also download the original model [`Lumina-mGPT-7B-768`](https://huggingface.co/Alpha-VLLM/Lumina-mGPT-7B-768) from Huggingface 🤗 and put them to the following directory:
        ```
        ckpts
        └── lumina_mgpt
            └── Lumina-mGPT-7B-768
                ├── config.json
                ├── generation_config.json
                ├── model-00001-of-00002.safetensors
                └── other files...
        ```
    2. **Anole**
        For Anole, download [`Anole-7b-v0.1-hf`](https://huggingface.co/leloy/Anole-7b-v0.1-hf), which is a huggingface style converted model from [`Anole`](https://huggingface.co/GAIR/Anole-7b-v0.1). 
        
        In addition, you should download the original VQ-VAE weights [provided by Meta](https://github.com/facebookresearch/chameleon) and put them to the following directory:

        ```
        ckpts
        └── anole
            ├── Anole-7b-v0.1-hf
            |   ├── config.json
            |   ├── generation_config.json
            |   ├── model-00001-of-00003.safetensors
            |   └── other files...
            └── chameleon
                └── tokenizer
                    ├── text_tokenizer.json
                    ├── vqgan.yaml
                    └── vqgan.ckpt
        ```

        **(Optional) Trained drafter**
        To use trained drafter, you need to download [`anole_drafter`](https://huggingface.co/jadohu/anole_drafter) and save it under trained_drafters directory.
        ```
        ckpts
        └── anole
            └── trained_drafters
                └── anole_drafter
                    ├── config.json
                    ├── generation_config.json
                    ├── pytorch_model.bin
                    └── other files...
        ```

---

## ✨ Usage



## ⚖️ License

This project is distributed under the Chameleon License by Meta Platforms, Inc. For more information, please see the `LICENSE` file in the repository.

---

## 🙏 Acknowledgement
This repository is built with extensive reference to [FoundationVision/LlamaGen](https://github.com/FoundationVision/LlamaGen), [Alpha-VLLM/Lumina-mGPT](https://github.com/Alpha-VLLM/Lumina-mGPT) and [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE), leveraging many of their core components and approaches.

<!-- ---

## 📄 Citation

```
@article{jang2024lantern,
  title={LANTERN: Accelerating Visual Autoregressive Models with Relaxed Speculative Decoding},
  author={Jang, Doohyuk and Park, Sihwan and Yang, June Yong and Jung, Yeonsung and Yun, Jihun and Kundu, Souvik and Kim, Sung-Yub and Yang, Eunho},
  journal={arXiv preprint arXiv:2410.03355},
  year={2024}
}
@article{park2025lanternenhancedrelaxedspeculative,
  title={LANTERN++: Enhanced Relaxed Speculative Decoding with Static Tree Drafting for Visual Auto-regressive Models}, 
  author={Sihwan Park and Doohyuk Jang and Sungyub Kim and Souvik Kundu and Eunho Yang},
  journal={arXiv preprint arXiv:2410.03355},
  year={2025}
}
``` -->