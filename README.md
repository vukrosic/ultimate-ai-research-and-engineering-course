# A Structured Path for Learning AI / ML Research & Engineering

I'm developing this course to contain **everything you need** to:

* Join elite labs like **OpenAI**, **Google**, or **MIT**
* Independently publish **groundbreaking open-source research**
* Build **world-class models**


## 📺 [My YouTube Videos](https://www.youtube.com/channel/UC7XJj9pv_11a11FUxCMz15g)

Follow along with course walkthroughs, video tutorials, and explanations.



## ⚡ Speedruns

For advanced learners, check out the [Speedruns](speedruns):
Research / engineering / optimization challenges that help you:

* Contribute to open-source
* Build real skills by doing


# 📚 Course

📘 [Introduction & Motivation](001%20Introduction%20%26%20Motivation)


## 🐍 Don't Know Python?

Start with the [Beginner Python Course](beginner-course) to get up to speed.

### 🗂️ Tip

Make a copy of the notebooks:
**Open notebook → File → Save a copy to Dive**

# Fundamentals (Most important)

- Intro course - Deep Learning by Professor Bryce - [YouTube](https://www.youtube.com/playlist?list=PLgPbN3w-ia_PeT1_c5jiLW3RJdR7853b9)

- PyTorch Fundamentals: From Linear Layers & Weight Intuition to LayerNorm, Variance, and Custom ML Blocks - [Google Colab](https://colab.research.google.com/drive/1Mk37B4ISgmhTNDEVTCB3R_Fwp5zEEqVS?usp=sharing) - [YouTube](https://youtu.be/QtlDV2r1ryE) - [Bilibili](https://www.bilibili.com/video/BV17LGczLED3/)

- Code Softmax, Cross-Entropy, and Gradients — From Scratch (No Torch) (_In development_) - [Googe Colab](https://colab.research.google.com/drive/1eCRAS6c0Fdy3PBitC2aztSDzp_CoSa_W?usp=sharing)

    ## Backpropagation

- Chain Rule & Backpropagation From Scratch [Google Colab](https://colab.research.google.com/drive/1wcNWayyiB8i4fjKEYmsiIuJmGEN5lTQK?usp=sharing)

    ## Matrix Multiplication

- Comparing MatMul: PyTorch Native vs Tiling vs Quantization (_In development_) - [Google Colab](https://colab.research.google.com/drive/1a_tkXxZ0gt3gFd52IP25bwrVvL8Cenyu?usp=sharing)

- Make Matrix Multiply 3x Faster by Padding Size to Power of 2 - [Google Colab](https://colab.research.google.com/drive/1VKIQS5ocefunYkoE-8uFz_0_xOtkBelG?usp=sharing)

- How Matrix Shape Affects Performance on Nvidia T4 Tensor Cores - _(in development)_ - [Google Colab](https://colab.research.google.com/drive/1eiWkfbrNv2GW7kDMty1jf_xzrEuqHOrO?usp=sharing)

- TODO: how to optimize matmuls on specific GPUs

    ## Training LLMs From Scratch

- Experimenting With Small Character-Level LLM:
Hyperparameters, Optimization, and Model Scaling - [Paper](https://drive.google.com/file/d/1sXN-c-L7z3ku29N4QVp6mAP7ZuX7B7Xf/view?usp=sharing) - [Google Colab](https://colab.research.google.com/drive/11bc71DzTe95XDq6IRbJPQtAy_pVD9fhC?usp=sharing)

- Train a Small LLM From Scratch In 50 Min - [Google Colab](https://colab.research.google.com/drive/1NUopXFOY_VDI_o72TEAnzfeSKmAPuz_9?usp=sharing)
    ## Diffusion Models
- Simplest diffusion model to generate points on a circle - [Google Colab](https://colab.research.google.com/drive/1alWuxOD8PiD1D7rbumKuFkEkzkj1JsbI?usp=sharing)
- Code & train a small diffusion model to calculate A mod B - [Google Colab](https://colab.research.google.com/drive/1lDMgngIQBL0btjavGcdwktZtixGSpE1I?usp=sharing)

    ## Other (important) Models
- Understand Simple Autoencoder - [Google Colab](https://colab.research.google.com/drive/18dZm4moQmuZOfVXKwlClmPhAcjV4rtIg?usp=sharing)
    > I had no idea autoencoders are so quick to train, a few seconds for autoencoder of numbers (0-10,000):

    > Encoder takes a number (56) -> vector embedding [0.3, 0.7, 0.42,...] -> decoder aims to predict the encoded number (56) from the vector embedding - these vector embeddings contain rich representation of the encoded number (token, sentence,...) that can be used in a models like LLMs, diffusion,...

    > I'm figuring out autoencoders as I think LLMs should process sentences, not tokens, as sentences can represent infinite number of concepts, as opposed to limited token vocabulary (usually about 150K)

    > Predicting over infinite distribution requires diffusion models (like seemingly infinite number of possible images), as autoregressive would just predict the blury average of the image, sentence, without any meaning.

    > Also diffusion model allows us to have truly unified training in the same latent space for visual and text data.


# High Performance on Hopper GPUs (H100, H200, H800)

- TMA (Tensor Memory Accelerator) alignment for fast memory on Hopper GPUs (DeepSeek's speed) - [Google Colab](https://colab.research.google.com/drive/1F6CNQND2F9a4yLLYqorNAkKEzVxQurCa?usp=sharing)

- High-Performance GPU Matrix Multiplication on H800, H100 & H200 from Scratch - [Google Colab](https://colab.research.google.com/drive/1zxrSNFySwuNycT30Huy3bjxvoEjHbrMa?usp=sharing)

# Fun experiments

- Looking for patterns in trained neural network weights - [Google Colab](https://colab.research.google.com/drive/1P7KreHpJcZL4vjDrRqd69eqsEFYl_2Oa?usp=sharing) - [Preview PDF Analysis](https://file.notion.so/f/f/795d9b1f-4854-4c8d-8295-2ca702b9d498/439a9db5-3835-4ef6-82df-d2576aed18a2/Looking_for_patterns_in_trained_neural_network_weights.pdf?table=block&id=22d7982f-d437-80e7-96f4-e1d80912ff49&spaceId=795d9b1f-4854-4c8d-8295-2ca702b9d498&expirationTimestamp=1752271200000&signature=fV4tMNJ0SwedGljjU3-guwNywDRrhy2XvFrDPWMgqBI&downloadName=Looking+for+patterns+in+trained+neural+network+weights.pdf) _In development_