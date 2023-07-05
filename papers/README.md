# Paper List
List of papers not just about speech synthesis 😀.


<h2>Content</h2>

* [TTS Frontend](#1)
* [Acoustic Model](#2)
    * [Autoregressive Model](#21)
    * [Non-Autoregressive Model](#22)
    * [Alignment Study](#23)
    * [Data Efficiency](#24)
* [Vocoder](#3)
    * [Autoregressive Model](#31)
    * [Non-Autoregressive Model](#32)
    * [Others](#33)
* [TTS towards Stylization](#4)
    * [Expressive TTS](#41)
    * [MultiSpeaker TTS](#42)
    * [New Perspective on TTS](#43)
* [Voice Conversion](#5)
    * [ASR & TTS Based](#51)
    * [VAE & Auto-Encoder Based](#52)
    * [GAN Based](#53)
* [Singing](#6)
    * [Singing Voice Synthesis](#61)
    * [Singing Voice Conversion](#62)
* [Speech Processing Related](#7)
    * [Speech Pretrained Model](#71)
    * [Speech Separation](#72)
    * [Speaker Verification](#73)
    * [Audio Super Resolution](#74)
    * [Tools](#75)
* [Natural Language Processing](#8)
    * [Sequence Modeling](#81)
    * [Pretrained Model](#82)
    * [Non-autoregressive Translation Model](#83)
    * [Speech2Speech Translation Model](#84)
    * [Neural Machine Reading Comprehension](#85)
* [VAE & GAN](#9)
    * [VAE](#91)
    * [GAN](#92)
* [Others](#10)


<h2 id="1">TTS Frontend</h2>

- [Pre-trained Text Representations for Improving Front-End Text Processing in Mandarin Text-to-Speech Synthesis](https://pdfs.semanticscholar.org/6abc/7dac0bdc50735b6d12f96400f59b5f084759.pdf) (Interspeech 2019)
- [A unified sequence-to-sequence front-end model for Mandarin text-to-speech synthesis](https://arxiv.org/pdf/1911.04111.pdf) (ICASSP 2020)
- [A hybrid text normalization system using multi-head self-attention for mandarin](https://arxiv.org/pdf/1911.04128.pdf) (ICASSP 2020)
- [Unified Mandarin TTS Front-end Based on Distilled BERT Model](https://arxiv.org/pdf/2012.15404.pdf) (2021-01)


<h2 id="2">Acoustic Model</h2>

<h3 id="21">Autoregressive Model</h3>

- Tacotron V1: [Tacotron: Towards End-to-End Speech Synthesis](https://arxiv.org/abs/1703.10135) (Interspeech 2017)
- Tacotron V2: [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/abs/1712.05884) (ICASSP 2018)
- Deep Voice V1: [Deep Voice: Real-time Neural Text-to-Speech](https://arxiv.org/abs/1702.07825) (ICML 2017)
- Deep Voice V2: [Deep Voice 2: Multi-Speaker Neural Text-to-Speech](https://arxiv.org/abs/1705.08947) (NeurIPS 2017)
- Deep Voice V3: [Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning](https://arxiv.org/abs/1710.07654) (ICLR 2018)
- Transformer-TTS: [Neural Speech Synthesis with Transformer Network](https://arxiv.org/abs/1809.08895) (AAAI 2019)
- DurIAN: [DurIAN: Duration Informed Attention Network For Multimodal Synthesis](https://arxiv.org/abs/1909.01700) (2019)
- [Location-Relative Attention Mechanisms For Robust Long-Form Speech Synthesis](https://arxiv.org/abs/1910.10288) (ICASSP 2020)
- Flowtron (flow based): [Flowtron: an Autoregressive Flow-based Generative Network for Text-to-Speech Synthesis](https://arxiv.org/abs/2005.05957) (2020)
- [Non-Attentive Tacotron: Robust and Controllable Neural TTS Synthesis Including Unsupervised Duration Modeling](https://arxiv.org/pdf/2010.04301v1.pdf) (under review ICLR 2021)
- RobuTrans (towards robust): [RobuTrans: A Robust Transformer-Based Text-to-Speech Model](https://ojs.aaai.org//index.php/AAAI/article/view/6337) (AAAI 2020)
- DeviceTTS: [DeviceTTS: A Small-Footprint, Fast, Stable Network for On-Device Text-to-Speech](https://arxiv.org/abs/2010.15311) (2020-10)
- Wave-Tacotron: [Wave-Tacotron: Spectrogram-free end-to-end text-to-speech synthesis](https://arxiv.org/abs/2011.03568) (2020-11)
- Streaming Acoustic Modeling: [Transformer-based Acoustic Modeling for Streaming Speech Synthesis](https://research.fb.com/wp-content/uploads/2021/06/Transformer-based-Acoustic-Modeling-for-Streaming-Speech-Synthesis.pdf) (2021-06)
- Apple TTS system: [On-device neural speech synthesis](https://arxiv.org/abs/2109.08710) (ASRU 2021)

<h3 id="22">Non-Autoregressive Model</h3>

- ParaNet: [Non-Autoregressive Neural Text-to-Speech](https://arxiv.org/pdf/1905.08459.pdf) (ICML 2020)
- FastSpeech: [FastSpeech: Fast, Robust and Controllable Text to Speech](https://arxiv.org/abs/1905.09263) (NeurIPS 2019)
- JDI-T: [JDI-T: Jointly trained Duration Informed Transformer for Text-To-Speech without Explicit Alignment](https://arxiv.org/abs/2005.07799) (2020)
- EATS: [End-to-End Adversarial Text-to-Speech](https://arxiv.org/pdf/2006.03575.pdf) (2020)
- FastSpeech 2: [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558) (2020)
- FastPitch: [FastPitch: Parallel Text-to-speech with Pitch Prediction](https://arxiv.org/pdf/2006.06873.pdf) (2020)
- Glow-TTS (flow based, Monotonic Attention): [Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search](https://arxiv.org/abs/2005.11129) (NeurIPS 2020)
- Flow-TTS (flow based): [Flow-TTS: A Non-Autoregressive Network for Text to Speech Based on Flow](https://ieeexplore.ieee.org/document/9054484) (ICASSP 2020)
- SpeedySpeech: [SpeedySpeech: Efficient Neural Speech Synthesis](https://arxiv.org/pdf/2008.03802.pdf) (Interspeech 2020)
- Parallel Tacotron: [Parallel Tacotron: Non-Autoregressive and Controllable TTS](https://arxiv.org/abs/2010.11439) (2020)
- BVAE-TTS: [Bidirectional Variational Inference for Non-Autoregressive Text-to-Speech](https://openreview.net/forum?id=o3iritJHLfO) (ICLR 2021)
- LightSpeech: [LightSpeech: Lightweight and Fast Text to Speech with Neural Architecture Search](https://arxiv.org/abs/2102.04040) (ICASSP 2021)
- Parallel Tacotron 2: [Parallel Tacotron 2: A Non-Autoregressive Neural TTS Model with Differentiable Duration Modeling](https://arxiv.org/pdf/2103.14574.pdf) (2021)
- Grad-TTS: [Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech](https://arxiv.org/abs/2105.06337) (ICML 2021)
- VITS (flow based): [Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech](https://arxiv.org/abs/2106.06103) (ICML 2021)
- RAD-TTS: [RAD-TTS: Parallel Flow-Based TTS with Robust Alignment Learning and Diverse Synthesis](https://openreview.net/pdf?id=0NQwnnwAORi) (ICML 2021 Workshop)
- WaveGrad 2: [WaveGrad 2: Iterative Refinement for Text-to-Speech Synthesis](https://arxiv.org/pdf/2106.09660.pdf) (Interspeech 2021)
- PortaSpeech: [PortaSpeech: Portable and High-Quality Generative Text-to-Speech](https://arxiv.org/abs/2109.15166) (NeurIPS 2021)
- DelightfulTTS (To synthesize natural and high-quality speech from text): [DelightfulTTS: The Microsoft Speech Synthesis System for Blizzard Challenge 2021](https://arxiv.org/pdf/2110.12612.pdf) (Blizzard Challenge 2021)
- DiffGAN-TTS: [DiffGAN-TTS: High-Fidelity and Efficient Text-to-Speech with Denoising Diffusion GANs](https://arxiv.org/abs/2201.11972) (2022-01)
- [BDDM: Bilateral Denoising Diffusion Models for Fast and High-Quality Speech Synthesis](https://arxiv.org/abs/2203.13508) (ICLR 2022)
- JETS: [JETS: Jointly Training FastSpeech2 and HiFi-GAN for End to End Text to Speech](https://arxiv.org/abs/2203.16852) (Interspeech 2022)
- WavThruVec: [WavThruVec: Latent speech representation as intermediate features for neural speech synthesis](https://arxiv.org/pdf/2203.16930.pdf) (2022-03)
- FastDiff: [FastDiff: A Fast Conditional Diffusion Model for High-Quality Speech Synthesis](https://arxiv.org/abs/2204.09934) (IJCAI 2022)
- NaturalSpeech: [NaturalSpeech: End-to-End Text to Speech Synthesis with Human-Level Quality](https://arxiv.org/abs/2205.04421) (2022-05)
- DelightfulTTS 2: [DelightfulTTS 2: End-to-End Speech Synthesis with Adversarial Vector-Quantized Auto-Encoders](https://arxiv.org/pdf/2207.04646.pdf) (Interspeech 2022)
- CLONE: [Controllable and Lossless Non-Autoregressive End-to-End Text-to-Speech](https://arxiv.org/abs/2207.06088) (2022-07)
- MB-iSTFT-VITS: [Lightweight and High-Fidelity End-to-End Text-to-Speech with Multi-Band Generation and Inverse Short-Time Fourier Transform](https://arxiv.org/abs/2210.15975) (2022-10)
- PITS: [Variational Pitch Inference without Fundamental Frequency for End-to-End Pitch-controllable TTS](https://arxiv.org/pdf/2302.12391.pdf) (2023-02)

<h3 id="23">Alignment Study</h3>

- Monotonic Attention: [Online and Linear-Time Attention by Enforcing Monotonic Alignments](https://arxiv.org/abs/1704.00784) (ICML 2017)
- Monotonic Chunkwise Attention: [Monotonic Chunkwise Attention](https://arxiv.org/abs/1712.05382) (ICLR 2018)
- [Forward Attention in Sequence-to-sequence Acoustic Modelling for Speech Synthesis](https://arxiv.org/abs/1807.06736) (ICASSP 2018)
- RNN-T for TTS: [Initial investigation of an encoder-decoder end-to-end TTS framework using marginalization of monotonic hard latent alignments](http://128.84.4.27/pdf/1908.11535) (2019)
- [Location-Relative Attention Mechanisms For Robust Long-Form Speech Synthesis](https://arxiv.org/abs/1910.10288) (ICASSP 2020)
- [Non-Attentive Tacotron: Robust and Controllable Neural TTS Synthesis Including Unsupervised Duration Modeling](https://arxiv.org/pdf/2010.04301v1.pdf) (under review ICLR 2021)
- EfficientTTS: [EfficientTTS: An Efficient and High-Quality Text-to-Speech Architecture](https://arxiv.org/abs/2012.03500) (2020-12)
- VAENAR-TTS: [VAENAR-TTS: Variational Auto-Encoder based Non-AutoRegressive Text-to-Speech Synthesis](https://arxiv.org/pdf/2107.03298.pdf) (2021-07)
- [One TTS Alignment To Rule Them All](https://arxiv.org/abs/2108.10447) (2021-08)

<h3 id="24">Data Efficiency</h3>

- [Semi-Supervised Training for Improving Data Efficiency in End-to-End Speech Synthesis](https://arxiv.org/abs/1808.10128) (2018)
- [Almost Unsupervised Text to Speech and Automatic Speech Recognition](https://arxiv.org/abs/1905.06791) (ICML 2019)
- [Unsupervised Learning For Sequence-to-sequence Text-to-speech For Low-resource Languages](https://arxiv.org/pdf/2008.04549.pdf) (Interspeech 2020)
- Multilingual Speech Synthesis: [One Model, Many Languages: Meta-learning for Multilingual Text-to-Speech](https://arxiv.org/abs/2008.00768) (Interspeech 2020)
- [Low-resource expressive text-to-speech using data augmentation](https://arxiv.org/abs/2011.05707) (2020-11)
- [One TTS Alignment To Rule Them All](https://arxiv.org/pdf/2108.10447.pdf) (2021-08)
- DenoiSpeech: [DenoiSpeech: Denoising Text to Speech with Frame-Level Noise Modeling](https://arxiv.org/abs/2012.09547) (ICASSP 2021)
- [Revisiting Over-Smoothness in Text to Speech](https://arxiv.org/pdf/2202.13066.pdf) (ACL 2022)
- [Unsupervised Text-to-Speech Synthesis by Unsupervised Automatic Speech Recognition](https://arxiv.org/pdf/2203.15796.pdf) (2022-03)
- [Simple and Effective Unsupervised Speech Synthesis](https://arxiv.org/pdf/2204.02524.pdf) (2022-04)
- [A Multi-Stage Multi-Codebook VQ-VAE Approach to High-Performance Neural TTS](https://arxiv.org/pdf/2209.10887.pdf) (Interspeech 2022)
- EPIC TTS Models (research on pruning): [EPIC TTS Models: Empirical Pruning Investigations Characterizing Text-To-Speech Models](https://arxiv.org/pdf/2209.10890.pdf) (Interspeech 2022)


<h2 id="3">Vocoder</h2>

<h3 id="31">Autoregressive Model</h3>

- WaveNet: [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499) (2016)
- WaveRNN: [Efficient Neural Audio Synthesis](https://arxiv.org/abs/1802.08435) (ICML 2018)
- WaveGAN: [Adversarial Audio Synthesis](https://arxiv.org/abs/1802.04208) (ICLR 2019)
- LPCNet: [LPCNet: Improving Neural Speech Synthesis Through Linear Prediction](https://arxiv.org/abs/1810.11846) (ICASSP 2019)
- [Towards achieving robust universal neural vocoding](https://arxiv.org/abs/1811.06292) (Interspeech 2019)
- GAN-TTS: [High Fidelity Speech Synthesis with Adversarial Networks](https://arxiv.org/pdf/1909.11646.pdf) (2019)
- MultiBand-WaveRNN: [DurIAN: Duration Informed Attention Network For Multimodal Synthesis](https://arxiv.org/abs/1909.01700) (2019)
- [Chunked Autoregressive GAN for Conditional Waveform Synthesis](https://arxiv.org/abs/2110.10139) (2021-10)
- Improved LPCNet: [Neural Speech Synthesis on a Shoestring: Improving the Efficiency of LPCNet](https://arxiv.org/pdf/2202.11169.pdf) (ICASSP 2022)
- Bunched LPCNet2: [Bunched LPCNet2: Efficient Neural Vocoders Covering Devices from Cloud to Edge](https://arxiv.org/pdf/2203.14416.pdf) (2022-03)

<h3 id="32">Non-Autoregressive Model</h3>

- Parallel-WaveNet: [Parallel WaveNet: Fast High-Fidelity Speech Synthesis](https://arxiv.org/pdf/1711.10433.pdf) (2017)
- WaveGlow: [WaveGlow: A Flow-based Generative Network for Speech Synthesis](https://arxiv.org/abs/1811.00002) (2018)
- Parallel-WaveGAN: [Parallel WaveGAN: A fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram](https://arxiv.org/abs/1910.11480) (2019)
- MelGAN: [MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis](https://arxiv.org/abs/1910.06711) (NeurIPS 2019)
- MultiBand-MelGAN: [Multi-band MelGAN: Faster Waveform Generation for High-Quality Text-to-Speech](https://arxiv.org/abs/2005.05106) (2020)
- VocGAN: [VocGAN: A High-Fidelity Real-time Vocoder with a Hierarchically-nested Adversarial Network](https://arxiv.org/abs/2007.15256) (Interspeech 2020)
- WaveGrad: [WaveGrad: Estimating Gradients for Waveform Generation](https://arxiv.org/pdf/2009.00713.pdf) (2020)
- DiffWave: [DiffWave: A Versatile Diffusion Model for Audio Synthesis](https://arxiv.org/abs/2009.09761) (2020)
- HiFi-GAN: [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://arxiv.org/pdf/2010.05646.pdf) (NeurIPS 2020)
- Parallel-WaveGAN (New): [Parallel waveform synthesis based on generative adversarial networks with voicing-aware conditional discriminators](https://arxiv.org/abs/2010.14151) (2020-10)
- StyleMelGAN: [StyleMelGAN: An Efficient High-Fidelity Adversarial Vocoder with Temporal Adaptive Normalization](https://arxiv.org/abs/2011.01557) (ICASSP 2021)
- [Improved parallel WaveGAN vocoder with perceptually weighted spectrogram loss](https://arxiv.org/abs/2101.07412) (SLT 2021)
- Fre-GAN: [Fre-GAN: Adversarial Frequency-consistent Audio Synthesis](https://arxiv.org/abs/2106.02297) (Interspeech 2021)
- UnivNet: [A Neural Vocoder with Multi-Resolution Spectrogram Discriminators for High-Fidelity Waveform Generation](https://arxiv.org/pdf/2106.07889.pdf) (2021-07)
- iSTFTNet: [iSTFTNet: Fast and Lightweight Mel-Spectrogram Vocoder Incorporating Inverse Short-Time Fourier Transform](https://arxiv.org/abs/2203.02395) (ICASSP 2022)
- [Parallel Synthesis for Autoregressive Speech Generation](https://arxiv.org/pdf/2204.11806.pdf) (2022-04)
- Avocodo: [Avocodo: Generative Adversarial Network for Artifact-free Vocoder](https://arxiv.org/pdf/2206.13404.pdf) (2022-06)

<h3 id="33">Others</h3>

- (Robust vocoder): [Towards Robust Neural Vocoding for Speech Generation: A Survey](https://arxiv.org/pdf/1912.02461.pdf) (2019)
- (Source-filter model based): [Neural source-filter waveform models for statistical parametric speech synthesis](https://arxiv.org/abs/1904.12088) (TASLP 2019)
- NHV: [Neural Homomorphic Vocoder](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/3188.pdf) (Interspeech 2020)
- Universal MelGAN: [Universal MelGAN: A Robust Neural Vocoder for High-Fidelity Waveform Generation in Multiple Domains](https://arxiv.org/abs/2011.09631) (2020)
- Binaural Speech Synthesis: [Neural Synthesis of Binaural Speech From Mono Audio](https://openreview.net/forum?id=uAX8q61EVRu) (ICLR 2021)
- Checkerboard artifacts in neural vocoder: [Upsampling artifacts in neural audio synthesis](https://arxiv.org/abs/2010.14356) (ICASSP 2021)
- Universal Vocoder Based on Parallel WaveNet: [Universal Neural Vocoding with Parallel WaveNet](https://arxiv.org/abs/2102.01106) (ICASSP 2021)
- (Comparison of discriminator): [GAN Vocoder: Multi-Resolution Discriminator Is All You Need](https://arxiv.org/abs/2103.05236) (2021-03)
- Vocoder Benchmark: [VocBench: A Neural Vocoder Benchmark for Speech Synthesis](https://arxiv.org/abs/2112.03099) (2021-12)
- BigVGAN (Universal vocoder): [BigVGAN: A Universal Neural Vocoder with Large-Scale Training](https://arxiv.org/abs/2206.04658) (2022-06)


<h2 id="4">TTS towards Stylization</h2>

<h3 id="41">Expressive TTS</h3>

- ReferenceEncoder-Tacotron: [Towards End-to-End Prosody Transfer for Expressive Speech Synthesis with Tacotron](https://arxiv.org/abs/1803.09047) (ICML 2018)
- GST-Tacotron: [Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis](https://arxiv.org/abs/1803.09017) (ICML 2018)
- [Predicting Expressive Speaking Style From Text In End-To-End Speech Synthesis](https://arxiv.org/pdf/1808.01410.pdf) (2018)
- GMVAE-Tacotron2: [Hierarchical Generative Modeling for Controllable Speech Synthesis](https://arxiv.org/abs/1810.07217) (ICLR 2019)
- BERT-TTS: [Towards Transfer Learning for End-to-End Speech Synthesis from Deep Pre-Trained Language Models](https://arxiv.org/abs/1906.07307) (2019)
- (Multi-style Decouple): [Multi-Reference Neural TTS Stylization with Adversarial Cycle Consistency](https://arxiv.org/abs/1910.11958) (2019)
- (Multi-style Decouple): [Multi-reference Tacotron by Intercross Training for Style Disentangling,Transfer and Control in Speech Synthesis](https://arxiv.org/abs/1904.02373) (Interspeech 2019)
- Mellotron: [Mellotron: Multispeaker expressive voice synthesis by conditioning on rhythm, pitch and global style tokens](https://arxiv.org/abs/1910.11997) (2019)
- [Robust and fine-grained prosody control of end-to-end speech synthesis](https://arxiv.org/abs/1811.02122) (ICASSP 2019)
- Flowtron (flow based): [Flowtron: an Autoregressive Flow-based Generative Network for Text-to-Speech Synthesis](https://arxiv.org/abs/2005.05957) (2020)
- (local style): [Fully-hierarchical fine-grained prosody modeling for interpretable speech synthesis](https://arxiv.org/abs/2002.03785) (ICASSP 2020)
- [Controllable Neural Prosody Synthesis](https://arxiv.org/pdf/2008.03388.pdf) (Interspeech 2020)
- GraphSpeech: [GraphSpeech: Syntax-Aware Graph Attention Network For Neural Speech Synthesis](https://arxiv.org/abs/2010.12423) (2020-10)
- BERT-TTS: [Improving Prosody Modelling with Cross-Utterance BERT Embeddings for End-to-end Speech Synthesis](https://arxiv.org/abs/2011.05161) (2020-11)
- (Global Emotion Style Control): [Controllable Emotion Transfer For End-to-End Speech Synthesis](https://arxiv.org/abs/2011.08679) (2020-11)
- (Phone Level Style Control): [Fine-grained Emotion Strength Transfer, Control and Prediction for Emotional Speech Synthesis](https://arxiv.org/abs/2011.08477) (2020-11)
- (Phone Level Prosody Modelling): [Mixture Density Network for Phone-Level Prosody Modelling in Speech Synthesis](https://arxiv.org/abs/2102.00851) (ICASSP 2021)
- (Phone Level Prosody Modelling): [Prosodic Clustering for Phoneme-level Prosody Control in End-to-End Speech Synthesis](https://arxiv.org/abs/2111.10177) (ICASSP 2021)
- PeriodNet: [PeriodNet: A non-autoregressive waveform generation model with a structure separating periodic and aperiodic components](https://arxiv.org/abs/2102.07786) (ICASSP 2021)
- PnG BERT: [PnG BERT: Augmented BERT on Phonemes and Graphemes for Neural TTS](https://arxiv.org/abs/2103.15060) (Interspeech 2021)
- [Towards Multi-Scale Style Control for Expressive Speech Synthesis](https://arxiv.org/abs/2104.03521) (2021-04)
- [Learning Robust Latent Representations for Controllable Speech Synthesis](https://arxiv.org/abs/2105.04458) (2021-05)
- [Diverse and Controllable Speech Synthesis with GMM-Based Phone-Level Prosody Modelling](https://arxiv.org/abs/2105.13086) (2021-05)
- [Improving Performance of Seen and Unseen Speech Style Transfer in End-to-end Neural TTS](https://arxiv.org/abs/2106.10003) (2021-06)
- (Conversational Speech Synthesis): [Controllable Context-aware Conversational Speech Synthesis](https://arxiv.org/abs/2106.10828) (Interspeech 2021)
- DeepRapper: [DeepRapper: Neural Rap Generation with Rhyme and Rhythm Modeling](https://arxiv.org/pdf/2107.01875.pdf) (ACL 2021)
- Referee: [Referee: Towards reference-free cross-speaker style transfer with low-quality data for expressive speech synthesis](https://arxiv.org/abs/2109.03439) (2021)
- (Text-Based Insertion TTS): [Zero-Shot Text-to-Speech for Text-Based Insertion in Audio Narration](https://arxiv.org/abs/2109.05426) (Interspeech 2021)
- [On the Interplay Between Sparsity, Naturalness, Intelligibility, and Prosody in Speech Synthesis](https://arxiv.org/abs/2110.01147) (2021-10)
- [Style Equalization: Unsupervised Learning of Controllable Generative Sequence Models](https://arxiv.org/pdf/2110.02891.pdf) (2021-10)
- TTS for dubbing: [Neural Dubber: Dubbing for Videos According to Scripts](https://arxiv.org/abs/2110.08243) (NeurIPS 2021)
- [Word-Level Style Control for Expressive, Non-attentive Speech Synthesis](https://arxiv.org/abs/2111.10173) (SPECOM 2021)
- MsEmoTTS: [MsEmoTTS: Multi-scale emotion transfer, prediction, and control for emotional speech synthesis](https://arxiv.org/pdf/2201.06460.pdf) (2022-01)
- [Disentangling Style and Speaker Attributes for TTS Style Transfer](https://arxiv.org/pdf/2201.09472.pdf) (2022-01)
- Word-level prosody modeling: [Unsupervised word-level prosody tagging for controllable speech synthesis](https://arxiv.org/pdf/2202.07200.pdf) (ICASSP 2022)
- ProsoSpeech: [ProsoSpeech: Enhancing Prosody With Quantized Vector Pre-training in Text-to-Speech](https://arxiv.org/abs/2202.07816) (ICASSP 2022)
- CampNet (speech editing):[CampNet: Context-Aware Mask Prediction for End-to-End Text-Based Speech Editing](https://arxiv.org/pdf/2202.09950.pdf) (2022-02)
- vTTS (visual text): [vTTS: visual-text to speech](https://arxiv.org/pdf/2203.14725.pdf) (2022-03)
- CopyCat2: [CopyCat2: A Single Model for Multi-Speaker TTS and Many-to-Many Fine-Grained Prosody Transfer](https://arxiv.org/abs/2206.13443) (Interspeech 2022)
- [Prosody Cloning in Zero-Shot Multispeaker Text-to-Speech](https://arxiv.org/abs/2206.12229) (Interspeech 2022)
- [Expressive, Variable, and Controllable Duration Modelling in TTS](https://arxiv.org/abs/2206.14165) (Interspeech 2022)

<h3 id="42">MultiSpeaker TTS</h3>

- Meta-Learning for TTS: [Sample Efficient Adaptive Text-to-Speech](https://arxiv.org/abs/1809.10460) (ICLR 2019)
- SV-Tacotron: [Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis](https://arxiv.org/abs/1806.04558) (NeurIPS 2018)
- Deep Voice V3: [Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning](https://arxiv.org/abs/1710.07654) (ICLR 2018)
- [Zero-Shot Multi-Speaker Text-To-Speech with State-of-the-art Neural Speaker Embeddings](https://arxiv.org/abs/1910.10838) (ICASSP 2020)
- MultiSpeech: [MultiSpeech: Multi-Speaker Text to Speech with Transformer](https://arxiv.org/abs/2006.04664) (2020)
- SC-WaveRNN: [Speaker Conditional WaveRNN: Towards Universal Neural Vocoder for Unseen Speaker and Recording Conditions](https://arxiv.org/pdf/2008.05289.pdf) (Interspeech 2020)
- MultiSpeaker Dataset: [AISHELL-3: A Multi-speaker Mandarin TTS Corpus and the Baselines](https://arxiv.org/abs/2010.11567) (2020)
- Life-long learning for multi-speaker TTS: [Continual Speaker Adaptation for Text-to-Speech Synthesis](https://arxiv.org/abs/2103.14512) (2021-03)
- [Meta-StyleSpeech : Multi-Speaker Adaptive Text-to-Speech Generation](https://arxiv.org/pdf/2106.03153.pdf) (ICML 2021)
- [Effective and Differentiated Use of Control Information for Multi-speaker Speech Synthesis](https://arxiv.org/pdf/2107.03065.pdf) (Interspeech 2021)
- [Speaker Generation](https://arxiv.org/abs/2111.05095) (2021-11)
- Meta-Voice: [Meta-Voice: Fast few-shot style transfer for expressive voice cloning using meta learning](https://arxiv.org/abs/2111.07218) (2021-11)

<h3 id="43">New Perspective on TTS</h3>

- PromptTTS: [PromptTTS: Controllable Text-to-Speech with Text Descriptions](https://arxiv.org/abs/2211.12171) (2022-11)
- VALL-E: [Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers](https://arxiv.org/pdf/2301.02111.pdf) (2023-01)
- InstructTTS: [InstructTTS: Modelling Expressive TTS in Discrete Latent Space with Natural Language Style Prompt](https://arxiv.org/abs/2301.13662) (2023-01)
- Spear-TTS: [Speak, Read and Prompt: High-Fidelity Text-to-Speech with Minimal Supervision](https://arxiv.org/abs/2302.03540) (2023-02)
- FoundationTTS: [FoundationTTS: Text-to-Speech for ASR Customization with Generative Language Model](https://arxiv.org/pdf/2303.02939v2.pdf) (2023-03)


<h2 id="5">Voice Conversion</h2>

<h3 id="51">ASR & TTS Based</h3>

- (introduce PPG into voice conversion): [Phonetic posteriorgrams for many-to-one voice conversion without parallel data training](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7552917) (2016)
- [A Vocoder-free WaveNet Voice Conversion with Non-Parallel Data](https://arxiv.org/pdf/1902.03705.pdf) (2019)
- TTS-Skins: [TTS Skins: Speaker Conversion via ASR](https://arxiv.org/pdf/1904.08983.pdf) (2019)
- [Non-Parallel Sequence-to-Sequence Voice Conversion with Disentangled Linguistic and Speaker Representations](https://arxiv.org/abs/1906.10508) (IEEE/ACM TASLP 2019)
- [One-shot Voice Conversion by Separating Speaker and Content Representations with Instance Normalization](https://arxiv.org/abs/1904.05742) (Interspeech 2019)
- Cotatron (combine text information with voice conversion system): [Cotatron: Transcription-Guided Speech Encoder for Any-to-Many Voice Conversion without Parallel Data](https://arxiv.org/abs/2005.03295) (Interspeech 2020)
- (TTS & ASR): [Voice Conversion by Cascading Automatic Speech Recognition and Text-to-Speech Synthesis with Prosody Transfer](https://arxiv.org/pdf/2009.01475.pdf) (Interspeech 2020)
- FragmentVC (wav to vec): [FragmentVC: Any-to-Any Voice Conversion by End-to-End Extracting and Fusing Fine-Grained Voice Fragments With Attention](https://arxiv.org/abs/2010.14150) (2020)
- [Towards Natural and Controllable Cross-Lingual Voice Conversion Based on Neural TTS Model and Phonetic Posteriorgram](https://arxiv.org/abs/2102.01991) (ICASSP 2021)
- (TTS & ASR): [On Prosody Modeling for ASR+TTS based Voice Conversion](https://arxiv.org/abs/2107.09477) (2021-07)
- [Cloning one's voice using very limited data in the wild](https://arxiv.org/pdf/2110.03347.pdf) (2021-10)

<h3 id="52">VAE & Auto-Encoder Based</h3>

- VAE-VC (VAE based): [Voice Conversion from Non-parallel Corpora Using Variational Auto-encoder](https://arxiv.org/pdf/1610.04019.pdf) (2016)
- (Speech representation learning by VQ-VAE): [Unsupervised speech representation learning using WaveNet autoencoders](https://arxiv.org/abs/1901.08810) (2019)
- Blow (Flow based): [Blow: a single-scale hyperconditioned flow for non-parallel raw-audio voice conversion](https://arxiv.org/abs/1906.00794) (NeurIPS 2019)
- AutoVC: [AUTOVC: Zero-Shot Voice Style Transfer with Only Autoencoder Loss](https://arxiv.org/abs/1905.05879) (2019)
- F0-AutoVC: [F0-consistent many-to-many non-parallel voice conversion via conditional autoencoder](https://arxiv.org/abs/2004.07370) (ICASSP 2020)
- [One-Shot Voice Conversion by Vector Quantization](https://ieeexplore.ieee.org/abstract/document/9053854) (ICASSP 2020)
- SpeechSplit (auto-encoder): [Unsupervised Speech Decomposition via Triple Information Bottleneck](https://arxiv.org/abs/2004.11284) (ICML 2020)
- NANSY: [Neural Analysis and Synthesis: Reconstructing Speech from Self-Supervised Representations](https://arxiv.org/pdf/2110.14513.pdf) (NeurIPS 2021)

<h3 id="53">GAN Based</h3>

- CycleGAN-VC V1: [Parallel-Data-Free Voice Conversion Using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1711.11293) (2017)
- StarGAN-VC: [StarGAN-VC: non-parallel many-to-many Voice Conversion Using Star Generative Adversarial Networks](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8639535&tag=1) (2018)
- CycleGAN-VC V2: [CycleGAN-VC2: Improved CycleGAN-based Non-parallel Voice Conversion](https://arxiv.org/abs/1904.04631) (2019)
- CycleGAN-VC V3: [CycleGAN-VC3: Examining and Improving CycleGAN-VCs for Mel-spectrogram Conversion](https://arxiv.org/abs/2010.11672) (2020)
- MaskCycleGAN-VC: [MaskCycleGAN-VC: Learning Non-parallel Voice Conversion with Filling in Frames](https://arxiv.org/abs/2102.12841) (ICASSP 2021)


<h2 id="6">Singing</h2>

<h3 id="61">Singing Voice Synthesis</h3>

- XiaoIce Band: [XiaoIce Band: A Melody and Arrangement Generation Framework for Pop Music](https://www.kdd.org/kdd2018/accepted-papers/view/xiaoice-banda-melody-and-arrangement-generation-framework-for-pop-music) (KDD 2018)
- Mellotron: [Mellotron: Multispeaker expressive voice synthesis by conditioning on rhythm, pitch and global style tokens](https://arxiv.org/abs/1910.11997) (2019)
- ByteSing: [ByteSing: A Chinese Singing Voice Synthesis System Using Duration Allocated Encoder-Decoder Acoustic Models and WaveRNN Vocoders](https://arxiv.org/abs/2004.11012) (2020)
- JukeBox: [Jukebox: A Generative Model for Music](https://arxiv.org/abs/2005.00341) (2020)
- XiaoIce Sing: [XiaoiceSing: A High-Quality and Integrated Singing Voice Synthesis System](https://arxiv.org/abs/2006.06261) (2020)
- HiFiSinger: [HiFiSinger: Towards High-Fidelity Neural Singing Voice Synthesis](https://arxiv.org/abs/2009.01776) (2019)
- [Sequence-to-sequence Singing Voice Synthesis with Perceptual Entropy Loss](https://arxiv.org/abs/2010.12024) (2020)
- Learn2Sing: [Learn2Sing: Target Speaker Singing Voice Synthesis by learning from a Singing Teacher](https://arxiv.org/abs/2011.08467) (2020-11)
- MusicBERT: [MusicBERT: Symbolic Music Understanding with Large-Scale Pre-Training](https://arxiv.org/pdf/2106.05630.pdf) (ACL 2021)
- SingGAN (Singing Voice Vocoder): [SingGAN: Generative Adversarial Network For High-Fidelity Singing Voice Generation](https://arxiv.org/abs/2110.07468) (AAAI 2022)
- Background music generation: [Video Background Music Generation with Controllable Music Transformer](https://arxiv.org/abs/2111.08380) (ACM Multimedia 2021)
- Multi-Singer (Singing Voice Vocoder): [Multi-Singer: Fast Multi-Singer Singing Voice Vocoder With A Large-Scale Corpus](https://arxiv.org/pdf/2112.10358.pdf) (ACM Multimedia 2021)
- Rapping-singing voice synthesis: [Rapping-Singing Voice Synthesis based on Phoneme-level Prosody Control](https://arxiv.org/pdf/2111.09146.pdf) (SSW 11)
- VISinger (VIST for Singing Voice Synthesis): [VISinger: Variational Inference with Adversarial Learning for End-to-End Singing Voice Synthesis](https://arxiv.org/pdf/2110.08813.pdf) (2021-10)
- Opencpop: [Opencpop: A High-Quality Open Source Chinese Popular Song Corpus for Singing Voice Synthesis](https://arxiv.org/pdf/2201.07429.pdf) (2022-01)
- [Learning the Beauty in Songs: Neural Singing Voice Beautifier](https://arxiv.org/pdf/2202.13277.pdf) (ACL 2022)
- [Learn2Sing 2.0: Diffusion and Mutual Information-Based Target Speaker SVS by Learning from Singing Teacher](https://arxiv.org/pdf/2203.16408.pdf) (2022-03)
- MusicLM: [MusicLM: Generating Music From Text](https://arxiv.org/abs/2301.11325) (2023-01)
- SingSong: [SingSong: Generating musical accompaniments from singing](https://arxiv.org/abs/2301.12662) (2023-01)

<h3 id="62">Singing Voice Conversion</h3>

- [A Universal Music Translation Network](https://arxiv.org/abs/1805.07848) (2018)
- [Unsupervised Singing Voice Conversion](https://arxiv.org/abs/1904.06590) (Interspeech 2019)
- PitchNet: [PitchNet: Unsupervised Singing Voice Conversion with Pitch Adversarial Network](https://arxiv.org/abs/1912.01852) (ICASSP 2020)
- DurIAN-SC: [DurIAN-SC: Duration Informed Attention Network based Singing Voice Conversion System](https://arxiv.org/abs/2008.03009) (Interspeech 2020)
- [Speech-to-Singing Conversion based on Boundary Equilibrium GAN](https://arxiv.org/abs/2005.13835) (Interspeech 2020)
- [PPG-based singing voice conversion with adversarial representation learning](https://arxiv.org/abs/2010.14804) (2020)


<h2 id="7">Speech Processing Related</h2>

<h3 id="71">Speech Pretrained Model</h3>

- Audio-Word2Vec: [Audio Word2Vec: Unsupervised Learning of Audio Segment Representations using Sequence-to-sequence Autoencoder](https://arxiv.org/pdf/1603.00982.pdf) (2016)
- [Unsupervised speech representation learning using WaveNet autoencoders](https://arxiv.org/abs/1901.08810) (2019)
- [Improving Transformer-based Speech Recognition Using Unsupervised Pre-training](https://arxiv.org/abs/1910.09932) (2019)
- SpeechBERT: [SpeechBERT: An Audio-and-text Jointly Learned Language Model for End-to-end Spoken Question Answering](https://arxiv.org/abs/1910.11559) (2019)
- DDSP: [DDSP: Differentiable Digital Signal Processing](https://arxiv.org/abs/2001.04643) (ICLR 2020)
- SoundStream: [SoundStream: An End-to-End Neural Audio Codec](https://arxiv.org/abs/2107.03312) (2021-07)
- NANSY: [Neural Analysis and Synthesis: Reconstructing Speech from Self-Supervised Representations](https://arxiv.org/abs/2110.14513) (NeurIPS 2021)
- Study of audio representations: [Audio representations for deep learning in sound synthesis: A review](https://arxiv.org/abs/2201.02490) (2022-01)
- MuLan (Music Text Embedding): [MuLan: A Joint Embedding of Music Audio and Natural Language](https://arxiv.org/abs/2208.12415) (2022-08)
- AudioLM: [AudioLM: a Language Modeling Approach to Audio Generation](https://arxiv.org/abs/2209.03143) (2022-09)
- AudioGen: [AudioGen: Textually Guided Audio Generation](https://arxiv.org/pdf/2209.15352.pdf) (2022-09)
- NANSY++: [NANSY++: Unified Voice Synthesis with Neural Analysis and Synthesis](https://arxiv.org/abs/2211.09407) (2022-11)

<h3 id="72">Speech Separation</h3>

- TasNet: [TasNet: time-domain audio separation network for real-time, single-channel speech separation](https://arxiv.org/abs/1711.00541) (ICASSP 2018)
- Conv-TasNet: [Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation](https://arxiv.org/abs/1809.07454)
- (Music Source Separation): [Decoupling Magnitude and Phase Estimation with Deep ResUNet for Music Source Separation](https://arxiv.org/pdf/2109.05418.pdf) (2021-09)

<h3 id="73">Speaker Verification</h3>

- DeepSpeaker: [Deep Speaker: an End-to-End Neural Speaker Embedding System](https://arxiv.org/pdf/1705.02304.pdf) (2017)
- GE2E Loss: [Generalized End-to-End Loss for Speaker Verification](https://arxiv.org/abs/1710.10467) (ICASSP 2018)

<h3 id="74">Audio Super Resolution</h3>

- [VoiceFixer: Toward General Speech Restoration With Neural Vocoder](https://arxiv.org/abs/2109.13731) (2021)

<h3 id="75">Tools</h3>

- ESPnet: [ESPnet: End-to-End Speech Processing Toolkit](https://arxiv.org/abs/1804.00015) (2018)
- SpeechBrain: [speechbrain](https://github.com/speechbrain/speechbrain)
- SpeechBrain Paper: [SpeechBrain: A General-Purpose Speech Toolkit](https://arxiv.org/abs/2106.04624)
- ESPnet2-TTS: [ESPnet2-TTS: Extending the Edge of TTS Research](https://arxiv.org/abs/2110.07840) (2021-10)
- WeNet 2.0: [WeNet 2.0: More Productive End-to-End Speech Recognition Toolkit](https://arxiv.org/pdf/2203.15455.pdf) (2022-03)


<h2 id="8">Natural Language Processing</h2>

<h3 id="81">Sequence Modeling</h3>

- LSTM: [Long Short-term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) (1997)
- GRU: [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078v3) (EMNLP 2014)
- TCN: [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271) (2018)
- Transformer: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (NIPS 2017)
- Transformer-XL: [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860) (ACL 2019)
- Reformer: [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451) (ICLR 2020)

<h3 id="82">Pretrained Model</h3>

- Awesome Repositories: [transformers](https://github.com/huggingface/transformers)
- BERT: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) (NAACL 2019)
- XLNET: [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237) (NeurIPS 2019)
- ALBERT: [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942) (ICLR 2020)
- [Masked Autoencoders that Listen](https://arxiv.org/abs/2207.06405) (2022-07)

<h3 id="83">Non-autoregressive Translation Model</h3>

- [A Study of Non-autoregressive Model for Sequence Generation](https://arxiv.org/abs/2004.10454) (ACL 2020)
- [Deterministic Non-Autoregressive Neural Sequence Modeling by Iterative Refinement](https://arxiv.org/abs/1802.06901) (EMNLP 2018)
- [Non-Autoregressive Neural Machine Translation](https://arxiv.org/abs/1711.02281v1) (ICLR 2018)
- [Non-Autoregressive Machine Translation with Auxiliary Regularization](https://arxiv.org/abs/1902.10245) (AAAI 2019)
- [Mask-Predict: Parallel Decoding of Conditional Masked Language Models](https://arxiv.org/abs/1904.09324) (EMNLP 2019)

<h3 id="84">Speech2Speech Translation Model</h3>

- Awesome Paper List: [awesome-speech-translation](https://github.com/dqqcasia/awesome-speech-translation)
- [Direct speech-to-speech translation with a sequence-to-sequence model](https://arxiv.org/abs/1904.06037) (Interspeech 2020)
- NeurST: [NeurST: Neural Speech Translation Toolkit](https://arxiv.org/pdf/2012.10018.pdf) (2020-12)
- Translatotron 2: [Translatotron 2: Robust direct speech-to-speech translation](https://arxiv.org/abs/2107.08661) (2021-07)

<h3 id="85">Neural Machine Reading Comprehension</h3>

- Review 2019: [Neural Machine Reading Comprehension: Methods and Trends](https://arxiv.org/abs/1907.01118) (2019)
- Review 2020: [A Survey on Machine Reading Comprehension: Tasks, Evaluation Metrics, and Benchmark Datasets](https://arxiv.org/pdf/2006.11880.pdf) (2019)
- NMRC first: [Teaching Machines to Read and Comprehend](https://arxiv.org/abs/1506.03340) (NIPS 2015)
- RACE dataset: [RACE: Large-scale ReAding Comprehension Dataset From Examinations](https://www.aclweb.org/anthology/D17-1082.pdf) (EMNLP 2017)
- Cloze test: [Large-scale Cloze Test Dataset Created by Teachers](https://arxiv.org/abs/1711.03225) (EMNLP 2018)
- HuggingFace: [HuggingFace's Transformers: State-of-the-art Natural Language Processing](https://arxiv.org/abs/1910.03771) (2019)


<h2 id="9">VAE & GAN</h2>

<h3 id="91">VAE</h3>

- VAE: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) (ICLR 2014)
- GM-VAE: [Deep Unsupervised Clustering with Gaussian Mixture Variational Autoencoders](https://arxiv.org/abs/1611.02648) (ICLR 2017)
- VQ-VAE: [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) (NIPS 2017)
- VQ-VAE 2: [Generating Diverse High-Fidelity Images with VQ-VAE-2](https://arxiv.org/abs/1906.00446) (NeurIPS 2019)

<h3 id="92">GAN</h3>

- GAN: [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) (NIPS 2014)
- Condition-GAN: [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) (2014)
- Info-GAN: [InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://arxiv.org/abs/1606.03657) (2016)
- SeqGAN: [SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient](https://arxiv.org/abs/1609.05473) (AAAI 2017)
- Cycle-GAN: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) (ICCV 2017)
- Star-GAN: [StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation](https://arxiv.org/abs/1711.09020) (CVPR 2018)
- BigGAN: [Large Scale GAN Training for High Fidelity Natural Image Synthesis](https://arxiv.org/abs/1809.11096) (ICLR 2019)
- Style-GAN: [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948) (CVPR 2019)


<h2 id="10">Others</h2>

- (Forgetting learning): [An Empirical Study of Example Forgetting during Deep Neural Network Learning](https://arxiv.org/abs/1812.05159) (ICLR 2019)
- ScaNN (search accelerating): [Accelerating Large-Scale Inference with Anisotropic Vector Quantization](https://arxiv.org/abs/1908.10396) (ICML 2020)
- (memory management): [Efficient Memory Management for Deep Neural Net Inference](https://arxiv.org/abs/2001.03288) (2020)
- Conformer: [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100) (InterSpeech 2020)
- Computational Arts: [When Creators Meet the Metaverse: A Survey on Computational Arts](https://arxiv.org/pdf/2111.13486.pdf) (2021-11)
