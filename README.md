# Speech Synthesis Paper
List of speech synthesis papers (-> [more papers](./papers/README.md) <-). Welcome to recommend more awesome papers 😀.

Repositories for collecting awesome speech paper:

- [awesome-speech-recognition-speech-synthesis-papers](https://github.com/zzw922cn/awesome-speech-recognition-speech-synthesis-papers) (from [ponyzhang](https://github.com/zzw922cn))
- [awesome-python-scientific-audio](https://github.com/faroit/awesome-python-scientific-audio) (from [Fabian-Robert Stöter](https://github.com/faroit))
- [TTS-papers](https://github.com/erogol/TTS-papers) (from [Eren Gölge](https://github.com/erogol))
- [awesome-speech-enhancement](https://github.com/Wenzhe-Liu/awesome-speech-enhancement) (from [Vincent Liu](https://github.com/Wenzhe-Liu))
- [speech-recognition-papers](https://github.com/xingchensong/speech-recognition-papers) (from [Xingchen Song](https://github.com/xingchensong))
- [awesome-tts-samples](https://github.com/seungwonpark/awesome-tts-samples) (from [Seung-won Park](https://github.com/seungwonpark))
- [awesome-speech-translation](https://github.com/dqqcasia/awesome-speech-translation) (from [dqqcasia](https://github.com/dqqcasia))

*What is the meaning of '<span class="star" style="color: red">&#x2605;</span>'? I add '<span class="star" style="color: red">&#x2605;</span>' to the papers which number of citations is over 50 (only in `Acoustic Model`, `Vocoder` and `TTS towards Stylization`). Beginner can read these paper first to get basic knowledge of Deep-Learning-based TTS model ([#1](https://github.com/xcmyz/speech-synthesis-paper/issues/1#issue-679044612)).*


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
* [Voice Conversion](#5)
    * [ASR & TTS Based](#51)
    * [VAE & Auto-Encoder Based](#52)
    * [GAN Based](#53)
* [Singing](#6)
    * [Singing Synthesis](#61)
    * [Singing Voice Conversion](#62)


<h2 id="1">TTS Frontend</h2>

- [Pre-trained Text Representations for Improving Front-End Text Processing in Mandarin Text-to-Speech Synthesis](https://pdfs.semanticscholar.org/6abc/7dac0bdc50735b6d12f96400f59b5f084759.pdf) (Interspeech 2019)
- [A unified sequence-to-sequence front-end model for Mandarin text-to-speech synthesis](https://arxiv.org/pdf/1911.04111.pdf) (ICASSP 2020)
- [A hybrid text normalization system using multi-head self-attention for mandarin](https://arxiv.org/pdf/1911.04128.pdf) (ICASSP 2020)
- [Unified Mandarin TTS Front-end Based on Distilled BERT Model](https://arxiv.org/pdf/2012.15404.pdf) (2021-01)


<h2 id="2">Acoustic Model</h2>

<h3 id="21">Autoregressive Model</h3>

- Tacotron V1<sup><span class="star" style="color: red">&#x2605;</span></sup>: [Tacotron: Towards End-to-End Speech Synthesis](https://arxiv.org/abs/1703.10135) (Interspeech 2017)
- Tacotron V2<sup><span class="star" style="color: red">&#x2605;</span></sup>: [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/abs/1712.05884) (ICASSP 2018)
- Deep Voice V1<sup><span class="star" style="color: red">&#x2605;</span></sup>: [Deep Voice: Real-time Neural Text-to-Speech](https://arxiv.org/abs/1702.07825) (ICML 2017)
- Deep Voice V2<sup><span class="star" style="color: red">&#x2605;</span></sup>: [Deep Voice 2: Multi-Speaker Neural Text-to-Speech](https://arxiv.org/abs/1705.08947) (NeurIPS 2017)
- Deep Voice V3<sup><span class="star" style="color: red">&#x2605;</span></sup>: [Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning](https://arxiv.org/abs/1710.07654) (ICLR 2018)
- Transformer-TTS<sup><span class="star" style="color: red">&#x2605;</span></sup>: [Neural Speech Synthesis with Transformer Network](https://arxiv.org/abs/1809.08895) (AAAI 2019)
- DurIAN: [DurIAN: Duration Informed Attention Network For Multimodal Synthesis](https://arxiv.org/abs/1909.01700) (2019)
- [Location-Relative Attention Mechanisms For Robust Long-Form Speech Synthesis](https://arxiv.org/abs/1910.10288) (ICASSP 2020)
- Flowtron (flow based): [Flowtron: an Autoregressive Flow-based Generative Network for Text-to-Speech Synthesis](https://arxiv.org/abs/2005.05957) (2020)
- [Non-Attentive Tacotron: Robust and Controllable Neural TTS Synthesis Including Unsupervised Duration Modeling](https://arxiv.org/pdf/2010.04301v1.pdf) (under review ICLR 2021)
- RobuTrans (towards robust): [RobuTrans: A Robust Transformer-Based Text-to-Speech Model](https://ojs.aaai.org//index.php/AAAI/article/view/6337) (AAAI 2020)
- DeviceTTS: [DeviceTTS: A Small-Footprint, Fast, Stable Network for On-Device Text-to-Speech](https://arxiv.org/abs/2010.15311) (2020-10)
- Wave-Tacotron: [Wave-Tacotron: Spectrogram-free end-to-end text-to-speech synthesis](https://arxiv.org/abs/2011.03568) (2020-11)

<h3 id="22">Non-Autoregressive Model</h3>

- ParaNet: [Non-Autoregressive Neural Text-to-Speech](https://arxiv.org/pdf/1905.08459.pdf) (ICML 2020)
- FastSpeech<sup><span class="star" style="color: red">&#x2605;</span></sup>: [FastSpeech: Fast, Robust and Controllable Text to Speech](https://arxiv.org/abs/1905.09263) (NeurIPS 2019)
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

<h3 id="23">Alignment Study</h3>

- Monotonic Attention<sup><span class="star" style="color: red">&#x2605;</span></sup>: [Online and Linear-Time Attention by Enforcing Monotonic Alignments](https://arxiv.org/abs/1704.00784) (ICML 2017)
- Monotonic Chunkwise Attention<sup><span class="star" style="color: red">&#x2605;</span></sup>: [Monotonic Chunkwise Attention](https://arxiv.org/abs/1712.05382) (ICLR 2018)
- [Forward Attention in Sequence-to-sequence Acoustic Modelling for Speech Synthesis](https://arxiv.org/abs/1807.06736) (ICASSP 2018)
- RNN-T for TTS: [Initial investigation of an encoder-decoder end-to-end TTS framework using marginalization of monotonic hard latent alignments](http://128.84.4.27/pdf/1908.11535) (2019)
- [Location-Relative Attention Mechanisms For Robust Long-Form Speech Synthesis](https://arxiv.org/abs/1910.10288) (ICASSP 2020)
- [Non-Attentive Tacotron: Robust and Controllable Neural TTS Synthesis Including Unsupervised Duration Modeling](https://arxiv.org/pdf/2010.04301v1.pdf) (under review ICLR 2021)
- EfficientTTS: [EfficientTTS: An Efficient and High-Quality Text-to-Speech Architecture](https://arxiv.org/abs/2012.03500) (2020-12)

<h3 id="24">Data Efficiency</h3>

- [Semi-Supervised Training for Improving Data Efficiency in End-to-End Speech Synthesis](https://arxiv.org/abs/1808.10128) (2018)
- [Almost Unsupervised Text to Speech and Automatic Speech Recognition](https://arxiv.org/abs/1905.06791) (ICML 2019)
- [Unsupervised Learning For Sequence-to-sequence Text-to-speech For Low-resource Languages](https://arxiv.org/pdf/2008.04549.pdf) (Interspeech 2020)
- Multilingual Speech Synthesis: [One Model, Many Languages: Meta-learning for Multilingual Text-to-Speech](https://arxiv.org/abs/2008.00768) (InterSpeech 2020)
- [Low-resource expressive text-to-speech using data augmentation](https://arxiv.org/abs/2011.05707) (2020-11)
- DenoiSpeech: [DenoiSpeech: Denoising Text to Speech with Frame-Level Noise Modeling](https://arxiv.org/abs/2012.09547) (ICASSP 2021)


<h2 id="3">Vocoder</h2>

<h3 id="31">Autoregressive Model</h3>

- WaveNet<sup><span class="star" style="color: red">&#x2605;</span></sup>: [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499) (2016)
- WaveRNN<sup><span class="star" style="color: red">&#x2605;</span></sup>: [Efficient Neural Audio Synthesis](https://arxiv.org/abs/1802.08435) (ICML 2018)
- WaveGAN<sup><span class="star" style="color: red">&#x2605;</span></sup>: [Adversarial Audio Synthesis](https://arxiv.org/abs/1802.04208) (ICLR 2019)
- LPCNet<sup><span class="star" style="color: red">&#x2605;</span></sup>: [LPCNet: Improving Neural Speech Synthesis Through Linear Prediction](https://arxiv.org/abs/1810.11846) (ICASSP 2019)
- [Towards achieving robust universal neural vocoding](https://arxiv.org/abs/1811.06292) (Interspeech 2019)
- GAN-TTS: [High Fidelity Speech Synthesis with Adversarial Networks](https://arxiv.org/pdf/1909.11646.pdf) (2019)
- MultiBand-WaveRNN: [DurIAN: Duration Informed Attention Network For Multimodal Synthesis](https://arxiv.org/abs/1909.01700) (2019)

<h3 id="32">Non-Autoregressive Model</h3>

- Parallel-WaveNet<sup><span class="star" style="color: red">&#x2605;</span></sup>: [Parallel WaveNet: Fast High-Fidelity Speech Synthesis](https://arxiv.org/pdf/1711.10433.pdf) (2017)
- WaveGlow<sup><span class="star" style="color: red">&#x2605;</span></sup>: [WaveGlow: A Flow-based Generative Network for Speech Synthesis](https://arxiv.org/abs/1811.00002) (2018)
- Parallel-WaveGAN<sup><span class="star" style="color: red">&#x2605;</span></sup>: [Parallel WaveGAN: A fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram](https://arxiv.org/abs/1910.11480) (2019)
- MelGAN<sup><span class="star" style="color: red">&#x2605;</span></sup>: [MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis](https://arxiv.org/abs/1910.06711) (NeurIPS 2019)
- MultiBand-MelGAN: [Multi-band MelGAN: Faster Waveform Generation for High-Quality Text-to-Speech](https://arxiv.org/abs/2005.05106) (2020)
- VocGAN: [VocGAN: A High-Fidelity Real-time Vocoder with a Hierarchically-nested Adversarial Network](https://arxiv.org/abs/2007.15256) (Interspeech 2020)
- WaveGrad: [WaveGrad: Estimating Gradients for Waveform Generation](https://arxiv.org/pdf/2009.00713.pdf) (2020)
- DiffWave: [DiffWave: A Versatile Diffusion Model for Audio Synthesis](https://arxiv.org/abs/2009.09761) (2020)
- HiFi-GAN: [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://arxiv.org/pdf/2010.05646.pdf) (NeurIPS 2020)
- Parallel-WaveGAN (New): [Parallel waveform synthesis based on generative adversarial networks with voicing-aware conditional discriminators](https://arxiv.org/abs/2010.14151) (2020-10)
- [Improved parallel WaveGAN vocoder with perceptually weighted spectrogram loss](https://arxiv.org/abs/2101.07412) (SLT 2021)

<h3 id="33">Others</h3>

- (Source-filter model based): [Neural source-filter waveform models for statistical parametric speech synthesis](https://arxiv.org/abs/1904.12088) (TASLP 2019)
- NHV: [Neural Homomorphic Vocoder](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/3188.pdf) (Interspeech 2020)
- Binaural Speech Synthesis: [Neural Synthesis of Binaural Speech From Mono Audio](https://openreview.net/forum?id=uAX8q61EVRu) (ICLR 2021)
- Checkerboard artifacts in neural vocoder: [Upsampling artifacts in neural audio synthesis](https://arxiv.org/abs/2010.14356) (ICASSP 2021)
- Universal MelGAN: [Universal MelGAN: A Robust Neural Vocoder for High-Fidelity Waveform Generation in Multiple Domains](https://arxiv.org/abs/2011.09631) (2020)
- Universal Vocoder Based on Parallel WaveNet: [Universal Neural Vocoding with Parallel WaveNet](https://arxiv.org/abs/2102.01106) (ICASSP 2021)
- (Comparison of discriminator): [GAN Vocoder: Multi-Resolution Discriminator Is All You Need](https://arxiv.org/abs/2103.05236) (2021-03)


<h2 id="4">TTS towards Stylization</h2>

<h3 id="41">Expressive TTS</h3>

- ReferenceEncoder-Tacotron<sup><span class="star" style="color: red">&#x2605;</span></sup>: [Towards End-to-End Prosody Transfer for Expressive Speech Synthesis with Tacotron](https://arxiv.org/abs/1803.09047) (ICML 2018)
- GST-Tacotron<sup><span class="star" style="color: red">&#x2605;</span></sup>: [Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis](https://arxiv.org/abs/1803.09017) (ICML 2018)
- [Predicting Expressive Speaking Style From Text In End-To-End Speech Synthesis](https://arxiv.org/pdf/1808.01410.pdf) (2018)
- GMVAE-Tacotron2<sup><span class="star" style="color: red">&#x2605;</span></sup>: [Hierarchical Generative Modeling for Controllable Speech Synthesis](https://arxiv.org/abs/1810.07217) (ICLR 2019)
- BERT-TTS: [Towards Transfer Learning for End-to-End Speech Synthesis from Deep Pre-Trained Language Models](https://arxiv.org/abs/1906.07307) (2019)
- (Multi-style Decouple): [Multi-Reference Neural TTS Stylization with Adversarial Cycle Consistency](https://arxiv.org/abs/1910.11958) (2019)
- (Multi-style Decouple): [Multi-reference Tacotron by Intercross Training for Style Disentangling,Transfer and Control in Speech Synthesis](https://arxiv.org/abs/1904.02373) (InterSpeech 2019)
- Mellotron: [Mellotron: Multispeaker expressive voice synthesis by conditioning on rhythm, pitch and global style tokens](https://arxiv.org/abs/1910.11997) (2019)
- Flowtron (flow based): [Flowtron: an Autoregressive Flow-based Generative Network for Text-to-Speech Synthesis](https://arxiv.org/abs/2005.05957) (2020)
- (local style): [Fully-hierarchical fine-grained prosody modeling for interpretable speech synthesis](https://arxiv.org/abs/2002.03785) (ICASSP 2020)
- [Controllable Neural Prosody Synthesis](https://arxiv.org/pdf/2008.03388.pdf) (Interspeech 2020)
- GraphSpeech: [GraphSpeech: Syntax-Aware Graph Attention Network For Neural Speech Synthesis](https://arxiv.org/abs/2010.12423) (2020-10)
- BERT-TTS: [Improving Prosody Modelling with Cross-Utterance BERT Embeddings for End-to-end Speech Synthesis](https://arxiv.org/abs/2011.05161) (2020-11)
- (Global Emotion Style Control): [Controllable Emotion Transfer For End-to-End Speech Synthesis](https://arxiv.org/abs/2011.08679) (2020-11)
- (Phone Level Style Control): [Fine-grained Emotion Strength Transfer, Control and Prediction for Emotional Speech Synthesis](https://arxiv.org/abs/2011.08477) (2020-11)
- (Phone Level Prosody Modelling): [Mixture Density Network for Phone-Level Prosody Modelling in Speech Synthesis](https://arxiv.org/abs/2102.00851) (ICASSP 2021)
- PeriodNet: [PeriodNet: A non-autoregressive waveform generation model with a structure separating periodic and aperiodic components](https://arxiv.org/abs/2102.07786) (ICASSP 2021)
- [Towards Multi-Scale Style Control for Expressive Speech Synthesis](https://arxiv.org/abs/2104.03521) (2021-04)
- [Learning Robust Latent Representations for Controllable Speech Synthesis](https://arxiv.org/abs/2105.04458) (2021-05)
- [Diverse and Controllable Speech Synthesis with GMM-Based Phone-Level Prosody Modelling](https://arxiv.org/abs/2105.13086) (2021-05)

<h3 id="42">MultiSpeaker TTS</h3>

- Meta-Learning for TTS<sup><span class="star" style="color: red">&#x2605;</span></sup>: [Sample Efficient Adaptive Text-to-Speech](https://arxiv.org/abs/1809.10460) (ICLR 2019)
- SV-Tacotron<sup><span class="star" style="color: red">&#x2605;</span></sup>: [Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis](https://arxiv.org/abs/1806.04558) (NeurIPS 2018)
- Deep Voice V3<sup><span class="star" style="color: red">&#x2605;</span></sup>: [Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning](https://arxiv.org/abs/1710.07654) (ICLR 2018)
- [Zero-Shot Multi-Speaker Text-To-Speech with State-of-the-art Neural Speaker Embeddings](https://arxiv.org/abs/1910.10838) (ICASSP 2020)
- MultiSpeech: [MultiSpeech: Multi-Speaker Text to Speech with Transformer](https://arxiv.org/abs/2006.04664) (2020)
- SC-WaveRNN: [Speaker Conditional WaveRNN: Towards Universal Neural Vocoder for Unseen Speaker and Recording Conditions](https://arxiv.org/pdf/2008.05289.pdf) (Interspeech 2020)
- MultiSpeaker Dataset: [AISHELL-3: A Multi-speaker Mandarin TTS Corpus and the Baselines](https://arxiv.org/abs/2010.11567) (2020)
- Life-long learning for multi-speaker TTS: [Continual Speaker Adaptation for Text-to-Speech Synthesis](https://arxiv.org/abs/2103.14512) (2021-03)
- [Meta-StyleSpeech : Multi-Speaker Adaptive Text-to-Speech Generation](https://arxiv.org/pdf/2106.03153.pdf) (ICML 2021)


<h2 id="5">Voice Conversion</h2>

<h3 id="51">ASR & TTS Based</h3>

- (introduce PPG into voice conversion): [Phonetic posteriorgrams for many-to-one voice conversion without parallel data training](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7552917) (2016)
- [A Vocoder-free WaveNet Voice Conversion with Non-Parallel Data](https://arxiv.org/pdf/1902.03705.pdf) (2019)
- TTS-Skins: [TTS Skins: Speaker Conversion via ASR](https://arxiv.org/pdf/1904.08983.pdf) (2019)
- [One-shot Voice Conversion by Separating Speaker and Content Representations with Instance Normalization](https://arxiv.org/abs/1904.05742) (InterSpeech 2019)
- Cotatron (combine text information with voice conversion system): [Cotatron: Transcription-Guided Speech Encoder for Any-to-Many Voice Conversion without Parallel Data](https://arxiv.org/abs/2005.03295) (Interspeech 2020)
- (TTS & ASR): [Voice Conversion by Cascading Automatic Speech Recognition and Text-to-Speech Synthesis with Prosody Transfer](https://arxiv.org/pdf/2009.01475.pdf) (InterSpeech 2020)
- FragmentVC (wav to vec): [FragmentVC: Any-to-Any Voice Conversion by End-to-End Extracting and Fusing Fine-Grained Voice Fragments With Attention](https://arxiv.org/abs/2010.14150) (2020)
- [Towards Natural and Controllable Cross-Lingual Voice Conversion Based on Neural TTS Model and Phonetic Posteriorgram](https://arxiv.org/abs/2102.01991) (ICASSP 2021)

<h3 id="52">VAE & Auto-Encoder Based</h3>

- VAE-VC (VAE based): [Voice Conversion from Non-parallel Corpora Using Variational Auto-encoder](https://arxiv.org/pdf/1610.04019.pdf) (2016)
- (Speech representation learning by VQ-VAE): [Unsupervised speech representation learning using WaveNet autoencoders](https://arxiv.org/abs/1901.08810) (2019)
- Blow (Flow based): [Blow: a single-scale hyperconditioned flow for non-parallel raw-audio voice conversion](https://arxiv.org/abs/1906.00794) (NeurIPS 2019)
- AutoVC: [AUTOVC: Zero-Shot Voice Style Transfer with Only Autoencoder Loss](https://arxiv.org/abs/1905.05879) (2019)
- F0-AutoVC: [F0-consistent many-to-many non-parallel voice conversion via conditional autoencoder](https://arxiv.org/abs/2004.07370) (ICASSP 2020)
- [One-Shot Voice Conversion by Vector Quantization](https://ieeexplore.ieee.org/abstract/document/9053854) (ICASSP 2020)
- SpeechFlow (auto-encoder): [Unsupervised Speech Decomposition via Triple Information Bottleneck](https://arxiv.org/abs/2004.11284) (ICML 2020)

<h3 id="53">GAN Based</h3>

- CycleGAN-VC V1: [Parallel-Data-Free Voice Conversion Using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1711.11293) (2017)
- StarGAN-VC: [StarGAN-VC: non-parallel many-to-many Voice Conversion Using Star Generative Adversarial Networks](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8639535&tag=1) (2018)
- CycleGAN-VC V2: [CycleGAN-VC2: Improved CycleGAN-based Non-parallel Voice Conversion](https://arxiv.org/abs/1904.04631) (2019)
- CycleGAN-VC V3: [CycleGAN-VC3: Examining and Improving CycleGAN-VCs for Mel-spectrogram Conversion](https://arxiv.org/abs/2010.11672) (2020)
- MaskCycleGAN-VC: [MaskCycleGAN-VC: Learning Non-parallel Voice Conversion with Filling in Frames](https://arxiv.org/abs/2102.12841) (ICASSP 2021)


<h2 id="6">Singing</h2>

<h3 id="61">Singing Synthesis</h3>

- XiaoIce Band: [XiaoIce Band: A Melody and Arrangement Generation Framework for Pop Music](https://www.kdd.org/kdd2018/accepted-papers/view/xiaoice-banda-melody-and-arrangement-generation-framework-for-pop-music) (KDD 2018)
- Mellotron: [Mellotron: Multispeaker expressive voice synthesis by conditioning on rhythm, pitch and global style tokens](https://arxiv.org/abs/1910.11997) (2019)
- ByteSing: [ByteSing: A Chinese Singing Voice Synthesis System Using Duration Allocated Encoder-Decoder Acoustic Models and WaveRNN Vocoders](https://arxiv.org/abs/2004.11012) (2020)
- JukeBox: [Jukebox: A Generative Model for Music](https://arxiv.org/abs/2005.00341) (2020)
- XiaoIce Sing: [XiaoiceSing: A High-Quality and Integrated Singing Voice Synthesis System](https://arxiv.org/abs/2006.06261) (2020)
- HiFiSinger: [HiFiSinger: Towards High-Fidelity Neural Singing Voice Synthesis](https://arxiv.org/abs/2009.01776) (2019)
- [Sequence-to-sequence Singing Voice Synthesis with Perceptual Entropy Loss](https://arxiv.org/abs/2010.12024) (2020)
- Learn2Sing: [Learn2Sing: Target Speaker Singing Voice Synthesis by learning from a Singing Teacher](https://arxiv.org/abs/2011.08467) (2020-11)

<h3 id="62">Singing Voice Conversion</h3>

- [A Universal Music Translation Network](https://arxiv.org/abs/1805.07848) (2018)
- [Unsupervised Singing Voice Conversion](https://arxiv.org/abs/1904.06590) (Interspeech 2019)
- PitchNet: [PitchNet: Unsupervised Singing Voice Conversion with Pitch Adversarial Network](https://arxiv.org/abs/1912.01852) (ICASSP 2020)
- DurIAN-SC: [DurIAN-SC: Duration Informed Attention Network based Singing Voice Conversion System](https://arxiv.org/abs/2008.03009) (Interspeech 2020)
- [Speech-to-Singing Conversion based on Boundary Equilibrium GAN](https://arxiv.org/abs/2005.13835) (Interspeech 2020)
- [PPG-based singing voice conversion with adversarial representation learning](https://arxiv.org/abs/2010.14804) (2020)
