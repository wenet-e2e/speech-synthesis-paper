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
* [Speech Processing Related](#7)
    * [Speech Pretrained Model](#71)
    * [Speech Separation](#72)
    * [Speaker Verification](#73)
    * [Speech Representation Learning](#74)
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
- [Location-Relative Attention Mechanisms For Robust Long-Form Speech Synthesis](https://arxiv.org/abs/1910.10288) (ICASSP 2020)
- DurIAN: [DurIAN: Duration Informed Attention Network For Multimodal Synthesis](https://arxiv.org/abs/1909.01700) (2019)
- Flowtron (flow based): [Flowtron: an Autoregressive Flow-based Generative Network for Text-to-Speech Synthesis](https://arxiv.org/abs/2005.05957) (2020)
- [Non-Attentive Tacotron: Robust and Controllable Neural TTS Synthesis Including Unsupervised Duration Modeling](https://arxiv.org/pdf/2010.04301v1.pdf) (under review ICLR 2021)
- DeviceTTS: [DeviceTTS: A Small-Footprint, Fast, Stable Network for On-Device Text-to-Speech](https://arxiv.org/abs/2010.15311) (2020-10)

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
- Wave-Tacotron: [Wave-Tacotron: Spectrogram-free end-to-end text-to-speech synthesis](https://arxiv.org/abs/2011.03568) (2020-11)

<h3 id="23">Alignment Study</h3>

- Monotonic Attention: [Online and Linear-Time Attention by Enforcing Monotonic Alignments](https://arxiv.org/abs/1704.00784) (ICML 2017)
- Monotonic Chunkwise Attention: [Monotonic Chunkwise Attention](https://arxiv.org/abs/1712.05382) (ICLR 2018)
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


<h2 id="3">Vocoder</h2>

<h3 id="31">Autoregressive Model</h3>

- WaveNet: [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499) (2016)
- WaveRNN: [Efficient Neural Audio Synthesis](https://arxiv.org/abs/1802.08435) (ICML 2018)
- WaveGAN: [Adversarial Audio Synthesis](https://arxiv.org/abs/1802.04208) (ICLR 2019)
- LPCNet: [LPCNet: Improving Neural Speech Synthesis Through Linear Prediction](https://arxiv.org/abs/1810.11846) (ICASSP 2019)
- GAN-TTS: [High Fidelity Speech Synthesis with Adversarial Networks](https://arxiv.org/pdf/1909.11646.pdf) (2019)
- MultiBand-WaveRNN: [DurIAN: Duration Informed Attention Network For Multimodal Synthesis](https://arxiv.org/abs/1909.01700) (2019)

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


<h2 id="4">TTS towards Stylization</h2>

<h3 id="41">Expressive TTS</h3>

- ReferenceEncoder-Tacotron: [Towards End-to-End Prosody Transfer for Expressive Speech Synthesis with Tacotron](https://arxiv.org/abs/1803.09047) (ICML 2018)
- GST-Tacotron: [Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis](https://arxiv.org/abs/1803.09017) (ICML 2018)
- [Predicting Expressive Speaking Style From Text In End-To-End Speech Synthesis](https://arxiv.org/pdf/1808.01410.pdf) (2018)
- GMVAE-Tacotron2: [Hierarchical Generative Modeling for Controllable Speech Synthesis](https://arxiv.org/abs/1810.07217) (ICLR 2019)
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

<h3 id="42">MultiSpeaker TTS</h3>

- Meta-Learning for TTS: [Sample Efficient Adaptive Text-to-Speech](https://arxiv.org/abs/1809.10460) (ICLR 2019)
- SV-Tacotron: [Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis](https://arxiv.org/abs/1806.04558) (NeurIPS 2018)
- Deep Voice V3: [Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning](https://arxiv.org/abs/1710.07654) (ICLR 2018)
- [Zero-Shot Multi-Speaker Text-To-Speech with State-of-the-art Neural Speaker Embeddings](https://arxiv.org/abs/1910.10838) (ICASSP 2020)
- MultiSpeech: [MultiSpeech: Multi-Speaker Text to Speech with Transformer](https://arxiv.org/abs/2006.04664) (2020)
- SC-WaveRNN: [Speaker Conditional WaveRNN: Towards Universal Neural Vocoder for Unseen Speaker and Recording Conditions](https://arxiv.org/pdf/2008.05289.pdf) (Interspeech 2020)
- MultiSpeaker Dataset: [AISHELL-3: A Multi-speaker Mandarin TTS Corpus and the Baselines](https://arxiv.org/abs/2010.11567) (2020)


<h2 id="5">Voice Conversion</h2>

<h3 id="51">ASR & TTS Based</h3>

- (introduce PPG into voice conversion): [Phonetic posteriorgrams for many-to-one voice conversion without parallel data training](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7552917) (2016)
- [A Vocoder-free WaveNet Voice Conversion with Non-Parallel Data](https://arxiv.org/pdf/1902.03705.pdf) (2019)
- TTS-Skins: [TTS Skins: Speaker Conversion via ASR](https://arxiv.org/pdf/1904.08983.pdf) (2019)
- [One-shot Voice Conversion by Separating Speaker and Content Representations with Instance Normalization](https://arxiv.org/abs/1904.05742) (InterSpeech 2019)
- Cotatron (combine text information with voice conversion system): [Cotatron: Transcription-Guided Speech Encoder for Any-to-Many Voice Conversion without Parallel Data](https://arxiv.org/abs/2005.03295) (Interspeech 2020)
- (TTS & ASR): [Voice Conversion by Cascading Automatic Speech Recognition and Text-to-Speech Synthesis with Prosody Transfer](https://arxiv.org/pdf/2009.01475.pdf) (InterSpeech 2020)
- FragmentVC (wav to vec): [FragmentVC: Any-to-Any Voice Conversion by End-to-End Extracting and Fusing Fine-Grained Voice Fragments With Attention](https://arxiv.org/abs/2010.14150) (2020)

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


<h2 id="7">Speech Processing Related</h2>

<h3 id="71">Speech Pretrained Model</h3>

- Audio-Word2Vec: [Audio Word2Vec: Unsupervised Learning of Audio Segment Representations using Sequence-to-sequence Autoencoder](https://arxiv.org/pdf/1603.00982.pdf) (2016)
- SpeechBERT: [SpeechBERT: An Audio-and-text Jointly Learned Language Model for End-to-end Spoken Question Answering](https://arxiv.org/abs/1910.11559) (2019)
- [Improving Transformer-based Speech Recognition Using Unsupervised Pre-training](https://arxiv.org/abs/1910.09932) (2019)

<h3 id="72">Speech Separation</h3>

- TasNet: [TasNet: time-domain audio separation network for real-time, single-channel speech separation](https://arxiv.org/abs/1711.00541) (ICASSP 2018)
- Conv-TasNet: [Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation](https://arxiv.org/abs/1809.07454)

<h3 id="73">Speaker Verification</h3>

- DeepSpeaker: [Deep Speaker: an End-to-End Neural Speaker Embedding System](https://arxiv.org/pdf/1705.02304.pdf) (2017)
- GE2E Loss: [Generalized End-to-End Loss for Speaker Verification](https://arxiv.org/abs/1710.10467) (ICASSP 2018)

<h3 id="74">Speech Representation Learning</h3>

- [Unsupervised speech representation learning using WaveNet autoencoders](https://arxiv.org/abs/1901.08810) (2019)


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

<h3 id="83">Non-autoregressive Translation Model</h3>

- [A Study of Non-autoregressive Model for Sequence Generation](https://arxiv.org/abs/2004.10454) (ACL 2020)
- [Deterministic Non-Autoregressive Neural Sequence Modeling by Iterative Refinement](https://arxiv.org/abs/1802.06901) (EMNLP 2018)
- [Non-Autoregressive Neural Machine Translation](https://arxiv.org/abs/1711.02281v1) (ICLR 2018)
- [Non-Autoregressive Machine Translation with Auxiliary Regularization](https://arxiv.org/abs/1902.10245) (AAAI 2019)
- [Mask-Predict: Parallel Decoding of Conditional Masked Language Models](https://arxiv.org/abs/1904.09324) (EMNLP 2019)

<h3 id="84">Speech2Speech Translation Model</h3>

- Awesome Paper List: [awesome-speech-translation](https://github.com/dqqcasia/awesome-speech-translation)
- [Direct speech-to-speech translation with a sequence-to-sequence model](https://arxiv.org/abs/1904.06037) (InterSpeech 2020)
- NeurST: [NeurST: Neural Speech Translation Toolkit](https://arxiv.org/pdf/2012.10018.pdf) (2020-12)

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
