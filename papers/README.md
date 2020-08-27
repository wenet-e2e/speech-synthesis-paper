# Paper List
List of papers not just about speech synthesis.

## TTS Frontend
- [Pre-trained Text Representations for Improving Front-End Text Processing in Mandarin Text-to-Speech Synthesis](https://pdfs.semanticscholar.org/6abc/7dac0bdc50735b6d12f96400f59b5f084759.pdf) (Interspeech 2019)
- [A unified sequence-to-sequence front-end model for Mandarin text-to-speech synthesis](https://arxiv.org/pdf/1911.04111.pdf) (ICASSP 2020)
- [A hybrid text normalization system using multi-head self-attention for mandarin](https://arxiv.org/pdf/1911.04128.pdf) (ICASSP 2020)


## Acoustic Model
### Autoregressive Model
- Tacotron V1: [Tacotron: Towards End-to-End Speech Synthesis](https://arxiv.org/abs/1703.10135) (Interspeech 2017)
- Tacotron V2: [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/abs/1712.05884) (ICASSP 2018)
- Deep Voice V1: [Deep Voice: Real-time Neural Text-to-Speech](https://arxiv.org/abs/1702.07825) (ICML 2017)
- Deep Voice V2: [Deep Voice 2: Multi-Speaker Neural Text-to-Speech](https://arxiv.org/abs/1705.08947) (NeurIPS 2017)
- Deep Voice V3: [Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning](https://arxiv.org/abs/1710.07654) (ICLR 2018)
- Transformer-TTS: [Neural Speech Synthesis with Transformer Network](https://arxiv.org/abs/1809.08895) (AAAI 2019)
- [Location-Relative Attention Mechanisms For Robust Long-Form Speech Synthesis](https://arxiv.org/abs/1910.10288) (ICASSP 2020)
- DurIAN: [DurIAN: Duration Informed Attention Network For Multimodal Synthesis](https://arxiv.org/abs/1909.01700) (2019)
- Flowtron (flow based): [Flowtron: an Autoregressive Flow-based Generative Network for Text-to-Speech Synthesis](https://arxiv.org/abs/2005.05957) (2020)

### Non-autoregressive Model
- ParaNet: [Non-Autoregressive Neural Text-to-Speech](https://arxiv.org/pdf/1905.08459.pdf) (ICML 2020)
- FastSpeech: [FastSpeech: Fast, Robust and Controllable Text to Speech](https://arxiv.org/abs/1905.09263) (NeurIPS 2019)
- JDI-T: [JDI-T: Jointly trained Duration Informed Transformer for Text-To-Speech without Explicit Alignment](https://arxiv.org/abs/2005.07799) (2020)
- EATS: [End-to-End Adversarial Text-to-Speech](https://arxiv.org/pdf/2006.03575.pdf) (2020)
- FastSpeech 2: [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558) (2020)
- FastPitch: [FastPitch: Parallel Text-to-speech with Pitch Prediction](https://arxiv.org/pdf/2006.06873.pdf) (2020)
- Glow-TTS (flow based, Monotonic Attention): [Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search](https://arxiv.org/abs/2005.11129) (2020)
- Flow-TTS (flow based): [Flow-TTS: A Non-Autoregressive Network for Text to Speech Based on Flow](https://ieeexplore.ieee.org/document/9054484) (ICASSP 2020)
- SpeedySpeech: [SpeedySpeech: Efficient Neural Speech Synthesis](https://arxiv.org/pdf/2008.03802.pdf) (Interspeech 2020)

### Alignment Study
- Monotonic Attention: [Online and Linear-Time Attention by Enforcing Monotonic Alignments](https://arxiv.org/abs/1704.00784) (ICML 2017)
- Monotonic Chunkwise Attention: [Monotonic Chunkwise Attention](https://arxiv.org/abs/1712.05382) (ICLR 2018)
- [Forward Attention in Sequence-to-sequence Acoustic Modelling for Speech Synthesis](https://arxiv.org/abs/1807.06736) (ICASSP 2018)
- RNN-T for TTS: [Initial investigation of an encoder-decoder end-to-end TTS framework using marginalization of monotonic hard latent alignments](http://128.84.4.27/pdf/1908.11535) (2019)
- [Location-Relative Attention Mechanisms For Robust Long-Form Speech Synthesis](https://arxiv.org/abs/1910.10288) (ICASSP 2020)

### Data Efficiency
- [Semi-Supervised Training for Improving Data Efficiency in End-to-End Speech Synthesis](https://arxiv.org/abs/1808.10128) (2018)
- [Almost Unsupervised Text to Speech and Automatic Speech Recognition](https://arxiv.org/abs/1905.06791) (ICML 2019)
- [Unsupervised Learning For Sequence-to-sequence Text-to-speech For Low-resource Languages](https://arxiv.org/pdf/2008.04549.pdf) (Interspeech 2020)


## Vocoder
### Autoregressive Model
- WaveNet: [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499) (2016)
- WaveRNN: [Efficient Neural Audio Synthesis](https://arxiv.org/abs/1802.08435) (ICML 2018)
- LPCNet: [LPCNet: Improving Neural Speech Synthesis Through Linear Prediction](https://arxiv.org/abs/1810.11846) (ICASSP 2019)
- GAN-TTS: [High Fidelity Speech Synthesis with Adversarial Networks](https://arxiv.org/pdf/1909.11646.pdf) (2019)
- WaveGAN: [Adversarial Audio Synthesis](https://arxiv.org/abs/1802.04208) (2018)
- MultiBand-WaveRNN: [DurIAN: Duration Informed Attention Network For Multimodal Synthesis](https://arxiv.org/abs/1909.01700) (2019)

### Non-autoregressive Model
- Parallel-WaveNet: [Parallel WaveNet: Fast High-Fidelity Speech Synthesis](https://arxiv.org/pdf/1711.10433.pdf) (2017)
- WaveGlow: [WaveGlow: A Flow-based Generative Network for Speech Synthesis](https://arxiv.org/abs/1811.00002) (2018)
- Parallel-WaveGAN: [Parallel WaveGAN: A fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram](https://arxiv.org/abs/1910.11480) (2019)
- MelGAN: [MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis](https://arxiv.org/abs/1910.06711) (NeurIPS 2019)
- MultiBand-MelGAN: [Multi-band MelGAN: Faster Waveform Generation for High-Quality Text-to-Speech](https://arxiv.org/abs/2005.05106) (2020)
- VocGAN: [VocGAN: A High-Fidelity Real-time Vocoder with a Hierarchically-nested Adversarial Network](https://arxiv.org/abs/2007.15256) (Interspeech 2020)


## TTS towards Stylization
### Expressive TTS
- ReferenceEncoder-Tacotron: [Towards End-to-End Prosody Transfer for Expressive Speech Synthesis with Tacotron](https://arxiv.org/abs/1803.09047) (ICML 2018)
- GST-Tacotron: [Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis](https://arxiv.org/abs/1803.09017) (ICML 2018)
- GMVAE-Tacotron2: [Hierarchical Generative Modeling for Controllable Speech Synthesis](https://arxiv.org/abs/1810.07217) (ICLR 2019)
- [Predicting Expressive Speaking Style From Text In End-To-End Speech Synthesis](https://arxiv.org/pdf/1808.01410.pdf) (2018)
- (Multi-style Decouple): [Multi-reference Tacotron by Intercross Training for Style Disentangling,Transfer and Control in Speech Synthesis](https://arxiv.org/abs/1904.02373) (InterSpeech 2019)
- Mellotron: [Mellotron: Multispeaker expressive voice synthesis by conditioning on rhythm, pitch and global style tokens](https://arxiv.org/abs/1910.11997) (2019)
- Flowtron  (flow based): [Flowtron: an Autoregressive Flow-based Generative Network for Text-to-Speech Synthesis](https://arxiv.org/abs/2005.05957) (2020)
- [Fully-hierarchical fine-grained prosody modeling for interpretable speech synthesis](https://arxiv.org/abs/2002.03785) (ICASSP 2020)
- [Controllable Neural Prosody Synthesis](https://arxiv.org/pdf/2008.03388.pdf) (Interspeech 2020)

### MultiSpeaker TTS
- [Sample Efficient Adaptive Text-to-Speech](https://arxiv.org/abs/1809.10460) (ICLR 2019)
- SV-Tacotron: [Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis](https://arxiv.org/abs/1806.04558) (NeurIPS 2018)
- Deep Voice V3: [Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning](https://arxiv.org/abs/1710.07654) (ICLR 2018)
- [Zero-Shot Multi-Speaker Text-To-Speech with State-of-the-art Neural Speaker Embeddings](https://arxiv.org/abs/1910.10838) (ICASSP 2020)
- MultiSpeech: [MultiSpeech: Multi-Speaker Text to Speech with Transformer](https://arxiv.org/abs/2006.04664) (2020)
- SC-WaveRNN: [Speaker Conditional WaveRNN: Towards Universal Neural Vocoder for Unseen Speaker and Recording Conditions](https://arxiv.org/pdf/2008.05289.pdf) (Interspeech 2020)


## Voice Conversion
### ASR Based
- (introduce PPG into voice conversion): [Phonetic posteriorgrams for many-to-one voice conversion without parallel data training](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7552917) (2016)
- [A Vocoder-free WaveNet Voice Conversion with Non-Parallel Data](https://arxiv.org/pdf/1902.03705.pdf) (2019)
- TTS-Skins: [TTS Skins: Speaker Conversion via ASR](https://arxiv.org/pdf/1904.08983.pdf) (2019)

### GAN/VAE Based
- AutoVC: [AUTOVC: Zero-Shot Voice Style Transfer with Only Autoencoder Loss](https://arxiv.org/abs/1905.05879) (2019)
- CycleGAN-VC V1: [Parallel-Data-Free Voice Conversion Using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1711.11293) (2017)
- CycleGAN-VC V2: [CycleGAN-VC2: Improved CycleGAN-based Non-parallel Voice Conversion](https://arxiv.org/abs/1904.04631) (2019)
- StarGAN-VC: [StarGAN-VC: non-parallel many-to-many Voice Conversion Using Star Generative Adversarial Networks](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8639535&tag=1) (2018)
- VAE-VC (VAE based): [Voice Conversion from Non-parallel Corpora Using Variational Auto-encoder](https://arxiv.org/pdf/1610.04019.pdf) (2016)

### Other
- [Blow: a single-scale hyperconditioned flow for non-parallel raw-audio voice conversion](https://arxiv.org/abs/1906.00794) (2019)
- [One-shot Voice Conversion by Separating Speaker and Content Representations with Instance Normalization](https://arxiv.org/abs/1904.05742) (2019)
- Cotatron (combine text information with voice conversion system): [Cotatron: Transcription-Guided Speech Encoder for Any-to-Many Voice Conversion without Parallel Data](https://arxiv.org/abs/2005.03295) (Interspeech 2020)


## Singing
### Singing Synthesis
- XiaoIce Band: [XiaoIce Band: A Melody and Arrangement Generation Framework for Pop Music](https://www.kdd.org/kdd2018/accepted-papers/view/xiaoice-banda-melody-and-arrangement-generation-framework-for-pop-music) (KDD 2018)
- Mellotron: [Mellotron: Multispeaker expressive voice synthesis by conditioning on rhythm, pitch and global style tokens](https://arxiv.org/abs/1910.11997) (2019)
- ByteSing: [ByteSing: A Chinese Singing Voice Synthesis System Using Duration Allocated Encoder-Decoder Acoustic Models and WaveRNN Vocoders](https://arxiv.org/abs/2004.11012) (2020)
- JukeBox: [Jukebox: A Generative Model for Music](https://arxiv.org/abs/2005.00341) (2020)
- XiaoIce Sing: [XiaoiceSing: A High-Quality and Integrated Singing Voice Synthesis System](https://arxiv.org/abs/2006.06261) (2020)

### Singing Voice Conversion
- [A Universal Music Translation Network](https://arxiv.org/abs/1805.07848) (2018)
- [Unsupervised Singing Voice Conversion](https://arxiv.org/abs/1904.06590) (Interspeech 2019)
- PitchNet: [PitchNet: Unsupervised Singing Voice Conversion with Pitch Adversarial Network](https://arxiv.org/abs/1912.01852) (ICASSP 2020)
- DurIAN-SC: [DurIAN-SC: Duration Informed Attention Network based Singing Voice Conversion System](https://arxiv.org/abs/2008.03009) (Interspeech 2020)
- [Speech-to-Singing Conversion based on Boundary Equilibrium GAN](https://arxiv.org/abs/2005.13835) (Interspeech 2020)


## Speech Processing Related
### Speech Pretrained Model
- Audio-Word2Vec: [Audio Word2Vec: Unsupervised Learning of Audio Segment Representations using Sequence-to-sequence Autoencoder](https://arxiv.org/pdf/1603.00982.pdf) (2016)
- SpeechBERT: [SpeechBERT: An Audio-and-text Jointly Learned Language Model for End-to-end Spoken Question Answering](https://arxiv.org/abs/1910.11559) (2019)
- [Improving Transformer-based Speech Recognition Using Unsupervised Pre-training](https://arxiv.org/abs/1910.09932) (2019)

### Speech Separation
- TasNet: [TasNet: time-domain audio separation network for real-time, single-channel speech separation](https://arxiv.org/abs/1711.00541) (ICASSP 2018)
- Conv-TasNet: [Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation](https://arxiv.org/abs/1809.07454)

### Speaker Verification
- DeepSpeaker: [Deep Speaker: an End-to-End Neural Speaker Embedding System](https://arxiv.org/pdf/1705.02304.pdf) (2017)
- GE2E Loss: [Generalized End-to-End Loss for Speaker Verification](https://arxiv.org/abs/1710.10467) (ICASSP 2018)


## Natural Language Processing
### Sequence Modeling
- LSTM: [Long Short-term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) (1997)
- GRU: [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078v3) (EMNLP 2014)
- TCN: [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271) (2018)
- Transformer: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (NIPS 2017)
- Transformer-XL: [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860) (ACL 2019)
- Reformer: [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451) (ICLR 2020)

### Non-autoregressive Translation Model
- [A Study of Non-autoregressive Model for Sequence Generation](https://arxiv.org/abs/2004.10454) (ACL 2020)
- [Deterministic Non-Autoregressive Neural Sequence Modeling by Iterative Refinement](https://arxiv.org/abs/1802.06901) (EMNLP 2018)
- [Non-Autoregressive Neural Machine Translation](https://arxiv.org/abs/1711.02281v1) (ICLR 2018)
- [Non-Autoregressive Machine Translation with Auxiliary Regularization](https://arxiv.org/abs/1902.10245) (AAAI 2019)
- [Mask-Predict: Parallel Decoding of Conditional Masked Language Models](https://arxiv.org/abs/1904.09324) (EMNLP 2019)


## VAE/GAN
### VAE
- VAE: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) (ICLR 2014)
- GM-VAE: [Deep Unsupervised Clustering with Gaussian Mixture Variational Autoencoders](https://arxiv.org/abs/1611.02648) (ICLR 2017)
- VQ-VAE: [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) (NIPS 2017)
- VQ-VAE 2: [Generating Diverse High-Fidelity Images with VQ-VAE-2](https://arxiv.org/abs/1906.00446) (NeurIPS 2019)

### GAN
- GAN: [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) (NIPS 2014)
- Condition-GAN: [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) (2014)
- Info-GAN: [InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://arxiv.org/abs/1606.03657) (2016)
- SeqGAN: [SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient](https://arxiv.org/abs/1609.05473) (AAAI 2017)
- Cycle-GAN: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) (ICCV 2017)
- Star-GAN: [StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation](https://arxiv.org/abs/1711.09020) (CVPR 2018)
- BigGAN: [Large Scale GAN Training for High Fidelity Natural Image Synthesis](https://arxiv.org/abs/1809.11096) (ICLR 2019)
- Style-GAN: [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948) (CVPR 2019)


## Others
- ScaNN (search accelerating): [Accelerating Large-Scale Inference with Anisotropic Vector Quantization](https://arxiv.org/abs/1908.10396) (ICML 2020)
- (memory management): [Efficient Memory Management for Deep Neural Net Inference](https://arxiv.org/abs/2001.03288) (2020)