# Speech Synthesis Paper
list of speech synthesis papers or papers related to speech processing.

## Acoustic Model
### Autoregressive Model
- Tacotron V1: [Tacotron: Towards End-to-End Speech Synthesis](https://arxiv.org/abs/1703.10135) (Interspeech 2017)
- Tacotron V2: [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/abs/1712.05884) (ICASSP 2018)
- Deep Voice V1: [Deep Voice: Real-time Neural Text-to-Speech](https://arxiv.org/abs/1702.07825) (ICML 2017)
- Deep Voice V2: [Deep Voice 2: Multi-Speaker Neural Text-to-Speech](https://arxiv.org/abs/1705.08947) (NeurIPS 2017)
- Deep Voice V3: [Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning](https://arxiv.org/abs/1710.07654) (ICLR 2018)
- Transformer-TTS: [Neural Speech Synthesis with Transformer Network](https://arxiv.org/abs/1809.08895) (AAAI 2019)
- [Location-Relative Attention Mechanisms For Robust Long-Form Speech Synthesis](https://arxiv.org/abs/1910.10288)
- DurIAN: [DurIAN: Duration Informed Attention Network For Multimodal Synthesis](https://arxiv.org/abs/1909.01700)
- Flowtron: [Flowtron: an Autoregressive Flow-based Generative Network for Text-to-Speech Synthesis](https://arxiv.org/abs/2005.05957) (flow based)

### Non-autoregressive Model
- ParaNet: [Non-Autoregressive Neural Text-to-Speech](https://arxiv.org/pdf/1905.08459.pdf) (ICML 2020)
- FastSpeech: [FastSpeech: Fast, Robust and Controllable Text to Speech](https://arxiv.org/abs/1905.09263) (NeurIPS 2019)
- JDI-T: [JDI-T: Jointly trained Duration Informed Transformer for Text-To-Speech without Explicit Alignment](https://arxiv.org/abs/2005.07799)
- EATS: [End-to-End Adversarial Text-to-Speech](https://arxiv.org/pdf/2006.03575.pdf)
- FastSpeech 2: [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558)
- FastPitch: [FastPitch: Parallel Text-to-speech with Pitch Prediction](https://arxiv.org/pdf/2006.06873.pdf)
- Glow-TTS: [Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search](https://arxiv.org/abs/2005.11129) (flow based, Monotonic Attention)

## Vocoder
### Autoregressive Model
- WaveNet: [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499)
- WaveRNN: [Efficient Neural Audio Synthesis](https://arxiv.org/abs/1802.08435)
- GAN-TTS: [High Fidelity Speech Synthesis with Adversarial Networks](https://arxiv.org/pdf/1909.11646.pdf)
- WaveGAN: [Adversarial Audio Synthesis](https://arxiv.org/abs/1802.04208)
- MultiBand-WaveRNN: [DurIAN: Duration Informed Attention Network For Multimodal Synthesis](https://arxiv.org/abs/1909.01700)

### Non-autoregressive Model
- Parallel-WaveNet: [Parallel WaveNet: Fast High-Fidelity Speech Synthesis](https://arxiv.org/pdf/1711.10433.pdf)
- WaveGlow: [WaveGlow: A Flow-based Generative Network for Speech Synthesis](https://arxiv.org/abs/1811.00002)
- Parallel-WaveGAN: [Parallel WaveGAN: A fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram](https://arxiv.org/abs/1910.11480)
- MelGAN: [MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis](https://arxiv.org/abs/1910.06711)
- MultiBand-MelGAN: [Multi-band MelGAN: Faster Waveform Generation for High-Quality Text-to-Speech](https://arxiv.org/abs/2005.05106)

## TTS towards stylization
### Expressive TTS
- ProsodyEncoder-Tacotron: [Towards End-to-End Prosody Transfer for Expressive Speech Synthesis with Tacotron](https://arxiv.org/abs/1803.09047) (ICML 2018)
- GST-Tacotron: [Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis](https://arxiv.org/abs/1803.09017) (ICML 2018)
- GMVAE-Tacotron2: [Hierarchical Generative Modeling for Controllable Speech Synthesis](https://arxiv.org/abs/1810.07217) (ICLR 2019)
- [Multi-reference Tacotron by Intercross Training for Style Disentangling,Transfer and Control in Speech Synthesis](https://arxiv.org/abs/1904.02373)
- Mellotron: [Mellotron: Multispeaker expressive voice synthesis by conditioning on rhythm, pitch and global style tokens](https://arxiv.org/abs/1910.11997)
- Flowtron: [Flowtron: an Autoregressive Flow-based Generative Network for Text-to-Speech Synthesis](https://arxiv.org/abs/2005.05957) (flow based)
- [Fully-hierarchical fine-grained prosody modeling for interpretable speech synthesis](https://arxiv.org/abs/2002.03785)

### MultiSpeaker TTS
- [Sample Efficient Adaptive Text-to-Speech](https://arxiv.org/abs/1809.10460)
- SV-Tacotron: [Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis](https://arxiv.org/abs/1806.04558)
- Deep Voice V3: [Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning](https://arxiv.org/abs/1710.07654) (ICLR 2018)

## Voice Conversion
### ASR Base
- [Phonetic posteriorgrams for many-to-one voice conversion without parallel data training](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7552917) (introduce PPG into voice conversion)
- [A Vocoder-free WaveNet Voice Conversion with Non-Parallel Data](https://arxiv.org/pdf/1902.03705.pdf)
- TTS-Skins: [TTS Skins: Speaker Conversion via ASR](https://arxiv.org/pdf/1904.08983.pdf)

### GAN/VAE Base
- AutoVC: [AUTOVC: Zero-Shot Voice Style Transfer with Only Autoencoder Loss](https://arxiv.org/abs/1905.05879)
- CycleGAN-VC V1: [Parallel-Data-Free Voice Conversion Using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1711.11293)
- CycleGAN-VC V2: [CycleGAN-VC2: Improved CycleGAN-based Non-parallel Voice Conversion](https://arxiv.org/abs/1904.04631)
- StarGAN-VC: [StarGAN-VC: non-parallel many-to-many Voice Conversion Using Star Generative Adversarial Networks](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8639535&tag=1)
- VAE-VC: [Voice Conversion from Non-parallel Corpora Using Variational Auto-encoder](https://arxiv.org/pdf/1610.04019.pdf)

### Other
- [Blow: a single-scale hyperconditioned flow for non-parallel raw-audio voice conversion](https://arxiv.org/abs/1906.00794)
- [One-shot Voice Conversion by Separating Speaker and Content Representations with Instance Normalization](https://arxiv.org/abs/1904.05742)
- Cotatron: [Cotatron: Transcription-Guided Speech Encoder for Any-to-Many Voice Conversion without Parallel Data](https://arxiv.org/abs/2005.03295) (combine text information with voice conversion system)

## Singing Synthesis
- XiaoIce Band: [XiaoIce Band: A Melody and Arrangement Generation Framework for Pop Music](https://www.kdd.org/kdd2018/accepted-papers/view/xiaoice-banda-melody-and-arrangement-generation-framework-for-pop-music)
- PitchNet: [PitchNet: Unsupervised Singing Voice Conversion with Pitch Adversarial Network](https://arxiv.org/abs/1912.01852)
- Mellotron: [Mellotron: Multispeaker expressive voice synthesis by conditioning on rhythm, pitch and global style tokens](https://arxiv.org/abs/1910.11997)
- ByteSing: [ByteSing: A Chinese Singing Voice Synthesis System Using Duration Allocated Encoder-Decoder Acoustic Models and WaveRNN Vocoders](https://arxiv.org/abs/2004.11012)
- JukeBox: [Jukebox: A Generative Model for Music](https://arxiv.org/abs/2005.00341)
- XiaoIce Singing: [XiaoiceSing: A High-Quality and Integrated Singing Voice Synthesis System](https://arxiv.org/abs/2006.06261)

## Speech Pretrained Model
- Audio-Word2Vec: [Audio Word2Vec: Unsupervised Learning of Audio Segment Representations using Sequence-to-sequence Autoencoder](https://arxiv.org/pdf/1603.00982.pdf)
- SpeechBERT: [SpeechBERT: An Audio-and-text Jointly Learned Language Model for End-to-end Spoken Question Answering](https://arxiv.org/abs/1910.11559)
- [Improving Transformer-based Speech Recognition Using Unsupervised Pre-training](https://arxiv.org/abs/1910.09932)