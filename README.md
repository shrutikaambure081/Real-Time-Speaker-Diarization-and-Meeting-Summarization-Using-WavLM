# Speaker Diarization and Summarization Using WavLM

## 1. Overview

This project is an AI-based speech processing system that analyzes multi-speaker audio and produces **speaker-wise transcripts, emotions, summaries, and speech analytics**.

### Pipeline

```text
Audio
 ↓
Voice Activity Detection
 ↓
WavLM Embeddings
 ↓
Speaker Diarization + Identification
 ↓
Whisper Transcription
 ↓
Voice + Text Emotion Detection
 ↓
BART Summarization
 ↓
Final Report
```

## 2. Objectives

- To identify different speakers in an audio recording.
- To determine when each speaker is speaking.
- To convert speech into text.
- To detect emotions from voice and text.
- To generate overall and speaker-wise summaries.
- To provide timestamps and speaker-wise speech analysis.

## 3. Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- WavLM
- Whisper
- Wav2Vec2
- DistilRoBERTa
- BART
- Librosa
- Scikit-learn
- NLP and Deep Learning

## 4. Models Used

| Model / Method | Used For | Purpose |
|---|---|---|
| **WavLM** | Speaker Identification | Extracts speaker embeddings. |
| **WavLM** | Speaker Diarization | Extracts embeddings from audio windows for speaker clustering. |
| **Neural Classifier Head** | Speaker Identification | Predicts the trained speaker class from WavLM embeddings. |
| **Agglomerative Clustering** | Speaker Diarization | Groups similar embeddings into speaker segments. |
| **Whisper-Small** | Speech-to-Text | Converts speaker audio segments into text. |
| **Wav2Vec2 Speech Emotion Model** | Voice Emotion | Detects emotion from audio. |
| **DistilRoBERTa Emotion Model** | Text Emotion | Detects emotion from the transcript. |
| **BART-Large-CNN** | Summarization | Generates concise summaries. |

## 5. Dataset

The project uses the **LibriSpeech Clean validation dataset**.

- 8 speakers
- 311 utterances
- 36.5 minutes of audio
- 80% training data
- 20% testing data
- 16 kHz mono audio
- 3–15 second utterances

Custom audio can also be provided using speaker-wise folders.

## 6. Speaker Identification

WavLM generates **512-dimensional speaker embeddings**.

A neural classifier is trained on 80% of the data and tested on the unseen 20%.

**Results:**
- Window Accuracy: **96.8%**
- Utterance Accuracy: **98.4%**

## 7. Speaker Diarization

The system identifies **who spoke when** using:

1. Energy-based Voice Activity Detection
2. 1.5-second audio windows
3. WavLM embeddings
4. Agglomerative clustering
5. Speaker segmentation

Automatic speaker count is estimated using silhouette scoring.

**Mean DER with known speaker count: 25.4%**

## 8. Speech Transcription

**Whisper-Small** converts each detected speaker segment into text.

The output contains:

- Speaker label
- Start time
- End time
- Transcribed speech

**Mean WER: 5.5%**

## 9. Emotion Detection

Two models are used:

- **Wav2Vec2 speech emotion model** → analyzes emotion from the speaker's voice.
- **DistilRoBERTa emotion model** → analyzes emotion from the transcribed text.

The system combines these results to produce the final expression/emotion.

## 10. Summarization

**BART-Large-CNN** generates:

- Overall conversation summary
- Speaker-wise summaries

Long transcripts are divided into smaller chunks before summarization.

## 11. Final Output

The system provides:

- Number of speakers
- Speaker-wise timestamps
- Speaker-wise transcript
- Detected emotions
- Overall summary
- Speaker-wise summary
- Talk-time information
- Evaluation metrics

### Output Files

```text
outputs/
├── *.rttm
├── *.json
└── final_results.csv
```

## 12. Final Results

| Metric | Result |
|---|---:|
| Speaker Identification – Utterance | **98.4%** |
| Speaker Identification – Window | **96.8%** |
| Mean DER – Known Speakers | **25.4%** |
| Mean WER | **5.5%** |

## 13. Limitations

- Current evaluation uses LibriSpeech read-aloud speech.
- Test conversations do not contain overlapping speech.
- Automatic speaker counting may underestimate or overestimate speakers.
- Pretrained emotion models may be less reliable on real-world conversations.
- Summaries may be less meaningful for read-aloud passages.

## 14. Future Scope

- To support overlapping speakers.
- To improve automatic speaker-count estimation.
- To use neural Voice Activity Detection.
- To fine-tune WavLM for specific domains.
- To use larger Whisper models.
- To support real-time meeting audio.

## 15. Conclusion

The project provides an end-to-end speech analytics pipeline combining **WavLM, Whisper, emotion models, clustering, and BART** to identify speakers, transcribe speech, analyze emotions, and generate summaries from multi-speaker audio.
