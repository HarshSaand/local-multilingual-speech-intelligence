# Local Multilingual Speech Intelligence

A privacy-conscious reference pipeline for running multilingual speech recognition locally. It can transcribe source-language audio, translate supported speech to English, attach optional speaker labels, and export plain text, JSON, and accessible HTML.

This repository is a portfolio-safe reference implementation. It contains no employer source code, recordings, customer transcripts, identifiers, private vocabulary, or production configuration.

## What it demonstrates

- Local inference with `faster-whisper`
- Multilingual transcription or English translation
- Optional two-or-more-speaker diarization with `pyannote.audio`
- Timestamped segments and speaker attribution by temporal overlap
- Structured TXT, JSON, and HTML output
- No automatic upload or network API for audio processing

## Install

Python 3.10 or newer and FFmpeg are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For optional speaker diarization:

```bash
pip install -r requirements-diarization.txt
export HF_TOKEN="your-hugging-face-token"
```

The diarization model may require accepting its access terms on Hugging Face.

## Run

```bash
python local_speech_intelligence.py recording.wav --out-dir outputs
```

Translate supported speech into English:

```bash
python local_speech_intelligence.py recording.wav --task translate --out-dir outputs
```

Add speaker diarization:

```bash
python local_speech_intelligence.py recording.wav --diarize --out-dir outputs
```

Useful options:

```text
--model small            Whisper model name or local model path
--language zh            Optional source-language hint
--device auto            auto, cpu, or cuda
--compute-type auto       auto, int8, float16, or another supported type
--task transcribe         transcribe or translate
--diarize                 Enable pyannote speaker diarization
--hf-token-env HF_TOKEN   Environment variable containing the model token
```

## Output contract

Each run creates three files named after the input recording:

- `.transcript.txt`: readable timestamped lines
- `.transcript.json`: structured metadata and segments
- `.transcript.html`: a responsive local review page

The JSON format is intentionally small:

```json
{
  "source": "recording.wav",
  "language": "en",
  "task": "transcribe",
  "segments": [
    {
      "start": 0.0,
      "end": 2.4,
      "speaker": "Speaker 1",
      "text": "Hello, how can I help?"
    }
  ]
}
```

The example in `examples/synthetic-transcript.json` is synthetic and contains no real conversation.

## Privacy boundaries

- Keep audio and generated transcripts out of version control.
- Review retention and consent requirements before processing real calls.
- Treat model output as a draft that may contain transcription, translation, or speaker-attribution errors.
- Do not use this reference implementation for identity verification or automated decisions.

## Test

The rendering tests use only Python's standard library and synthetic data:

```bash
python -m unittest discover -s tests
```

