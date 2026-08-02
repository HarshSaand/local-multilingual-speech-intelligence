#!/usr/bin/env python3
"""Local multilingual transcription with optional speaker diarization.

This portfolio-safe reference contains no organisation-specific prompts,
vocabulary, role rules, recordings, or transcript data.
"""

from __future__ import annotations

import argparse
import html
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


SUPPORTED_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".ogg",
    ".wav",
    ".webm",
}


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str
    speaker: str = "Transcript"


@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str


def format_timestamp(seconds: float) -> str:
    total_ms = max(0, round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def overlap_seconds(segment: TranscriptSegment, turn: SpeakerTurn) -> float:
    return max(0.0, min(segment.end, turn.end) - max(segment.start, turn.start))


def attach_speakers(
    segments: Iterable[TranscriptSegment], turns: Iterable[SpeakerTurn]
) -> list[TranscriptSegment]:
    diarization = list(turns)
    output: list[TranscriptSegment] = []
    for segment in segments:
        best = max(
            diarization,
            key=lambda turn: overlap_seconds(segment, turn),
            default=None,
        )
        speaker = segment.speaker
        if best and overlap_seconds(segment, best) > 0:
            speaker = best.speaker
        output.append(
            TranscriptSegment(
                start=segment.start,
                end=segment.end,
                text=segment.text,
                speaker=speaker,
            )
        )
    return output


def transcribe(
    audio_path: Path,
    model_name: str,
    task: str,
    language: str | None,
    device: str,
    compute_type: str,
) -> tuple[list[TranscriptSegment], str | None]:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    raw_segments, info = model.transcribe(
        str(audio_path),
        task=task,
        language=language,
        vad_filter=True,
        beam_size=5,
    )
    segments = [
        TranscriptSegment(
            start=float(segment.start),
            end=float(segment.end),
            text=segment.text.strip(),
        )
        for segment in raw_segments
        if segment.text.strip()
    ]
    return segments, getattr(info, "language", language)


def diarize(audio_path: Path, token: str) -> list[SpeakerTurn]:
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token,
    )
    annotation = pipeline(str(audio_path))
    labels: dict[str, str] = {}
    turns: list[SpeakerTurn] = []
    for turn, _, raw_label in annotation.itertracks(yield_label=True):
        label = labels.setdefault(raw_label, f"Speaker {len(labels) + 1}")
        turns.append(SpeakerTurn(float(turn.start), float(turn.end), label))
    return turns


def transcript_payload(
    source: Path,
    language: str | None,
    task: str,
    segments: list[TranscriptSegment],
) -> dict:
    return {
        "source": source.name,
        "language": language,
        "task": task,
        "segments": [asdict(segment) for segment in segments],
    }


def render_text(segments: Iterable[TranscriptSegment]) -> str:
    return "\n".join(
        f"[{format_timestamp(segment.start)} - {format_timestamp(segment.end)}] "
        f"{segment.speaker}: {segment.text}"
        for segment in segments
    ) + "\n"


def render_html(title: str, segments: Iterable[TranscriptSegment]) -> str:
    rows = "\n".join(
        "<article class=\"turn\">"
        f"<p class=\"meta\">{html.escape(segment.speaker)} "
        f"<time>{format_timestamp(segment.start)}</time></p>"
        f"<p>{html.escape(segment.text)}</p>"
        "</article>"
        for segment in segments
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
    body {{ max-width: 760px; margin: auto; padding: 32px 20px 72px; line-height: 1.55; }}
    h1 {{ font-size: clamp(2rem, 6vw, 3.5rem); line-height: 1; }}
    .turn {{ padding: 18px 0; border-top: 1px solid color-mix(in srgb, currentColor 20%, transparent); }}
    .meta {{ display: flex; justify-content: space-between; gap: 16px; font-size: .82rem; font-weight: 700; }}
    .turn > p:last-child {{ margin-bottom: 0; }}
  </style>
</head>
<body>
  <main>
    <h1>{html.escape(title)}</h1>
    {rows}
  </main>
</body>
</html>
"""


def write_outputs(
    source: Path,
    out_dir: Path,
    language: str | None,
    task: str,
    segments: list[TranscriptSegment],
) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = source.stem
    text_path = out_dir / f"{stem}.transcript.txt"
    json_path = out_dir / f"{stem}.transcript.json"
    html_path = out_dir / f"{stem}.transcript.html"
    text_path.write_text(render_text(segments), encoding="utf-8")
    json_path.write_text(
        json.dumps(transcript_payload(source, language, task, segments), indent=2),
        encoding="utf-8",
    )
    html_path.write_text(
        render_html(f"Transcript: {source.name}", segments),
        encoding="utf-8",
    )
    return text_path, json_path, html_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="Audio or video file to process")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model", default="small")
    parser.add_argument("--task", choices=("transcribe", "translate"), default="transcribe")
    parser.add_argument("--language", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compute-type", default="auto")
    parser.add_argument("--diarize", action="store_true")
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source = args.audio.expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"Input file not found: {source}")
    if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise SystemExit(f"Unsupported input extension. Expected one of: {allowed}")

    segments, detected_language = transcribe(
        source,
        model_name=args.model,
        task=args.task,
        language=args.language,
        device=args.device,
        compute_type=args.compute_type,
    )

    if args.diarize:
        token = os.environ.get(args.hf_token_env)
        if not token:
            raise SystemExit(
                f"Diarization needs a Hugging Face token in {args.hf_token_env}."
            )
        segments = attach_speakers(segments, diarize(source, token))

    paths = write_outputs(
        source,
        args.out_dir.expanduser().resolve(),
        detected_language,
        args.task,
        segments,
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

