import json
import tempfile
import unittest
from pathlib import Path

from local_speech_intelligence import (
    SpeakerTurn,
    TranscriptSegment,
    attach_speakers,
    format_timestamp,
    write_outputs,
)


class PipelineTests(unittest.TestCase):
    def test_timestamp_format(self):
        self.assertEqual(format_timestamp(62.345), "00:01:02.345")

    def test_speaker_assignment_uses_temporal_overlap(self):
        segments = [TranscriptSegment(0.0, 2.0, "Hello")]
        turns = [SpeakerTurn(0.5, 2.2, "Speaker 1")]
        self.assertEqual(attach_speakers(segments, turns)[0].speaker, "Speaker 1")

    def test_outputs_escape_html_and_preserve_json(self):
        segments = [TranscriptSegment(0.0, 1.0, "Use <local> inference", "Speaker 1")]
        with tempfile.TemporaryDirectory() as directory:
            paths = write_outputs(
                Path("synthetic.wav"), Path(directory), "en", "transcribe", segments
            )
            self.assertEqual(len(paths), 3)
            payload = json.loads(paths[1].read_text(encoding="utf-8"))
            self.assertEqual(payload["segments"][0]["text"], "Use <local> inference")
            markup = paths[2].read_text(encoding="utf-8")
            self.assertIn("Use &lt;local&gt; inference", markup)


if __name__ == "__main__":
    unittest.main()
