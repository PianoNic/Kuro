"""Dynamic transcript context manager.

Keeps the transcript buffer lean and relevant by:
1. Trimming old context after Kuro responds (conversation is "done")
2. Clearing stale context after silence gaps
3. Removing filler/noise lines that add no value
"""

import asyncio
import time
import re
from collections import deque

import config

# Lines matching these patterns are pure filler and can be dropped during cleanup
FILLER_PATTERNS = re.compile(
    r"^[^:]+:\s*("
    r"mhm|mh|hm+|ja\.?|ok(ay)?\.?|ah\.?|oh\.?|ähm?|tschüss\.?|"
    r"bye\.?|silent\.?|\.+|wow\.?"
    r")$",
    re.IGNORECASE,
)

# How many lines to keep after Kuro responds (the recent exchange)
KEEP_AFTER_RESPONSE = 10

# Seconds of no new transcription before we consider the conversation "stale"
SILENCE_TIMEOUT = 60.0

# Minimum lines to keep even during cleanup (so context isn't empty)
MIN_CONTEXT_LINES = 3


class ContextManager:
    """Manages dynamic cleanup of the transcript deque."""

    def __init__(self, transcript: deque):
        self.transcript = transcript
        self.last_transcription_time = time.time()
        self._cleanup_task: asyncio.Task | None = None

    def on_new_transcription(self):
        """Called when a new transcription arrives."""
        self.last_transcription_time = time.time()

    def on_kuro_responded(self):
        """Called after Kuro generates and queues a response.

        Trims old context: keeps only the last few lines around the response,
        so the next LLM call sees fresh context instead of stale history.
        """
        lines = list(self.transcript)

        if len(lines) <= KEEP_AFTER_RESPONSE:
            return  # Nothing to trim

        # Find the last Kuro response
        last_kuro_idx = None
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith("Kuro: "):
                last_kuro_idx = i
                break

        if last_kuro_idx is None:
            return

        # Keep: a few lines before Kuro's response + Kuro's response + anything after
        start = max(0, last_kuro_idx - 2)
        kept = lines[start:]

        # Also remove filler from the kept lines (except Kuro's own lines)
        cleaned = [
            line for line in kept
            if line.startswith("Kuro: ") or not FILLER_PATTERNS.match(line)
        ]

        # Ensure minimum context
        if len(cleaned) < MIN_CONTEXT_LINES:
            cleaned = kept[-MIN_CONTEXT_LINES:]

        # Replace transcript contents
        self.transcript.clear()
        self.transcript.extend(cleaned)

        trimmed = len(lines) - len(cleaned)
        if trimmed > 0:
            print(f"[Context] Trimmed {trimmed} stale lines, kept {len(cleaned)}")

    def start(self, loop: asyncio.AbstractEventLoop):
        """Start the background silence detector."""
        self._cleanup_task = loop.create_task(self._silence_detector())

    def stop(self):
        """Stop the background task."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()

    async def _silence_detector(self):
        """Background task: clear old context after prolonged silence."""
        while True:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds

                if not self.transcript:
                    continue

                elapsed = time.time() - self.last_transcription_time

                if elapsed > SILENCE_TIMEOUT:
                    count = len(self.transcript)
                    if count > 0:
                        self.transcript.clear()
                        print(f"[Context] Cleared {count} lines after {elapsed:.0f}s silence")

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[Context] Error: {e}")
