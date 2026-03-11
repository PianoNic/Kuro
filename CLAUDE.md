# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kuro is a Discord voice bot that listens to voice chat, transcribes speech in real-time using Whisper, decides whether to respond via an LLM (OpenRouter API), and speaks back using edge-tts. It acts as a conversational participant in Discord voice channels.

## Running the Bot

```bash
# Install dependencies (requires Python 3.10+, CUDA-capable GPU recommended)
pip install -r requirements.txt

# Copy .env.example to .env and fill in DISCORD_TOKEN and OPENROUTER_API_KEY
python bot.py
```

Requires FFmpeg on PATH for audio playback. First run downloads the Whisper model (~1.5GB).

## Architecture

The bot follows a linear voice pipeline: **Discord audio → VAD → Whisper → LLM → TTS → Discord playback**.

- **`bot.py`** — Entry point. Discord bot commands (`.join`, `.leave`, `.say`, `.clear_transcript`), manages the transcript buffer (deque), orchestrates the pipeline, handles TTS playback via FFmpegPCMAudio.
- **`audio_pipeline.py`** — `StreamSink` (receives raw PCM from Discord via `voice_recv`) and `AudioPipeline` (runs in a background thread: per-user VAD via Silero → speech segmentation → Whisper transcription → async callback to bot).
- **`llm_handler.py`** — Sends transcript context to OpenRouter (OpenAI-compatible API) with Kuro's system prompt. Returns `None` (maps to `[SILENT]`) when Kuro shouldn't speak.
- **`tts_handler.py`** — Generates speech via `edge-tts`, writes temp .mp3 files to `temp_audio/`.
- **`config.py`** — All configuration from environment variables with defaults.

## Key Design Details

- Audio from Discord is 48kHz stereo PCM; `AudioPipeline._process_chunk` converts to 16kHz mono for VAD/Whisper.
- VAD and speech state are tracked **per-user** (`_user_states` dict) so overlapping speakers are handled independently.
- The pipeline thread communicates back to the async bot via `asyncio.run_coroutine_threadsafe`.
- Response cooldown (`RESPONSE_COOLDOWN`) prevents spam; the wake word overrides the cooldown.
- Temp audio files are cleaned up in the `after` callback of `vc.play()`.

## Configuration

All config is via `.env` (see `.env.example`). Key settings: `LLM_MODEL` (default: `anthropic/claude-haiku-4-5`), `WHISPER_MODEL` (default: `distil-large-v3`), `WHISPER_DEVICE` (`cuda`/`cpu`), `TTS_VOICE`, `WAKE_WORD` (default: `kuro`).
