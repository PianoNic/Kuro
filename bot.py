import discord
from discord.ext import commands, voice_recv
import asyncio
import logging
import re
import struct
import time
import os
from collections import deque

import config
from audio_pipeline import StreamSink, AudioPipeline
from llm_handler import LLMHandler
from tts_handler import generate_speech
from context_manager import ContextManager

# --- Monkey-patches for discord-ext-voice-recv bugs ---

# 1) Fix BEDE header parser: bounds check to prevent struct.error on short data
from discord.ext.voice_recv.rtp import RTPPacket

def _patched_parse_bede_header(self, data: bytes, length: int) -> None:
    offset = 4
    end = 4 + length * 4
    n = 0
    while n < length and offset < end:
        if offset >= len(data):
            break
        next_byte = data[offset : offset + 1]
        if not next_byte:
            break
        if next_byte == b'\x00':
            offset += 1
            continue
        header = struct.unpack('>B', next_byte)[0]
        element_id = header >> 4
        element_len = 1 + (header & 0b0000_1111)
        self.extension_data[element_id] = data[offset + 1 : offset + 1 + element_len]
        offset += 1 + element_len
        n += 1

RTPPacket._parse_bede_header = _patched_parse_bede_header

# 2) PacketRouter._do_run: catch OpusError so one corrupt packet doesn't kill the thread
from discord.ext.voice_recv.router import PacketRouter

def _patched_do_run(self):
    while not self._end_thread.is_set():
        self.waiter.wait()
        with self._lock:
            for decoder in self.waiter.items:
                try:
                    data = decoder.pop_data()
                except discord.opus.OpusError:
                    decoder.reset()
                    continue
                if data is not None:
                    self.sink.write(data.source, data)

PacketRouter._do_run = _patched_do_run

# Suppress noisy voice_recv logs (unknown ssrc, packet loss, extra WS keys)
logging.getLogger("discord.ext.voice_recv").setLevel(logging.ERROR)

# Ensure temp_audio directory exists
os.makedirs("temp_audio", exist_ok=True)

intents = discord.Intents.all()
bot = commands.Bot(command_prefix=".", intents=intents)

# Global state
llm = LLMHandler()
pipeline = None
transcript = deque(maxlen=config.MAX_TRANSCRIPT_LINES)
ctx_manager = ContextManager(transcript)
last_response_time = 0
_debounce_task: asyncio.Task | None = None
_pending_wake_word = False
_llm_busy = False  # Prevent concurrent LLM calls

# Pre-compile alias pattern for replacing wake word variants with "Kuro"
_alias_pattern = re.compile(
    "|".join(re.escape(a) for a in config.WAKE_WORD_ALIASES if a != config.WAKE_WORD),
    re.IGNORECASE,
) if len(config.WAKE_WORD_ALIASES) > 1 else None


def get_username(user_id: int) -> str:
    """Resolve user ID to display name."""
    user = bot.get_user(user_id)
    return user.display_name if user else f"User-{user_id}"


async def on_transcription(user_id: int, text: str, language: str):
    """Called by AudioPipeline when speech is transcribed."""
    global _debounce_task, _pending_wake_word

    username = get_username(user_id)
    transcript.append(f"{username}: {text}")
    ctx_manager.on_new_transcription()

    text_lower = text.lower()
    wake_word_here = any(alias in text_lower for alias in config.WAKE_WORD_ALIASES)
    if wake_word_here:
        _pending_wake_word = True

    # Cancel previous debounce timer
    if _debounce_task and not _debounce_task.done():
        _debounce_task.cancel()

    # Wake word: respond immediately. Otherwise: debounce and let LLM decide.
    if wake_word_here:
        _debounce_task = asyncio.create_task(_respond_now())
    else:
        _debounce_task = asyncio.create_task(_debounced_respond())


def _is_kuro_active() -> bool:
    """Check if Kuro spoke very recently (within last 2 transcript lines).

    Only returns True for immediate follow-ups to Kuro, not general chatter.
    """
    recent = list(transcript)[-2:]
    return any(line.startswith("Kuro: ") for line in recent)


async def _respond_now():
    """Call LLM immediately (wake word detected, no debounce)."""
    await _do_respond()


async def _debounced_respond():
    """Wait briefly for more transcriptions, then call the LLM."""
    try:
        await asyncio.sleep(config.DEBOUNCE_DELAY)
    except asyncio.CancelledError:
        return
    await _do_respond()


async def _do_respond():
    """Build context and call the LLM."""
    global last_response_time, _pending_wake_word, _llm_busy

    if _llm_busy:
        return

    # Check cooldown (wake word overrides)
    now = time.time()
    if now - last_response_time < config.RESPONSE_COOLDOWN:
        if not _pending_wake_word:
            return

    wake_word_triggered = _pending_wake_word
    _pending_wake_word = False

    # Build context from transcript buffer, replacing wake word aliases with "Kuro"
    lines = list(transcript)
    if _alias_pattern:
        lines = [_alias_pattern.sub("Kuro", line) for line in lines]
    context = "\n".join(lines)

    # Stage 1: Gate check
    # Skip gate if: wake word detected, OR Kuro is actively in the conversation
    skip_gate = wake_word_triggered or _is_kuro_active()
    if skip_gate:
        reason = "wake word" if wake_word_triggered else "active conversation"
        print(f"[Gate] Skipped ({reason})")
    else:
        print(f"[Gate] Checking {len(lines)} lines...")
        should = await llm.should_respond(context)
        if not should:
            print("[Gate] -> Staying silent")
            return

    print(f"[LLM] Generating response for {len(lines)} lines of context...")

    # Stage 2: Generate actual response (pass lines for proper user/assistant turns)
    _llm_busy = True
    try:
        response = await llm.get_response(lines)
    except Exception as e:
        print(f"[LLM] Error: {e}")
        return
    finally:
        _llm_busy = False

    if not response:
        return

    last_response_time = time.time()
    transcript.append(f"Kuro: {response}")
    print(f"[Kuro] {response}")

    # Trim stale context now that Kuro has responded
    ctx_manager.on_kuro_responded()

    # Generate TTS and play
    await _speak(response)


# Pre-compiled emoji pattern
_EMOJI_RE = re.compile(
    "["
    "\U00002600-\U000027BF"  # misc symbols
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0001F000-\U0001FFFF"  # all supplemental symbols (emoticons, dingbats, etc.)
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)


def _strip_emojis(text: str) -> str:
    """Remove emojis and other non-speech characters for TTS."""
    return _EMOJI_RE.sub("", text).strip()


async def _speak(text: str):
    """Generate TTS audio and play it in the voice channel."""
    try:
        text = _strip_emojis(text)
        if not text:
            return
        audio_path = await generate_speech(text)
        print(f"[TTS] Generated: {audio_path}")
        for vc in bot.voice_clients:
            if vc.is_connected():
                # Wait for current playback to finish
                while vc.is_playing():
                    await asyncio.sleep(0.2)
                source = discord.FFmpegPCMAudio(audio_path)
                vc.play(
                    source,
                    after=lambda e, p=audio_path: _cleanup_audio(p, e),
                )
                break
    except Exception as e:
        print(f"[TTS] Error: {e}")


def _cleanup_audio(path: str, error=None):
    if error:
        print(f"[TTS] Playback error: {error}")
    try:
        os.unlink(path)
    except OSError:
        pass


@bot.event
async def on_ready():
    global pipeline
    print(f"[Bot] Kuro is online as {bot.user}")
    print(f"[Bot] Whisper model: {config.WHISPER_MODEL}")
    print(f"[Bot] LLM model: {config.LLM_MODEL}")
    print(f"[Bot] TTS voice: {config.TTS_VOICE}")

    # Preload models in a background thread so they don't block the event loop
    if pipeline is None:
        print("[Bot] Preloading models (this may take a moment)...")
        pipeline = await asyncio.to_thread(
            AudioPipeline, on_transcription=on_transcription
        )
        print("[Bot] Models ready! Use .join to enter a voice channel.")


@bot.command()
async def join(ctx):
    """Join the voice channel and start listening."""
    global pipeline

    if not ctx.author.voice:
        await ctx.send("You're not in a voice channel!")
        return

    if pipeline is None:
        await ctx.send("Still loading models, please wait...")
        return

    # Disconnect from existing voice if any
    if ctx.voice_client:
        pipeline.stop()
        await ctx.voice_client.disconnect()

    # Connect using VoiceRecvClient for audio receiving
    vc = await ctx.author.voice.channel.connect(cls=voice_recv.VoiceRecvClient)

    # Start listening with our custom sink
    sink = StreamSink()
    vc.listen(sink)

    # Start audio processing thread
    pipeline.start(sink, asyncio.get_event_loop())

    # Start context cleanup background task
    ctx_manager.start(asyncio.get_event_loop())

    await ctx.send("I'm here! Listening~ ✨")


@bot.command()
async def leave(ctx):
    """Stop listening and leave the voice channel."""
    global pipeline

    if not ctx.voice_client:
        await ctx.send("I'm not in a voice channel!")
        return

    if pipeline:
        pipeline.stop()

    ctx_manager.stop()

    try:
        ctx.voice_client.stop_listening()
    except Exception:
        pass

    await ctx.voice_client.disconnect()
    transcript.clear()

    await ctx.send("Bye bye~ 👋")


@bot.command()
async def say(ctx, *, text: str):
    """Make Kuro say something in the voice channel (for testing)."""
    if not ctx.voice_client:
        await ctx.send("I'm not in a voice channel!")
        return

    transcript.append(f"Kuro: {text}")
    await _speak(text)
    await ctx.message.add_reaction("🔊")


@bot.command()
async def clear_transcript(ctx):
    """Clear the conversation transcript buffer."""
    transcript.clear()
    await ctx.send("Transcript cleared!")


if __name__ == "__main__":
    if not config.DISCORD_TOKEN:
        print("ERROR: DISCORD_TOKEN not set in .env file!")
        print("Copy .env.example to .env and fill in your tokens.")
        exit(1)

    if not config.OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY not set in .env file!")
        exit(1)

    bot.run(config.DISCORD_TOKEN)
