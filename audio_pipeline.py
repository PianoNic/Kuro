import os
import queue
import threading
import asyncio

# Add NVIDIA pip-installed DLL paths so faster-whisper/CTranslate2 can find cuBLAS on Windows
try:
    import nvidia.cublas
    _cublas_dir = os.path.join(os.path.dirname(nvidia.cublas.__path__[0]), "cublas", "bin")
    if os.path.isdir(_cublas_dir):
        os.add_dll_directory(_cublas_dir)
        os.environ["PATH"] = _cublas_dir + os.pathsep + os.environ.get("PATH", "")
except (ImportError, OSError):
    pass

import numpy as np
import torch
from discord.ext import voice_recv
from silero_vad import load_silero_vad, VADIterator
from faster_whisper import WhisperModel
from faster_whisper.utils import download_model

import config


class StreamSink(voice_recv.AudioSink):
    """AudioSink that pushes decoded PCM packets to a queue in real-time."""

    def __init__(self):
        super().__init__()
        self.audio_queue = queue.Queue()
        self.finished = False

    def wants_opus(self) -> bool:
        return False  # We want decoded PCM

    def write(self, user, data):
        if data and data.pcm:
            user_id = user.id if user else 0
            self.audio_queue.put((user_id, data.pcm))

    def cleanup(self):
        self.finished = True


class AudioPipeline:
    """Processes audio from StreamSink: VAD -> Whisper -> callback."""

    def __init__(self, on_transcription):
        """
        on_transcription: async callback(user_id: int, text: str, language: str)
        """
        print(f"[AudioPipeline] Downloading Whisper model '{config.WHISPER_MODEL}'...")
        print("[AudioPipeline] (this only happens once, the model will be cached)")
        model_path = download_model(config.WHISPER_MODEL)
        print("[AudioPipeline] Download complete. Loading model onto GPU...")
        self.whisper = WhisperModel(
            model_path,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
            local_files_only=True,
        )
        print("[AudioPipeline] Whisper model loaded.")

        print("[AudioPipeline] Loading VAD model...")
        self.vad_model = load_silero_vad()
        print("[AudioPipeline] VAD model loaded.")

        self.on_transcription = on_transcription
        self.running = False
        self._thread = None
        self._loop = None
        self._user_states = {}

    def _get_user_state(self, user_id):
        if user_id not in self._user_states:
            self._user_states[user_id] = {
                "vad": VADIterator(
                    self.vad_model,
                    sampling_rate=16000,
                    min_silence_duration_ms=config.VAD_SILENCE_MS,
                    speech_pad_ms=100,
                ),
                "vad_buffer": np.array([], dtype=np.float32),
                "speech_buffer": np.array([], dtype=np.float32),
                "is_speaking": False,
            }
        return self._user_states[user_id]

    def start(self, sink: StreamSink, loop: asyncio.AbstractEventLoop):
        self.running = True
        self._loop = loop
        self._thread = threading.Thread(
            target=self._process_loop, args=(sink,), daemon=True
        )
        self._thread.start()
        print("[AudioPipeline] Processing started.")

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=3)
        self._user_states.clear()
        print("[AudioPipeline] Processing stopped.")

    def _process_loop(self, sink: StreamSink):
        while self.running and not sink.finished:
            try:
                user_id, data = sink.audio_queue.get(timeout=0.5)
                self._process_chunk(user_id, data)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[AudioPipeline] Error processing audio: {e}")

    def _process_chunk(self, user_id, data: bytes):
        # discord-ext-voice-recv gives 48kHz, 16-bit PCM, stereo
        audio_int16 = np.frombuffer(data, dtype=np.int16)

        # Stereo -> mono (average L+R channels)
        if len(audio_int16) >= 2:
            audio_int16 = audio_int16.reshape(-1, 2).mean(axis=1).astype(np.int16)

        # Downsample 48kHz -> 16kHz (take every 3rd sample)
        audio_int16 = audio_int16[::3]

        # Convert to float32 normalized [-1, 1]
        audio_float = audio_int16.astype(np.float32) / 32768.0

        state = self._get_user_state(user_id)

        # Accumulate in VAD buffer
        state["vad_buffer"] = np.concatenate([state["vad_buffer"], audio_float])

        # Process in 512-sample chunks (32ms at 16kHz)
        vad_chunk_size = 512
        while len(state["vad_buffer"]) >= vad_chunk_size:
            chunk = state["vad_buffer"][:vad_chunk_size]
            state["vad_buffer"] = state["vad_buffer"][vad_chunk_size:]

            chunk_tensor = torch.from_numpy(chunk)
            speech_dict = state["vad"](chunk_tensor, return_seconds=True)

            # Start of speech
            if speech_dict and "start" in speech_dict:
                state["is_speaking"] = True
                state["speech_buffer"] = np.array([], dtype=np.float32)

            # Accumulate speech audio
            if state["is_speaking"]:
                state["speech_buffer"] = np.concatenate(
                    [state["speech_buffer"], chunk]
                )

            # End of speech -> transcribe
            if speech_dict and "end" in speech_dict:
                state["is_speaking"] = False
                if len(state["speech_buffer"]) > 1600:  # >0.1s of audio
                    self._transcribe(user_id, state["speech_buffer"])
                state["speech_buffer"] = np.array([], dtype=np.float32)

            # Safety: force transcribe if speech buffer > 30 seconds
            if state["is_speaking"] and len(state["speech_buffer"]) > 16000 * 30:
                self._transcribe(user_id, state["speech_buffer"])
                state["speech_buffer"] = np.array([], dtype=np.float32)
                state["is_speaking"] = False
                state["vad"].reset_states()

    # Common Whisper hallucinations (phantom subtitles generated from silence/noise)
    HALLUCINATION_PATTERNS = {
        "untertitelung", "untertitel", "copyright", "vielen dank",
        "thank you", "thanks for watching", "subtitles by",
        "amara.org", "sdf", "zdf", "ard", "br 20",
    }

    def _is_hallucination(self, text: str) -> bool:
        text_lower = text.lower().strip()
        return any(pattern in text_lower for pattern in self.HALLUCINATION_PATTERNS)

    def _transcribe(self, user_id, audio: np.ndarray):
        try:
            segments, info = self.whisper.transcribe(
                audio, beam_size=3, language=config.WHISPER_LANGUAGE
            )
            text = " ".join(seg.text for seg in segments).strip()

            if text and len(text) > 1 and not self._is_hallucination(text):
                print(f"[Transcription] User {user_id} ({info.language}): {text}")
                asyncio.run_coroutine_threadsafe(
                    self.on_transcription(user_id, text, info.language),
                    self._loop,
                )
        except Exception as e:
            print(f"[AudioPipeline] Transcription error: {e}")
