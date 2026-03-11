import os
import tempfile

import edge_tts

import config


async def generate_speech(text: str) -> str:
    """Generate speech audio from text using edge-tts.

    Returns the path to a temporary .mp3 file.
    Caller is responsible for deleting the file after use.
    """
    temp_file = tempfile.NamedTemporaryFile(
        suffix=".mp3", delete=False, dir="temp_audio"
    )
    temp_path = temp_file.name
    temp_file.close()

    communicate = edge_tts.Communicate(text, config.TTS_VOICE, rate=config.TTS_RATE)
    await communicate.save(temp_path)

    return temp_path
