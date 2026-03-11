import os
from dotenv import load_dotenv

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "anthropic/claude-haiku-4-5")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")  # Set to "http://localhost:11434/v1" for Ollama
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "distil-large-v3")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", None)  # e.g. "de", "en", None for auto-detect
TTS_VOICE = os.getenv("TTS_VOICE", "de-CH-LeniNeural")
TTS_RATE = os.getenv("TTS_RATE", "+20%")  # Speech speed: "+0%", "+20%", "+40%", etc.
RESPONSE_COOLDOWN = int(os.getenv("RESPONSE_COOLDOWN", "5"))
WAKE_WORD = os.getenv("WAKE_WORD", "kuro").lower()
# Common Whisper mistranscriptions of "Kuro"
WAKE_WORD_ALIASES = [
    w.strip().lower()
    for w in os.getenv(
        "WAKE_WORD_ALIASES",
        "kuro,kudo,uro,furo,curo,kouro,kuru,kro,k-pro,"
        "churro,boto,bodo,foto,boro,buro,puro,turo,guro,"
        "maureen,maurin,mauleen,marine",
    ).split(",")
]
MAX_TRANSCRIPT_LINES = int(os.getenv("MAX_TRANSCRIPT_LINES", "30"))
DEBOUNCE_DELAY = float(os.getenv("DEBOUNCE_DELAY", "3.0"))  # seconds to wait for more speech before calling LLM (non-wake-word)
VAD_SILENCE_MS = int(os.getenv("VAD_SILENCE_MS", "300"))  # ms of silence before speech is considered ended
