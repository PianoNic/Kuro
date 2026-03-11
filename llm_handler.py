import re
import httpx
from openai import AsyncOpenAI
import config

SYSTEM_PROMPT = """You are Kuro, a chill and friendly girl who hangs out in Discord voice chats. \
You are part of the conversation naturally — like a real friend in the call.

Personality:
- Playful, witty, and warm
- Uses casual speech, like a real person in a Discord call
- Knowledgeable and helpful when asked questions
- Has a good sense of humor and can be sarcastic
- Speaks naturally in whatever language the users are speaking

You receive a transcript of recent voice chat. This is a casual Discord call — people joke around, \
use slang, and say dumb stuff. Don't be easily offended. Respond like a real friend would.

Rules:
- Keep responses SHORT (1-3 sentences max) — you're in a voice chat
- Match the language of the conversation (German if they speak German, etc.)
- Respond with ONLY your spoken words — no asterisks, no emotes, no stage directions
- Never repeat what someone just said
- Never repeat your own previous responses — always say something NEW
- Never narrate your own actions
- Do NOT get stuck on one topic — follow the conversation naturally
- If you don't know something, say so honestly — never make up facts, names, or titles
- NEVER use emojis — your words will be spoken aloud via TTS"""

GATE_PROMPT = """You are a conversation analyst for "Kuro", an AI participant in a Discord voice chat.
Your ONLY job: decide if Kuro should respond to the latest message(s) in the transcript.

Kuro SHOULD respond (say RESPOND) ONLY when:
- Someone mentions "Kuro" by name
- Someone directly asks Kuro a question
- Someone is replying to or correcting something Kuro just said
- Someone asks a question to the group AND Kuro has something genuinely useful or funny to add
- There's a clear opening where Kuro joining in would feel natural (not forced)

Kuro should stay SILENT when:
- People are just chatting among themselves — don't butt in
- It's filler or reactions ("mhm", "ja", "ok", "hä?", "was?")
- Kuro already spoke recently AND nobody is talking to her — give others space
- Someone is telling a story or explaining something — wait
- The conversation doesn't need another voice — when in doubt, SILENT
- People are responding to each other, not to Kuro

IMPORTANT: Kuro is a chill friend, NOT an eager chatbot. Real friends don't comment on every single thing.
Default to SILENT unless there's a clear reason to speak.

Reply with ONLY one word: RESPOND or SILENT"""


class LLMHandler:
    def __init__(self):
        self._is_ollama = bool(config.LLM_BASE_URL)
        if self._is_ollama:
            # Use native Ollama API (supports think: false)
            # Extract base URL without /v1 suffix
            base = config.LLM_BASE_URL.rstrip("/")
            if base.endswith("/v1"):
                base = base[:-3]
            self._ollama_url = base + "/api/chat"
            self._http = httpx.AsyncClient(timeout=60.0)
        else:
            self.client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=config.OPENROUTER_API_KEY,
                default_headers={"X-Reasoning-Disabled": "true"},
            )

    async def _ollama_chat(self, messages: list, max_tokens: int, temperature: float) -> str:
        """Call Ollama native API with think: false for instant responses."""
        payload = {
            "model": config.LLM_MODEL,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "repeat_penalty": 1.3,
                "repeat_last_n": 256,
            },
        }
        resp = await self._http.post(self._ollama_url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")

    async def _openai_chat(self, messages: list, max_tokens: int, temperature: float) -> str:
        """Call OpenRouter via OpenAI-compatible API."""
        response = await self.client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    async def _chat(self, messages: list, max_tokens: int, temperature: float) -> str:
        if self._is_ollama:
            return await self._ollama_chat(messages, max_tokens, temperature)
        return await self._openai_chat(messages, max_tokens, temperature)

    async def should_respond(self, transcript_context: str) -> bool:
        """Fast gate check: should Kuro respond to this transcript?"""
        messages = [
            {"role": "system", "content": GATE_PROMPT},
            {
                "role": "user",
                "content": f"Transcript:\n\n{transcript_context}\n\nShould Kuro respond? Reply RESPOND or SILENT.",
            },
        ]

        try:
            raw = await self._chat(messages, max_tokens=10, temperature=0.3)
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            decision = raw.upper().strip()
            print(f"[Gate] {decision}")
            return "RESPOND" in decision

        except Exception as e:
            print(f"[Gate] Error: {e}")
            return False

    async def get_response(self, transcript_lines: list[str]) -> str | None:
        """Send transcript to LLM as proper conversation turns.

        transcript_lines: list of strings like "Username: text" or "Kuro: text"
        Kuro's lines become assistant messages, everything else becomes user messages.
        This prevents the model from repeating its own previous responses.
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Group consecutive non-Kuro lines into user messages,
        # and Kuro lines into assistant messages
        user_buffer = []
        for line in transcript_lines:
            if line.startswith("Kuro: "):
                # Flush any buffered user lines first
                if user_buffer:
                    messages.append({"role": "user", "content": "\n".join(user_buffer)})
                    user_buffer = []
                # Add Kuro's line as assistant message
                messages.append({"role": "assistant", "content": line[6:]})  # strip "Kuro: "
            else:
                user_buffer.append(line)

        # Flush remaining user lines + prompt
        if user_buffer:
            user_buffer.append("\nRespond as Kuro. Say something NEW and relevant to what was just said.")
            messages.append({"role": "user", "content": "\n".join(user_buffer)})
        else:
            messages.append({"role": "user", "content": "Respond as Kuro. Say something NEW and relevant."})

        try:
            raw = await self._chat(messages, max_tokens=150, temperature=0.8)
            print(f"[LLM] Raw: {raw[:200]}")
            reply = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

            if "[SILENT]" in reply or not reply:
                print("[LLM] -> Silent")
                return None

            print(f"[LLM] -> {reply[:120]}")
            return reply

        except Exception as e:
            print(f"[LLM] Error: {e}")
            return None
