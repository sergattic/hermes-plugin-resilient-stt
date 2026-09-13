"""Resilient STT plugin — cost-ordered failover across xAI/Groq/OpenAI/DeepInfra.

Registers a single wrapper provider ("resilient-stt") that tries each backend
in a fixed ladder, catching both raised exceptions and {"success": False}
envelopes and advancing to the next rung. Closes a real gap: Hermes's native
STT dispatch has no runtime fallback (unlike the main chat model).
"""

from __future__ import annotations

from .provider import ResilientSTTProvider


def register(ctx) -> None:
    ctx.register_transcription_provider(ResilientSTTProvider())
