"""Resilient STT provider — cost-ordered failover ladder.

Registers a single TranscriptionProvider ("resilient-stt") that tries each
rung in order and returns on the first success, catching both raised
exceptions and {"success": False} envelopes and advancing to the next rung.
This gives transparent per-call failover (Hermes's native STT dispatch has
none) without the agent having to react to a failed transcription itself.

Deliberately reuses Hermes's own internal provider implementations
(tools.transcription_tools._transcribe_*) rather than reimplementing each
provider's HTTP/auth/retry logic — see plugin.yaml for the full rationale
and the fragility trade-off this accepts.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from agent.transcription_provider import TranscriptionProvider

logger = logging.getLogger(__name__)

# Cost-ordered per owner decision 2026-09-13: free-via-subscription rungs
# first, paid rungs held in reserve. Each entry: (rung_id, model, env_var,
# provider_id_for_key_resolution, display_label).
_LADDER = [
    ("xai", None, "XAI_API_KEY", "xai", "xAI Grok (free via OAuth sub)"),
    ("groq", "whisper-large-v3", "GROQ_API_KEY", "groq", "Groq (whisper-large-v3, NOT turbo)"),
    ("openai", "gpt-4o-transcribe", "OPENAI_API_KEY", "openai", "OpenAI gpt-4o-transcribe"),
    ("deepinfra", None, "DEEPINFRA_API_KEY", "deepinfra", "DeepInfra Voxtral"),
]


def _rung_available(env_var: str, provider_id: str) -> bool:
    """Rung-specific availability check.

    NOT a uniform `_resolve_provider_key` check for every rung — xAI and
    OpenAI each have a non-static-key credential path that a naive env-var
    check misses entirely:

    - xAI: velis's real, working, FREE path is the xai-oauth subscription
      credential (`tools.xai_http.resolve_xai_http_credentials()`), not a
      standalone XAI_API_KEY — matches Hermes's own auto-detect logic
      (transcription_tools.py ~line 1118). A plain env-var check here
      would report xAI unavailable even when it's the one rung actually
      free-and-working, which defeats the entire cost-first ordering.
    - OpenAI: mirrors `_resolve_openai_audio_client_config()`'s own real
      resolution (direct key OR Nous-managed-gateway OR local override),
      not just a bare OPENAI_API_KEY.
    """
    try:
        if provider_id == "xai":
            from tools.xai_http import resolve_xai_http_credentials
            direct_key = ""
            try:
                from tools.transcription_tools import _resolve_provider_key
                direct_key = _resolve_provider_key(env_var, provider_id) or ""
            except Exception:  # noqa: BLE001
                pass
            if direct_key:
                return True
            return bool(resolve_xai_http_credentials().get("api_key"))
        if provider_id == "openai":
            from tools.transcription_tools import _resolve_openai_audio_client_config
            try:
                _resolve_openai_audio_client_config()
                return True
            except ValueError:
                return False
        from tools.transcription_tools import _resolve_provider_key
        return bool(_resolve_provider_key(env_var, provider_id))
    except Exception:  # noqa: BLE001 — availability check must never raise
        return False


class ResilientSTTProvider(TranscriptionProvider):
    @property
    def name(self) -> str:
        return "resilient-stt"

    @property
    def display_name(self) -> str:
        return "Resilient STT (xAI -> Groq -> OpenAI -> DeepInfra)"

    def is_available(self) -> bool:
        return any(_rung_available(env_var, pid) for _, _, env_var, pid, _ in _LADDER)

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {"id": rung_id, "display": label}
            for rung_id, _model, _env, _pid, label in _LADDER
        ]

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Resilient STT",
            "badge": "ladder",
            "tag": "xAI -> Groq -> OpenAI -> DeepInfra, first success wins",
            "env_vars": [
                {"key": "XAI_API_KEY", "prompt": "xAI API key (optional — OAuth sub also works)"},
                {"key": "GROQ_API_KEY", "prompt": "Groq API key", "url": "https://console.groq.com/keys"},
                {"key": "OPENAI_API_KEY", "prompt": "OpenAI API key", "url": "https://platform.openai.com/api-keys"},
                {"key": "DEEPINFRA_API_KEY", "prompt": "DeepInfra API key", "url": "https://deepinfra.com/dash/api_keys"},
            ],
        }

    def transcribe(
        self,
        file_path: str,
        *,
        model: Optional[str] = None,
        language: Optional[str] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        from tools.transcription_tools import (
            _transcribe_xai,
            _transcribe_groq,
            _transcribe_openai,
            _transcribe_deepinfra,
        )

        prompt = extra.get("prompt")
        errors: List[str] = []

        for rung_id, default_model, env_var, provider_id, label in _LADDER:
            if not _rung_available(env_var, provider_id):
                logger.debug("resilient-stt: skipping %s (no credentials)", rung_id)
                continue

            # Deliberately ignore the caller-supplied `model` param: it's
            # meaningful for a single-backend provider (pick a model from
            # list_models()), but this provider's list_models() returns rung
            # IDs, not real model names — each rung always uses its own
            # correct default (e.g. Groq gets whisper-large-v3, never the
            # weaker turbo default) rather than a value meant for whatever
            # rung actually ends up serving the call.
            try:
                if rung_id == "xai":
                    # xAI is the one rung that honors a language hint.
                    result = _transcribe_xai(file_path, default_model or "", language=language, prompt=prompt)
                elif rung_id == "groq":
                    result = _transcribe_groq(file_path, default_model, language=None, prompt=prompt)
                elif rung_id == "openai":
                    result = _transcribe_openai(file_path, default_model, language=None, prompt=prompt)
                elif rung_id == "deepinfra":
                    result = _transcribe_deepinfra(file_path, default_model or "", language=None, prompt=prompt)
                else:  # pragma: no cover — defensive, ladder table is static
                    continue
            except Exception as exc:  # noqa: BLE001 — a rung must never take the whole ladder down
                logger.warning("resilient-stt: rung '%s' raised: %s", rung_id, exc, exc_info=True)
                errors.append(f"{rung_id}: {exc}")
                continue

            if isinstance(result, dict) and result.get("success"):
                result.setdefault("provider", "resilient-stt")
                result["rung"] = rung_id
                logger.info("resilient-stt: succeeded via '%s' rung", rung_id)
                return result

            err = (result or {}).get("error", "unknown error") if isinstance(result, dict) else "non-dict result"
            logger.info("resilient-stt: rung '%s' failed (%s), advancing", rung_id, err)
            errors.append(f"{rung_id}: {err}")

        return {
            "success": False,
            "transcript": "",
            "provider": "resilient-stt",
            "error": "All STT rungs failed or unavailable: " + "; ".join(errors) if errors
            else "No STT rung has credentials configured (XAI_API_KEY/GROQ_API_KEY/OPENAI_API_KEY/DEEPINFRA_API_KEY all unset).",
        }
