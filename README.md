# hermes-plugin-resilient-stt

A [Hermes Agent](https://hermes-agent.nousresearch.com/) plugin that adds a cost-ordered
speech-to-text (STT) failover ladder: **xAI Grok → Groq (whisper-large-v3) → OpenAI
(gpt-4o-transcribe) → DeepInfra (Voxtral) → Mistral (Voxtral)**. First success wins; a down
or unavailable rung is skipped transparently instead of failing the whole transcription.

## Why this exists

Hermes's native STT dispatch has **no runtime fallback chain** — an explicit `stt.provider`
is honored with no retry if it's unavailable (unlike the main chat model's
`fallback_providers`). This plugin closes that gap using Hermes's `TranscriptionProvider`
plugin ABC (a real extension point, distinct from the 6 built-in STT backends), registering
a single provider that tries each backend in order internally.

## Ladder order (cost-first)

1. **xAI Grok** — free via an existing xAI OAuth subscription entitlement (no separate
   `XAI_API_KEY` required, though one is honored if set).
2. **Groq**, pinned to `whisper-large-v3` — deliberately *not* the weaker default
   `whisper-large-v3-turbo`, which mislabels some non-English speech.
3. **OpenAI**, `gpt-4o-transcribe` — needs a genuine OpenAI **Platform** API key
   (`OPENAI_API_KEY`); a ChatGPT/Codex subscription does not grant this.
4. **DeepInfra**, Voxtral.
5. **Mistral**, `voxtral-small-latest` — added 2026-09-14, appended at the end without a
   cost/quality comparison against the other paid rungs; genuinely just "last resort" by
   default, not a considered ranking.

Reorder `_LADDER` in `provider.py` if your own cost/quality priorities differ.

## Language handling

All 5 rungs auto-detect rather than force a language. Only xAI's underlying handler
accepts a language hint at all; forcing it can suppress detection of a second language a
bilingual speaker uses (found the hard way — forcing English/Ukrainian broke detection of
whichever wasn't forced). A caller-supplied language hint is still forwarded to the xAI rung
specifically; every other rung ignores it.

## Install

```
hermes plugins install sergattic/hermes-plugin-resilient-stt
```

Then set in `config.yaml`:

```yaml
stt:
  provider: resilient-stt
```

Configure whichever of `XAI_API_KEY` / `GROQ_API_KEY` / `OPENAI_API_KEY` /
`DEEPINFRA_API_KEY` / `MISTRAL_API_KEY` you have — a rung with no credentials is skipped,
not an error, as long as at least one rung further down the ladder has one.

## Design note: reuses Hermes's own internal provider code

This plugin deliberately calls Hermes's own internal
`tools.transcription_tools._transcribe_xai` / `_transcribe_groq` / `_transcribe_openai` /
`_transcribe_deepinfra` / `_transcribe_mistral` functions rather than reimplementing each
provider's HTTP/auth/retry
logic from scratch — including xAI's OAuth-credential-refresh-and-retry-once behavior on a
401/403, which would be a real correctness risk to duplicate imperfectly.

**Trade-off, disclosed rather than hidden:** these are underscore-prefixed "private" Hermes
internals, not a stable public plugin API. A future Hermes release could change their
signature without notice. If `hermes plugins doctor resilient-stt` or a live transcription
starts failing after a Hermes upgrade, check this before assuming your API keys are wrong.

## License

MIT
