---
title: YouTube Shorts Voice Mixer
emoji: 🎬
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "5.33.0"
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
---

# 🎬 YouTube Shorts Voice Mixer — API

AI voice generation using **Kokoro-82M** (hexgrad) with audio effects pipeline.

**API-only** — designed for n8n / HTTP integration. Gradio UI available at `/ui`.

## API Endpoint

```
POST https://<your-space>.hf.space/synthesize
Content-Type: application/json

{
  "text": "[Shock] Breaking news! [Fast] Here is the story.",
  "seed": 42,
  "speed": 1.2,
  "voice": ""
}
```

## Style Tags

| Tag | Default Voice | Style |
|---|---|---|
| `[Shock]` | `am_adam` | High energy, shock |
| `[Fast]` | `am_michael` | Rapid-fire narrator |
| `[Deep]` | `am_fenrir` | Slow, dramatic |
| `[Sarcastic]` | `am_eric` | Dry humor |
| `[Question]` | `am_liam` | Rising intonation |

## Available Voices

**American Male:** `am_adam`, `am_michael`, `am_echo`, `am_eric`, `am_fenrir`, `am_liam`, `am_onyx`  
**American Female:** `af_heart`, `af_bella`, `af_nicole`, `af_sarah`, `af_sky`, `af_alloy`, `af_nova`  
**British Male:** `bm_george`, `bm_lewis`  
**British Female:** `bf_emma`, `bf_isabella`
