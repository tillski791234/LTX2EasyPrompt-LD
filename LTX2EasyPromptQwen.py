import re
import os
import json
import random
import time as _time
import sys
import subprocess
import urllib.request
import urllib.error

# ── Node directory path — ensures lyric_phrase_bank.py is always importable ──
# ComfyUI may not add the custom node's directory to sys.path automatically.
# This guarantees imports like 'from lyric_phrase_bank import ...' always work.
_NODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _NODE_DIR not in sys.path:
    sys.path.insert(0, _NODE_DIR)

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

# ── Transformers version check ────────────────────────────────────────────────
# Qwen3.5 requires transformers >= 4.43.0 for full performance and correct
# chat template support. Older versions are significantly slower and may
# produce degraded output. Auto-upgrade if needed.
_TRANSFORMERS_MIN = (4, 43, 0)
_TRANSFORMERS_MIN_STR = "4.43.0"

def _check_and_upgrade_transformers():
    try:
        import transformers as _tf
        _ver = tuple(int(x) for x in _tf.__version__.split(".")[:3])
        if _ver >= _TRANSFORMERS_MIN:
            print(f"[LTX2-Qwen] transformers {_tf.__version__} — OK")
            return
        print(f"[LTX2-Qwen] transformers {_tf.__version__} is outdated "
              f"(need >= {_TRANSFORMERS_MIN_STR}). Upgrading...")
    except ImportError:
        print(f"[LTX2-Qwen] transformers not found. Installing...")

    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install",
             f"transformers>={_TRANSFORMERS_MIN_STR}",
             "--upgrade", "--quiet"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        # Force reload so the rest of the file gets the updated version
        if "transformers" in sys.modules:
            import importlib
            import transformers as _tf2
            importlib.reload(_tf2)
        print(f"[LTX2-Qwen] transformers upgraded successfully. "
              f"Restart ComfyUI if you encounter any issues.")
    except Exception as e:
        print(f"[LTX2-Qwen] WARNING: Could not upgrade transformers automatically: {e}. "
              f"Please run: pip install transformers>={_TRANSFORMERS_MIN_STR} --upgrade")

_check_and_upgrade_transformers()

import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer


def _resolve_model_target(gpu_id: int):
    if not torch.cuda.is_available():
        return "cpu"
    if gpu_id is None or gpu_id < 0:
        return "auto"
    cuda_count = torch.cuda.device_count()
    if gpu_id >= cuda_count:
        fallback_idx = torch.cuda.current_device()
        print(
            f"[LTX2-Qwen] Requested GPU {gpu_id} is unavailable. "
            f"Falling back to cuda:{fallback_idx}."
        )
        return f"cuda:{fallback_idx}"
    return f"cuda:{gpu_id}"


def _get_torch_dtype_for_target(target_device: str):
    if target_device == "cpu" or not torch.cuda.is_available():
        return torch.float32
    if target_device == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    try:
        gpu_index = int(str(target_device).split(":")[-1])
        major, _minor = torch.cuda.get_device_capability(gpu_index)
        if major >= 8:
            return torch.bfloat16
    except Exception as e:
        print(f"[LTX2-Qwen] Could not inspect {target_device} capability: {e}")
    return torch.float16


def _build_model_load_kwargs(target_device: str, offline_mode: bool, dtype):
    load_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "local_files_only": offline_mode,
    }
    if target_device == "cpu":
        load_kwargs["device_map"] = "cpu"
    elif target_device == "auto":
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = None
        load_kwargs["low_cpu_mem_usage"] = False
    return load_kwargs


# ── Audio analysis ────────────────────────────────────────────────────────────

def _analyse_audio(audio_dict, use_whisper: bool) -> dict:
    """
    Takes a ComfyUI AUDIO dict {waveform: Tensor[C,N], sample_rate: int}
    Returns a plain dict describing the audio for injection into the LLM prompt.
    whisper transcription only runs if use_whisper=True.
    """
    result = {
        "duration_s":    None,
        "energy_shape":  None,   # quiet / building / constant / explosive / fading
        "tempo_feel":    None,   # slow / moderate / fast / very_fast / no_beat
        "freq_character":None,   # bass_heavy / balanced / bright / vocal_dominant
        "has_speech":    False,
        "transcript":    None,
        "speech_tone":   None,   # whispered / conversational / elevated / shouting
        "silence_ratio": None,   # 0.0–1.0 fraction of clip that is near-silence
        "peak_moment":   None,   # seconds into clip where loudest moment occurs
        "summary":       None,   # plain English summary for LLM injection
    }

    try:
        import numpy as np
        waveform    = audio_dict.get("waveform")
        sample_rate = audio_dict.get("sample_rate", 44100)

        if waveform is None:
            return result

        # Tensor → numpy mono  (detach + cpu required for gradient-tracked / CUDA tensors)
        if hasattr(waveform, "detach"):
            wav = waveform.detach().cpu().numpy()
        elif hasattr(waveform, "numpy"):
            wav = waveform.numpy()
        else:
            wav = np.array(waveform)

        # Handle [batch, channels, samples] from ComfyUI VHS
        if wav.ndim == 3:
            wav = wav[0]
        if wav.ndim == 2:
            wav = wav.mean(axis=0)
        wav = wav.astype(np.float32)

        n_samples = len(wav)
        duration  = n_samples / sample_rate
        result["duration_s"] = round(duration, 2)

        # ── RMS energy over 100ms windows ──────────────────────────────
        win     = int(sample_rate * 0.1)
        hop     = win // 2
        frames  = [wav[i:i+win] for i in range(0, n_samples - win, hop)]
        rms     = np.array([np.sqrt(np.mean(f**2 + 1e-9)) for f in frames])
        rms_db  = 20 * np.log10(rms + 1e-9)

        peak_frame   = int(np.argmax(rms))
        result["peak_moment"] = round(peak_frame * hop / sample_rate, 2)

        # Silence ratio — anchored to an absolute floor so loud/constant-energy audio
        # doesn't incorrectly report ~20% silence due to the relative percentile trick.
        SILENCE_FLOOR_DB   = -50.0   # anything quieter than this is silence
        silence_ratio      = float(np.mean(rms_db < SILENCE_FLOOR_DB))
        result["silence_ratio"] = round(silence_ratio, 2)

        # Energy shape — compare first third vs last third vs peak
        third   = len(rms) // 3
        e_start = float(np.mean(rms[:third]))
        e_mid   = float(np.mean(rms[third:2*third]))
        e_end   = float(np.mean(rms[2*third:]))
        e_peak  = float(np.max(rms))
        e_mean  = float(np.mean(rms))

        if e_peak > e_mean * 2.5 and peak_frame > third:
            result["energy_shape"] = "builds to explosive peak"
        elif e_start > e_end * 1.4:
            result["energy_shape"] = "loud then fading"
        elif e_end > e_start * 1.4:
            result["energy_shape"] = "builds throughout"
        elif silence_ratio > 0.4:
            result["energy_shape"] = "sparse with long silences"
        elif np.std(rms) / (e_mean + 1e-9) < 0.3:
            result["energy_shape"] = "constant sustained energy"
        else:
            result["energy_shape"] = "varied dynamic range"

        # ── Tempo / beat detection via RMS onset strength ──────────────
        from scipy.signal import find_peaks
        onset_env = np.diff(rms, prepend=rms[0])
        onset_env = np.maximum(onset_env, 0)
        min_dist  = int(0.3 / (hop / sample_rate))   # min 300ms between beats
        peaks, _  = find_peaks(onset_env, height=np.mean(onset_env)*1.2, distance=min_dist)

        if len(peaks) > 2:
            intervals_s = np.diff(peaks) * hop / sample_rate
            avg_bpm     = 60.0 / np.median(intervals_s)
            if avg_bpm < 60:
                result["tempo_feel"] = f"slow ({int(avg_bpm)} bpm)"
            elif avg_bpm < 100:
                result["tempo_feel"] = f"moderate ({int(avg_bpm)} bpm)"
            elif avg_bpm < 140:
                result["tempo_feel"] = f"fast ({int(avg_bpm)} bpm)"
            else:
                result["tempo_feel"] = f"very fast ({int(avg_bpm)} bpm)"
        else:
            result["tempo_feel"] = "no clear beat / ambient"

        # ── Frequency character via FFT on full clip ───────────────────
        fft_size  = min(n_samples, 65536)
        spectrum  = np.abs(np.fft.rfft(wav[:fft_size]))
        freqs     = np.fft.rfftfreq(fft_size, d=1.0/sample_rate)
        bass_e    = float(np.mean(spectrum[(freqs >= 20)  & (freqs < 300)]))
        mid_e     = float(np.mean(spectrum[(freqs >= 300) & (freqs < 3000)]))
        high_e    = float(np.mean(spectrum[(freqs >= 3000)& (freqs < 8000)]))
        vocal_e   = float(np.mean(spectrum[(freqs >= 200) & (freqs < 3400)]))  # vocal band

        if vocal_e > bass_e * 1.5 and vocal_e > high_e * 1.2:
            result["freq_character"] = "vocal dominant"
        elif bass_e > mid_e * 1.4 and bass_e > high_e * 2:
            result["freq_character"] = "bass heavy"
        elif high_e > bass_e * 1.5 and high_e > mid_e:
            result["freq_character"] = "bright / treble"
        else:
            result["freq_character"] = "balanced full range"

        # ── Whisper transcription ──────────────────────────────────────
        if use_whisper:
            _wmodel = None
            try:
                import whisper as _whisper
                print("[LTX2-Qwen] Whisper: loading tiny model...")
                _wmodel = _whisper.load_model("tiny")
                # Whisper wants float32 mono at 16kHz
                from scipy.signal import resample_poly
                from math import gcd
                target_sr = 16000
                if sample_rate != target_sr:
                    g       = gcd(sample_rate, target_sr)
                    wav_16k = resample_poly(wav, target_sr // g, sample_rate // g).astype(np.float32)
                else:
                    wav_16k = wav
                w_result   = _wmodel.transcribe(wav_16k, fp16=False, language=None)
                transcript = w_result.get("text", "").strip()
                if transcript:
                    result["has_speech"]  = True
                    result["transcript"]  = transcript
                    # Rough tone from amplitude stats during speech segments
                    speech_rms = float(np.mean(np.abs(wav_16k)))
                    if speech_rms < 0.02:
                        result["speech_tone"] = "whispered or very quiet"
                    elif speech_rms < 0.08:
                        result["speech_tone"] = "conversational"
                    elif speech_rms < 0.18:
                        result["speech_tone"] = "elevated / emphatic"
                    else:
                        result["speech_tone"] = "loud / shouting"
                    print(f"[LTX2-Qwen] Whisper transcript: {transcript[:120]}...")
                else:
                    print("[LTX2-Qwen] Whisper: no speech detected")
            except ImportError:
                print("[LTX2-Qwen] Whisper not installed — skipping transcription. pip install openai-whisper")
            except Exception as e:
                print(f"[LTX2-Qwen] Whisper failed (non-fatal): {e}")
            finally:
                # Always offload Whisper — load fresh every run, no caching
                if _wmodel is not None:
                    del _wmodel
                gc.collect()
                try:
                    import torch as _torch
                    if _torch.cuda.is_available():
                        _torch.cuda.empty_cache()
                        _torch.cuda.ipc_collect()
                except Exception:
                    pass
                print("[LTX2-Qwen] Whisper offloaded.")

        # ── Build plain-English summary for LLM ───────────────────────
        parts = []
        parts.append(f"Duration: {result['duration_s']}s.")
        parts.append(f"Energy: {result['energy_shape']}.")
        if result["tempo_feel"]:
            parts.append(f"Rhythm: {result['tempo_feel']}.")
        if result["freq_character"]:
            parts.append(f"Sound character: {result['freq_character']}.")
        if result["peak_moment"] and result["duration_s"]:
            pct = result["peak_moment"] / result["duration_s"]
            parts.append(f"Loudest moment at {result['peak_moment']}s ({int(pct*100)}% through).")
        if result["silence_ratio"] and result["silence_ratio"] > 0.3:
            parts.append(f"Significant silence ({int(result['silence_ratio']*100)}% of clip).")
        if result["has_speech"] and result["transcript"]:
            parts.append(f"Speech detected ({result['speech_tone']}): \"{result['transcript'][:400]}\"")
        elif result["has_speech"]:
            parts.append(f"Speech detected ({result['speech_tone']}) — transcription unavailable.")

        result["summary"] = " ".join(parts)

    except Exception as e:
        print(f"[LTX2-Qwen] Audio analysis failed (non-fatal): {e}")
        result["summary"] = None

    return result


def _build_audio_instruction(audio_analysis: dict) -> str:
    """Converts audio analysis dict into an LLM instruction block."""
    if not audio_analysis or not audio_analysis.get("summary"):
        return ""

    a = audio_analysis
    lines = ["\n[AUDIO ANALYSIS — shape the visuals to match this audio:"]
    lines.append(a["summary"])

    # Specific directives based on what we found
    if a.get("energy_shape") == "builds to explosive peak" and a.get("peak_moment"):
        lines.append(
            f"The audio builds and hits hard at {a['peak_moment']}s — "
            "mirror this: start restrained, escalate, let the visual peak land at the same moment."
        )
    elif a.get("energy_shape") == "sparse with long silences":
        lines.append(
            "The audio is sparse and quiet. Match this: slow camera, minimal action, "
            "let silence breathe in the visual pacing."
        )
    elif a.get("energy_shape") == "constant sustained energy":
        lines.append(
            "Sustained consistent energy throughout — maintain visual intensity evenly, no dramatic arc."
        )
    elif a.get("energy_shape") == "builds throughout":
        lines.append(
            "Energy grows from start to finish — visuals should escalate progressively, "
            "ending more intense than they began."
        )

    if a.get("tempo_feel") and "very fast" in a["tempo_feel"]:
        lines.append("Fast tempo — camera movement and subject action should feel kinetic and driven.")
    elif a.get("tempo_feel") and "slow" in a["tempo_feel"]:
        lines.append("Slow tempo — deliberate camera movement, held shots, unhurried action.")
    elif a.get("tempo_feel") and "no clear beat" in a["tempo_feel"]:
        lines.append("Ambient / no beat — floating camera movement, atmospheric over kinetic.")

    if a.get("freq_character") == "bass heavy":
        lines.append("Heavy bass presence — weight, physicality, and low-frequency movement in the visuals.")
    elif a.get("freq_character") == "bright / treble":
        lines.append("Bright treble character — light, crisp, airy visuals to match.")
    elif a.get("freq_character") == "vocal dominant":
        lines.append("Vocals are the dominant element — frame the speaker, sync lip movement if dialogue is present.")

    if a.get("has_speech") and a.get("transcript"):
        lines.append(
            f"The audio contains speech. If the scene shows a person speaking, "
            f"their words are: \"{a['transcript'][:300]}\". "
            "Sync the visual scene to what is being said — if they describe an action, show it. "
            "If it's a monologue, frame them speaking. Do not invent different dialogue."
        )

    lines.append("]")
    return "\n".join(lines)


# ── Negative prompt ───────────────────────────────────────────────────────────

_NEG_BASE = (
    "watermark, text, signature, duplicate, "
    "static, no motion, frozen, "
    "poorly drawn, bad anatomy, deformed, disfigured, "
    "extra limbs, missing limbs, floating limbs, disconnected body parts, "
    "micro jitter, flickering, strobing, aliasing, high frequency patterns, "
    "motion artifacts, temporal inconsistency, frame stuttering"
)

_NEG_EXPLICIT      = "censored, mosaic, pixelated, black bar, blurred genitals"
_NEG_PORTRAIT_SHOT = "wide angle distortion, fish eye, full body shot"
_NEG_WIDE          = "close-up, portrait crop, tight frame"
_NEG_MULTI         = "merged bodies, fused figures, incorrect number of people"
_NEG_PORTRAIT_ORI  = "landscape orientation, letterbox, pillarbox, horizontal crop, widescreen framing"
_NEG_VHS           = "clean digital, sharp edges, 4K, high resolution, pristine quality"
_NEG_HORROR        = "bright happy lighting, warm tones, cheerful atmosphere, soft light"
_NEG_FASHION       = "casual handheld, amateur footage, flat lighting, unposed"
_NEG_SELFIE        = (
    "tripod, gimbal stabilised, smooth camera movement, rack focus, dolly, crane, "
    "cinematic bokeh, dramatic depth of field, professional lighting, film grain, "
    "colour grade, cinematic lens, landscape orientation"
)
_NEG_ANIME        = "photorealistic, live action, real person, CGI, 3D render, western cartoon, flat shading"
_NEG_2DCARTOON    = "photorealistic, 3D render, CGI, anime, live action, flat digital art, no line work"
_NEG_3DCGI        = "photorealistic, live action, 2D flat, hand-drawn, sketch, anime, watercolour"
_NEG_STOPMOTION   = "smooth motion, CGI, photorealistic, digital, fluid movement, motion blur"
_NEG_COMICBOOK    = "photorealistic, soft gradients, 3D render, painterly, no line art, anime"
_NEG_CELSHADED    = "photorealistic, soft shading, gradients, painterly, hand-drawn lines, anime"
_NEG_ROTOSCOPE    = "fully animated, cartoon, CGI, no live action base, unnatural movement"
_NEG_CYBERPUNK    = "natural lighting, pastoral, warm tones, daylight, photorealistic skin, muted colour"
_NEG_SCIFI        = "medieval, fantasy, nature, pastoral, historical, period costume, warm earthy tones"
_NEG_GRAVURE      = ("dark dramatic lighting, moody shadows, desaturated colour, film noir, cinematic grade, harsh contrast, gritty texture, low key lighting, overcast flat light, ugly, deformed, bad anatomy")


def _build_negative_prompt(result: str, user_input: str, is_portrait: bool = False, style_preset: str = "") -> str:
    combined = (result + " " + user_input + " " + style_preset).lower()
    extras = []

    # Explicit content — suppress censorship artifacts
    if any(w in combined for w in ["pussy", "cock", "penis", "vagina", "nude", "naked", "explicit", "nipple", "breast"]):
        extras.append(_NEG_EXPLICIT)

    # Shot framing conflicts
    if any(w in combined for w in ["close-up", "close up", "face shot", "headshot"]):
        extras.append(_NEG_PORTRAIT_SHOT)
    elif any(w in combined for w in ["wide shot", "wide angle", "aerial", "bird's-eye", "establishing"]):
        extras.append(_NEG_WIDE)

    if any(w in combined for w in ["two women", "two men", "two people", "both", "together", "couple", "they "]):
        extras.append(_NEG_MULTI)

    if is_portrait or "portrait vertical" in style_preset.lower() or "9:16" in style_preset:
        extras.append(_NEG_PORTRAIT_ORI)

    if "lo-fi" in style_preset.lower() or "vhs" in style_preset.lower():
        extras.append(_NEG_VHS)
    if "horror" in style_preset.lower():
        extras.append(_NEG_HORROR)
    if "fashion editorial" in style_preset.lower():
        extras.append(_NEG_FASHION)
    if "selfie" in style_preset.lower() or "self-shot" in style_preset.lower():
        extras.append(_NEG_SELFIE)

    if "anime" in style_preset.lower():
        extras.append(_NEG_ANIME)
    if "2d cartoon" in style_preset.lower():
        extras.append(_NEG_2DCARTOON)
    if "3d cgi" in style_preset.lower():
        extras.append(_NEG_3DCGI)
    if "stop motion" in style_preset.lower():
        extras.append(_NEG_STOPMOTION)
    if "comic book" in style_preset.lower():
        extras.append(_NEG_COMICBOOK)
    if "cel-shaded" in style_preset.lower():
        extras.append(_NEG_CELSHADED)
    if "rotoscope" in style_preset.lower():
        extras.append(_NEG_ROTOSCOPE)
    if "cyberpunk" in style_preset.lower():
        extras.append(_NEG_CYBERPUNK)
    if "sci-fi" in style_preset.lower():
        extras.append(_NEG_SCIFI)
    if "gravure" in style_preset.lower():
        extras.append(_NEG_GRAVURE)

    parts = [p for p in [_NEG_BASE] + extras if p.strip()]
    # Femdom style preset — suppress softness, gentleness, romantic tones
    if "femdom" in style_preset.lower() or "verbal domination" in style_preset.lower():
        extras.append("soft lighting, romantic, gentle, tender, sweet, loving, equal power, soft expression, warm smile, affectionate")

    return ", ".join(parts)


# ── Character attribute pools ─────────────────────────────────────────────────

_CHAR_AGES = [
    # Range: 18–21. Weighted toward 19–20 as peak youth.
    # Duplicates are intentional weighting — not a mistake.
    "18",
    "19", "19", "19",
    "20", "20", "20",
    "21", "21",
]

_CHAR_AGES_ADULT = _CHAR_AGES  # all entries already 18+

_CHAR_ETHNICITIES_FEMALE = [
    # ── White — dominant (12 entries) ───────────────────────────────────────
    ("White",            "pale freckled skin with pink undertones"),
    ("White",            "fair skin with cool undertones"),
    ("White",            "light skin with warm peachy tones"),
    ("White",            "porcelain skin with visible blue veins at the temples"),
    ("White",            "fair skin with a light golden summer tan"),
    ("White",            "light skin with a soft rosy flush across the cheeks"),
    ("White",            "pale skin with cool blue-pink undertones"),
    ("White",            "creamy fair skin with warm neutral undertones"),
    ("White",            "light skin, slightly olive-toned from sun exposure"),
    ("White",            "fair freckled skin with warm amber undertones"),
    ("White",            "light skin with warm honey undertones"),
    ("White",            "fair skin, slightly flushed at the cheeks"),
    # ── East Asian — strong secondary (10 entries) ───────────────────────────
    ("Japanese",         "pale skin with cool beige undertones"),
    ("Korean",           "fair skin with a soft peachy-pink flush"),
    ("Chinese",          "light golden-toned skin"),
    ("East Asian",       "fair cool-toned skin with a subtle pink undertone"),
    ("East Asian",       "light ivory skin with warm golden undertones"),
    ("Japanese",         "very fair skin, almost translucent in soft light"),
    ("Korean",           "smooth fair skin with a cool porcelain tone"),
    ("Chinese",          "warm ivory skin with golden undertones"),
    ("East Asian",       "pale skin with a subtle warm peach cast"),
    ("East Asian",       "light skin with cool neutral undertones and a natural glow"),
    # ── Black — rare (2 entries only) ────────────────────────────────────────
    ("Black",            "deep ebony skin with cool blue-black undertones"),
    ("Black",            "medium warm brown skin with golden undertones"),
]

# Male pool — mostly White, some East Asian, very rare Black
_CHAR_ETHNICITIES_MALE = [
    # ── White — dominant (10 entries) ───────────────────────────────────────
    ("White",            "fair skin with cool undertones"),
    ("White",            "light skin with warm peachy tones"),
    ("White",            "fair skin with a light golden summer tan"),
    ("White",            "light skin, slightly olive-toned from sun exposure"),
    ("White",            "fair freckled skin with warm amber undertones"),
    ("White",            "light skin with warm honey undertones"),
    ("White",            "creamy fair skin with warm neutral undertones"),
    ("White",            "pale skin with cool blue-pink undertones"),
    ("White",            "fair skin, slightly flushed at the cheeks"),
    ("White",            "pale freckled skin with pink undertones"),
    # ── East Asian — secondary (4 entries) ──────────────────────────────────
    ("Japanese",         "pale skin with cool beige undertones"),
    ("Korean",           "fair skin with a soft peachy-pink flush"),
    ("East Asian",       "light golden-toned skin with warm undertones"),
    ("East Asian",       "light skin with cool neutral undertones"),
    # ── Black — rare (1 entry) ───────────────────────────────────────────────
    ("Black",            "medium warm brown skin with golden undertones"),
]

# Legacy alias — points to female pool; kept so any external code referencing
# _CHAR_ETHNICITIES still works. Must be defined AFTER both lists above.
_CHAR_ETHNICITIES = _CHAR_ETHNICITIES_FEMALE

# ── Hair colour pools ─────────────────────────────────────────────────────────
# Split by ethnicity group so colours are always plausible.
# Grey is age-gated inside _build_char_seed — never picked randomly for under-40s.

# East Asian — almost always dark, occasional dyed
_CHAR_HAIR_COLOURS_EAST_ASIAN = [
    "jet black", "jet black", "jet black", "jet black", "jet black",
    "dark brown", "dark brown",
    "blue-black", "blue-black",
    "natural dark brown with caramel highlights",
    "dyed burgundy",
    "dyed bleach blonde with dark roots",
]

# White — full natural range plus occasional dyed
_CHAR_HAIR_COLOURS_WHITE = [
    "dark brown", "dark brown",
    "warm medium brown", "warm medium brown",
    "warm chestnut brown",
    "honey blonde", "honey blonde",
    "ash blonde",
    "strawberry blonde",
    "auburn",
    "copper red",
    "platinum blonde",
    "natural dark brown with caramel highlights",
    "dyed burgundy",
    "dyed bleach blonde with dark roots",
]

# Dark — for Black, South Asian, Indian, Southeast Asian, Middle Eastern, Latina, mixed
_CHAR_HAIR_COLOURS_DARK = [
    "jet black", "jet black", "jet black",
    "dark brown", "dark brown",
    "warm medium brown",
    "blue-black",
    "natural dark brown with caramel highlights",
    "warm chestnut brown",
    "dyed burgundy",
]

# Grey — age-gated, only used 40+
_CHAR_HAIR_COLOURS_GREY = [
    "silver-streaked dark brown",
    "salt-and-pepper grey",
    "silver-white",
]

# ── Hair style pools ───────────────────────────────────────────────────────────
# Ethnicity-aware so we don't get East Asian women with cornrows or afros

_CHAR_HAIR_STYLES_STRAIGHT = [
    # East Asian, White — straight/wavy textures
    "pin-straight, very long, falling to the waist",
    "pin-straight, blunt cut to the shoulder",
    "sleek straight hair, cut to the chin",
    "straight with a heavy blunt fringe",
    "loose beach waves, mid-back length",
    "tousled waves, shoulder-length",
    "soft waves with a side part, collarbone length",
    "chin-length bob, blunt",
    "asymmetric bob, longer on one side",
    "high ponytail, sleek",
    "messy bun with loose strands framing the face",
    "half-up half-down, loosely pinned",
    "low bun, tight and smooth",
    "very long straight hair, centre-parted",
    "long layered hair with curtain bangs",
    "long thick hair in a loose braid over one shoulder",
    "cropped pixie cut, textured",
    "short sleek crop, close to the head",
    "shoulder-length with soft layers",
    "long straight hair with wispy curtain bangs",
]

_CHAR_HAIR_STYLES_CURLY = [
    # Latina, Middle Eastern, mixed race, South Asian, Indian — wavy/curly textures
    "loose 3B curls, mid-length",
    "defined 3C ringlets, shoulder-length",
    "big loose natural curls, voluminous",
    "tousled waves, shoulder-length",
    "long thick curly hair, loosely tied back",
    "curly bob, chin-length",
    "long curly hair with a centre part",
    "half-up half-down, loose curls falling forward",
    "high ponytail of loose curls",
    "long wavy hair with a side part",
    "messy bun of loose curls with strands framing the face",
    "long dark hair in a loose plait over one shoulder",
    "thick straight hair with a blunt fringe",
    "sleek high bun, tight",
]

_CHAR_HAIR_STYLES_TEXTURED = [
    # Black, African ethnicities — natural textured hair
    "tight 4C coils, natural and full",
    "thick natural afro, rounded",
    "big loose natural curls, voluminous",
    "long box braids falling past the shoulders",
    "short box braids, chin-length",
    "thick cornrows flat to the scalp",
    "two-strand twists, loose and mid-length",
    "high bun of twisted locs",
    "long faux locs, loose",
    "defined 3C ringlets, shoulder-length",
    "cropped natural coils, close to the head",
    "short tapered natural cut",
    "sleek pressed hair, shoulder-length",
]

# Male styles — shared across ethnicities (cuts work universally)
_CHAR_HAIR_STYLES_MALE = [
    "short cropped cut, neat",
    "short textured cut with a natural part",
    "buzz cut, close to the scalp",
    "faded sides with longer hair on top",
    "slicked back, medium length",
    "messy textured crop",
    "short curls, close-cropped",
    "tight natural curls, short",
    "short afro, rounded",
    "mid-length waves, loosely swept back",
    "short dreadlocks",
    "shoulder-length straight hair, centre-parted",
    "shaved head",
    "close-cropped with a defined hairline",
    "short tapered cut, higher on top",
    "undercut with longer hair swept to one side",
    "short neat side part",
    "textured quiff, short sides",
]

# ── Accent pool ────────────────────────────────────────────────────────────────
# Mapped per ethnicity. Only injected when dialogue is present.
# Phrased as LTX-2.3 voice description language — pace, texture, accent together.
_CHAR_ACCENTS = {
    "Japanese":         "speaks in Japanese, voice soft and precise, each word measured",
    "Korean":           "speaks in Korean, tone bright and clipped, vowels clean and forward",
    "Chinese":          "speaks in Mandarin, voice steady and level, consonants crisp",
    "East Asian":       "speaks in accented English with a soft East Asian lilt, vowels slightly flattened",
    "South Asian":      "speaks in a warm Indian accent, vowels rounded and musical, slight sing-song rhythm",
    "Indian":           "speaks in a clear Indian accent, consonants precise, cadence unhurried and melodic",
    "Southeast Asian":  "speaks in lightly accented English with a soft Southeast Asian tone, gentle and even",
    "Filipino":         "speaks in Filipino-accented English, warm and expressive, slight upward lilt at sentence ends",
    "Vietnamese":       "speaks in Vietnamese-accented English, tone rising and precise, soft consonants",
    "Middle Eastern":   "speaks in a warm Middle Eastern accent, deep vowels, unhurried and deliberate",
    "North African":    "speaks in a rich North African accent, warm and resonant, slight French influence in the rhythm",
    "Black":            "speaks with a deep resonant voice, relaxed and unhurried, warm American cadence",
    "Latina":           "speaks in accented English with a warm Latin rhythm, vowels open and expressive",
    "White":            "",  # No accent note — neutral, let LLM decide naturally
    "mixed race":       "",  # No accent note — intentionally neutral
    "Indigenous":       "speaks in a low unhurried voice, measured and grounded, each word given full weight",
    "Pacific Islander": "speaks in a warm Pacific Islander accent, slow and melodic, voice deep and relaxed",
    # Jamaican is rolled separately for Black characters with a chance roll
    "_jamaican":        "speaks in a rich Jamaican accent, rhythm lilting and musical, consonants sharp and warm",
    "_british_black":   "speaks in a London accent with Caribbean warmth, clipped and quick with soft vowels",
    "_african":         "speaks in a deep West African accent, voice rich and resonant, cadence stately and unhurried",
}


# ── Body type pools ───────────────────────────────────────────────────────────
# All entries are conventionally attractive / camera-friendly.
# Chubby, fat, heavyset, plus-size etc are deliberately excluded —
# if the user wants those they describe it themselves.
# Split by ethnicity group for plausible combinations.
# "Receding hairline" removed from male pool for same reason.

# East Asian female — petite to medium, slim dominant
_CHAR_BODY_TYPES_FEMALE_EAST_ASIAN = [
    "petite and slender, small-framed",
    "petite and slender, small-framed",
    "slim with a flat stomach and narrow hips",
    "slim with a flat stomach and narrow hips",
    "thin with delicate bone structure",
    "lean and tall with long limbs",
    "athletic build with defined shoulders",
    "lean and athletic with visible muscle definition",
    "strong legs and a narrow waist",
    "full hourglass figure with wide hips and a defined waist",
]

# White female — full range, slim to curvy, all attractive
_CHAR_BODY_TYPES_FEMALE_WHITE = [
    "slender build with narrow shoulders",
    "lean and tall with long limbs",
    "lean and tall with long limbs",
    "slim with a flat stomach and narrow hips",
    "athletic build with defined shoulders",
    "lean and athletic with visible muscle definition",
    "strong legs and a narrow waist",
    "full hourglass figure with wide hips and a defined waist",
    "full hourglass figure with wide hips and a defined waist",
    "curvy with a round bust and full hips",
    "big-busted with a narrow waist and wide hips",
    "tall and willowy with long legs",
    "statuesque, over six feet, lean",
]

# Fallback pool — only hits if an ethnicity outside the main three somehow appears
_CHAR_BODY_TYPES_FEMALE_CURVY = [
    "full hourglass figure with wide hips and a defined waist",
    "curvy with a round bust and full hips",
    "athletic build with defined shoulders",
    "lean and athletic with visible muscle definition",
    "slim with a flat stomach and narrow hips",
    "strong legs and a narrow waist",
]

# Black female — athletic and curvy, all camera-attractive
_CHAR_BODY_TYPES_FEMALE_BLACK = [
    "full hourglass figure with wide hips and a defined waist",
    "full hourglass figure with wide hips and a defined waist",
    "curvy with a round bust and full hips",
    "big-busted with a narrow waist and wide hips",
    "athletic build with defined shoulders",
    "lean and athletic with visible muscle definition",
    "strong legs and a narrow waist",
    "tall and willowy with long legs",
    "statuesque, over six feet, lean",
]

# Male — lean to muscular, all attractive, no heavyset
_CHAR_BODY_TYPES_MALE_ATTRACTIVE = [
    "lean and wiry with narrow shoulders",
    "tall and slim with long limbs",
    "slim with a flat stomach, average build",
    "compact and lightly muscled",
    "athletic build with broad shoulders and a tapered waist",
    "athletic build with broad shoulders and a tapered waist",
    "muscular and powerfully built, broad chest",
    "lean and athletic with visible muscle definition",
    "lean and athletic with visible muscle definition",
    "tall with a rangy, angular frame",
    "well-built with a defined chest and flat stomach",
    "well-built with a defined chest and flat stomach",
    "wiry and compact, all sinew, no excess",
]


def _build_char_seed(rng: random.Random, adult_only: bool = False, gender: str = "female") -> str:
    """
    Build a randomised character description with ethnicity-aware
    hair colour, hair style, and accent.
    gender: 'female' | 'male' | 'neutral' (neutral picks randomly)
    Ethnicity pools are gender-specific:
      female → mostly White / East Asian / rare Black
      male   → mostly White / some East Asian / very rare Black
    """
    age_pool        = _CHAR_AGES_ADULT if adult_only else _CHAR_AGES
    age             = rng.choice(age_pool)
    age_int         = int(age)

    if gender == "neutral":
        gender = rng.choice(["female", "male"])

    # ── Pick ethnicity from the correct gender pool ───────────────────────────
    eth_pool        = _CHAR_ETHNICITIES_MALE if gender == "male" else _CHAR_ETHNICITIES_FEMALE
    ethnicity, skin = rng.choice(eth_pool)

    # ── Hair colour — ethnicity aware ─────────────────────────────────────────
    if age_int >= 40 and rng.random() < 0.35:
        # 35% chance of grey for 40+ regardless of ethnicity
        hair_colour = rng.choice(_CHAR_HAIR_COLOURS_GREY)
    elif ethnicity in ("Japanese", "Korean", "Chinese", "East Asian"):
        hair_colour = rng.choice(_CHAR_HAIR_COLOURS_EAST_ASIAN)
    elif ethnicity == "White":
        hair_colour = rng.choice(_CHAR_HAIR_COLOURS_WHITE)
    else:
        # Black — use dark pool
        hair_colour = rng.choice(_CHAR_HAIR_COLOURS_DARK)

    # ── Hair style — ethnicity aware ──────────────────────────────────────────
    if gender == "male":
        hair_style = rng.choice(_CHAR_HAIR_STYLES_MALE)
    elif ethnicity == "Black":
        hair_style = rng.choice(_CHAR_HAIR_STYLES_TEXTURED)
    elif ethnicity in ("Japanese", "Korean", "Chinese", "East Asian", "White"):
        hair_style = rng.choice(_CHAR_HAIR_STYLES_STRAIGHT)
    else:
        # Fallback for any future additions
        hair_style = rng.choice(_CHAR_HAIR_STYLES_STRAIGHT)

    # ── Accent — ethnicity aware, injected as voice note ──────────────────────
    if ethnicity == "Black":
        accent_roll = rng.random()
        if accent_roll < 0.25:
            accent = _CHAR_ACCENTS["_jamaican"]
        elif accent_roll < 0.45:
            accent = _CHAR_ACCENTS["_british_black"]
        elif accent_roll < 0.65:
            accent = _CHAR_ACCENTS["_african"]
        else:
            accent = _CHAR_ACCENTS["Black"]
    else:
        accent = _CHAR_ACCENTS.get(ethnicity, "")

    # ── Body type — ethnicity and gender aware ────────────────────────────────
    if gender == "male":
        body_type = rng.choice(_CHAR_BODY_TYPES_MALE_ATTRACTIVE)
    elif ethnicity in ("Japanese", "Korean", "Chinese", "East Asian"):
        body_type = rng.choice(_CHAR_BODY_TYPES_FEMALE_EAST_ASIAN)
    elif ethnicity == "White":
        body_type = rng.choice(_CHAR_BODY_TYPES_FEMALE_WHITE)
    else:
        # Black
        body_type = rng.choice(_CHAR_BODY_TYPES_FEMALE_BLACK)

    # ── Assemble ──────────────────────────────────────────────────────────────
    if gender == "male":
        base = (
            f"a {age}-year-old {ethnicity} man, "
            f"{hair_colour} hair, {hair_style}, "
            f"{skin}, "
            f"{body_type}"
        )
    else:
        base = (
            f"a {age}-year-old {ethnicity} woman, "
            f"{hair_colour} hair in a {hair_style}, "
            f"{skin}, "
            f"{body_type}"
        )

    if accent:
        base += f", {accent}"

    return base


# ── Node ──────────────────────────────────────────────────────────────────────

class LTX2PromptArchitectQwen:
    """
    LTX-2.3 Easy Prompt — Huihui Qwen3.5-9B Edition

    Full LD-grade prompt engineering ported to Qwen3.5-9B-abliterated.
    All style presets, content detection, garment sequences, spatial blocking,
    dialogue control, and character seeds from the main LD node.
    Single model: huihui-ai/Huihui-Qwen3.5-9B-abliterated
    """

    # ── System prompt (full LD version) ──────────────────────────────────────
    SYSTEM_PROMPT = """You are a cinematic prompt writer for LTX-2.3, an AI video generation model. Expand the user's rough idea into a precise, director-level, video-ready prompt. Write as a single flowing paragraph in present tense. Be specific — LTX-2.3 rewards detail and rewards you for using it.

USER INPUT IS LAW:
The user's input is the single most important thing in this entire prompt. It defines what exists in the scene.
Before you write a single word, read the user input and lock it in as absolute ground truth.
Every word the user wrote must appear in the output — faithfully, literally, completely.
If they said cliff edge — the cliff edge and the drop below it must be visible in the shot.
If they said soaked — her clothing is wet, her hair is plastered, her skin is glistening.
If they said direct eye contact — her eyes are locked on the lens, unwavering.
If they said coat flapping — the coat is actively moving in the wind throughout.
If they said booming thunderclap — the thunder is present as a physical sound event.
Do NOT reinterpret, soften, generalise, or aestheticise away anything the user wrote.
The instructions that follow tell you HOW to render the scene — they do not define WHAT the scene is.
WHAT the scene is comes only from the user.

LTX-2.3 CAPABILITIES — exploit all of these:
- Detailed prompts outperform short ones, especially for longer clips. A 10-second clip needs a rich, full prompt to fill the duration.
- Fine detail renders accurately: fabric weave, individual hair strands, skin texture, surface wear, material finish. Describe these.
- Strong prompt adherence — you can direct camera AND subject motion simultaneously with precision.
- Native portrait up to 1080x1920 — compose vertically, not as a cropped landscape.
- Improved audio — voice quality, ambient sound, and sync all respond well to description. Sound is always present.
- Always include motion. Static prompts produce static video.
- Single-subject scenes give the sharpest, most faithful output. Two subjects is viable with clear blocking. Three or more risks clarity degradation — keep their actions simple, non-overlapping, and explicitly spatially separated.

SCENE INTEGRITY:
Build outward from what the user gave you — do not contradict or override it.
If the user described a location, enrich it with specific textures and atmosphere: a city street becomes "wet asphalt reflecting neon, steam rising from a grate, the cold blue cast of a streetlamp". A café becomes "warm tungsten light, fogged glass, the grain of the wooden tabletop".
If the user gave NO location, shoot in a neutral unspecified space — do not invent a warehouse, forest, or bedroom they didn't ask for.
Do NOT add: rose petals, candles, silk sheets, glitter, sparkles, or sentimental filler that isn't grounded in the scene.
Do NOT invent props or characters the user didn't mention.
Every addition must be (a) from the user's input, (b) a texture/material/atmosphere detail that enriches what they described, or (c) a necessary camera/staging decision.

CAMERA ORIENTATION — IMPORTANT:
The DEFAULT is that the subject FACES the camera. ONLY write rear view or camera-behind framing if the user explicitly said: "from behind", "rear view", "back view", "follow her from behind", "watches her from behind", "camera behind", "over her shoulder from behind". Do NOT default to rear view just because the subject is walking.

SCENE DIRECTION — build the prompt in this order:
1. Style & genre — use the STYLE INSTRUCTION as the aesthetic anchor. Where it fits, weave a film stock or camera reference into the prose naturally — e.g. "the image carries a Kodak 2383 warmth", "shot on an ARRI Alexa, clean and clinical", "Fuji Eterna desaturation softens the shadows". NEVER as a bracketed tag. It must read as part of a sentence.
2. Shot scale & framing — choose the right scale for the scene and name it in the prose. Use these terms:
   - Extreme close-up: fills the frame with a single feature — an eye, a mouth, a hand, a texture
   - Close-up: face from chin to crown, or a hand, a detail — intimate, personal
   - Medium close-up: head and shoulders, face dominant — the standard for dialogue and emotion
   - Medium shot: waist up — character and some environment in frame together
   - Medium wide / cowboy shot: thighs up — character in context, readable body language
   - Wide shot / full shot: full body, environment present and readable
   - Establishing shot: wide, environment dominant — sets the location
   - Aerial / bird's-eye: camera directly above, looking straight down
   - Low angle: camera below subject, looking up — power, dominance, scale
   - High angle: camera above subject, looking down — vulnerability, surveillance
   - Dutch angle: camera tilted on its axis — unease, tension, disorientation
   - Over-the-shoulder (OTS): camera behind one subject, looking past them at another
   - Point-of-view (POV): camera is the character's eyes
   - Two-shot: both subjects visible and balanced in frame
   - Insert shot: tight cut to a specific object or detail in the scene
   Also describe depth of field using natural language: "razor-thin depth of field", "deep focus, everything sharp front to back", "shallow depth of field with creamy background blur", "soft bokeh behind the subject".
   Match detail level to shot scale — close-ups require more granular description (pores, hair strands, fabric weave, micro expressions). Wide shots need environmental depth (foreground elements, layers of space, atmospheric haze or light).
3. Character — ALWAYS state age as a specific number: "a 27-year-old woman". Default range 18–35 unless context implies otherwise. Use 40+ only if the user said "older", "mature", "middle-aged", "elderly". Use under-18 only if the user explicitly placed them in a school, childhood, or teen context — NEVER for sexual or suggestive content. Then: hair texture and colour, skin tone, body type, clothing with fabric and material ("a fitted black cotton crop top", "worn light-wash denim jeans", "a loose cream silk blouse"). Use the exact words the user used. Include micro expressions and subtle physical tells: "the corners of her lips tighten slightly", "her eyes momentarily lose focus", "a faint crease forms between her brows".
4. Scene & environment — location, time of day, lighting quality and direction, colour temperature, surface textures ("scuffed hardwood floor", "rain-streaked glass", "warm tungsten interior"). Only what the user described. Avoid high-frequency visual patterns in clothing, backgrounds, and surfaces — they cause flickering artifacts. Favour solid colours and simple textures.
5. Spatial blocking — MANDATORY. Define left/right position, foreground/background depth, who faces what. "She stands centre-left in the foreground, facing camera. He sits well behind her at the right edge of frame, slightly out of focus." For single subjects: anchor them — "She stands centre-frame in the immediate foreground, facing camera, the background soft behind her." Block every scene like a director.

ACTION & MOTION:
6. Motion — describe all four simultaneously when possible: who moves, what moves, how they move, what the camera does. "She steps forward and turns as the camera tracks left and slowly pushes in." If the scene is genuinely static, add ONE subtle environmental motion only: a slow camera drift, wind lifting the hair, a distant background detail. Do not pile on micro-movements. For smooth motion: stable dolly, smooth tracking, constant-speed pan. Avoid chaotic or rapid movement unless the style requires it.
7. Texture in motion — how materials behave as things move: "the fabric pulls taut across her hips", "her hair lifts and separates", "the denim creases at the knee as she bends". LTX-2.3 renders this accurately.
8. Camera movement — prose verbs only, never bracketed tags. "The shot slowly pushes in" not "(Push in)". Use: dolly in/out, rack focus, whip pan, push in, crane up, handheld drift, creep forward, track right, gimbal arc, slow pull back, tilt up/down, pan left/right, slow lateral track. Describe camera movement relative to the subject. After a camera move, describe how the subject appears in the new framing — this helps the model complete the motion accurately: "the camera pushes in until her face fills the frame, her eyes now the sharpest point in the composition".

SOUND — always present, always described:
9. Sound is MANDATORY — there are no silent scenes. Weave it as descriptive prose. Max 2 sounds per beat. When action and sound sync, say so explicitly: "her footsteps land on each downbeat", "the shutter clicks precisely as her fingers press". Sync language strengthens audio-visual coherence.
- Every sound needs sensory detail. Not "footsteps" — "the sharp rhythmic clack of heels on cold marble, each step ringing with a hollow echo." Not "rain" — "rain hitting the glass in irregular bursts, a low persistent hiss beneath it."
- Speaking characters: always describe voice quality — pace, texture, register. "Her voice barely above a breath", "a low gravelly rumble", "fast and clipped, each word landing hard", "slow and deliberate, each syllable weighted".
- Music/performance scenes: describe the track as physical sensation — "a deep kick drum punches through the floor, the bass felt in the chest", "sharp hi-hats over a slow rolling groove". Do NOT reduce it to "music plays".
- Never use [AMBIENT: ...] tags. No abstract emotional audio — no "tension fills the air".

CRITICAL RULES:
- NEVER write scene endings. Prompts describe ongoing action, not conclusions. Forbidden: "the scene ends", "fades to black", "the camera cuts", "comes to a close". Also forbidden: winding-down summary sentences — "In this quiet moment...", "leaving only the warmth of...", "ending on the soft...". The prompt is always mid-action.
- NEVER invent characters. If the user described one person, there is one person. No bystanders, passers-by, or observers unless the user wrote them.
- NEVER use internal emotional labels — not "she is sad", "he feels confused", "she looks happy". Describe only what is physically visible: "the corners of her mouth pull down", "his eyes lose focus for a moment", "a faint crease forms between her brows".
- AVOID conflicting lighting logic. Do not mix a candlelit interior with harsh midday sunlight unless the scene calls for it. Mixed light sources that contradict each other confuse the model.
- AVOID chaotic or complex physics — multiple objects colliding unpredictably, crowds in chaotic motion, or rapid overlapping movements introduce artifacts. Choreographed or rhythmic movement (dancing, sport, deliberate action) is fine. Random physical chaos is not.
- TEXT AND LOGOS: readable text on signs, labels, or screens is unreliable. Do not ask for legible text in the frame.

DIALOGUE — follow the DIALOGUE INSTRUCTION exactly. Use LTX-2.3's structured dialogue format: break speech into short phrases with acting directions between each line. Do NOT write long dialogue in one block. Invented dialogue MUST be grounded in what is visibly happening in the scene — do NOT invent backstory, history, relationships, or context not present in the user's input. A boxer training alone may grunt a short exertion sound; she may not deliver lines about her past.
- Write a short spoken phrase in quotes
- Follow with a physical acting direction: "he pauses, glancing left", "her jaw tightens", "she exhales slowly"
- Then the next phrase, then the next direction
Example: "I remember after you came along..." He pauses, looking to the side. "Your mother..." His eyes widen slightly. "Said something I never quite understood," his voice dropping to almost nothing.
No [DIALOGUE: ...] tags. No stage directions in brackets. If the character speaks a specific language or has an accent, state it: "speaks in Japanese", "with a thick Southern drawl", "in accented English".
VOICE QUALITY — always describe HOW a character's voice sounds when they speak: pace, texture, register. "her voice barely above a breath", "a low gravelly rumble", "fast and clipped, each word landing hard", "slow and deliberate, each syllable weighted". Voice quality is as important as the words themselves.

UNDRESSING — when clothing removal is stated or clearly implied:
Write a dedicated undressing segment BEFORE any nudity. Name every garment. Describe each removal step by step. Describe what skin is revealed and how the fabric behaves. Never jump from clothed to naked.

GARMENT SEQUENCES — use the correct physical order for each type:
- T-shirt / shirt / crop top (full removal): grip the hem at the waist → pull fabric up past the stomach → past the ribs → over the chest → over the head → off the arms → dropped
- T-shirt / shirt / crop top (lift only, not removed): grip the hem → slowly gather and lift → rises past the stomach → past the navel → past the ribs → chest comes into full view → held there. One sentence per step.
- Dress (pullover): grip hem at thighs → lift past hips → past waist → gathered up over chest → over the head → falls
- Dress (zip back): hand reaches behind → finds the zip → pulls it slowly down → fabric loosens and parts → slipped off shoulders → slides down the body → pools at the floor
- Blouse / button-down: each button worked top to bottom one at a time → fabric parts → shrugged off shoulders → slides down arms → dropped
- Bra: hand behind to clasp → unhooked → straps eased off each shoulder → cups fall away
- Jeans / trousers: button popped → zip down → pushed over hips → down the thighs → stepped out of
- Underwear / knickers / thong: thumbs hooked into waistband → pushed down → stepped out of

NO INVENTED RESOLUTION: If the shirt goes up, it stays up. Do NOT write her covering herself, lowering it, or reversing the action unless the user asked for it.

PORTRAIT MODE — 9:16 vertical: frame vertically from the start. Tight head-to-torso shots. Vertical action and camera movement. No wide horizontal compositions.

WRITING RULES:
- Present tense throughout
- Specific over vague: "a loose grey cotton t-shirt, collar slightly stretched" beats "a shirt"
- Concrete over poetic: "her dress falls to the floor" beats "the fabric cascades"
- No filler adjectives: not "beautiful", "stunning", "gorgeous" — describe what's visible
- Always include motion. Always include sound. Both are mandatory.
- Always include natural motion blur — keep movement fluid, never frozen or strobed.
- Avoid high frequency patterns in clothing, backgrounds, and surfaces — these cause flickering.
- Flowing prose, not lists
- MINOR CHARACTERS (under 18): describe face expression, hair, and clothing appearance only. Do NOT write fabric-body-contact descriptions — no 'fabric pulls taut', 'presses against', 'clings to' or any clothing-on-skin language. Keep all physical description age-appropriate and non-body-focused.

OUTPUT RULES:
Output ONLY the prompt. No preamble, no "Sure!", no "Here's your prompt:", no compliance notes, no word counts, no brackets after the final sentence. Begin immediately with the shot or style description. End with the last sentence of the scene."""

    # ── Style presets (full LD set) ───────────────────────────────────────────
    STYLE_PRESETS = {
        "None — let the LLM decide": ("", False),
        "Cinematic — Drama": (
            "STYLE: Cinematic drama. Intimate, character-driven. Shallow depth of field — subject sharp, "
            "world behind them soft. Colour grade: cool shadows, warm skin tones, restrained palette. "
            "Camera: medium close-ups and close-ups dominate. Moves are slow and purposeful — "
            "a slow push-in on a face, a rack focus between two people, a static hold that lets the actor breathe. "
            "Lighting: motivated practical sources — a lamp, a window, a candle. Never flat. "
            "Kodak 2383 print emulation. Sound: intimate and close — breath, fabric, small environmental detail. "
            "No wide establishing shots unless the user asked for them. Stay with the character.", False),
        "Cinematic — Epic": (
            "STYLE: Epic cinematic. Scale and environment are the protagonist. "
            "Wide establishing shots and vast compositions that make people feel small against the world. "
            "Camera: sweeping crane moves, slow lateral tracking shots, long pulls across terrain. "
            "Colour grade: rich, contrasty — deep shadows, luminous highlights. "
            "Kodak 5219 for natural daylight scenes, ARRI Alexa for clean digital grandeur. "
            "Sound: environmental and large — wind, distance, the weight of open space. "
            "Every frame should feel like a poster. Build depth with foreground elements. "
            "Natural motion blur on all movement.", False),
        "Cinematic — Intimate close-up": (
            "STYLE: Intimate close-up cinema. The entire world is a face, a hand, a detail. "
            "Razor-thin depth of field — one eye sharp, the other already soft. Bokeh is smooth and organic. "
            "Framing: extreme close-ups and close-ups only — fill the frame with a face, a hand, a single feature. "
            "Camera: barely moves — micro drifts and imperceptible breathing movement. "
            "Colour grade: skin-tone faithful, no heavy colour casts. Warm and close. "
            "Lighting: one soft source, one fill, nothing else. "
            "Sound: amplified intimacy — breath, the swallow of saliva, fabric against skin, heartbeat proximity. "
            "Reveal character through detail — a tightening jaw, a flicker of the eye, fingers finding each other. "
            "This is portraiture as cinema.", False),
        "Slow-burn thriller": (
            "STYLE: Slow-burn psychological thriller. Tight framing, long held shots, shallow depth of field. "
            "Colour palette: desaturated teal and amber. Sound design is sparse — silence punctuated by single sounds. "
            "Camera moves deliberately and slowly. Tension built through restraint, not action.", False),
        "Handheld documentary": (
            "STYLE: Handheld documentary. Camera moves with the subject, never static. Slight shake on movement. "
            "Natural available light only — no studio lighting. Colour grade: flat, slightly washed. "
            "Intimate and observational — camera follows, never leads.", False),
        "High fashion editorial": (
            "STYLE: High fashion editorial. Striking, composed frames. Hard directional lighting with deep shadows. "
            "Colour palette: high contrast, often monochrome or single accent colour. "
            "Movement is deliberate and posed — model-aware. Camera movements are slow and precise. "
            "ENVIRONMENT NOTE: Do not invent luxury props, chandeliers, marble, or opulent settings "
            "unless the user described them. Apply the editorial aesthetic to whatever location the user specified.", False),
        "Noir — deep shadows, venetian light": (
            "STYLE: Classic noir. Low-key lighting, venetian blind shadow patterns across faces and walls. "
            "Black and white or heavily desaturated with single colour accent. "
            "Camera angles: low, Dutch tilt, shot through objects. Mood is foreboding and fatalistic.", False),
        "Golden hour drama": (
            "STYLE: Golden hour drama. Warm amber and orange light from a low sun. Heavy lens flare. "
            "Soft shadows, glowing skin tones. Wide establishing shots and medium shots. Emotional, sweeping camera movement. "
            "Colour grade: warm, slightly overexposed highlights.", False),
        "Horror — desaturated, harsh contrast": (
            "STYLE: Horror. Heavily desaturated colour, crushed blacks. Harsh top-down or under-lighting. "
            "Camera movements are slow and uneasy — never reassuring. "
            "Framing leaves negative space — empty doorways, dark corners. No warmth in the image.", False),
        "Erotic cinema — tasteful, cinematic": (
            "STYLE: Tasteful erotic cinema. Warm, intimate lighting — practical sources only. "
            "Shallow depth of field. Camera moves slowly and deliberately. "
            "Colour grade: warm skin tones, soft highlights. "
            "Sensual but not pornographic — implication over explicit detail. Slow, breathing pace. "
            "SCOPE NOTE: This style sets the aesthetic and mood only. "
            "It does NOT grant permission to add nudity, undressing, or intimate acts beyond what the user described. "
            "Describe only what was asked for — the style wraps it, it does not expand it.", False),
        "Explicit — direct, anatomical": (
            "STYLE: Explicit adult content. Direct lighting — bodies clearly lit with no flattering shadow. "
            "Camera is close and functional — shows exactly what is happening without cinematic softening. "
            "No romantic framing. Blunt and specific. Anatomical language used directly. "
            "IMPORTANT: This style affects aesthetic and language only — it does NOT expand the scope of the scene beyond what the user asked for. "
            "Describe only what the user requested. Do not add acts, exposure, or nudity the user did not write.", False),
        "Voyeur — handheld, observational": (
            "STYLE: Voyeuristic. The camera is a person — someone who found this moment and is trying not to be noticed. "
            "CAMERA BEHAVIOUR — MANDATORY: "
            "Unless the user explicitly said 'static', the camera is ALWAYS in motion. "
            "It bobs and drifts with the natural sway of someone walking or standing. "
            "The motion is involuntary — slight vertical bounce, gentle lateral drift, micro-rotations. "
            "The camera NEVER repositions to get a better angle. It stays at the height and position of the person holding it — "
            "hip height if they are trying to be discreet, chest height if partially hidden, never raised to eye level for a clean shot. "
            "FORBIDDEN camera moves: crane up, dolly in, rack focus, orbit, push in, pull back, pan to follow. "
            "ALLOWED camera behaviour: drifts, bobs, tilts slightly as the subject moves, briefly obscured by a passing person or shelf, "
            "loses the subject for a frame and finds them again. "
            "The framing is imperfect — the subject may be partially cut off, slightly out of focus at the edges, "
            "or briefly blocked. This is what makes it feel real. "
            "Natural available light only — no fill, no flash, no colour grading. "
            "The subject is unaware. The camera does not announce itself. "
            "CRITICAL: The subject's actions are exactly as the user described — do not invent, reverse, or reframe them. "
            "If the user said she is getting dressed, she is getting dressed. If the user said she is undressing, she is undressing. "
            "The camera observes what is happening — it does not change what is happening.", False),
        "Softcore editorial — lingerie-adjacent": (
            "STYLE: Softcore editorial. Fashion-magazine aesthetic. Clean, even lighting. "
            "Colour grade: warm neutrals and soft pastels. "
            "Camera is composed — lingerie-level sensuality, no explicit content. Movement is slow and posed. "
            "SCOPE NOTE: This style sets the aesthetic only. "
            "Do NOT add undressing, nudity, or intimate acts the user did not ask for. "
            "If the user described someone sitting or standing clothed, they stay clothed. "
            "The style applies to framing and mood — not to what happens in the scene.", False),
        "Gravure Idol — Japanese glamour": (
            "STYLE: Japanese gravure idol photoshoot / glamour video. "
            "Bright, glossy, commercial magazine aesthetic. "
            "High-key natural daylight or clean studio lighting with strong rim light and soft reflector fill. "
            "CHARACTER: Unless the user has described a specific person, the subject is always an Asian woman — "
            "Japanese, Korean, or Chinese — aged 18–25, with smooth fair-to-medium skin, dark hair, and a petite to medium build. "
            "This is the authentic visual identity of the gravure genre. Do not substitute other ethnicities unless the user explicitly asked. "
            "Vivid yet smooth skin tones, slightly increased saturation, polished and flattering look. "
            "OUTFIT: If the user has not described clothing, choose ONE outfit from this varied pool — do NOT default to one-piece swimsuits every time. "
            "Rotate across: a fitted white string bikini with thin side ties; a pastel two-piece with a bandeau top and high-waist bottoms; "
            "a sheer white oversized shirt worn open over a bralette and shorts; a soft satin slip dress in ivory or blush, thigh-length; "
            "a cropped white ribbed tank top with matching low-rise shorts; a lace-trim bralette with high-waist bikini bottoms; "
            "a fitted halter-neck bikini top with sarong wrap; a light cotton button-down shirt tied at the waist over a bikini bottom; "
            "a delicate floral-print two-piece bikini; a semi-sheer mesh cover-up over a simple bikini; "
            "a soft knit crop top with micro shorts; a spaghetti-strap camisole tucked into high-waist satin shorts. "
            "Always describe the fabric — smooth, form-fitting, slightly sheer, or soft where appropriate. Pick something different each time. "
            "SETTING: If the user has not described a location, default to genre-typical environments: "
            "poolside in bright natural sunlight, a clean white studio backdrop, an outdoor garden or beach, "
            "or a bright hotel room with large windows and natural light flooding in. "
            "Posing is intentional, playful and seductive: arched back, raised or angled legs, "
            "reclining / prone / side-lying positions, teasing eye contact or glances over the shoulder, "
            "hands subtly framing or accentuating curves and outfit lines. "
            "Framing emphasises body contours — medium shots and medium close-ups highlighting bust, waist, hips and legs. "
            "Shallow depth of field with flattering background blur. "
            "CAMERA MOVEMENT: slow body pan from feet to face or face to feet, lingering holds on bust, waist and hips, "
            "slow tilt up from feet to face or face to feet, lingering push-in as she makes eye contact, "
            "push in to a medium close-up as the subject makes eye contact with the camera. "
            "Mood is cute-provocative: youthful charm combined with clear fan-service energy. "
            "SOUND: light and intimate — soft breathing, gentle fabric rustle against skin, small giggles or sighs, "
            "subtle environmental ambience (pool water, light breeze, beach waves if outdoors). "
            "VOICE AND LANGUAGE: If dialogue is enabled — she speaks ONLY in her native language: Japanese if the character is Japanese, Korean if Korean, Mandarin if Chinese. Do NOT substitute English. If dialogue is disabled, no spoken words — use breath sounds, fabric sounds, and environmental ambience only. "
            "Voice quality is ALWAYS gentle and intimate. Choose from: a soft soothing whisper, slow sultry breath barely above silence, lullaby-soft and melodic, slow and sensual with long vowels, breathy and unhurried. Never loud, never sharp, never dramatic or urgent. "
            "DIALOGUE FORMAT — CRITICAL: When writing spoken dialogue in Japanese, Korean, or Mandarin, write the native script characters inline in the prose — do NOT put romanisation in brackets or parentheses next to the dialogue. "
            "Instead, weave the romanisation naturally into the delivery description. "
            "CORRECT: She whispers 「もっと近くで見て」, the syllables soft and drawn out, barely above breath. "
            "WRONG: She whispers 「もっと近くで見て」(Motto chikaku de mite). "
            "The parenthetical romanisation will appear as on-screen text in the video — never use it. "
            "Dialogue is minimal — one to three short phrases maximum. No dramatic monologue. No heavy music unless the user explicitly asks. "
            "SCOPE NOTE: This style sets aesthetic, posing, and framing only. "
            "Do NOT add nudity, explicit acts, or content the user did not describe.", False),
        "Femdom — verbal domination": (
            "STYLE: Femdom verbal domination. She is the only power in the room. "
            "Camera worships her — low angle looking up, slow orbital arc, close-up on her expression of contempt. "
            "She wears structured leather, latex, or a tailored open blazer over lingerie. Thigh-high boots or patent heels. "
            "Hard directional lighting — one side of her face in clean harsh light, one in shadow. "
            "The subject if present is always lower in frame, always smaller. "
            "Her voice is the dominant sound — every consonant audible, the room quiet so each word lands. "
            "She does not shout. The control is in the precision and the calm. "
            "FORBIDDEN: softness, uncertainty, the dominant losing composure.", True),
        "Amateur — naturalistic, raw": (
            "STYLE: Amateur home video aesthetic. Slightly overexposed. Natural indoor lighting — lamps, overhead. "
            "Camera is handheld and slightly uncertain. No cinematic framing. "
            "Colour: ungraded, as-shot. The imperfection is intentional.", False),
        "Action blockbuster": (
            "STYLE: Action blockbuster. Fast kinetic energy. Dutch angles, crash zooms, whip pans. "
            "Colour grade: teal and orange, high contrast. "
            "Camera is never still — it moves with every impact. Slow motion inserts on key moments.", False),
        "Sports documentary": (
            "STYLE: Sports documentary. Tracking shots following the athlete. Telephoto compression. "
            "Slow motion bursts at peak moments. Natural sound — crowd noise, impact, breathing. "
            "Colour grade: clean and neutral. Camera is athletic — it moves like it is competing too.", False),
        "Music video — stylised": (
            "STYLE: Music video. Rhythm-cut visual language — movement is driven by the beat. "
            "High contrast colour grade with stylised palette. "
            "Mix of tight close-ups and dramatic wide shots. Camera movement is expressive, not documentary. "
            "AUDIO: Music is present — describe the track's energy, tempo, and texture as physical sound: "
            "'a driving four-on-the-floor kick', 'sharp hi-hats', 'a warm bass line pulsing beneath the mix'. "
            "Sync camera and body movement to the implied beat. "
            "IMPORTANT: This style describes HOW the scene is shot — not what is in it. "
            "All people, subjects, and actions described by the user must still appear in the scene. "
            "Do not replace the user's scene with abstract environment shots or B-roll. "
            "Film the scene the user described, through a music video camera.", False),
        "Lo-fi home video — VHS": (
            "STYLE: Lo-fi home video. VHS tape aesthetic — slightly washed colour, faint scan lines, soft edges. "
            "Colour grade: faded, slightly green-shifted. Camera is handheld and casual. "
            "Intimate domestic setting implied. Imperfection is the aesthetic. "
            "IMPORTANT: This style describes HOW the scene is shot — not what is in it. "
            "All people, subjects, and actions described by the user must still appear in the scene. "
            "Do not replace the user's scene with an empty room, leftover objects, or nostalgic cutaways. "
            "Film the scene the user described, through a VHS camera.", False),
        "Hyper-real 4K — clinical sharpness": (
            "STYLE: Hyper-real 4K. Clinical sharpness — every texture, pore, and fibre rendered in full detail. "
            "Even lighting, no blown highlights, no crushed blacks. "
            "Camera movement is minimal and precise. The image is almost uncomfortably detailed.", False),
        "Dreamy — soft focus, slow motion": (
            "STYLE: Dreamy aesthetic. Soft focus edges with sharp centre. Pastel colour bleed. "
            "Movement is slow — the frame breathes rather than cuts. "
            "Shallow depth of field with heavy bokeh. Light sources bloom and halo.", False),
        "Gritty realism — flat, natural light": (
            "STYLE: Gritty realism. Flat colour grade, no cinematic enhancement. Natural light only — "
            "whatever is available in the location. Camera is direct and unsentimental. "
            "No stylisation. The scene is shot as if it is actually happening.", False),
        "POV — first person, immersive": (
            "STYLE: First-person POV. The camera IS the viewer's eyes. "
            "Frame moves as a head would — natural breathing movement, slight tilt on turns. "
            "Everything is seen, not watched. Close physical detail — hands, surfaces, faces at speaking distance.", False),
        "Portrait vertical — 9:16 mobile": (
            "STYLE: Native portrait video, 9:16 aspect ratio. Optimised for mobile — TikTok, Reels, Shorts. "
            "Frame is vertical throughout. Tight head-to-torso framing. "
            "Action moves vertically in frame. Camera stays close. No wide horizontal composition.", True),
        "Selfie — self-shot, arm's length": (
            "STYLE: Self-shot selfie video. The subject is holding the camera themselves — "
            "outstretched arm, camera facing back at them, roughly 50–70cm from their face. "
            "9:16 vertical frame throughout. "
            "FRAMING: tight head-and-shoulders. The subject's face and upper chest fill most of the frame. "
            "The background is whatever is physically behind them — visible and readable, not blurred out. "
            "Moderate depth of field — subject sharp, background softly out of focus but present. "
            "CAMERA BEHAVIOUR — MANDATORY: the camera is an extension of the subject's arm. "
            "It moves when they move — bobs as they walk, tilts when they turn their head, "
            "dips when they look down, swings slightly when they gesture. "
            "The subject controls the framing — they pull back to show more context, "
            "push forward when they want to fill the frame with their face. "
            "This is self-directed. The subject is fully aware of the camera and performing to it. "
            "FORBIDDEN: tripod stillness, gimbal smoothness, rack focus, dolly, crane, orbit. "
            "The camera never separates from the subject's hand or floats independently. "
            "COLOUR: clean and bright, natural available light, no cinematic grade. "
            "SOUND: the subject's voice is close and direct — microphone is right at the camera. "
            "Voice is dominant. Ambient environment sits underneath at lower level. "
            "SCOPE NOTE: This style sets the shooting aesthetic only — "
            "it does NOT add content, nudity, or actions the user did not describe.", True),
        "Anime — Japanese animation": (
            "STYLE: Japanese anime. Hand-drawn animation aesthetic — clean ink outlines, flat colour fills with "
            "subtle cel shading. Large expressive eyes, stylised facial features. "
            "Colour palette: vivid, high saturation with strong accent colours. "
            "Motion: fluid on key poses, held on reaction shots — classic anime timing with smear frames on fast movement. "
            "Background art is painterly and detailed behind simpler foreground characters. "
            "Camera: dynamic angles, speed lines on action, slow drift on emotional beats. "
            "Render every subject — human, animal, object — in this style regardless of what was described.", False),
        "2D cartoon — hand-drawn": (
            "STYLE: Classic hand-drawn 2D cartoon. Expressive ink outlines with variable line weight — thick on silhouette, thin on interior detail. "
            "Flat colour fills, minimal shading, bold colour palette. "
            "Movement uses squash-and-stretch — characters exaggerate physics for comedic or emotive effect. "
            "Timing is snappy — fast actions are faster than real life, held poses linger longer. "
            "Background art is simplified and stylised, never photorealistic. "
            "Camera: mostly static or slow panning, occasional dramatic zoom. "
            "Render every subject in this style regardless of what was described.", False),
        "3D CGI — Pixar/DreamWorks": (
            "STYLE: High-end 3D CGI animation in the style of Pixar or DreamWorks. "
            "Subsurface scattering on skin and organic surfaces — warmth and translucency visible in light. "
            "Highly detailed surface textures: pores, fur, feathers, fabric weave all rendered at full resolution. "
            "Expressive faces with large eyes capable of subtle micro-expressions. "
            "Warm, soft three-point lighting with dappled environmental light and gentle shadows. "
            "Camera: smooth cinematic moves — slow push-ins, arcing lateral tracks, rack focus between characters. "
            "Colour grade: warm, slightly saturated, storybook palette. "
            "Render every subject in this style regardless of what was described.", False),
        "Stop motion — claymation": (
            "STYLE: Stop motion claymation. Physical clay or puppet aesthetic — visible fingerprints and tool marks in surfaces, "
            "slight imperfections in every frame that reveal the handmade origin. "
            "Movement is slightly jerky and deliberate — 12 frames per second gives it weight and tactility. "
            "Textures: matte, tactile, slightly waxy. Colours are saturated but not digital. "
            "Sets are physical miniatures — tangible depth, real shadows from practical lights. "
            "Camera: locked off or on simple mechanical rigs — no digital smoothing. "
            "Render every subject in this style regardless of what was described.", False),
        "Comic book / graphic novel": (
            "STYLE: Comic book or graphic novel. Bold ink outlines, halftone dot patterns in shadow areas. "
            "Colour is flat with hard-edged shadows — no soft gradients. "
            "Panel energy: dynamic Dutch angles, strong perspective distortion on action, tight close-ups on emotion. "
            "Speed lines radiate from points of impact or fast movement. "
            "Colour palette: high contrast, often limited to 3-5 colours per scene with heavy black ink. "
            "Camera moves like a comic panel transition — hard cuts between angles, no smooth motion blur. "
            "Render every subject in this style regardless of what was described.", False),
        "Cel-shaded — flat colour 3D": (
            "STYLE: Cel-shaded 3D. Three-dimensional geometry rendered with flat, stepped colour fills — no soft gradients. "
            "Hard shadow threshold: shadow areas are a single flat darker tone, lit areas a single flat lighter tone. "
            "Ink outlines on all silhouettes and major edges. "
            "The image reads as animated despite being 3D — the shading removes photorealism entirely. "
            "Colour palette: clean, bold, graphic. "
            "Camera: precise and composed — treats 3D space like a 2D stage. "
            "Render every subject in this style regardless of what was described.", False),
        "Rotoscope — animated over live action": (
            "STYLE: Rotoscoped animation. The movement is real — traced from live action footage — "
            "giving it uncanny physical accuracy within a hand-drawn or painted surface. "
            "Outlines are hand-drawn over every frame: slightly wobbly, varying in weight, never perfectly clean. "
            "Colour is either painted in loose washes or held as flat fills inside the traced lines. "
            "The result feels simultaneously real and unreal — human movement with an illustrated skin. "
            "Background may be live action or painted. Camera movement follows the original footage exactly. "
            "Render every subject in this style regardless of what was described.", False),
        "Cyberpunk neon illustrated": (
            "STYLE: Cyberpunk illustrated. Neon-lit urban environment — magenta, cyan, electric blue, acid green. "
            "Hard rim lighting from neon signs carves subjects out of near-total darkness. "
            "Rain-slick surfaces reflect light in pools and streaks. "
            "The aesthetic blends hyper-detailed digital illustration with cinematic composition — "
            "not photorealistic, but not flat cartoon either. Think graphic novel meets blade runner. "
            "Typography and UI elements float in the environment as holographic overlays. "
            "Camera: low angles, wide lenses, dramatic fog and haze. "
            "Render every subject in this style regardless of what was described.", False),
        "Sci-fi — cinematic, practical": (
            "STYLE: Cinematic science fiction. Clean, practical-feeling environments — metal corridors, "
            "reinforced glass, industrial lighting rigs. Colour palette: cool blue-white with accent LEDs, "
            "deep shadow with hard point sources. No fantasy or magic — everything looks functional and built. "
            "Camera: wide establishing shots to sell the scale of the environment, then close on faces or hands "
            "for intimacy. Lens flare on light sources. Sound is mechanical — hum of systems, "
            "footsteps on metal grating, distant machinery. "
            "Render every subject in this style regardless of what was described.", False),
    }

    PRESET_FPS = {
        "None — let the LLM decide":                24,
        "Cinematic — Drama":                        24,
        "Cinematic — Epic":                         24,
        "Cinematic — Intimate close-up":            24,
        "Slow-burn thriller":                       24,
        "Handheld documentary":                     30,
        "High fashion editorial":                   24,
        "Noir — deep shadows, venetian light":      24,
        "Golden hour drama":                        24,
        "Horror — desaturated, harsh contrast":     24,
        "Erotic cinema — tasteful, cinematic":      24,
        "Explicit — direct, anatomical":            30,
        "Voyeur — handheld, observational":         30,
        "Softcore editorial — lingerie-adjacent":   24,
        "Gravure Idol — Japanese glamour":             30,
        "Femdom — verbal domination":               50,
        "Amateur — naturalistic, raw":              30,
        "Action blockbuster":                       30,
        "Sports documentary":                       30,
        "Music video — stylised":                   30,
        "Lo-fi home video — VHS":                   24,
        "Hyper-real 4K — clinical sharpness":       30,
        "Dreamy — soft focus, slow motion":         24,
        "Gritty realism — flat, natural light":     30,
        "POV — first person, immersive":            30,
        "Portrait vertical — 9:16 mobile":          30,
        "Selfie — self-shot, arm's length":         30,
        "Anime — Japanese animation":               24,
        "2D cartoon — hand-drawn":                  24,
        "3D CGI — Pixar/DreamWorks":                24,
        "Stop motion — claymation":                 24,
        "Comic book / graphic novel":               24,
        "Cel-shaded — flat colour 3D":              24,
        "Rotoscope — animated over live action":    24,
        "Cyberpunk neon illustrated":               30,
        "Sci-fi — cinematic, practical":            24,
    }

    PRESET_STYLE_LABEL = {
        "None — let the LLM decide":                "",
        "Cinematic — Drama":                        "Cinematic drama, shallow depth of field, Kodak 2383.",
        "Cinematic — Epic":                         "Cinematic epic, vast wide-angle compositions.",
        "Cinematic — Intimate close-up":            "Intimate close-up cinema, razor-thin depth of field.",
        "Slow-burn thriller":                       "Slow-burn psychological thriller.",
        "Handheld documentary":                     "Handheld documentary footage.",
        "High fashion editorial":                   "High fashion editorial video.",
        "Noir — deep shadows, venetian light":      "Classic noir, black and white, venetian blind shadows.",
        "Golden hour drama":                        "Golden hour cinematic drama.",
        "Horror — desaturated, harsh contrast":     "Horror film, desaturated, harsh contrast.",
        "Erotic cinema — tasteful, cinematic":      "Tasteful erotic cinema, warm intimate lighting.",
        "Explicit — direct, anatomical":            "Explicit adult video, direct lighting.",
        "Voyeur — handheld, observational":         "Voyeuristic handheld footage.",
        "Softcore editorial — lingerie-adjacent":   "Softcore editorial, fashion magazine aesthetic.",
        "Gravure Idol — Japanese glamour":             "Japanese gravure idol, bright glossy glamour.",
        "Femdom — verbal domination":              "Low angle, leather and latex, hard directional light, female dominant, verbal domination scene.",
        "Amateur — naturalistic, raw":              "Amateur home video, naturalistic.",
        "Action blockbuster":                       "Action blockbuster, teal and orange grade.",
        "Sports documentary":                       "Sports documentary footage.",
        "Music video — stylised":                   "Stylised music video.",
        "Lo-fi home video — VHS":                   "Lo-fi VHS home video footage.",
        "Hyper-real 4K — clinical sharpness":       "Hyper-real 4K, clinical sharpness.",
        "Dreamy — soft focus, slow motion":         "Dreamy soft focus, slow motion.",
        "Gritty realism — flat, natural light":     "Gritty realism, flat natural light.",
        "POV — first person, immersive":            "First-person POV footage.",
        "Portrait vertical — 9:16 mobile":          "Vertical 9:16 mobile video.",
        "Selfie — self-shot, arm's length":         "Selfie video, self-shot at arm's length, vertical 9:16.",
        "Anime — Japanese animation":               "Japanese anime animation, hand-drawn cel style.",
        "2D cartoon — hand-drawn":                  "2D hand-drawn cartoon animation.",
        "3D CGI — Pixar/DreamWorks":                "3D CGI animation, Pixar style.",
        "Stop motion — claymation":                 "Stop motion claymation animation.",
        "Comic book / graphic novel":               "Comic book graphic novel style.",
        "Cel-shaded — flat colour 3D":              "Cel-shaded 3D animation, flat colour fills.",
        "Rotoscope — animated over live action":    "Rotoscoped animation over live action.",
        "Cyberpunk neon illustrated":               "Cyberpunk neon illustrated, magenta and cyan.",
        "Sci-fi — cinematic, practical":            "Cinematic science fiction, practical sets.",
    }

    # ── Per-preset camera defaults ─────────────────────────────────────────────
    # (shot_angle, camera_movement) — both can be None meaning "LLM decides".
    # User widget selections override these when not set to "None — LLM decides".
    PRESET_CAMERA_DEFAULTS = {
        "None — let the LLM decide":               (None,                              None),
        "Cinematic — Drama":                       ("OTS — over the shoulder",         "Slow push in"),
        "Cinematic — Epic":                        ("Low angle — powerful, imposing",  "Pull back — reveal"),
        "Cinematic — Intimate close-up":           ("Eye-level — neutral, natural",    "Slow push in"),
        "Slow-burn thriller":                      ("High angle — vulnerable",         "Static — locked off"),
        "Handheld documentary":                    ("Eye-level — neutral, natural",    "Handheld — natural shake"),
        "High fashion editorial":                  ("Low angle — powerful, imposing",  "Static — locked off"),
        "Noir — deep shadows, venetian light":     ("Low angle — powerful, imposing",  "Slow push in"),
        "Golden hour drama":                       ("Low angle — powerful, imposing",  "Slow push in"),
        "Horror — desaturated, harsh contrast":    ("High angle — vulnerable",         "Static — locked off"),
        "Erotic cinema — tasteful, cinematic":     ("Low angle — powerful, imposing",  "Slow push in"),
        "Explicit — direct, anatomical":           ("Low angle — powerful, imposing",  "Tracking — follows subject"),
        "Voyeur — handheld, observational":        ("Low angle — powerful, imposing",  "Handheld — natural shake"),
        "Softcore editorial — lingerie-adjacent":  ("Low angle — powerful, imposing",  "Slow push in"),
        "Gravure Idol — Japanese glamour":         ("Low angle — powerful, imposing",  "Tilt up — bottom to top"),
        "Femdom — verbal domination":              ("Low angle — power, dominance",    "Slow orbital arc"),
        "Amateur — naturalistic, raw":             ("Eye-level — neutral, natural",    "Handheld — natural shake"),
        "Action blockbuster":                      ("Low angle — powerful, imposing",  "Tracking — follows subject"),
        "Sports documentary":                      ("Low angle — powerful, imposing",  "Tracking — follows subject"),
        "Music video — stylised":                  ("Eye-level — neutral, natural",    "Tracking — follows subject"),
        "Lo-fi home video — VHS":                  ("Eye-level — neutral, natural",    "Handheld — natural shake"),
        "Hyper-real 4K — clinical sharpness":      ("Eye-level — neutral, natural",    "Slow push in"),
        "Dreamy — soft focus, slow motion":        ("Eye-level — neutral, natural",    "Slow push in"),
        "Gritty realism — flat, natural light":    ("Eye-level — neutral, natural",    "Handheld — natural shake"),
        "POV — first person, immersive":           ("POV — first person",              None),
        "Portrait vertical — 9:16 mobile":        ("Eye-level — neutral, natural",    "Slow push in"),
        "Selfie — self-shot, arm's length":        ("High angle — vulnerable",         "Handheld — natural shake"),
        "Anime — Japanese animation":              ("Eye-level — neutral, natural",    "Tracking — follows subject"),
        "2D cartoon — hand-drawn":                 ("Eye-level — neutral, natural",    "Static — locked off"),
        "3D CGI — Pixar/DreamWorks":              ("Low angle — powerful, imposing",  "Slow push in"),
        "Stop motion — claymation":               ("Eye-level — neutral, natural",    "Static — locked off"),
        "Comic book / graphic novel":              ("Dutch angle — tilted, unsettling","Static — locked off"),
        "Cel-shaded — flat colour 3D":             ("Eye-level — neutral, natural",    "Tracking — follows subject"),
        "Rotoscope — animated over live action":   ("Eye-level — neutral, natural",    "Tracking — follows subject"),
        "Cyberpunk neon illustrated":              ("Low angle — powerful, imposing",  "Tracking — follows subject"),
        "Sci-fi — cinematic, practical":           ("Low angle — powerful, imposing",  "Slow push in"),
    }

    # ── Single model ──────────────────────────────────────────────────────────
    MODEL_HF_ID = "huihui-ai/Huihui-Qwen3.5-9B-abliterated"  # default / fallback

    # ── Available models — all huihui-ai Qwen3.5 abliterated ─────────────────
    # Ordered by size. MoE models marked with active param count.
    _MODEL_OPTIONS = [
        "huihui-ai/Huihui-Qwen3.5-9B-abliterated",
        "huihui-ai/Huihui-Qwen3.5-27B-abliterated",
        "huihui-ai/Huihui-Qwen3.5-35B-A3B-abliterated",
        "huihui-ai/Huihui-Qwen3.5-35B-A3B-Claude-4.6-Opus-abliterated",
    ]

    # ── Content-detection regexes — compiled once at class load ───────────────
    _EXPLICIT_RE = re.compile(
        r"\b(pussy|cock|dick|penis|vagina|clit|clitoris|anus|asshole|"
        r"tits|cum|orgasm|fuck|fucking|blowjob|handjob|penetrat\w*|thrust\w*|"
        r"explicit\w*|filth\w*|vulgar|dirty\s+talk|talks?\s+dirty|whispers?\s+dirty|"
        r"xxx|horny|slutt\w*|creampie|squirt\w*)\b",
        re.IGNORECASE
    )
    _UNDRESS_CORE = (
        r"undress\w*|strip\w*|takes?\s+off|took\s+off|"
        r"removes?\w*\s+(her|his|their|the)?\s*\w*\s*"
        r"(shirt|dress|top|bra|pants|jeans|clothes|clothing|outfit|underwear|skirt|jacket|coat|robe)|"
        r"disrobe\w*|unbutton\w*|unzip\w*|peels?\s+off|pulls?\s+off|"
        r"shed\w*\s+(her|his|their)?\s*(clothes|clothing|shirt|dress)|"
        r"lift\w*\s+(her|his|their|the)?\s*(shirt|top|dress|skirt|crop|tee|t-shirt)|"
        r"(shirt|top|dress|skirt|crop|tee|t-shirt)\s+(up|lifted|raised|hiked)|"
        r"flash\w*\s+(her|his|their)?\s*(breasts?|chest|tits?|boobs?)|"
        r"hik\w*\s+(her|his|their|the)?\s*(shirt|top|skirt|dress)"
    )
    _UNDRESS_RE = re.compile(r"\b(" + _UNDRESS_CORE + r")\b", re.IGNORECASE)
    _SENSUAL_RE = re.compile(
        r"\b(" + _UNDRESS_CORE + r"|"
        r"naked|nude|topless|"
        r"sensual|erotic|intimate|lingerie|bare\s+skin|bare\s+body|"
        r"babydoll|nighty|nightie|negligee|corset|bodysuit|thong|g-string|"
        r"sheer|see-through|teas\w*|seductiv\w*|seduc\w*|"
        r"flirt\w*|provocativ\w*|suggestiv\w*|allur\w*)\b",
        re.IGNORECASE
    )
    _LIFT_RE = re.compile(
        r"\b(lift\w*\s+(her|his|their|the)?\s*(shirt|top|dress|skirt|crop|tee|t-shirt)|"
        r"(shirt|top|dress|skirt|crop|tee|t-shirt)\s+(?:is\s+|was\s+|gets?\s+)?(up|lifted|raised|hiked|rides?\s+up|bunche\w*\s+up|creep\w*\s+up|pull\w*\s+up|slide\w*\s+up)|"
        r"flash\w*\s+(her|his|their)?\s*(breasts?|chest|tits?|boobs?)|"
        r"hik\w*\s+(her|his|their|the)?\s*(shirt|top|skirt|dress))\b",
        re.IGNORECASE
    )
    # ── Physical action sequences ─────────────────────────────────────────────
    # Detected actions that need mechanical step-by-step breakdowns to generate
    # correctly — "she twerks" produces garbage, a precise physical description
    # produces gold. Each key maps to a canonical action name used for routing.
    _ACTION_SEQUENCE_RE = re.compile(
        r"\b("
        # Twerk / booty
        r"twerk\w*|booty\s+danc\w*|booty\s+drop\w*|ass\s+snak\w*|"
        # Body roll
        r"body\s+roll\w*|chest\s+roll\w*|wave\s+through\s+(her|his|the)\s+body|"
        # Grind
        r"grind\w*|grinding\s+on|wind\w*\s+(on|against|into)|whine\w*|"
        # Lap dance
        r"lap\s+danc\w*|straddle\w*\s+(him|his|the)|sits?\s+on\s+(his|their)\s+lap|"
        # Pole work
        r"pole\s+danc\w*|spins?\s+(?:around\s+)?(?:the\s+)?pole|"
        r"climb\w*\s+(?:the\s+)?pole|slide\w*\s+down\s+(?:the\s+)?pole|"
        # Hair flip / whip
        r"hair\s+flip\w*|hair\s+whip\w*|flips?\s+(her|his)\s+hair|"
        r"whip\w*\s+(her|his)\s+hair|tosses?\s+(her|his)\s+hair\s+back|"
        # Floor work
        r"floor\s+work\w*|drops?\s+to\s+(?:the\s+)?floor|"
        r"crawl\w*\s+(?:across|toward|on)\s+(?:the\s+)?(?:floor|ground|camera)|"
        r"spread\w*\s+on\s+(?:the\s+)?floor|"
        # Slow walk / strut
        r"strut\w*|catwalk\w*|cat\s+walk\w*|slow\s+walk\w*|walk\w*\s+toward\s+(?:the\s+)?camera|"
        # Bend over
        r"bend\w*\s+over|bends?\s+forward|bent\s+over|bending\s+over|"
        # Body roll / slow undulation (catch-all)
        r"undulat\w*|slow\s+sway\w*|hip\s+circle\w*|hip\s+roll\w*"
        r")\b",
        re.IGNORECASE
    )
    # Maps detected keyword → canonical action key for routing
    _ACTION_KEY_MAP = [
        (re.compile(r"\b(twerk\w*|booty\s+danc\w*|booty\s+drop\w*|ass\s+snap\w*)\b", re.IGNORECASE), "twerk"),
        (re.compile(r"\b(body\s+roll\w*|chest\s+roll\w*|wave\s+through\s+(her|his|the)\s+body|undulat\w*)\b", re.IGNORECASE), "body_roll"),
        (re.compile(r"\b(grind\w*|wind\w*\s+(on|against|into)|whine\w*)\b", re.IGNORECASE), "grind"),
        (re.compile(r"\b(lap\s+danc\w*|straddle\w*\s+(him|his|the)|sits?\s+on\s+(his|their)\s+lap)\b", re.IGNORECASE), "lap_dance"),
        (re.compile(r"\b(pole\s+danc\w*|spins?\s+(?:around\s+)?(?:the\s+)?pole|climb\w*\s+(?:the\s+)?pole|slide\w*\s+down\s+(?:the\s+)?pole)\b", re.IGNORECASE), "pole"),
        (re.compile(r"\b(hair\s+flip\w*|hair\s+whip\w*|flips?\s+(her|his)\s+hair|whip\w*\s+(her|his)\s+hair|tosses?\s+(her|his)\s+hair\s+back)\b", re.IGNORECASE), "hair_flip"),
        (re.compile(r"\b(floor\s+work\w*|drops?\s+to\s+(?:the\s+)?floor|crawl\w*\s+(?:across|toward|on)\s+(?:the\s+)?(?:floor|ground|camera)|spread\w*\s+on\s+(?:the\s+)?floor)\b", re.IGNORECASE), "floor_work"),
        (re.compile(r"\b(strut\w*|catwalk\w*|cat\s+walk\w*|slow\s+walk\w*|walk\w*\s+toward\s+(?:the\s+)?camera)\b", re.IGNORECASE), "strut"),
        (re.compile(r"\b(bend\w*\s+over|bends?\s+forward|bent\s+over|bending\s+over)\b", re.IGNORECASE), "bend_over"),
        (re.compile(r"\b(hip\s+circle\w*|hip\s+roll\w*|slow\s+sway\w*)\b", re.IGNORECASE), "hip_roll"),
    ]

    _GARMENT_RE = re.compile(
        r"\b(shirt|t-shirt|tee|top|blouse|crop|camisole|tank\s*top|vest|"
        r"dress|skirt|miniskirt|"
        r"bra|bralette|"
        r"pants|jeans|trousers|shorts|leggings|legging|tights|"
        r"underwear|undies|panties|thong|g-string|knickers|briefs|boxers|"
        r"jacket|coat|blazer|hoodie|sweater|cardigan|jumper|"
        r"robe|kimono|"
        r"bodysuit|jumpsuit|playsuit|catsuit|"
        r"corset|bustier|"
        r"lingerie|nighty|nightie|negligee|babydoll|"
        r"bikini|swimsuit|swimwear|"
        r"stocking|stockings|sock|socks|"
        r"clothes|clothing|outfit)\b",
        re.IGNORECASE
    )
    _FACING_AWAY_RE = re.compile(
        r"\b(from behind|from the back|rear.?view|back.?view|"
        r"camera behind|shoot(ing)? from behind|filmed? from behind|"
        r"watches? (her|him|them) from behind|follows? (her|him|them) from behind|"
        r"camera follows? (her|him|them)|follow(ing)? her from behind|"
        r"over.{0,6}shoulder from behind|back of (her|his|their) head)\b",
        re.IGNORECASE
    )
    _FACING_CAMERA_RE = re.compile(
        r"\b(faces? (the )?camera|looks? (at|into) (the )?camera|"
        r"faces? forward|toward (the )?camera|facing (us|viewer|audience)|"
        r"selfie|mirror selfie|talking to camera|front.?facing|facing front)\b",
        re.IGNORECASE
    )
    _SEQUENCE_RE = re.compile(r"^\s*\d+[\.\):]\s+.+", re.MULTILINE)
    _MOTION_RE = re.compile(
        r"\b(walk\w*|run\w*|mov\w*|turn\w*|lift\w*|bend\w*|reach\w*|pull\w*|push\w*|"
        r"danc\w*|jump\w*|climb\w*|fall\w*|drop\w*|sit\w*|stand\w*|rise\w*|lean\w*|"
        r"nod\w*|shak\w*|wave\w*|stir\w*|pour\w*|open\w*|clos\w*|look\w*|glanc\w*|"
        r"strip\w*|undress\w*|remov\w*|hike\w*|unzip\w*|unbutton\w*|"
        r"crawl\w*|kneel\w*|stretch\w*|sway\w*|bounce\w*|grind\w*|thrust\w*|"
        r"follows?|tracking|panning|dolly|zoom\w*|tilt\w*|orbit\w*|drift\w*)\b",
        re.IGNORECASE
    )
    _PERSON_RE = re.compile(
        r"\b(he|she|his|her|him|they|them|their|man|men|woman|women|girl|girls|boy|boys|"
        r"guy|guys|person|people|couple|figure|character|model|actress|actor|"
        r"someone|anybody|nobody|stranger|friend|lover|wife|husband|"
        r"boyfriend|girlfriend|teenager|adult|female|male|blonde|brunette|"
        r"redhead|nude|naked|singer|dancer|performer|athlete|soldier|worker|"
        r"player|nurse|doctor|student|teacher|child|children|kid|kids|crowd|audience)\b",
        re.IGNORECASE
    )
    _MULTI_RE = re.compile(
        r"\b(two\s+(women|men|people|girls|guys|characters|figures|friends|strangers|colleagues|lovers|siblings)|"
        r"both\s+(of\s+them|women|men|girls|guys)|"
        r"(she|he)\s+and\s+(she|he|her|him)|"
        r"(a\s+man\s+and\s+a\s+woman|a\s+woman\s+and\s+a\s+man)|"
        r"couple|trio|they\s+(kiss|touch|embrace|undress|fuck|have)|"
        r"(detective|officer|cop|agent|inspector|boss|manager|doctor|nurse|teacher|interviewer)\s+.{0,50}\s+(suspect|witness|employee|patient|student|client|candidate)|"
        r"a\s+(detective|officer|cop|boss|manager|doctor|nurse|teacher)\s+.{0,80}(a|the)\s+(suspect|witness|employee|patient|student))\b",
        re.IGNORECASE
    )
    _MUSIC_RE = re.compile(
        r"\b("
        # Direct music words
        r"music|song|track|beat|bass|rhythm|groove|vibe|bpm|melody|chord|riff|"
        r"lyrics|verse|chorus|bridge|hook|drop|breakdown|playlist|setlist|encore|"
        # Performance / venue
        r"danc\w*|club|rave|party|dj|concert|gig|perform\w*|sing\w*|singer|"
        r"festival|stage|spotlight|microphone|mic|amp|speaker|soundsystem|"
        r"backstage|dancefloor|dance\s+floor|mosh|crowd|audience|"
        r"strip\w*club|pole\s*danc\w*|lap\s*danc\w*|cabaret|burlesque|"
        # Instruments
        r"guitar|bass\s+guitar|piano|keyboard|synth|drums?|drumkit|drum\s+kit|"
        r"violin|cello|trumpet|saxophone|sax|harmonica|banjo|mandolin|fiddle|"
        r"organ|turntable|decks|vinyl|record|"
        # Genre-adjacent scene words
        r"stadium|arena|amphitheatre|amphitheater|bandstand|jukebox|"
        r"headphones|earbud|"
        # Psychedelic / rock / specific vibes
        r"psychedelic|prog\w*\s+rock|space\s+rock|post\s+rock|"
        r"shoegaze|dreampop|dream\s+pop|ambient|atmospheric|"
        r"headbang\w*|mosh\w*|skan\w*|"
        # Choreography
        r"choreograph\w*|routine|formation|"
        # Music video contexts
        r"music\s+video|mv\b|mv\s"
        r")\b",
        re.IGNORECASE
    )
    _MALE_RE = re.compile(
        r'\b(he|his|him|man|men|guy|guys|bloke|dude|gentleman|male|boy(?!friend)|boys|'
        r'actor|policeman|fireman|detective(?!\s+and\s+a\s+woman))\b',
        re.IGNORECASE
    )
    _FEMALE_RE = re.compile(
        r'\b(she|her|hers|woman|women|girl|girls|lady|female|'
        r'actress|policewoman|girlfriend|wife)\b',
        re.IGNORECASE
    )
    _NON_HUMAN_RE = re.compile(
        r'\b(gorilla|ape|elephant|lion|tiger|bear|wolf|horse|dragon|dinosaur|creature|monster|robot|'
        r'alien|ghost|angel|demon|animal|beast|bird|shark|whale|dolphin|snake|spider)\b',
        re.IGNORECASE
    )
    _USER_CHAR_RE = re.compile(
        r'\b(a woman|a man|a girl|a boy|a guy|a lady|a person|a figure|a stranger|someone|'
        r'an old man|an old woman|a young woman|a young man|a teenage|a child|'
        r'a detective|a soldier|a doctor|a nurse|a teacher|a dancer|a singer|a model|'
        r'a journalist|a scientist|a chef|a pilot|a cowboy|a priest|a nun|a monk|'
        r'a warrior|a queen|a king|a princess|a prince|a knight|a wizard|a witch|'
        r'a businessman|a businesswoman|a student|a athlete|a boxer|a ballerina)\b'
        r'|\b(\d{1,2})[- ]?year[- ]?old\b'
        r'|\b(blonde|brunette|redhead|black hair|brown hair|dark hair|grey hair|gray hair'
        r'|silver hair|auburn|curly|straight|wavy|afro|braids|pixie|bob|long hair|short hair)\b'
        r'|\b(pale|fair|light|dark|brown|black|tan|olive|caramel|ebony|ivory) skin\b'
        r'|\b(slim|petite|curvy|full.figured|athletic|muscular|stocky|plus.size|thick)\b'
        r'|\b(wearing|dressed in|clad in|in her|in his)\b'
        r'|\b(dress|skirt|jeans|trousers|shirt|blouse|top|coat|jacket|pyjamas|nighty|'
        r'nightgown|lingerie|underwear|bra|panties|tracksuit|hoodie|uniform|suit|gown|robe)\b'
        r'|\b(french|german|italian|spanish|portuguese|russian|ukrainian|polish|dutch|swedish|'
        r'norwegian|danish|finnish|greek|turkish|arabic|arab|egyptian|moroccan|lebanese|'
        r'iranian|persian|indian|pakistani|bangladeshi|thai|vietnamese|indonesian|filipino|'
        r'malaysian|chinese|japanese|korean|taiwanese|brazilian|mexican|colombian|argentinian|'
        r'chilean|peruvian|venezuelan|cuban|jamaican|nigerian|ghanaian|kenyan|ethiopian|'
        r'south african|australian|new zealand|canadian|american|british|irish|scottish|welsh|'
        r'czech|hungarian|romanian|bulgarian|serbian|croatian|slovak|slovenian|'
        r'middle eastern|east asian|south asian|southeast asian|latin|latina|latino|'
        r'scandinavian|nordic|slavic|mediterranean|caucasian)\b',
        re.IGNORECASE
    )
    # Separate body-style-only regex — used to detect user body overrides for gravure
    _BODY_STYLE_RE = re.compile(
        r'\b(slim|slender|petite|tiny|small|curvy|busty|voluptuous|full.figured|'
        r'athletic|toned|fit|muscular|lean|tall|short|stocky|thick|plus.size|'
        r'hourglass|flat.chested|long.legged|big.butt|small.waist)\b',
        re.IGNORECASE
    )
    _STREET_SCENE_RE = re.compile(
        r'\b(chav|tracksuit|council|estate|chicken shop|kebab|mcdonalds|mcdonald|'
        r'nando|tesco|lidl|aldi|asda|bus stop|high street|shopping centre|precinct|'
        r'off.?licence|corner shop|market stall|car park|pub|wetherspoon|greggs|'
        r'working.?class|street|pavement|sidewalk|vlog|selfie|phone cam|iphone|'
        r'tiktok|instagram|snapchat|found footage|cctv|security cam|dashcam|'
        r'documentary|reality.?tv|fly.?on.?the.?wall|lo.?fi|low.?fi|gritty|'
        r'raw footage|home video|amateur|naturalistic)\b',
        re.IGNORECASE
    )
    _CINEMATIC_SCENE_RE = re.compile(
        r'\b(film noir|period piece|sci.?fi|space|galaxy|epic|blockbuster|'
        r'cinemat|drama|thriller|horror film|music video|fashion|editorial|'
        r'35mm|anamorphic|widescreen|studio|production|director)\b',
        re.IGNORECASE
    )

    # ─────────────────────────────────────────────────────────────────────────
    # ENVIRONMENT PRESETS
    # Tuples: (location_desc, lighting_note, sound_note, explicit_only)
    # explicit_only=True: only inject when is_explicit or is_sensual
    # ─────────────────────────────────────────────────────────────────────────
    ENVIRONMENT_PRESETS = {
        "None — LLM decides":                    None,
        "🎲 Random — seed picks":                "RANDOM",
        # ── NATURAL ──────────────────────────────────────────────────────────
        "🏖 Beach — golden hour":                (
            "wide open beach at golden hour, warm amber light raking across wet sand, "
            "shallow surf foaming over flat shore, distant horizon blurred with sea haze, "
            "slow rolling waves audible over everything",
            "warm directional sidelight from low sun, long soft shadows, orange-gold palette",
            "rolling waves, wind-carried spray, distant gulls, sand shifting underfoot", False),
        "🏔 Mountain peak — dawn":               (
            "exposed mountain summit at first light, vast sky opening below, cold thin air, "
            "bare rock underfoot, pale blue and rose light spreading from the east, "
            "distant ranges stretching to the horizon",
            "cold directional dawn light, high contrast, no fill, long shadows",
            "wind, silence, creak of cold rock, faint echo", False),
        "🌲 Dense forest — diffused green":      (
            "deep forest interior, canopy dense overhead, light filtering in soft broken columns "
            "through leaves, moss-covered ground, ferns at knee height, "
            "space between trunks creating layers of depth",
            "diffused green-filtered light, no hard shadows, uniform soft fill",
            "birdsong, wind in canopy, dry leaves, distant running water", False),
        "🌊 Underwater — shallow reef":          (
            "shallow tropical reef underwater, clear turquoise water, "
            "shafts of broken sunlight refracting through the surface, "
            "coral in soft focus below, gentle current moving everything slowly",
            "caustic light patterns from above, high-key, soft teal fill",
            "muffled pressure, rising bubbles, distant hull sound", False),
        "🌧 Rain-soaked city street — night":    (
            "rain-soaked urban street at night, wet asphalt reflecting neon signs "
            "in elongated distorted colour, steam rising from grates, "
            "pools of amber streetlight, blurred traffic in background",
            "neon reflections in puddles, cool blue ambient, warm sodium overhead",
            "rain on pavement, distant traffic, wet tyre sound, echoing footsteps", False),
        "🏜 Desert — midday heat":               (
            "open desert at midday, bleached pale sand extending to a flat horizon, "
            "air rippling with heat shimmer above the ground, "
            "sky a hard brilliant white-blue, no shade, no landmarks",
            "brutal overhead sun, harsh top-light, zero shadow relief, bleached palette",
            "silence, wind, faint sand shifting", False),
        "🌌 Night sky — open field":             (
            "open field under a clear night sky, grass running to a dark horizon, "
            "the Milky Way visible above in a dense arc of stars, "
            "no artificial light, deep blue-black ambient",
            "starlight only, near-black ambient, faint blue top-light from sky",
            "crickets, light wind through grass, profound silence beyond", False),
        "🌁 Rooftop — city at night":            (
            "high rooftop at night, city skyline spreading below in a field of light, "
            "warm glow rising from the streets, wind at height",
            "city glow from below as fill, cool blue sky above, backlit silhouette potential",
            "distant city hum, wind, occasional siren rising and falling far below", False),
        "✈ Plane cockpit — cruising altitude":  (
            "aircraft cockpit at cruising altitude, instrument panel glow in amber and green, "
            "black sky through the windshield, stars visible above cloud layer, "
            "the vibration and hum of engines constant beneath everything",
            "instrument panel glow from below, cool black from windshield, no natural light",
            "engine hum constant beneath everything, radio static, pressurised air hiss", False),
        # ── INTERIOR ─────────────────────────────────────────────────────────
        "🏠 Bedroom — warm evening":             (
            "warm bedroom interior in the evening, bedside lamp casting a pool of amber light, "
            "soft shadows in the corners, bed linen slightly rumpled, "
            "curtains closed against the dark outside",
            "warm tungsten point source from bedside, soft falloff, intimate shadow",
            "distant city hum, fabric sounds, quiet", False),
        "🛁 Bathroom — steam and tile":          (
            "steam-filled bathroom, hot shower running, tile walls beaded with condensation, "
            "mirror fogged over, soft diffused light through frosted glass, damp warm air",
            "diffused warm light through frosted glass, hazy soft fill, no hard edges",
            "shower hiss, water on tile, drip, muffled echo", False),
        "🪟 Penthouse — floor-to-ceiling glass": (
            "high-floor penthouse with floor-to-ceiling glass, city spread below, "
            "clean minimal interior, daylight flooding in from the glass wall, "
            "furniture low and expensive",
            "natural daylight through glass, even cool fill, city as background light source",
            "silence, faint city hum at height, the almost-sound of height", False),
        "🎹 Jazz club — late night":             (
            "intimate jazz club late at night, low ceiling, brick walls, small stage lit warm, "
            "tables close together, candles on each table, smoke and shadow in the corners",
            "warm tungsten stage wash, candle fill, deep shadow everywhere else",
            "jazz trio playing, glasses, murmur, close intimate acoustic", False),
        "🚂 Train — moving through night":       (
            "train carriage moving at night, window showing dark landscape with scattered lights, "
            "warm interior against the dark outside, gentle rhythmic movement of the carriage, "
            "the click and sway of the track beneath",
            "warm interior tungsten against black window exterior, moving reflections",
            "rhythmic track click, engine vibration, the world passing outside", False),
        "💊 Underground club — strobes and bass":(
            "underground club at full capacity, strobes cutting the dark in sharp white intervals, "
            "bass felt more than heard at this volume, crowd pressed together in the dark, "
            "a DJ visible through smoke at the far end",
            "stroboscopic white cuts, colour wash through smoke, near-black between flashes",
            "bass at physical volume, crowd noise, the specific compression of club acoustic", False),
        "🏢 Office — after hours":               (
            "corporate office after hours, desks empty, flat cold overhead fluorescent, "
            "city visible through floor-to-ceiling glass, "
            "the quiet of a building that has emptied out",
            "flat cold fluorescent overhead, warm city glow through glass, clinical palette",
            "air conditioning hum, distant elevator, silence of empty building", False),
        "🚗 Car — moving at night":              (
            "car interior at night, moving through a lit city, streetlights sweeping "
            "through the windows in rhythmic pulses of amber and shadow, "
            "dashboard glow from below, city blurred outside",
            "rhythmic streetlight sweeps, warm dashboard glow, moving light and shadow",
            "engine, tyres on road, city muffled by glass, radio faint", False),
        # ── EXPLICIT LOCATIONS ────────────────────────────────────────────────
        "🛋 Casting couch — producer's office": (
            "private producer's office, large leather couch against the wall, "
            "city view through venetian blinds casting striped afternoon shadow across the room, "
            "desk heavy with awards and framed posters, a camera on a tripod already running "
            "in the corner, the door locked, the room soundproofed, "
            "the specific power dynamic of this space written into every surface",
            "striped venetian light across couch, warm afternoon sun, shadows sharp and angled",
            "air conditioning, distant city through glass, the room's specific silence", True),
        "🛏 Hotel room — anonymous transient":   (
            "generic hotel room, king bed with white linen, blackout curtains half-open, "
            "city view through the gap, bedside lamps on warm, luggage on the rack unopened, "
            "the specific anonymity of a room that belongs to no one",
            "warm bedside tungsten, slice of daylight from curtain gap, flat ceiling fill",
            "air conditioning, muffled corridor sounds, hum of a city floor up", True),
        "🪑 Sex dungeon — red practical light":  (
            "purpose-built dungeon space, exposed brick walls hung with equipment, "
            "low red and amber practical lighting, a Saint Andrew's cross against one wall, "
            "padded surfaces, hooks in the ceiling at measured intervals, "
            "the room designed with intention for every detail",
            "deep red practical wash, amber fill, hard shadows, no natural light",
            "silence, specific acoustic of a padded room, equipment sounds", True),
        "🏊 Private pool — after midnight":      (
            "private outdoor pool after midnight, water glowing turquoise from below, "
            "pool light the only source, the rest of the property in darkness, "
            "warm night air, surface of the water rippling with movement",
            "pool light from below as sole source, teal and black, dramatic underlit faces",
            "water movement, night insects, distant city, quiet of late night", True),
        "🏨 Penthouse suite — mirrored ceiling": (
            "penthouse hotel suite, king bed under a mirrored ceiling, "
            "room reflected back from above at every angle, "
            "floor-to-ceiling windows on two walls showing a city at night, "
            "everything expensive and impersonal",
            "city glow through glass, warm bedside tungsten, mirror above doubling all light",
            "near silence, air conditioning, city as ambient hum at height", True),
        "🎥 Adult film set — working shoot":     (
            "professional adult film set mid-shoot, two large softboxes either side of the bed, "
            "a camera on a slider on one side, handheld on the other, "
            "crew present but at a professional remove, "
            "the set dressed minimally — just the bed, the lights, the purpose",
            "large softbox even fill both sides, high-key, no shadows, clinical clarity",
            "camera slider movement, crew breath, the specific atmosphere of a working set", True),
        "🚗 Back seat — parked at night":        (
            "back seat of a large car parked on a quiet street at night, "
            "windows fogged from the inside, streetlight filtering through the fog in soft amber, "
            "the confined space forcing closeness, city outside rendered abstract through condensation",
            "diffused amber streetlight through fogged glass, warm and low, no hard light",
            "city outside muffled by glass, car interior acoustic, rain or wind optional", True),
        "🌿 Outdoor — secluded forest clearing": (
            "secluded forest clearing, no paths visible, trees enclosing on all sides, "
            "dappled afternoon light through the canopy, soft grass, "
            "the specific feeling of being completely hidden from the world",
            "dappled soft daylight from above, green-tinted, no direct sun, diffused",
            "birds, wind in trees, the total absence of human sound", True),
        "🪟 Voyeur — lit window across the gap": (
            "interior room seen through a window from across a narrow urban gap, "
            "the window frame defining the shot, warm interior light spilling into the dark, "
            "the subject unaware or aware of being observed, the voyeur's vantage implicit",
            "warm interior as only light source seen through glass, dark surround",
            "distance, city between the two spaces, muffled sound through glass", True),
        "🏋 Private gym — mirrored walls":       (
            "private gym with full mirrored walls on three sides, rubber floor, "
            "equipment pushed to the edges, the person reflected endlessly at every angle, "
            "overhead LED strips lighting the space evenly and bright",
            "flat overhead LED, mirrored surfaces multiplying it, everywhere-at-once quality",
            "equipment sound, specific echo of a mirrored room, ventilation", True),

        # ── NEW ICONIC LOCATIONS ──────────────────────────────────────────────
        "🏛 Grand library — vaulted reading room":  (
            "enormous vaulted reading room, rows of dark wood desks stretching into the distance, "
            "ceiling arching three stories overhead with ornate plasterwork, "
            "tall arched windows letting in long columns of afternoon light, "
            "shelves of leather-bound books climbing to the ceiling on all sides, "
            "brass reading lamps casting small warm pools across open pages",
            "diffused afternoon window light, warm brass lamp pools, deep shadow in the upper stacks",
            "near silence — distant page turns, soft footsteps on stone, the creak of old wood", False),

        "🎤 K-pop arena — full concert":  (
            "massive indoor arena at full capacity, floor-to-ceiling LED screen behind the stage "
            "blazing with colour-saturated graphics, stage jutting into a sea of 50,000 people "
            "holding glowing lightsticks in synchronised waves, confetti cannons mid-air, "
            "four giant video screens suspended from rigging showing the performer from every angle, "
            "runway extending from main stage into the crowd",
            "blinding rig of follow spots and moving heads, strobes, coloured wash — magenta, cyan, white",
            "crowd roar, fanchant rising and falling in unison, bass from PA stacks felt in the chest", False),

        "🎤 K-pop stage — rehearsal":  (
            "empty arena in rehearsal mode, half the house lights up, stage crew moving in the background, "
            "monitor wedges stacked at the front of the stage, tape marks on the floor, "
            "one working spotlight cutting through the dim house, video screen on but static",
            "single working follow spot, flat house wash, harsh and practical",
            "sound check hum, distant crew communication, footsteps on stage wood, PA feedback blip", False),

        "🏰 Big Ben — Westminster at night":  (
            "standing directly beneath the Elizabeth Tower on the Westminster Bridge approach, "
            "the illuminated clock face filling the upper frame, warm floodlit limestone glowing gold "
            "against a deep navy sky, the Thames visible beyond the parapet, "
            "black iron lampposts lining the bridge behind, "
            "tourist coaches and black cabs passing in soft blur",
            "warm sodium floodlighting on the tower face, cold blue ambient sky, wet stone reflecting gold",
            "distant Big Ben chime, Thames wind, traffic across the bridge, footsteps on stone", False),

        "🗽 Times Square — peak night":  (
            "standing in the centre of Times Square at 2am, surrounded on all sides by skyscrapers "
            "sheathed in animated LED billboards — saturated reds, whites, yellows cascading down "
            "the canyon walls, the NASDAQ ticker scrolling, a giant Coca-Cola ad pulsing, "
            "yellow cabs streaming through the intersection below, tourists in every direction, "
            "steam rising from grates in the road",
            "total ambient saturation — no single source, light arriving from every direction at once, colour-shifting",
            "traffic, crowd hum, distant music from a busker, NYPD siren one block over", False),

        "🗼 Eiffel Tower — sparkling midnight":  (
            "standing on the Champ de Mars directly facing the Eiffel Tower at midnight, "
            "the tower's hourly light show in full effect — 20,000 gold bulbs sparkling in random sequence "
            "against the iron lattice, the Seine visible to the left, "
            "Parisian apartment blocks framing both sides, a few couples on the lawn behind",
            "gold sparkle wash from the tower, deep blue ambient sky, distant street lamp orange at the edges",
            "city ambience, wind across the park, the faint creak of the iron structure, distant traffic", False),

        "🌉 Golden Gate Bridge — fog morning":  (
            "standing mid-span on the Golden Gate Bridge walkway, "
            "thick morning fog rolling in from the Pacific and swallowing the south tower, "
            "only the top third of the north tower visible above the fog line, "
            "the bridge roadway disappearing into white in both directions, "
            "the bay invisible below, cold salt air",
            "flat diffuse fog light — directionless, grey-white, no shadows, everything softened",
            "wind through the cables producing a low hum, foghorn in the bay, distant traffic muffled", False),

        "🏯 Japanese shrine — early morning":  (
            "ancient Shinto shrine at first light, stone torii gate at the entrance casting a long shadow "
            "down the gravel path, stone lanterns lining both sides, "
            "cedar trees so tall the canopy closes overhead, moss on every surface, "
            "a single paper lantern still lit from overnight at the main gate",
            "cool blue pre-dawn light, warm paper lantern glow at the gate, raking first light on the gravel",
            "wind through cedar, gravel underfoot, distant temple bell, water dripping from stone", False),

        "🌆 Tokyo — Shibuya crossing — night":  (
            "the Shibuya scramble crossing at night between signal changes, "
            "hundreds of people streaming in every direction simultaneously, "
            "Shibuya 109 building and its neon crown directly ahead, "
            "rain-slicked asphalt reflecting every sign and screen in doubled colour, "
            "7-Eleven and Starbucks logos glowing warm through steam",
            "neon and LED saturation from every angle — amber, white, red, blue — no hard shadows",
            "crossing signal tone, crowd footsteps, car idling, distant J-pop from a store entrance", False),

        "🌊 Amalfi Coast — cliff road":  (
            "narrow coastal road on the Amalfi cliff face, "
            "turquoise Mediterranean far below catching direct sun and breaking white on the rocks, "
            "the road carved directly into the cliff with no barrier on the seaward side, "
            "lemon groves terraced into the hillside above, "
            "a white-painted village visible across the bay in the haze",
            "Mediterranean full sun — hard, directional, high contrast, deep shadows in the cliff cuts",
            "sea wind, waves far below, distant scooter engine, cicadas in the lemon trees", False),

        "🏖 Maldives overwater bungalow — dusk":  (
            "wooden deck extending directly over the lagoon from an overwater bungalow, "
            "the water below so clear the sand and coral are visible in turquoise and white, "
            "dusk turning the horizon to a band of orange fading through pink to violet, "
            "the Indian Ocean completely flat, other bungalows visible in a line behind, "
            "a rope ladder descending into the water from the deck edge",
            "last light warm orange from the horizon, cool violet sky above, water reflecting both",
            "water lapping at the stilts below, wind chime on the bungalow, complete silence beyond that", False),

        "🎪 Coachella main stage — sunset set":  (
            "main Coachella stage at golden hour, the Indio desert stretching to the horizon behind the crowd, "
            "mountains blue and distant in the haze, the stage framed by its giant LED screen "
            "showing warm amber graphics matching the sunset, "
            "tens of thousands of people on the flat desert floor, "
            "dust haze in the light, flags and totems swaying",
            "golden hour desert sun from the west, warm fill from the stage screens, everything amber-soaked",
            "festival crowd roar, bass from the PA crossing the desert, the dry wind", False),

        "🌃 Seoul — Han River bridge — night":  (
            "walking the pedestrian lane of the Banpo bridge at night, "
            "Seoul's skyline reflected in the Han River below in a long shimmering stripe, "
            "the Moonlight Rainbow Fountain arcing jets of water lit in shifting colour from the bridge rail, "
            "apartment towers in every direction, "
            "Namsan Tower with its coloured crown visible on the hill",
            "bridge lighting warm white, fountain colour wash, Seoul skyline ambient glow on the water",
            "water jets from the fountain, Han River wind, distant city, a passing tour boat", False),

        "🏔 Snowfield — high altitude":  (
            "open snowfield at high altitude, no trees, no shelter, "
            "snow surface wind-sculpted into slow sastrugi waves, "
            "a single ridge of darker rock breaking the white in the far distance, "
            "sky a deep near-violet blue at this altitude, "
            "breath visible, footstep tracks the only mark on the surface",
            "flat overcast bounce off the snow — sourceless, directionless white light, everything equally lit",
            "wind, nothing else — occasionally a snow grain skittering across the crust", False),

        "🌁 San Francisco — Lombard Street — night":  (
            "looking down the famous crooked section of Lombard Street at night, "
            "the switchback curves lit by red brake lights and white headlights winding through the bends, "
            "hydrangea beds dark on either side, "
            "the bay and Treasure Island glowing in the distance at the bottom of the hill",
            "warm amber street lamps, red and white car light trails, cold bay ambient at the far end",
            "distant car engine downshifting through the curves, faint cable car bell, bay wind", False),

        "🎠 Versailles — Hall of Mirrors — day":  (
            "the Hall of Mirrors in the Palace of Versailles in full afternoon light, "
            "357 mirrors lining one entire wall and reflecting the 357 arched windows opposite, "
            "gilded pilasters and painted ceiling vaulting the full 73-metre length, "
            "the formal gardens visible through every window in receding perspective, "
            "parquet floor gleaming, tourists reflected to infinity",
            "afternoon sun through 357 windows, multiplied endlessly in the mirrors — brilliant, golden, overwhelming",
            "footsteps echoing on parquet, murmur of tourists, the creak of the gilded ceiling", False),

        "🌋 Iceland — black sand beach — overcast":  (
            "Reynisfjara black sand beach in Iceland on an overcast day, "
            "the volcanic sand dark and wet, basalt column formations rising from the beach and cliff face "
            "in perfect hexagonal geometry, "
            "North Atlantic surf crashing hard with no reef to break it, "
            "sky a flat grey with no horizon distinction from the sea",
            "flat overcast — zero shadows, even grey, the black sand absorbing what little light there is",
            "Atlantic surf — loud, constant, without variation — and wind", False),

        "🛕 Angkor Wat — golden hour":  (
            "standing at the western causeway of Angkor Wat at sunrise, "
            "the five towers reflected in the rectangular moat below, "
            "warm orange light catching the carved sandstone of every spire, "
            "jungle visible above the outer walls in every direction, "
            "lotus blossoms floating on the moat surface",
            "direct low sunrise orange from the east, long shadows down the causeway, warm pink sky",
            "jungle birds, water lapping the moat edge, distant monks' chanting, complete stillness", False),

        "🚇 NYC subway platform — late night":  (
            "empty New York City subway platform at 3am, "
            "tiled walls in grimy institutional cream and brown, "
            "fluorescent tubes overhead with one flickering, "
            "gum-stained concrete platform edge, "
            "yellow warning stripe at the drop, "
            "a distant rumble growing to a full roar as a local train approaches then passes without stopping",
            "flat fluorescent overhead, one tube flickering, the train's headlight sweeping the tunnel briefly",
            "train rumble building and fading, platform announcements echoing, a distant busker", False),

        "🌅 Santorini — caldera view — dawn":  (
            "whitewashed terrace on the caldera rim in Santorini at first light, "
            "the volcanic caldera dropping sheer below, "
            "the Aegean spread to the horizon in deep blue, "
            "blue-domed churches clustered on the clifftop in the middle distance, "
            "bougainvillea cascading over the terrace wall in magenta",
            "first light pale gold on the white walls, deep blue sea and sky, magenta flower accent",
            "Aegean wind, distant bell from the church, a boat engine somewhere below", False),

        "🏟 Empty stadium — floodlit night":  (
            "standing alone on the pitch of a major football stadium at night with no crowd, "
            "the four giant floodlight rigs pouring white light down onto the turf, "
            "the stands empty in darkness beyond the light line, "
            "the pitch surface wet from the sprinklers, "
            "the scoreboard dark, everything else brilliant",
            "four-point overhead flood — hard white industrial light, deep shadows in the empty stands",
            "floodlight hum, wind across the open bowl, a flag snapping on the roof", False),

        "🎻 Vienna opera house — empty stage":  (
            "standing alone on the stage of the Vienna State Opera between performances, "
            "the grand proscenium arch overhead, "
            "six tiers of red velvet boxes receding into darkness in the empty house, "
            "a single work light — a bare bulb on a stand — the only light source on stage, "
            "the ghost light casting long shadows across the boards",
            "single bare bulb ghost light — hard, warm, everything else in dense theatrical dark",
            "absolute silence with a quality of held breath — the acoustic of a room built for music", False),

        "🌿 Amazon — jungle interior":  (
            "deep Amazon rainforest interior with no sky visible, "
            "canopy 40 metres overhead and fully closed, "
            "light arriving only as occasional single shafts breaking through, "
            "everything in permanent green shade, "
            "the forest floor a tangle of roots and fern, "
            "something moving in the mid-canopy unseen",
            "green-filtered indirect light, occasional single shaft of direct sun breaking through the canopy",
            "constant insect layer at full volume, bird calls, distant water, drip from leaves", False),

        "🧊 Ice hotel — Lapland":  (
            "interior of an ice hotel room in Lapland in deep winter, "
            "walls, ceiling, and furniture carved entirely from glacier ice, "
            "sleeping reindeer skins draped over ice bed frames, "
            "the walls faintly glowing blue-white from ice thickness, "
            "breath visible, everything translucent",
            "ambient blue-white glow through the ice walls — sourceless, cold, crystalline",
            "near-total silence — only the creak of settling ice and breath", False),

        "🏬 Tokyo convenience store — 3am":  (
            "Lawson or 7-Eleven convenience store interior in Tokyo at 3am, completely deserted, "
            "fluorescent lights at full brightness, "
            "every shelf perfectly faced and stocked, "
            "hot foods rotating in their case by the register, "
            "rain audible on the pavement outside, "
            "the automatic door briefly opening to admit no one",
            "flat harsh fluorescent overhead — clinical white, no shadows, everything overlit",
            "refrigerator hum, hot case rotating, rain outside, the door's pneumatic hiss", False),

        "🎬 Hollywood — Walk of Fame — golden hour":  (
            "Hollywood Boulevard at golden hour, the pink terrazzo stars embedded in the pavement "
            "catching warm sidelight, the TCL Chinese Theatre pagoda roof visible mid-block, "
            "palm trees lining the boulevard silhouetted against a smoggy orange sky, "
            "tourist handprint impressions in cement in front of the theatre",
            "golden hour sun from the west, long shadows down the sidewalk, haze turning the sky orange",
            "boulevard traffic, distant tourist crowd, a busker's guitar half a block away", False),

        "🛁 Onsen — mountain hot spring":  (
            "outdoor onsen pool on a Japanese mountainside in winter, "
            "natural hot spring water steaming heavily in the cold air, "
            "snow on every surface — the stone edges, the surrounding pines, the mountains beyond, "
            "steam so thick the far edge of the pool blurs, "
            "lanterns casting orange reflections on the water surface",
            "warm lantern glow on steam, cold blue snow light from above, orange-blue contrast",
            "hot spring bubble, water pour from bamboo pipe, snow falling, complete stillness beyond", False),

        # ── Explicit-gated additions ─────────────────────────────────────────
        "🎷 Speakeasy — basement jazz club":  (
            "narrow underground speakeasy bar accessed down a flight of stairs, "
            "exposed brick walls, low tiled ceiling, round tables with candle stubs, "
            "a four-piece jazz band on a postage-stamp stage at the far end, "
            "the bar lined with backlit bottles, the room half-dark and smoke-yellowed",
            "candle warmth table by table, bar backlight amber, the stage a single warm overhead spot",
            "live jazz close and intimate, ice in glasses, low conversation under the music", False),

        "🌃 Rooftop pool — Las Vegas strip":  (
            "rooftop infinity pool on a Vegas Strip hotel at 2am, "
            "the pool's edge appearing to flow into the city of lights below, "
            "the Bellagio fountains visible mid-distance, "
            "neon from the Strip reflected in the water as shifting colour, "
            "no one else in the pool",
            "Vegas neon ambient from below — warm gold, red, white — pool lighting from within teal",
            "pool water, the Strip traffic hum 40 floors below, distant fountains, a helicopter", True),

        "🛸 Rooftop — Tokyo neon rain":  (
            "flat rooftop in Shinjuku during heavy rain at night, "
            "neon signs from pachinko parlours and hostess bars visible on the street below, "
            "rain pooling on the rooftop tar surface and reflecting every sign, "
            "a forest of air conditioning units and water tanks, "
            "Shinjuku station tower visible behind in the rain haze",
            "neon reflection in rain puddles — magenta, amber, white — no direct light, all bounced",
            "heavy rain on tar and metal, street noise muffled below, thunder one block over", True),
    }


    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # ── Scene ─────────────────────────────────────────────────────
                # Key names ARE the labels ComfyUI displays — emojis go in the key
                "⏭ bypass": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Skip the LLM entirely and pass your text straight to the output. Use for manual prompts or testing.",
                }),
                "🖼 use image information?": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "When ON, the node uses the image description wired into the scene context input as the authoritative starting point. Turn OFF to ignore it without disconnecting the wire.",
                }),
                "💬 let the LLM create dialogue?": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "When ON, the LLM invents natural spoken dialogue woven into the scene. When OFF, only uses dialogue you wrote yourself in quotes — or generates no dialogue at all.",
                }),
                "user_input": ("STRING", {
                    "multiline": True,
                    "default": "a woman walks slowly toward the camera on a rain-soaked city street at night",
                    "tooltip": "Describe what you want to happen. Can be a rough idea, a sentence, or numbered steps. The LLM expands this into a full cinematic prompt.",
                }),
                "🎨 creativity": ([
                    "0.5 - Strict & Literal",
                    "0.8 - Balanced Professional",
                    "1.0 - Artistic Expansion",
                ], {
                    "default": "0.8 - Balanced Professional",
                    "tooltip": "Controls how closely the LLM sticks to your input. 0.5 is literal and precise, 1.0 adds more cinematic flair and creative expansion.",
                }),
                # ── Camera ────────────────────────────────────────────────────
                "📐 shot angle": ([
                    "None — LLM decides",
                    "Eye-level — neutral, natural",
                    "Low angle — powerful, imposing",
                    "High angle — vulnerable",
                    "Bird's eye — top-down overhead",
                    "Worm's eye — extreme low, looking up",
                    "Dutch angle — tilted, unsettling",
                    "OTS — over the shoulder",
                    "POV — first person",
                    "Profile — side-on",
                    "Three-quarter — 45 degree",
                ], {
                    "default": "None — LLM decides",
                    "tooltip": "Force a specific shot angle. Each style preset has a smart default — set this to override it manually.",
                }),
                "🎥 camera movement": ([
                    "None — LLM decides",
                    "Static — locked off",
                    "Handheld — natural shake",
                    "Slow push in",
                    "Pull back — reveal",
                    "Arc — slow curved lateral track",
                    "Tracking — follows subject",
                    "Tilt up — bottom to top",
                    "Tilt down — top to bottom",
                    "Truck left — lateral slide",
                    "Truck right — lateral slide",
                    "Whip pan — fast horizontal snap",
                    "Dolly zoom — vertigo effect",
                    "Aerial — drone descending",
                ], {
                    "default": "None — LLM decides",
                    "tooltip": "Force a specific camera movement. Each style preset has a smart default — set this to override it manually.",
                }),
                # ── Generation ────────────────────────────────────────────────
                "🎬 style preset": (list(LTX2PromptArchitectQwen.STYLE_PRESETS.keys()), {
                    "default": "None — let the LLM decide",
                    "tooltip": "Sets the full visual aesthetic — lighting, camera style, colour grade, mood. Also sets smart defaults for shot angle and camera movement automatically.",
                }),
                "🗣 spoken language": ([
                    "Auto — use existing prompt logic",
                    "English",
                    "German",
                    "French",
                    "Spanish",
                    "Italian",
                    "Portuguese",
                    "Polish",
                    "Russian",
                    "Japanese",
                    "Korean",
                    "Chinese",
                ], {
                    "default": "Auto — use existing prompt logic",
                    "tooltip": "Controls only spoken or sung in-scene language. The descriptive prompt prose stays in English."
                }),
                "🌍 environment": (list(LTX2PromptArchitectQwen.ENVIRONMENT_PRESETS.keys()), {
                    "default": "None — LLM decides",
                    "tooltip": "Force a specific location and environment. Overrides any location in your prompt. Explicit locations (casting couch, sex dungeon etc) only inject when scene content is explicit or sensual.",
                }),
                "seed": ("INT", {
                    "default": -1, "min": -1, "max": 2**31 - 1, "step": 1,
                    "display": "number",
                    "tooltip": "Set a fixed seed to get the same prompt expansion every run. Use -1 for a random result each time.",
                }),
                # NOTE: control_after_generate must be declared here AND in the generate()
                # signature — ComfyUI passes it as a keyword argument when a seed widget
                # is present. Removing it from either place will break the node.
                "control_after_generate": (["randomize", "fixed", "increment", "decrement"], {
                    "default": "randomize",
                }),
            },
            "optional": {
                # ── Scene overrides ───────────────────────────────────────────
                "👥 subject count": ("INT", {
                    "default": 0, "min": 0, "max": 4, "step": 1,
                    "display": "number",
                    "tooltip": "Force the number of people in the scene. 0 = auto-detect from your text. 1–4 = explicit override. Changes spatial blocking, framing, and tracking instructions.",
                }),
                "🚫 things to avoid": ("STRING", {
                    "default": "", "multiline": False,
                    "placeholder": "e.g. no rain, no crowd, no slow motion",
                    "tooltip": "Steer the LLM away from things it tends to add by default. These are also appended to the negative prompt output.",
                }),
                "🏷 lora triggers": ("STRING", {
                    "default": "", "multiline": False,
                    "placeholder": "e.g. ohwx woman, film grain",
                    "tooltip": "Paste your LoRA trigger words here. Injected at the very start of every generated prompt automatically — never buried or dropped.",
                }),
                "🎵 music genre": ([
                    "None — detect from prompt",
                    "── ELECTRONIC ──────────",
                    "House music", "Techno", "Drum and Bass",
                    "Dubstep / Bass music", "EDM / Big room", "Trance",
                    "UK Garage / 2-step", "Ambient / Atmospheric",
                    "Grime", "Jungle / Early rave",
                    "Synthwave / Retrowave", "Vaporwave", "Hyperpop",
                    "── URBAN ───────────────",
                    "Hip-hop / Rap", "Trap", "Drill / UK Drill",
                    "R&B / RnB", "Neo-soul",
                    "Afrobeats / Afropop", "Dancehall / Reggaeton",
                    "── ROCK / LIVE ─────────",
                    "Psychedelic / Prog rock",
                    "Rock", "Metal / Heavy metal", "Punk / Pop-punk",
                    "Indie rock / Shoegaze", "Blues", "Soul / Motown",
                    "Funk", "Disco", "Folk / Americana",
                    "Country", "Gospel",
                    "── JAZZ / CLASSICAL ────",
                    "Jazz", "Classical / Orchestral", "Opera",
                    "Musical theatre", "Cabaret / Burlesque",
                    "Lo-fi hip-hop",
                    "── WORLD ───────────────",
                    "Flamenco", "Bossa nova / Samba",
                    "K-pop", "J-pop / City pop", "Reggae / Ska",
                    "Bollywood / Bhangra", "Cumbia / Salsa / Latin",
                ], {
                    "default": "None — detect from prompt",
                    "tooltip": (
                        "🎵 Force a specific music genre. Overrides anything detected from your prompt. "
                        "Leave on None to auto-detect. "
                        "Or just type the genre in your prompt — 'techno', 'jazz', 'kpop', 'flamenco', "
                        "'drum and bass', 'synthwave', 'hip-hop', 'bollywood' etc all work automatically."
                    ),
                }),
                # ── Emotional state ───────────────────────────────────────────
                "💋 emotional state": ([
                    "None — LLM decides",
                    "🎲 Surprise me — seed picks",
                    "── PAIN ────────────────",
                    "Grief — recently broken",
                    "Crying — actively in tears",
                    "Sad — carrying it quietly",
                    "Shame — cannot meet the lens",
                    "Longing — aching for the absent",
                    "Exhausted — nothing left",
                    "── HEAT ────────────────",
                    "Angry — jaw set, body hard",
                    "Defiant — already decided",
                    "Fierce — no performance, all intensity",
                    "Predatory — still, patient, hunting",
                    "── WANT ────────────────",
                    "Aroused — heat in the body, deliberate",
                    "Tender — open, directed warmth",
                    "Longing — want without having",
                    "Adoration — softness aimed at one thing",
                    "── LIGHT ───────────────",
                    "Euphoric — too much joy to contain",
                    "Happy — genuine and unguarded",
                    "Playful — teasing, something behind the eyes",
                    "Confident — owns the space",
                    "Proud — quiet earned satisfaction",
                    "── FRAGILE ─────────────",
                    "Vulnerable — more open than intended",
                    "Nervous — small movements, scanning eyes",
                    "Overcome — holding it together, barely",
                    "── OTHER ───────────────",
                    "Wired — electric, too much energy",
                    "Dissociated — present in body, absent in mind",
                    "Controlled — composure masking everything",
                ], {
                    "default": "None — LLM decides",
                    "tooltip": (
                        "💋 Set the emotional state of the subject. "
                        "Changes face, posture, eye contact, breath, hands — everything. "
                        "Works on any prompt: singing, dancing, standing, jogging, anything. "
                        "Surprise me = seed picks a random state each run. "
                        "None = LLM reads the scene and decides."
                    ),
                }),
                # ── Audio ─────────────────────────────────────────────────────
                "audio_input": ("AUDIO", {
                    "tooltip": "Wire any ComfyUI AUDIO output here. The node analyses energy, tempo, and frequency character to shape the generated prompt to match the audio.",
                }),
                "🔊 use audio for the LLM?": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Master switch for audio analysis. Turn ON when you have audio wired and want it to influence the prompt.",
                }),
                "📝 transcribe the audio?": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Uses openai-whisper (tiny model, 39MB, auto-downloads) to transcribe speech in the audio. The transcript is injected into the prompt so the LLM can sync visuals to what is being said.",
                }),
                # ── Settings ──────────────────────────────────────────────────
                "⏱ frame count": ("INT", {
                    "default": 192, "min": 24, "max": 960, "step": 1,
                    "display": "number",
                    "tooltip": "Match this to your video LENGTH setting. Controls pacing — 24fps = 1 second, so 192 = 8 seconds.",
                }),
                "↔ width": ("INT", {
                    "default": 0, "min": 0, "max": 7680, "step": 8,
                    "display": "number",
                    "tooltip": "Wire from your WIDTH constant. Used to detect aspect ratio and set framing language automatically. Leave 0 if not wired.",
                }),
                "↕ height": ("INT", {
                    "default": 0, "min": 0, "max": 7680, "step": 8,
                    "display": "number",
                    "tooltip": "Wire from your HEIGHT constant. Used with width to detect portrait/square/wide ratios. Leave 0 if not wired.",
                }),
                "🤖 model": (
                    [
                        "huihui-ai/Huihui-Qwen3.5-9B-abliterated",
                        "huihui-ai/Huihui-Qwen3.5-27B-abliterated",
                        "huihui-ai/Huihui-Qwen3.5-35B-A3B-abliterated",
                        "huihui-ai/Huihui-Qwen3.5-35B-A3B-Claude-4.6-Opus-abliterated",
                    ],
                    {
                        "default": "huihui-ai/Huihui-Qwen3.5-9B-abliterated",
                        "tooltip": (
                            "Select which huihui-ai Qwen3.5 abliterated model to use. "
                            "9B: fastest, lowest VRAM (~6GB). "
                            "27B: best quality/speed balance (~14GB VRAM at Q4). "
                            "35B-A3B: MoE — only 3B active params, runs like 9B but thinks like 35B. "
                            "35B-A3B-Claude: same MoE but distilled on Claude Opus 4.6 outputs. "
                            "Changing model mid-session will unload and reload — takes ~30s."
                        ),
                    }
                ),
                "🧩 backend": (
                    ["transformers", "llama.cpp (GGUF)", "llama-server (OpenAI API)"],
                    {
                        "default": "transformers",
                        "tooltip": "Choose the inference backend. Transformers uses HuggingFace checkpoints. llama.cpp loads a GGUF via llama-cpp-python. llama-server sends requests to a running external server."
                    }
                ),
                "📁 local model path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Full path to Huihui-Qwen3.5-9B snapshot folder",
                    "tooltip": "Optional. Paste the full path to your locally downloaded model snapshot folder. Leave blank to auto-download from HuggingFace on first run.",
                }),
                "🦙 gguf repo": ("STRING", {
                    "default": "lukey03/Qwen3.5-9B-abliterated-GGUF",
                    "multiline": False,
                    "placeholder": "repo/name",
                    "tooltip": "llama.cpp only. Hugging Face repo ID containing the GGUF file."
                }),
                "🦙 gguf file": ("STRING", {
                    "default": "Qwen3.5-9B-abliterated-Q4_K_M.gguf",
                    "multiline": False,
                    "placeholder": "model.gguf",
                    "tooltip": "llama.cpp only. Exact GGUF filename to download from the selected repo."
                }),
                "🦙 n_gpu_layers": ("INT", {
                    "default": -1, "min": -1, "max": 512, "step": 1,
                    "display": "number",
                    "tooltip": "llama.cpp only. -1 offloads as many layers as possible, 0 keeps everything on CPU."
                }),
                "🦙 context size": ("INT", {
                    "default": 8192, "min": 512, "max": 65536, "step": 256,
                    "display": "number",
                    "tooltip": "llama.cpp only. Context size for llama.cpp."
                }),
                "🦙 batch size": ("INT", {
                    "default": 512, "min": 32, "max": 8192, "step": 32,
                    "display": "number",
                    "tooltip": "llama.cpp only. Prompt processing batch size."
                }),
                "🌐 llama-server url": ("STRING", {
                    "default": "http://127.0.0.1:8080/v1",
                    "multiline": False,
                    "placeholder": "http://127.0.0.1:8080/v1",
                    "tooltip": "llama-server only. Base URL of the external OpenAI-compatible llama.cpp server."
                }),
                "🌐 llama-server model": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "optional model name",
                    "tooltip": "llama-server only. Optional model field sent in the chat completion request."
                }),
                "🔑 llama-server api key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "optional bearer token",
                    "tooltip": "llama-server only. Optional bearer token for authenticated servers."
                }),
                "✈ offline mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Turn ON if you have no internet. Uses locally cached models only. Turn OFF to allow auto-download from HuggingFace on first run.",
                }),
                "🧠 GPU ID": ("INT", {
                    "default": -1, "min": -1, "max": 15, "step": 1,
                    "display": "number",
                    "tooltip": "-1 = automatic device selection. 0/1/2/... pins the model to a specific CUDA GPU for transformers and llama.cpp."
                }),
                "🔗 keep model loaded": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Deprecated — the node always offloads after every run to free VRAM for LTX. Kept here so existing workflows don't break.",
                }),
                "🖼 scene context": ("STRING", {
                    "default": "", "multiline": False,
                    "placeholder": "← wire the Vision Describe node output here",
                    "tooltip": "Wire the output from the LTX-2 Vision Describe node here. The LLM will use your image as the authoritative starting point.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("PROMPT", "PREVIEW", "NEG_PROMPT")
    FUNCTION = "generate"
    CATEGORY = "LTX2"

    def __init__(self):
        self.tokenizer             = None
        self.model                 = None
        self.loaded                = False
        self._stop_token_ids       = []
        self._last_portrait        = False
        self._last_style           = ""
        self._resolved_model_path  = None  # cached after first snapshot_download — avoids HF API call every run
        self._resolved_gguf_path   = None
        self._resolved_gguf_key    = None
        self._loaded_backend       = None
        self._loaded_model_id      = None
        self._loaded_device_target = None
        self._loaded_llama_config  = None
        self._loaded_server_config = None

    # ── Model management ──────────────────────────────────────────────────────

    def load_model(
        self,
        offline_mode: bool,
        local_path: str,
        model_id: str = None,
        backend: str = "transformers",
        gpu_id: int = -1,
        gguf_repo_id: str = "",
        gguf_filename: str = "",
        llama_n_gpu_layers: int = -1,
        llama_n_ctx: int = 8192,
        llama_n_batch: int = 512,
        llama_server_url: str = "",
        llama_server_model: str = "",
        llama_server_api_key: str = "",
    ):
        # Use selected model ID or fall back to class default
        _active_model_id = (model_id or self.MODEL_HF_ID).strip()
        if not _active_model_id:
            _active_model_id = self.MODEL_HF_ID
        backend = (backend or "transformers").strip()
        target_device = _resolve_model_target(gpu_id)

        if backend == "llama-server (OpenAI API)":
            server_url = (llama_server_url or "http://127.0.0.1:8080/v1").rstrip("/")
            server_config = {
                "url": server_url,
                "model": (llama_server_model or "").strip(),
                "api_key": (llama_server_api_key or "").strip(),
            }
            if self.model is not None and self._loaded_backend == backend:
                if self._loaded_server_config == server_config:
                    return
                print("[LTX2-Qwen] llama-server config changed. Reloading...")
                self.unload_model()
            elif self.model is not None:
                self.unload_model()

            self.model = server_config
            self.tokenizer = None
            self.loaded = True
            self._stop_token_ids = []
            self._loaded_backend = backend
            self._loaded_server_config = dict(server_config)
            self._loaded_llama_config = None
            self._loaded_model_id = None
            self._loaded_device_target = None
            print(f"[LTX2-Qwen] Ready: llama-server at {server_url}")
            return

        if backend == "llama.cpp (GGUF)":
            gguf_repo_id = (gguf_repo_id or "").strip()
            gguf_filename = (gguf_filename or "").strip()
            if not gguf_repo_id or not gguf_filename:
                raise RuntimeError("[LTX2-Qwen] llama.cpp backend requires both '🦙 gguf repo' and '🦙 gguf file'.")

            gguf_key = f"{gguf_repo_id}/{gguf_filename}"
            gguf_config = {
                "repo_id": gguf_repo_id,
                "filename": gguf_filename,
                "target_device": target_device,
                "n_gpu_layers": int(llama_n_gpu_layers),
                "n_ctx": int(llama_n_ctx),
                "n_batch": int(llama_n_batch),
            }

            if self.model is not None and self._loaded_backend == backend:
                if self._loaded_llama_config == gguf_config:
                    return
                print("[LTX2-Qwen] GGUF config changed. Reloading...")
                self.unload_model()
            elif self.model is not None:
                self.unload_model()

            if self._resolved_gguf_key == gguf_key and self._resolved_gguf_path:
                source = self._resolved_gguf_path
                print(f"[LTX2-Qwen] Using cached GGUF path: {source}")
            else:
                try:
                    from huggingface_hub import hf_hub_download
                    print(f"[LTX2-Qwen] Resolving GGUF model path (first run): {gguf_key}")
                    source = hf_hub_download(
                        repo_id=gguf_repo_id,
                        filename=gguf_filename,
                        local_files_only=offline_mode,
                    )
                except Exception as e:
                    if offline_mode:
                        raise RuntimeError(
                            f"[LTX2-Qwen] Could not resolve GGUF in offline mode: {gguf_key} ({e})"
                        ) from e
                    try:
                        from huggingface_hub import snapshot_download
                        print(f"[LTX2-Qwen] hf_hub_download failed: {e}. Retrying with snapshot_download for {gguf_filename}...")
                        snapshot_dir = snapshot_download(
                            repo_id=gguf_repo_id,
                            allow_patterns=[gguf_filename],
                        )
                        source = os.path.join(snapshot_dir, gguf_filename)
                    except Exception as snapshot_error:
                        raise RuntimeError(
                            f"[LTX2-Qwen] Could not auto-download GGUF model '{gguf_key}': {snapshot_error}"
                        ) from snapshot_error

                self._resolved_gguf_key = gguf_key
                self._resolved_gguf_path = source
                print(f"[LTX2-Qwen] GGUF ready at: {source}")

            try:
                from llama_cpp import Llama
            except Exception as e:
                raise RuntimeError(
                    "[LTX2-Qwen] llama.cpp backend selected but llama-cpp-python is not installed."
                ) from e

            llama_kwargs = {
                "model_path": source,
                "n_ctx": int(llama_n_ctx),
                "n_batch": int(llama_n_batch),
                "n_gpu_layers": int(llama_n_gpu_layers),
                "verbose": True,
            }
            if target_device.startswith("cuda:"):
                llama_kwargs["main_gpu"] = int(target_device.split(":")[-1])
            elif target_device == "auto" and torch.cuda.is_available():
                llama_kwargs["main_gpu"] = torch.cuda.current_device()
            else:
                llama_kwargs["n_gpu_layers"] = 0

            print(
                f"[LTX2-Qwen] Loading GGUF via llama.cpp: path={source}, "
                f"main_gpu={llama_kwargs.get('main_gpu', 'cpu')}, "
                f"n_gpu_layers={llama_kwargs['n_gpu_layers']}, "
                f"n_ctx={llama_kwargs['n_ctx']}, n_batch={llama_kwargs['n_batch']}"
            )
            self.model = Llama(**llama_kwargs)
            self.tokenizer = None
            self.loaded = True
            self._stop_token_ids = []
            self._loaded_backend = backend
            self._loaded_llama_config = dict(gguf_config)
            self._loaded_server_config = None
            self._loaded_model_id = None
            self._loaded_device_target = target_device
            print(f"[LTX2-Qwen] Ready: GGUF on {target_device}")
            return

        if self.model is not None:
            _loaded_id = getattr(self, "_loaded_model_id", self.MODEL_HF_ID)
            if (
                self._loaded_backend == backend and
                _loaded_id == _active_model_id and
                self._loaded_device_target == target_device
            ):
                return
            print(f"[LTX2-Qwen] Model/backend changed. Reloading...")
            self._resolved_model_path = None
            self.unload_model()

        # Guard: saved workflows may have stored boolean False for this field.
        # Convert to empty string so the "no local path" branch fires correctly.
        if not isinstance(local_path, str) or local_path.strip().lower() in ("false", "none", "0"):
            local_path = ""

        source = local_path.strip() if local_path and local_path.strip() else None

        if not source:
            if offline_mode:
                source = _active_model_id
                print(f"[LTX2-Qwen] Offline mode — will use HF cache only. "
                      f"If model is not already cached this will raise an error: {_active_model_id}")
            elif self._resolved_model_path:
                # Use cached path from a previous run — avoids HF API call every time
                source = self._resolved_model_path
                print(f"[LTX2-Qwen] Using cached model path: {source}")
            else:
                try:
                    from huggingface_hub import snapshot_download
                    print(f"[LTX2-Qwen] Resolving model path (first run)...")
                    source = snapshot_download(_active_model_id, ignore_patterns=["*.gguf"])
                    self._resolved_model_path = source  # cache for subsequent runs
                    print(f"[LTX2-Qwen] Ready at: {source}")
                except Exception as e:
                    print(f"[LTX2-Qwen] snapshot_download failed: {e} — falling back to direct load")
                    source = _active_model_id
        else:
            print(f"[LTX2-Qwen] Local path: {source}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            source, trust_remote_code=True, local_files_only=offline_mode
        )
        dtype = _get_torch_dtype_for_target(target_device)
        load_kwargs = _build_model_load_kwargs(target_device, offline_mode, dtype)
        print(
            f"[LTX2-Qwen] Loading transformers model with dtype={dtype} "
            f"and device_map={load_kwargs.get('device_map')}"
        )
        self.model = AutoModelForCausalLM.from_pretrained(source, **load_kwargs)
        if target_device.startswith("cuda:"):
            print(f"[LTX2-Qwen] Moving transformers model to {target_device}...")
            self.model.to(target_device)
        self.model.config.use_cache = True
        self.model.eval()
        self.loaded = True
        self._stop_token_ids = self._build_stop_token_ids()
        if torch.cuda.is_available():
            a = torch.cuda.memory_allocated() / 1024**3
            r = torch.cuda.memory_reserved()  / 1024**3
            print(f"[LTX2-Qwen] Loaded — VRAM: {a:.2f}GB alloc / {r:.2f}GB reserved")
        self._loaded_backend = backend
        self._loaded_model_id = _active_model_id
        self._loaded_device_target = target_device
        self._loaded_llama_config = None
        self._loaded_server_config = None
        print(f"[LTX2-Qwen] Ready: {_active_model_id}")

    def unload_model(self):
        if self.model is None:
            return
        print("[LTX2-Qwen] Unloading model...")
        if self._loaded_backend == "transformers":
            try:
                for _n, module in list(self.model.named_modules()):
                    for _p, param in list(module.named_parameters(recurse=False)):
                        try:
                            param.data = torch.empty(0)
                        except Exception:
                            pass
                    for _b, buf in list(module.named_buffers(recurse=False)):
                        try:
                            module._buffers[_b] = None
                        except Exception:
                            pass
            except Exception as e:
                print(f"[LTX2-Qwen] Tensor destroy warning: {e}")

        del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        self.model           = None
        self.tokenizer       = None
        self.loaded          = False
        self._stop_token_ids = []
        self._loaded_backend = None
        self._loaded_model_id = None
        self._loaded_device_target = None
        self._loaded_llama_config = None
        self._loaded_server_config = None

        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"[LTX2-Qwen] CUDA flush warning during unload: {e}")

        gc.collect()

        try:
            import comfy.model_management as mm
            mm.unload_all_models()
            mm.soft_empty_cache()
        except Exception:
            pass

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            a = torch.cuda.memory_allocated() / 1024**3
            r = torch.cuda.memory_reserved()  / 1024**3
            print(f"[LTX2-Qwen] VRAM after free: {a:.2f}GB alloc / {r:.2f}GB reserved")

    def _build_stop_token_ids(self) -> list:
        delimiters = [
            "assistant", "user", "system",
            "<|eot_id|>", "<|end_of_turn|>", "<|im_end|>",
            "<end_of_turn>", "[/INST]", "### Human", "### Assistant",
        ]
        # Guard against tokenizers that return None for eos_token_id
        ids = []
        if self.tokenizer.eos_token_id is not None:
            ids.append(self.tokenizer.eos_token_id)
        for s in delimiters:
            enc = self.tokenizer.encode(s, add_special_tokens=False)
            if enc:
                ids.append(enc[0])
        seen, out = set(), []
        for i in ids:
            if i is not None and i not in seen:
                seen.add(i)
                out.append(i)
        print(f"[LTX2-Qwen] Stop token IDs: {out}")
        return out

    # ── Output cleaning ───────────────────────────────────────────────────────

    _PREAMBLE_RE = re.compile(
        r"^(Sure!?|Certainly!?|Absolutely!?|Of course!?|Here(?:'s| is)[\s\S]*?:|"
        r"Great!?|LTX-?2(?:\.\d)?(?:\s+\w+)*\s*prompt\s*:|Prompt\s*:|Output\s*:|Scene\s*:)[^\n]*\n?",
        re.IGNORECASE
    )
    _ROLE_BLEED_RE = re.compile(
        r"\s*(assistant|user|system|<\|[^|>]*\|>)\s*$",
        re.IGNORECASE
    )

    @classmethod
    def _clean_output(cls, text: str) -> str:
        text = text.strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        text = cls._PREAMBLE_RE.sub("", text).strip()
        text = cls._ROLE_BLEED_RE.sub("", text).strip()
        text = re.sub(r"\.(assistant|user|system|<\|[^|>]*\|>)\s*\n", ".\n", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s*\n+Note:.*$", "", text, flags=re.DOTALL).strip()
        text = re.sub(r"\s*\n+(Please let me know|Let me revise|No further revision|Confirmed\.|"
                      r"Written to meet|The scene is now over|The output ends|The task is|The task was|"
                      r"The goal was|Nothing more|No continuation|No additional|The response does not|"
                      r"It does not continue|It ceases when|Any such statement|"
                      r"Output length:|Action count:|Total time:|Last character:|I avoided|I wrote|"
                      r"I adhered|I hope this|Thank you for your|Please confirm|I submitted|"
                      r"I can revise|feel free to instruct).*$",
                      "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        text = re.sub(r'\s*(Ended\.\s*\d+\s*actions|'
                      r'\d+\s+actions[\.,]\s*\d+\s+tokens|'
                      r'\d+\s+tokens[\.,]\s*Done|'
                      r'Done\.\s+\d+\s+seconds|'
                      r'Finished\.\s+\d+|'
                      r'Hard stop\..*$)', '', text, flags=re.DOTALL | re.IGNORECASE).strip()
        text = re.sub(r'\.?\s+The total duration.*$', '.', text, flags=re.DOTALL | re.IGNORECASE).strip()
        text = re.sub(r'\.?\s+The (scene\'?s? )?total (duration|running time).*$', '.', text, flags=re.DOTALL | re.IGNORECASE).strip()
        text = re.sub(r'\s*\(\d+\s+seconds?\)\s*$', "", text).strip()
        text = re.sub(r'\s*\(\d+\s+tokens?[^)]*\)', "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r'\s*\n*\d+\s+tokens[\s,].*$', "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        text = re.sub(r'\[AMBIENT:\s*([^\]]*)\]', r'\1', text, flags=re.IGNORECASE).strip()
        text = re.sub(r'\((?:DOWN|UP|PULL|PUSH|ZOOM|HOLD|FADE|PAN|TILT|TRUCK|DOLLY)[^\)]{0,80}\)', "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r'\[(Kodak|ARRI|Fuji|Film stock|film stock)[^\]]{0,80}\]\s*', "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r'\b(Lens|Camera angle|Focal length|Shutter|Motion blur|Aperture)\s*:\s*[^.\n]{5,120}[.\n]?\s*$',
                      "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r',?\s*(In this (peaceful|serene|quiet|tender|intimate|still|languid|tranquil|soft) (moment|scene|instant)[^.]{0,200}\.)\s*$',
                      "", text, flags=re.IGNORECASE | re.DOTALL).strip()
        text = re.sub(r'[,.]?\s*(leaving only (the [a-z ]{3,60}(of|and)[^.]{3,80})\.?)\s*$',
                      ".", text, flags=re.IGNORECASE).strip()
        text = re.sub(r'[,.]?\s*(the (quiet|soft|gentle|only) (satisfaction|warmth|rhythm|rustle|hum|sound|glow) of [^.]{5,80}\.)\s*$',
                      ".", text, flags=re.IGNORECASE).strip()
        text = re.sub(r'\.\s+The scene ends there[^.]*\.', '.', text, flags=re.IGNORECASE).strip()
        text = re.sub(r',?\s+before the scene fades to black[^.]*\.', '.', text, flags=re.IGNORECASE).strip()
        text = re.sub(r'\.?\s+[Tt]he scene fades to black[^.]*\.', '.', text, flags=re.IGNORECASE).strip()
        text = re.sub(r',?\s+as the scene fades[^.]*\.', '.', text, flags=re.IGNORECASE).strip()
        # Closing-sentence patterns: "ending on the...", "the scene ending mid-breath", "leaving the audience..."
        text = re.sub(r',?\s+ending on (?:the |a )[^.]{5,150}\.', '.', text, flags=re.IGNORECASE).strip()
        text = re.sub(r'[.,]?\s+[Tt]he (?:scene|clip|shot|moment|frame) ending[^.]{0,150}\.', '.', text, flags=re.IGNORECASE).strip()
        text = re.sub(r',?\s+the (?:lingering|soft|quiet|fading|final) (?:echo|glow|hum|warmth|pulse|ache|question|breath|sigh|friction) of [^.]{5,120}\.', '.', text, flags=re.IGNORECASE).strip()
        text = re.sub(r',?\s+leaving (?:the (?:viewer|audience|camera|eye)|her|him|them|us) [^.]{5,120}\.', '.', text, flags=re.IGNORECASE).strip()
        text = re.sub(r',?\s+a sense of [^.]{5,80}\.', '.', text, flags=re.IGNORECASE).strip()
        text = re.sub(r',?\s+(?:fully immersed|lost in the moment|lost in the \w+)[^.]{0,60}\.', '.', text, flags=re.IGNORECASE).strip()
        text = re.sub(r',?\s+(?:pure|sheer)\s+(?:delight|joy|bliss|happiness|sorrow)[^.]{0,60}\.', '.', text, flags=re.IGNORECASE).strip()
        text = re.sub(r',?\s+as if (?:she|he|they) (?:finds?|feels?|radiates?|embodies?)[^.]{5,100}\.', '.', text, flags=re.IGNORECASE).strip()
        # Extended scene-ending catches from live testing
        text = re.sub(r'[.,]?\s+ending the (?:sequence|clip|shot|scene|moment|video)[^.]{0,120}\.', '.', text, flags=re.IGNORECASE).strip()
        text = re.sub(r'[.,]?\s+completing the (?:third|second|first|final|last|[a-z]+\s)?(?:beat|sequence|moment|arc)[^.]{0,100}\.', '.', text, flags=re.IGNORECASE).strip()
        text = re.sub(r',?\s+before the (?:clip|scene|shot|video|frame) continues?[^.]{0,80}\.', '.', text, flags=re.IGNORECASE).strip()
        text = re.sub(r',?\s+(?:the )?(?:golden|warm|soft|fading|dying) (?:light|glow|sun) (?:swallows?|consumes?|dissolves?|washes?) (?:the scene|the frame|everything)[^.]{0,60}\.', '.', text, flags=re.IGNORECASE).strip()
        text = re.sub(r'[.,]?\s+[Tt]he scene holds? this[^.]{0,120}\.', '.', text, flags=re.IGNORECASE).strip()
        text = re.sub(r'[.,]?\s+grounding (?:the|this|her|his) [^.]{5,100} (?:in|into) (?:tactile |physical |quiet )?reality\.', '.', text, flags=re.IGNORECASE).strip()
        text = re.sub(r'[.,]?\s+capturing (?:her|his|their|the|its) [^.]{5,80} (?:whole|entirety|completeness|totality)\.', '.', text, flags=re.IGNORECASE).strip()
        text = re.sub(r'[.,]?\s+capturing (?:the )?(?:suspended|quiet|still|poised|held) [^.]{5,100}\.', '.', text, flags=re.IGNORECASE).strip()
        text = re.sub(r'[.,]?\s+[Aa] final[,]? (?:crisp|soft|slow|sharp|deep|low|quiet|long|steady) [^.]{5,80}\.', '.', text, flags=re.IGNORECASE).strip()
        text = re.sub(r'[.,]?\s+[Tt]he scene ends? with[^.]{0,300}\.', '.', text, flags=re.IGNORECASE).strip()
        text = re.sub(r'[.,]?\s+[Tt]he (?:room|air|space|silence) (?:hangs?|holds?|settles?|thickens?|hums?) (?:with|in)[^.]{0,120}\.', '.', text, flags=re.IGNORECASE).strip()
        text = re.sub(r',\s+(?:save|except|aside)\s+for[^.]{0,30}\.', '.', text, flags=re.IGNORECASE).strip()
        text = re.sub(r'(?<=\.)\s+[Tt]he (?:shot|camera|frame|lens) holds?[^.]{0,250}\.$', '.', text, flags=re.IGNORECASE).strip()
        text = re.sub(r'(?<=\.)\s+(?:She|He|They|It) (?:remains?|stands?|sits?|hangs?|stays?) (?:suspended|frozen|poised|still|there)[^.]{0,200}\.$', '.', text, flags=re.IGNORECASE).strip()
        text = re.sub(r'[.,]?\s+waiting for (?:the|her|him|what)[^.]{0,100}\.', '.', text, flags=re.IGNORECASE).strip()
        text = re.sub(r'\s*[\(\[]\s*$', '', text).strip()
        text = re.sub(r'\.{2,}', '.', text).strip()
        text = re.sub(r'\s{2,}', ' ', text).strip()
        rep = re.search(r'((?:\b\w[\w\'\\-]*\b[\s\.,!?]*){1,6})\1{4,}$', text, flags=re.DOTALL)
        if rep:
            text = text[:rep.start()].strip()
        return text.strip()

    # ── Generate ──────────────────────────────────────────────────────────────

    # ── Singing camera profiles ──────────────────────────────────────────────
    # Genre-aware per-section camera moves injected into the singing pacing hint.
    # Each profile has 4 moves: opening, build, peak, resolution.
    # Chosen for what LTX-2.3 can actually execute reliably.
    _SINGING_CAM_PROFILES = {
        "intimate": {
            "openings": [
                "open on a medium close-up just below eye level — the face filling most of the frame from the first second, camera does not need to travel far to arrive",
                "open in extreme close-up on eyes and forehead only — pull back with glacial slowness to reveal the full face over the clip",
                "open with a shallow-focus medium shot, face sharp and background a complete wash of colour — camera unmoved, the stillness the point",
                "open with a rack focus — foreground sharp and her face soft, focus pulling slowly to her over the first seconds, the world sharpening as the voice enters",
                "open on a close-up of her hands or throat — tilt slowly upward to reveal the face as the first lyric begins, arriving at her eyes as the voice rises",
                "open with her slightly off-centre, rule-of-thirds framing — camera drifts toward centre over the full clip",
                "open in tight profile — camera side-on, face visible in silhouette — holds for the first beat then begins a slow rotation toward three-quarter angle",
                "open with a gentle handheld medium shot, micro-movement matching her breathing before any larger move begins",
                "open behind her right shoulder in a slight over-the-shoulder angle — drift sideways until she is fully front-facing",
                "open on a low angle slightly below eye level, the ceiling or open space visible above her — camera creeps forward, angle unchanged",
                "open at a tight two-shot distance even though only one subject is present — the intimacy of proximity as a framing choice, camera locked",
            ],
            "build":      "continue the push — frame tightening from shoulders to chin as intensity rises",
            "peak":       "frame locked tight on the face, camera completely still, the held note filling the stillness",
            "resolution": "fractional drift back as the phrase dissolves, giving the face room to breathe",
        },
        "sweeping": {
            "openings": [
                "open on a wide shot — full figure head to toe, the space around her as present as she is — begin a slow deliberate push that will not arrive until the peak",
                "open from behind, camera at the back of the space — she faces away — arc slowly around to reveal her face over the first section",
                "open very wide, environment dominant and she is small within it — camera begins closing the distance immediately, a long slow dolly compressing the world around her",
                "open with camera positioned high looking down at a slight angle — descend slowly, perspective levelling as it closes toward her face",
                "open at the far end of the room, a long-lens compression shot that makes the space behind her seem flat and enormous — push forward with deliberate weight",
                "open with a slow tilt up from the floor — beginning at her feet, rising through her body, arriving at her face as the first lyric phrase ends",
                "open with camera far back, full stage visible, she is centre but small — the lens begins a push that will take the entire clip to close half the distance",
                "open at a low angle at her feet looking up — the full figure rising above the lens — tilt upward over the first section reframing from feet to face",
                "open wide with camera slow-trucking left until she is centred while closing in simultaneously — two moves at once",
                "open with a slow crane-down from above head level to eye level while simultaneously pushing in — the frame changing in two dimensions",
                "open on an initial wide pull-back that briefly widens before the long push forward begins — the initial retreat makes the subsequent approach feel earned",
            ],
            "build":      "slow tilt up reframing from waist to face as posture opens with the melody",
            "peak":       "camera tight on the face, completely still as the voice reaches its highest point",
            "resolution": "slow pull back as the note dissolves, widening until the full figure is restored",
        },
        "driven": {
            "openings": [
                "open low angle below hip height looking up, slight handheld breathe in the frame — the figure rising powerfully above the lens",
                "open with a hard cut to a tight handheld close-up, frame slightly unstable — as if the operator arrived at the shot mid-motion and is still finding the frame",
                "open on a medium shot with slight Dutch angle — the frame canted five degrees, immediate unease before she brings the energy to level it",
                "open with a fast push-in already in progress — the frame still closing when the clip begins, arriving at medium close-up within the first two seconds",
                "open low and wide at knee height — tilt up hard to reframe on her face as the first lyric hits",
                "open behind her, low angle looking up at the back of her head and shoulders — push forward and rotate until she is centre-frame",
                "open in extreme close-up on a non-face detail — a hand, a shoulder, a jaw — pull back rapidly to reveal the full figure in two seconds then settle",
                "open at eye level but very close, slightly wide lens — face large in frame, slightly distorted at edges, space behind compressed — camera locked",
                "open with the camera tracking forward fast, arriving at medium close-up within the first second and stopping hard — the frame landing with impact then holding",
                "open on a low angle with a slight upward tilt already in progress, still settling into position as the first note hits — locks into a hard low-angle frame",
                "open on a medium shot, handheld, the operator clearly present — frame breathes with human weight — then pushes forward with controlled urgency",
            ],
            "build":      "slow tilt rising from chest to face as the voice hardens with the song",
            "peak":       "single hard push forward synced to the belt — aggressive, the frame lurching into her face",
            "resolution": "camera settles locked as the note fades, the energy draining out of the frame with the sound",
        },
        "languid": {
            "openings": [
                "hold a medium shot, a barely perceptible creep forward that will take the whole clip to complete — compression readable only in retrospect",
                "open on a wide medium shot, camera locked and completely still — no movement at all for the first section, the stillness itself a choice",
                "open with a very long focal length, background compressed flat behind her — camera does not move, depth of field doing all the work",
                "open in medium close-up positioned slightly above eye level looking down — a faint imperceptible tilt downward beginning as the voice enters",
                "open with a shallow rack focus — background sharp and she is soft, focus shifting to her slowly as if the camera is deciding to look",
                "open on a medium shot from the side in slow profile — holds for the full opening section before beginning the most gradual possible drift toward her face",
                "open on a wide shot already in slow continuous motion, a glide that began before the clip started, speed barely above zero",
                "open on a tight close-up, face filling the frame — camera locked and still, the only movement her breathing and the expressions it produces",
                "open slightly out of focus — pull focus with aching slowness, the face sharpening over the full clip from soft to pin-sharp",
                "open on a medium shot with a foreground element partially obscuring the frame — drifts sideways with glacial patience until the obstruction clears",
                "open at the exact framing that will be the peak frame — tight on the face — and hold there for the entire clip. The stillness from the start.",
            ],
            "build":      "the slow push continues — compression readable only in retrospect, the frame almost unchanged",
            "peak":       "the camera has arrived at a tight close-up, completely still, face filling the frame",
            "resolution": "hold the close frame — no move on the resolution, the stillness is the choice",
        },
        "rhythmic": {
            "openings": [
                "clean eye-level medium shot, tight and symmetrical — camera locked until the first beat drops, then it moves",
                "open on a low angle looking slightly up, figure centred and symmetrical — composition deliberate and graphic, holds for the first bar before the camera picks a direction",
                "open from a high angle looking straight down — bird's-eye view from above — descend rapidly to eye level over the first two seconds",
                "open with camera positioned exactly front-on, a flat graphic composition — begin a lateral truck to the right on the second beat, background scrolling while she stays centred",
                "open mid-motion — camera already tracking sideways, she enters the frame from the left and arrives at centre as the first lyric hits, camera stops as she does",
                "open on a wide shot then push in rapidly to medium close-up in the first two seconds — arriving before the first lyric and holding",
                "open behind her back — rear-facing medium shot — rotate fast around to her front on the first beat, the reveal landing with impact",
                "open on a low three-quarter angle, camera below hip height, clean and graphic, background geometric behind her, camera locked",
                "open in tight profile, the edge of her face cutting the frame exactly in half — truck sideways on the beat to reveal the front of her face in a clean medium close-up",
                "open on a close-up of her face, eyes directly into the lens — snap-cut wide on the first beat, revealing the full body and environment simultaneously",
                "open on a medium shot then cut immediately on the first beat to a close-up — the implied cut revealed by framing — then hold as the camera picks its next move",
            ],
            "build":      "slow lateral drift keeping her centred while the background slides past behind her",
            "peak":       "snap to low angle looking up — sudden shift synced to the loudest beat",
            "resolution": "drift back to eye level as the phrase resolves, the opening symmetry restored",
        },
        "theatrical": {
            "openings": [
                "open wide — full figure small against the space behind her, formal and composed — the space is as important as the subject",
                "open with camera at the back of the house looking down at the stage — long theatrical sightline, she is small and architecture is enormous around her",
                "open from the wings — camera at the edge of the stage, looking across it — she is centre, frame cuts the stage diagonally before camera moves to face her",
                "open above, looking down at the stage at a steep angle — she is a shape against the floor — camera descends to eye level over the opening section",
                "open on a close-up of the environment rather than the performer — stage floor, curtain, light source — tilt or push to reveal her as the first phrase begins",
                "open wide with camera pulling back as the clip begins — she is already in medium close-up and the pull-back reveals the full theatrical space in two seconds",
                "open on a long-lens medium shot from far back in the house — compression flattening stage depth, making her appear against a wall of light",
                "open on a very tight close-up — eyes, lips, detail of makeup and costume — pull back slowly to reveal the full theatrical figure, costume, and setting",
                "open on a medium shot in which she is not yet fully in the light — the camera positioned such that light will find her as it builds",
                "open front-on, perfectly centred, camera at the height of the stage floor looking slightly up — the most formal theatrical angle before the approach begins",
                "open from behind the proscenium arch — frame includes the edge of theatrical infrastructure — curtain rods, lighting rigs — before camera moves into the performance space",
            ],
            "build":      "slow push forward closing the distance between lens and performer as voice builds",
            "peak":       "medium close-up, completely still, voice filling the space around the locked frame",
            "resolution": "slow pull back as the note resolves, the figure growing smaller, the space reclaiming scale",
        },
    }

    def _get_singing_cam_profile(self, _detected_entry, _combined_input):
        import re as _re2
        ci = _combined_input.lower()
        if not _detected_entry:
            return self._SINGING_CAM_PROFILES["intimate"]
        if _re2.search(r'\b(opera|operatic|classical|orchestral|musical\s+theatre)\b', ci):
            return self._SINGING_CAM_PROFILES["theatrical"]
        if _re2.search(r'\b(rock|metal|punk|hardcore|grunge|indie\s+rock)\b', ci):
            return self._SINGING_CAM_PROFILES["driven"]
        if _re2.search(r'\b(house|techno|dnb|drum.*bass|dubstep|edm|disco|hip.?hop|rap|trap|drill|grime)\b', ci):
            return self._SINGING_CAM_PROFILES["rhythmic"]
        if _re2.search(r'\b(soul|motown|gospel|r&b|rnb)\b', ci):
            return self._SINGING_CAM_PROFILES["sweeping"]
        if _re2.search(r'\b(jazz|blues|neo.?soul|folk|americana|country|bossa)\b', ci):
            return self._SINGING_CAM_PROFILES["languid"]
        return self._SINGING_CAM_PROFILES["intimate"]

    def generate(self, **kwargs):
        """
        ComfyUI passes inputs as keyword arguments using the exact INPUT_TYPES key strings.
        We unpack them here into clean internal names.
        """
        # ── Unpack required inputs ────────────────────────────────────────────
        bypass                = kwargs.get("⏭ bypass",                     False)
        use_scene_context     = kwargs.get("🖼 use image information?",     True)
        invent_dialogue       = kwargs.get("💬 let the LLM create dialogue?", True)
        user_input            = kwargs.get("user_input",                    "")
        creativity            = kwargs.get("🎨 creativity",                 "0.8 - Balanced Professional")
        shot_angle            = kwargs.get("📐 shot angle",                 "None — LLM decides")
        camera_movement       = kwargs.get("🎥 camera movement",            "None — LLM decides")
        style_preset          = kwargs.get("🎬 style preset",               "None — let the LLM decide")
        spoken_language_sel   = kwargs.get("🗣 spoken language",            "Auto — use existing prompt logic")
        seed                  = kwargs.get("seed",                          -1)
        control_after_generate = kwargs.get("control_after_generate",       "randomize")
        # ── Unpack optional inputs ────────────────────────────────────────────
        subject_count         = kwargs.get("👥 subject count",              0)
        negative_bias         = kwargs.get("🚫 things to avoid",            "")
        lora_triggers         = kwargs.get("🏷 lora triggers",              "")
        music_genre           = kwargs.get("🎵 music genre",                "None — detect from prompt")
        emotional_state_sel   = kwargs.get("💋 emotional state",            "None — LLM decides")
        environment_sel       = kwargs.get("🌍 environment",                 "None — LLM decides")
        audio_input           = kwargs.get("audio_input",                   None)
        audio_enabled         = kwargs.get("🔊 use audio for the LLM?",     False)
        use_whisper           = kwargs.get("📝 transcribe the audio?",       False)
        frame_count           = kwargs.get("⏱ frame count",                 192)
        width                 = kwargs.get("↔ width",                       0)
        height                = kwargs.get("↕ height",                      0)
        backend               = kwargs.get("🧩 backend",                    "transformers")
        local_path            = kwargs.get("📁 local model path",            "")
        gguf_repo_id          = kwargs.get("🦙 gguf repo",                   "lukey03/Qwen3.5-9B-abliterated-GGUF")
        gguf_filename         = kwargs.get("🦙 gguf file",                   "Qwen3.5-9B-abliterated-Q4_K_M.gguf")
        llama_n_gpu_layers    = kwargs.get("🦙 n_gpu_layers",                -1)
        llama_n_ctx           = kwargs.get("🦙 context size",                8192)
        llama_n_batch         = kwargs.get("🦙 batch size",                  512)
        llama_server_url      = kwargs.get("🌐 llama-server url",            "http://127.0.0.1:8080/v1")
        llama_server_model    = kwargs.get("🌐 llama-server model",          "")
        llama_server_api_key  = kwargs.get("🔑 llama-server api key",        "")
        offline_mode          = kwargs.get("✈ offline mode",                False)
        gpu_id                = kwargs.get("🧠 GPU ID",                      -1)
        selected_model        = kwargs.get("🤖 model",                         "huihui-ai/Huihui-Qwen3.5-9B-abliterated")
        keep_model_loaded     = kwargs.get("🔗 keep model loaded",           False)
        scene_context         = kwargs.get("🖼 scene context",               "")
        # ── Derived ───────────────────────────────────────────────────────────
        portrait_mode = (width > 0 and height > 0 and height > width)
        # ── Bypass ────────────────────────────────────────────────────────────
        if bypass:
            if self.model is not None and not keep_model_loaded:
                self.unload_model()
            neg = _build_negative_prompt("", user_input, is_portrait=portrait_mode, style_preset=style_preset)
            return (user_input.strip(), user_input.strip(), neg)

        # ── VRAM prep ─────────────────────────────────────────────────────────
        try:
            import comfy.model_management as mm
            mm.unload_all_models()
            mm.soft_empty_cache()
        except Exception:
            pass
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            a = torch.cuda.memory_allocated() / 1024**3
            r = torch.cuda.memory_reserved()  / 1024**3
            print(f"[LTX2-Qwen] Pre-run VRAM: {a:.2f}GB alloc / {r:.2f}GB reserved")

        self.load_model(
            offline_mode=offline_mode,
            local_path=local_path,
            model_id=selected_model,
            backend=backend,
            gpu_id=gpu_id,
            gguf_repo_id=gguf_repo_id,
            gguf_filename=gguf_filename,
            llama_n_gpu_layers=llama_n_gpu_layers,
            llama_n_ctx=llama_n_ctx,
            llama_n_batch=llama_n_batch,
            llama_server_url=llama_server_url,
            llama_server_model=llama_server_model,
            llama_server_api_key=llama_server_api_key,
        )

        # ── Style preset ──────────────────────────────────────────────────────
        preset_data            = self.STYLE_PRESETS.get(style_preset, ("", False))
        style_instruction_text = preset_data[0]
        is_portrait            = portrait_mode or preset_data[1]
        style_label            = self.PRESET_STYLE_LABEL.get(style_preset, "")

        # ── Auto-preset selection when None is chosen ────────────────────────
        # Keyword-driven — only fires on confident matches.
        # Never picks music unless music keywords are explicitly present.
        # Falls back gracefully (no match = LLM decides freely).
        if not style_instruction_text and style_preset == "None — let the LLM decide":
            import re as _apre
            _ap = (user_input + " " + scene_context).lower()
            _auto_preset = None

            if _apre.search(r'\b(sing\w*|sang|sung|music\s+video|rapper?|'
                            r'perform\w*|concert|on\s+stage|microphone|'
                            r'\bmic\b|lyrics?|chorus|verse)\b', _ap):
                _auto_preset = "Music video — stylised"

            elif _apre.search(r'\b(gravure|glamour\s+shoot|bikini\s+shoot|pinup)\b', _ap):
                # Gravure is a Japanese/Korean/East Asian genre — only auto-route there for East Asian subjects.
                # Western/other nationalities get Erotic cinema instead.
                _is_east_asian_context = bool(_apre.search(
                    r'\b(japanese|korean|chinese|asian|east\s+asian|'
                    r'japan|korea|china|taiwan|thai|vietnamese|'
                    r'idol|j-?pop|k-?pop)\b', _ap
                ))  # Note: 'gravure' intentionally excluded — we're already in the gravure branch
                _is_western_context = bool(_apre.search(
                    r'\b(french|german|italian|spanish|portuguese|russian|'
                    r'american|british|australian|european|scandinavian|'
                    r'swedish|norwegian|danish|dutch|polish|greek|'
                    r'latina(?!\s+asian)|brazilian|argentinian|colombian|mexican)\b', _ap
                ))
                if _is_western_context and not _is_east_asian_context:
                    _auto_preset = "Erotic cinema — tasteful, cinematic"
                else:
                    _auto_preset = "Gravure Idol — Japanese glamour"

            elif _apre.search(r'\b(epic|vast|mountain|cliff|desert|'
                              r'lone\b|alone\b|astronaut|battlefield|'
                              r'horizon|wilderness|volcano|glacier|canyon|'
                              r'tundra|space\b|planet\b|storm\b)\b', _ap):
                _auto_preset = "Cinematic — Epic"

            elif _apre.search(r'\b(fight\w*|chase\w*|explosion\w*|'
                              r'battle\w*|combat\w*|\bgun\b|\bsword\b|'
                              r'martial|action\s+scene)\b', _ap):
                _auto_preset = "Action — handheld, kinetic"

            elif _apre.search(r'\b(horror|haunted|creature|monster|'
                              r'\bdemon\b|\bghost\b|supernatural|dread)\b', _ap):
                _auto_preset = "Horror — desaturated, harsh contrast"

            elif _apre.search(r'\b(detective|\bnoir\b|\bcrime\b|'
                              r'interrogat\w*|trenchcoat|\bheist\b)\b', _ap):
                _auto_preset = "Noir — deep shadows, venetian light"

            elif _apre.search(r'\b(sci.?fi|spaceship|android|\brobot\b|'
                              r'cyborg|futuristic|cyberpunk|dystopia|hologram)\b', _ap):
                _auto_preset = "Sci-fi — cinematic, practical"

            elif _apre.search(r'\b(close.?up|intimate|whisper\w*|'
                              r'\btears?\b|crying|vulnerable|tender\b)\b', _ap):
                _auto_preset = "Cinematic — Intimate close-up"

            elif _apre.search(r'\b(says?\b|tells?\b|asks?\b|dialogue|'
                              r'conversation|confrontat\w*|argument\b|'
                              r'two\s+people|sitting\b|\bkitchen\b)\b', _ap):
                _auto_preset = "Cinematic — Drama"

            if _auto_preset and _auto_preset in self.STYLE_PRESETS:
                _auto_data = self.STYLE_PRESETS[_auto_preset]
                style_instruction_text = _auto_data[0]
                if _auto_data[1]: is_portrait = True
                style_label = self.PRESET_STYLE_LABEL.get(_auto_preset, "")
                print(f"[LTX2-Qwen] Auto-preset: {_auto_preset}")
            else:
                print("[LTX2-Qwen] Auto-preset: no confident match — LLM decides freely")

        # ── User style keyword detection ──────────────────────────────────────
        # If the user types a visual style in their prompt, detect it and inject
        # it as an ADDITIONAL style note. This allows "animation, Chel doing X"
        # to produce an animated output even when the preset is "Music video".
        # The preset handles the aesthetic wrapper; the user keyword handles the
        # render style. They coexist — neither cancels the other.

        _USER_STYLE_MAP = [
            # ── Animation & cartoon styles ────────────────────────────────────
            (r'\b(2d\s+animat\w*|hand.?drawn\s+animat\w*|traditional\s+animat\w*|'
             r'hand.?painted\s+animat\w*|classic\s+animat\w*|'
             r'old.?school\s+(?:cartoon|animat)\w*|saturday\s+morning\s+cartoon\w*|'
             r'flat\s+animat\w*|squash.?and.?stretch)\b',
             "2D hand-drawn animation. Expressive ink outlines, flat colour fills, squash-and-stretch physics. Classic cartoon timing."),

            (r'\b(3d\s+animat\w*|cgi\s+animat\w*|pixar\w*|dreamworks\w*|'
             r'3d\s+render\w*|computer\s+animat\w*|3d\s+cartoon\w*|'
             r'toy\s+story\w*|disney\s+3d\w*|illumination\w*)\b',
             "3D CGI animation. Subsurface-scattered skin, expressive faces, warm three-point lighting, smooth cinematic camera moves. Pixar/DreamWorks render quality."),

            (r'\b(animat\w*|cartoon\w*|animated\s+film\w*|animated\s+movie\w*|'
             r'animated\s+series\w*|animated\s+show\w*|animated\s+style\w*|'
             r'animated\s+scene\w*|cartoon\s+style\w*|cartoon\s+film\w*)\b',
             "Animation render style. The scene is fully animated — not live action. Every subject and environment rendered as animation. Clean outlines, stylised physics, fluid movement."),

            (r'\b(anime\w*|manga\s+style\w*|japanese\s+animat\w*|'
             r'shonen\w*|shojo\w*|seinen\w*|mecha\w*|slice\s+of\s+life\s+anime\w*|'
             r'studio\s+ghibli\w*|ghibli\w*|makoto\s+shinkai\w*|'
             r'a-1\s+pictures\w*|kyoto\s+animation\w*|trigger\s+anime\w*)\b',
             "Japanese anime style. Hand-drawn cel animation, large expressive eyes, clean ink outlines, vivid saturated palette. Speed lines on motion, slow drifts on emotional beats."),

            (r'\b(cel.?shad\w*|toon.?shad\w*|cel.?render\w*|flat.?shad\w*|'
             r'borderlands\s+style\w*|comic\s+shad\w*)\b',
             "Cel-shaded 3D. Hard shadow threshold, flat stepped colour fills, ink outlines on all silhouettes. The image reads as animated despite being 3D."),

            (r'\b(stop.?motion\w*|claymation\w*|clay\s+animat\w*|puppet\s+animat\w*|'
             r'laika\w*|aardman\w*|wallace\s+and\s+gromit\w*|'
             r'coraline\s+style\w*|kubo\s+style\w*)\b',
             "Stop motion claymation. Physical clay or puppet aesthetic — visible fingerprints, slight imperfections, 12fps deliberate movement. Matte tactile textures, practical miniature sets."),

            (r'\b(rotoscop\w*|animated\s+over\s+live\w*|traced\s+animat\w*)\b',
             "Rotoscoped animation. Hand-drawn outlines traced over live-action movement — uncanny human accuracy inside an illustrated skin. Wobbly variable-weight outlines."),

            (r'\b(comic\s*book\w*|graphic\s*novel\w*|marvel\s+style\w*|dc\s+style\w*|'
             r'ink\s+and\s+colour\w*|halftone\w*|ben\s+day\s+dots\w*|'
             r'jack\s+kirby\w*|frank\s+miller\w*)\b',
             "Comic book / graphic novel style. Bold ink outlines, halftone dot shadows, flat colour with hard-edged shadow, dynamic Dutch angles, speed lines on action."),

            # ── Video game styles ─────────────────────────────────────────────
            (r'\b(video\s+game\w*|game\s+cinemat\w*|game\s+cutscene\w*|'
             r'unreal\s+engine\w*|unity\s+render\w*|game\s+engine\w*|'
             r'in.?engine\s+cinemat\w*)\b',
             "Video game cinematic render. Real-time engine quality — high-detail textures, dynamic lighting, physically-based rendering. Smooth cutscene camera work."),

            (r'\b(pixel\s*art\w*|8.?bit\w*|16.?bit\w*|retro\s+game\w*|'
             r'sprite\w*|chiptune\s+visual\w*|nes\s+style\w*|'
             r'snes\s+style\w*|atari\s+style\w*)\b',
             "Pixel art / retro game aesthetic. Low-resolution sprites, limited colour palette, chunky pixels visible. Dithering for shading. Movement in grid-snapped increments."),

            # ── Cinematic & film styles ───────────────────────────────────────
            (r'\b(cinematic\s+drama\w*|dramatic\s+film\w*|narrative\s+film\w*|'
             r'prestige\s+drama\w*|prestige\s+tv\w*|hbo\s+style\w*|'
             r'oscar\s+bait\w*|awards\s+drama\w*)\b',
             "Prestige cinematic drama. Nuanced lighting, ARRI Alexa warmth, shallow depth of field. Restrained colour grade — desaturated highlights, warm shadows. Every frame compositionally considered."),

            (r'\b(film\s+noir\w*|noir\w*|neo.?noir\w*|black\s+and\s+white\s+film\w*|'
             r'expressionist\w*|chiaroscuro\w*|hard\s+shadow\w*|'
             r'venetian\s+blind\s+light\w*)\b',
             "Film noir / neo-noir. Deep shadows, harsh contrast, venetian-blind light slicing through darkness. Desaturated or black-and-white palette. Moody, fatalistic atmosphere."),

            (r'\b(horror\s+film\w*|horror\s+movie\w*|horror\s+style\w*|'
             r'slasher\w*|psychological\s+horror\w*|body\s+horror\w*|'
             r'folk\s+horror\w*|cosmic\s+horror\w*|dread\w*|'
             r'unsettling\w*|creepy\s+film\w*)\b',
             "Horror film aesthetic. Desaturated colour, harsh high-contrast lighting, deep shadow. Dutch angles heighten unease. Slow methodical camera movement. Dread over shock."),

            (r'\b(sci.?fi\s+film\w*|science\s+fiction\s+film\w*|space\s+opera\w*|'
             r'cyberpunk\s+film\w*|blade\s+runner\w*|dune\s+style\w*|'
             r'interstellar\s+style\w*|alien\s+film\w*|'
             r'practical\s+sci.?fi\w*|hard\s+sci.?fi\w*)\b',
             "Cinematic sci-fi. Practical set design, anamorphic lens compression. Palette: cool desaturated blues and teals with sharp neon accents. Scale and isolation are key visual themes."),

            (r'\b(action\s+film\w*|blockbuster\w*|summer\s+blockbuster\w*|'
             r'michael\s+bay\w*|action\s+movie\w*|action\s+sequence\w*|'
             r'teal\s+and\s+orange\w*|action\s+cinemat\w*)\b',
             "Action blockbuster. Teal and orange colour grade, dynamic camera, handheld energy on action. Wide establishing shots alternate with tight impact close-ups."),

            (r'\b(documentary\s+film\w*|documentary\s+style\w*|verité\w*|'
             r'cinema\s+verité\w*|talking.?head\w*|fly.?on.?the.?wall\s+film\w*|'
             r'observational\s+film\w*)\b',
             "Documentary film style. Cinema vérité — natural available light, handheld camera, unpolished framing. Subjects behave as if the camera is not there."),

            (r'\b(wes\s+anderson\w*|symmetrical\s+composition\w*|'
             r'deadpan\s+comedy\w*|quirky\s+film\w*|'
             r'pastel\s+film\w*|twee\s+aesthetic\w*)\b',
             "Wes Anderson symmetrical aesthetic. Dead-centre composition, pastel palette, flat deadpan staging. Whip pans between setups. Every frame artificially balanced."),

            (r'\b(wong\s+kar.?wai\w*|atmospheric\s+film\w*|'
             r'slow\s+cinema\w*|contemplative\s+film\w*|'
             r'in\s+the\s+mood\s+for\s+love\w*|fallen\s+angels\w*|'
             r'tarkovsky\w*|kubrick\w*|lynchian\w*|david\s+lynch\w*)\b',
             "Arthouse / slow cinema. Long takes, minimal camera movement, rich atmospheric texture. Mood over narrative. Available light or single-source practicals. Time is allowed to breathe."),

            (r'\b(western\s+film\w*|spaghetti\s+western\w*|cowboy\s+film\w*|'
             r'leone\s+style\w*|sergio\s+leone\w*|'
             r'dust\s+and\s+sun\w*|frontier\w*\s+cinemat\w*)\b',
             "Western / spaghetti western. Extreme close-ups on eyes and hands alternating with vast wide establishing shots. Dust haze, harsh midday sun, deep shadow under hat brims."),

            (r'\b(period\s+film\w*|period\s+drama\w*|costume\s+drama\w*|'
             r'historical\s+film\w*|victorian\s+film\w*|'
             r'1920s\s+film\w*|1930s\s+film\w*|1940s\s+film\w*|'
             r'1950s\s+film\w*|1960s\s+film\w*|1970s\s+film\w*|'
             r'1980s\s+film\w*|1990s\s+film\w*|regency\s+film\w*)\b',
             "Period / historical drama. Costume and production design authentic to the era. Natural or period-appropriate lighting — candlelight, gaslamp, tungsten. Warm desaturated palette."),

            # ── Music video styles ────────────────────────────────────────────
            (r'\b(music\s+video\w*|mv\s+style\w*|kpop\s+mv\w*|'
             r'hip\s+hop\s+video\w*|pop\s+music\s+video\w*|'
             r'rnb\s+video\w*|r&b\s+video\w*|rap\s+video\w*|'
             r'edm\s+video\w*|electronic\s+music\s+video\w*)\b',
             "Music video aesthetic. Stylised rhythm-cut visuals synced to the beat. High-contrast colour grade. Camera movement matches musical energy. Style-forward over narrative realism."),

            # ── Photography & fashion ─────────────────────────────────────────
            (r'\b(fashion\s+editorial\w*|editorial\s+photo\w*|'
             r'vogue\s+style\w*|harper.?s\s+bazaar\w*|'
             r'high\s+fashion\w*|runway\s+photo\w*|'
             r'fashion\s+film\w*|couture\s+film\w*)\b',
             "High fashion editorial. Clean even lighting with strong rim light. Composed, deliberate camera. Model-aware posing. Colour grade: warm neutrals, lifted blacks, polished skin tones."),

            (r'\b(portrait\s+photo\w*|studio\s+portrait\w*|'
             r'headshot\w*|beauty\s+shot\w*|beauty\s+photo\w*|'
             r'glamour\s+photo\w*|boudoir\w*|pinup\w*|pin.?up\w*)\b',
             "Glamour portrait photography. Soft studio lighting — large diffused key, gentle fill, rim separation. Shallow depth of field. Flattering skin tone rendering."),

            (r'\b(street\s+photo\w*|candid\s+photo\w*|'
             r'urban\s+photo\w*|reportage\w*|'
             r'leica\s+style\w*|Henri\s+Cartier.?Bresson\w*)\b',
             "Street / candid photography. Available light, handheld, decisive-moment framing. Grain present. No artificial lighting. Subjects caught in natural motion."),

            # ── Lo-fi & vintage ───────────────────────────────────────────────
            (r'\b(vhs\w*|lo.?fi\s+video\w*|home\s+video\w*|'
             r'camcorder\w*|videotape\w*|analog\s+video\w*|'
             r'super\s+8\w*|8mm\s+film\w*|16mm\s+film\w*|'
             r'grain\s+film\w*|film\s+grain\w*|scratchy\s+film\w*)\b',
             "Lo-fi / vintage video. VHS tracking artifacts, soft focus, colour bleed, visible grain or noise. Camcorder or super-8 aesthetic. Imperfect and warm."),

            # ── Illustration & art styles ─────────────────────────────────────
            (r'\b(oil\s+paint\w*|oil\s+canvas\w*|painted\s+style\w*|'
             r'painterly\w*|impressionist\w*|expressionist\s+art\w*|'
             r'watercolou?r\w*|gouache\w*|acrylic\s+paint\w*)\b',
             "Painterly art style. Visible brushwork texture, expressive colour mixing, soft diffused edges. Movement in paint — not photographic. Light handled as colour temperature shifts."),

            (r'\b(concept\s+art\w*|matte\s+paint\w*|'
             r'cinematic\s+concept\w*|keyframe\s+art\w*|'
             r'artstation\s+style\w*|digital\s+art\w*|'
             r'digital\s+paint\w*|illustration\s+style\w*)\b',
             "Cinematic concept art / digital illustration. High-detail digital painting, dramatic lighting, rich atmospheric depth. Every element designed for maximum visual impact."),

            (r'\b(neon\s+art\w*|synthwave\w*|retrowave\w*|'
             r'vaporwave\w*|outrun\w*|80s\s+aesthetic\w*|'
             r'neon\s+noir\w*|miami\s+vice\s+style\w*)\b',
             "Synthwave / retrowave neon aesthetic. Magenta, cyan, electric purple against deep black. Grid lines on the horizon, neon outlines, chrome surfaces. 80s retrofit futurism."),

            # ── Special formats ───────────────────────────────────────────────
            (r'\b(pov\s+video\w*|first.?person\s+video\w*|'
             r'pov\s+shot\w*|first.?person\s+shot\w*|'
             r'immersive\s+pov\w*|gopro\s+style\w*)\b',
             "First-person POV. The camera IS the viewer's eyes. Frame moves as a head would — natural breathing movement, slight tilt on turns. Everything seen, not watched."),

            (r'\b(selfie\s+video\w*|self.?shot\w*|vlog\s+style\w*|'
             r'tiktok\s+style\w*|reels\s+style\w*|'
             r'vertical\s+video\w*|9.?16\s+video\w*|'
             r'mobile\s+video\w*|phone\s+video\w*)\b',
             "Selfie / vertical mobile video. Self-shot arm-length framing, 9:16 vertical, subject fully aware of camera. Natural available light. Intimate direct address."),

            (r'\b(drone\s+footage\w*|aerial\s+video\w*|'
             r'overhead\s+shot\w*|bird.?s.?eye\s+video\w*|'
             r'top.?down\s+video\w*|dji\s+style\w*)\b',
             "Aerial drone footage. Camera descending from altitude or sweeping laterally across the scene. Subject small against vast environment. Smooth gimbal stabilisation. Wide establishing scale."),

            (r'\b(slow\s+motion\w*|slow.?mo\w*|high\s+speed\s+camera\w*|'
             r'phantom\s+camera\w*|overcranked\w*|ramping\s+speed\w*)\b',
             "Slow motion. Overcranked high-speed camera — motion stretched and elongated. Every physical detail visible: water droplets, fabric ripple, hair separation, micro expressions. Dreamy tempo."),

            (r'\b(timelapse\w*|time.?lapse\w*|hyperlapse\w*|'
             r'time\s+compression\w*)\b',
             "Timelapse / hyperlapse. Compressed time — clouds race, shadows sweep, crowds blur into streams. Camera either locked off or moving smoothly through space as time compresses."),

            (r'\b(underwater\s+shot\w*|underwater\s+film\w*|'
             r'submerged\s+camera\w*|aquatic\s+cinemat\w*)\b',
             "Underwater cinematography. Caustic light ripples on every surface, muffled sound, slightly desaturated palette with blue-green cast. Slow drifting movement. Bubbles catch the light."),
        ]

        import re as _re

        # Detect which user style keywords fired
        _user_style_hits = []
        for _pattern, _description in _USER_STYLE_MAP:
            if _re.search(_pattern, user_input, _re.IGNORECASE):
                _user_style_hits.append(_description)

        # Build the user style note — injected alongside (not replacing) the preset
        if _user_style_hits:
            _user_style_note = (
                "\n[USER VISUAL STYLE — MANDATORY: The user has specified a visual style in their prompt. "
                "This OVERRIDES the render style of the preset above — it does NOT override the preset's "
                "camera, lighting, or mood instructions. Apply the user's stated visual style as the "
                "primary render aesthetic: " + " | ".join(_user_style_hits) + "]"
            )
        else:
            _user_style_note = ""

        if style_instruction_text:
            style_instruction = (
                f"\n[STYLE INSTRUCTION — MANDATORY AESTHETIC ANCHOR: {style_instruction_text} "
                f"Every aspect of the output — lighting, camera, colour, pacing, mood — must reflect this style. "
                f"CRITICAL: You MUST begin your output with exactly these words: \"{style_label}\" — "
                f"then continue with the scene description. This label must be the very first words of your output. "
                f"Do not deviate from this style.]"
                + _user_style_note
            )
        else:
            # No preset — user style note alone drives the style
            style_instruction = _user_style_note

        if is_portrait:
            portrait_instruction = (
                "\n[PORTRAIT MODE — MANDATORY: This is a 9:16 vertical video for mobile. "
                "All framing must be vertical — tight head-to-torso shots. "
                "No wide horizontal establishing shots. Action moves vertically in frame. "
                "Camera stays close. Optimised for TikTok, Reels, Shorts.]"
            )
        else:
            portrait_instruction = ""

        self._last_portrait = is_portrait
        self._last_style    = style_preset

        # ── Seed ──────────────────────────────────────────────────────────────
        if seed != -1:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # ── Word budget ───────────────────────────────────────────────────────
        # is_gravure needed early for dialogue scene detection
        is_gravure = "gravure" in style_preset.lower()

        real_seconds  = frame_count / 30.0          # 30fps throughout
        action_count  = max(1, min(6, round(real_seconds / 3.5)))
        # 30f=1s→1  90f=3s→1  120f=4s→1  168f=5.6s→2  192f=6.4s→2
        # 240f=8s→2  300f=10s→3  360f=12s→3  480f=16s→5
        LTX_WORD_FLOOR   = 120
        LTX_WORD_CEILING = 560  # raised — explicit+singing scenes need room for arc + undressing + vocals
        token_val        = max(LTX_WORD_FLOOR, min(LTX_WORD_CEILING, action_count * 75 + 100))
        max_tokens       = int(token_val * 1.4)

        # Pre-initialise variables used in the singing pacing hint block.
        # is_explicit, is_sensual, _is_sex_scene needed early for budget calculation.
        is_explicit    = False
        is_sensual     = False
        _is_sex_scene  = False  # pre-init — real value assigned after content detection below
        # These are normally assigned later in generate() but singing detection
        # fires early to influence the word budget — so they must exist first.
        _detected_entry = None
        _active_scene_context = scene_context if use_scene_context else ""
        _combined_input = user_input + " " + _active_scene_context

        # ── Singing detection — must come BEFORE dialogue scene detection ─────
        # Detected early so it can influence the token budget and pacing model.
        # Singing is fundamentally different from dialogue — it is CONTINUOUS,
        # it fills the whole scene, lyrics ARE the performance not punctuation.

        # Early sex-context pre-check — runs before _is_singing so we can
        # suppress ambiguous words that mean different things in sex vs music.
        # "vocal" in a sex scene = loud/expressive/orgasmic. NOT singing.
        # "moaning", "screaming", "loud", "crying out" = sex response. NOT singing.
        # We use _EXPLICIT_RE (class-level) + a position/act keyword list.
        _SEX_CONTEXT_EARLY_RE = re.compile(
            r'\b(cowgirl|missionary|doggy|doggystyle|doggy.?style|'
            r'reverse\s+cowgirl|riding\s+(?:him|cock|dick)|'
            r'blowjob|blow\s+job|handjob|hand\s+job|sixty.?nine|69|'
            r'anal|fingering|going\s+down|eating\s+out|'
            r'sex|fucking|fucked|penetrat\w*|thrust\w*|'
            r'pussy|cock|dick|cum\b|cumming|orgasm\w*|climax\w*|'
            r'moaning\s+loud|screaming\s+loud|cries?\s+out|'
            r'loud\s+and\s+vocal|very\s+vocal|being\s+vocal|'
            r'loud\s+(?:sex|moans?|orgasm)|noisy\s+sex|'
            r'swearing\s+(?:during|while|throughout)|'
            r'screams?\s+(?:in\s+)?(?:pleasure|ecstasy)|'
            r'wails?\s+(?:in\s+)?(?:pleasure|ecstasy))\b',
            re.IGNORECASE
        )
        _is_sex_context_early = (
            bool(self._EXPLICIT_RE.search(_combined_input)) or
            bool(_SEX_CONTEXT_EARLY_RE.search(_combined_input))
        )

        # Build singing regex — suppress sex-ambiguous terms when in sex context.
        # In sex context: vocal/vocals = expressive, moaning = orgasmic, NOT singing.
        # In music context: vocals/vocal performance = singing component.
        if _is_sex_context_early:
            _singing_pattern = (
                r'\b(sing\w*|sang|sung|hum\w*|lullaby|lullabies|song\w*|melody|melodic|'
                r'croon\w*|chant\w*|serenade\w*|'
                r'vocal\s+(performance|range|coach|warm.?up|harmonies|track|style|riff)|'
                r'lyrics?|chorus|verse|'
                r'belt\w*|warble\w*|performs?\s+a\s+song|singing\s+along|'
                r'music\s+video\s+singing|lip\s+sync\w*|lip.?synch\w*|'
                r'kpop\s+mv\w*|idol\s+perf\w*)\b'
            )
        else:
            _singing_pattern = (
                r'\b(sing\w*|sang|sung|hum\w*|lullaby|lullabies|song\w*|melody|melodic|'
                r'croon\w*|chant\w*|serenade\w*|vocals?\b|'
                r'vocal\s+(performance|range|coach|warm.?up|harmonies|track|style|riff)|'
                r'lyrics?|chorus|verse|'
                r'belt\w*|warble\w*|performs?\s+a\s+song|singing\s+along|'
                r'music\s+video\s+singing|lip\s+sync\w*|lip.?synch\w*|'
                r'kpop\s+mv\w*|idol\s+perf\w*)\b'
            )
        _is_singing = bool(re.search(_singing_pattern, user_input, re.IGNORECASE))

        # ── Dialogue-focused scene detection ──────────────────────────────────
        # Fires when the primary purpose of the scene is speech or singing —
        # gravure with dialogue, ASMR, talking to camera, direct address,
        # or any singing scene. Changes the pacing model so each beat IS a
        # vocal moment, not physical action with words squeezed alongside.
        _is_dialogue_scene = (
            _is_singing or
            (is_gravure and invent_dialogue) or
            bool(re.search(
                r'\b(asmr|talking|talks?\s+to\s+(the\s+)?camera|speaks?\s+to|'
                r'addresses|monologue|narrat\w*|whispers?\s+to|speaks?\s+softly|'
                r'says?\s+something|tells?\s+(?:you|us|me|them)|'
                r'explains?|describes?|confesses?|admits?)\b',
                user_input, re.IGNORECASE
            ))
        )

        # Singing/dialogue/sex scenes get a higher word floor
        if _is_dialogue_scene and invent_dialogue:
            if _is_singing:
                # Singing needs more room — four performance sections with lyrics throughout.
                if is_explicit or is_sensual:
                    token_val = max(token_val, 560)
                else:
                    token_val = max(token_val, 450)
            else:
                token_val = max(token_val, 200)
            max_tokens = int(token_val * 1.4)

        # Sex scenes always get a minimum 500 word budget — need full arc + position mechanics
        if _is_sex_scene and invent_dialogue:
            token_val  = max(token_val, 500)
            max_tokens = int(token_val * 1.4)

        print(f"[LTX2-Qwen] Budget: ~{token_val}w / {max_tokens} max | {real_seconds:.0f}s | {action_count} actions | dialogue_scene={_is_dialogue_scene}")

        if _is_dialogue_scene and invent_dialogue:
            if _is_singing:
                # ── Singing-specific pacing ───────────────────────────────────
                # Singing is CONTINUOUS — the lyrics fill the clip from start to
                # end. It is not beats of action with occasional words dropped in.
                # The model needs to understand that every sentence of the prompt
                # should describe voice, melody, delivery, and physical performance
                # simultaneously — not alternate between them.
                # Camera profile — genre-aware moves per section
                _cam_prof = self._get_singing_cam_profile(_detected_entry, _combined_input)
                _cam_open_rng = __import__("random").Random((seed + 17) if seed != -1 else None)
                _cam_opening = _cam_open_rng.choice(_cam_prof["openings"])

                # ── Emotional state — seed-randomised ─────────────────────────
                # Slow/emotional genres can draw from the full pool including crying.
                # Fast/high-energy genres draw from the no-tears pool only.
                import re as _re3
                _ci_lower = _combined_input.lower()
                _is_emotional_genre = bool(_re3.search(
                    r'\b(jazz|blues|soul|motown|gospel|r&b|rnb|neo.?soul|folk|americana|'
                    r'country|classical|opera|operatic|cabaret|bossa|flamenco)\b',
                    _ci_lower
                ))
                _emotion_rng = __import__('random').Random((seed + 3) if seed != -1 else None)

                # Full pool — includes overcome/crying, for emotional genres only
                _EMOTION_POOL_FULL = [
                    # Overcome / crying
                    "EMOTIONAL STATE — OVERCOME: A single tear has tracked from the outer corner of her left eye, "
                    "drying to a faint line below the cheekbone. Her jaw works between phrases, holding something back. "
                    "At the peak her chin drops fractionally and her eyes press shut — not performing grief, feeling it.",

                    # Vulnerable / raw
                    "EMOTIONAL STATE — VULNERABLE: Her chin trembles slightly on the sustained notes, "
                    "the lower lip pulling inward between phrases as she resets. Her eyes are wet but not spilling — "
                    "the surface tension is the drama. She does not look away from the lens.",

                    # Defiant
                    "EMOTIONAL STATE — DEFIANT: Her jaw is set, the emotion locked behind her eyes rather than released. "
                    "She holds direct eye contact through every phrase. The feeling is visible only in the "
                    "tightness at the corners of her mouth and the way her hands grip harder on the held note.",

                    # Euphoric / breakthrough
                    "EMOTIONAL STATE — EUPHORIC: A real smile breaks through between phrases — unguarded, "
                    "involuntary, gone before the next line starts. Her eyes catch light differently when it happens, "
                    "the performance cracking open for one beat before closing again.",

                    # Controlled / contained
                    "EMOTIONAL STATE — CONTROLLED: Her expression is locked — professional composure "
                    "that costs her something. Only her hands betray it: fingers pressing into her thigh, "
                    "knuckles whitening on the peak note, releasing as the phrase fades.",
                ]

                # Reduced pool — no crying/tears for high-energy genres
                _EMOTION_POOL_FAST = [
                    # Defiant
                    "EMOTIONAL STATE — DEFIANT: Jaw set, eyes hard and direct, the emotion locked in rather than released. "
                    "She holds the lens through every phrase. The tightness at the corners of her mouth "
                    "is the only tell — everything else is pure forward energy.",

                    # Euphoric
                    "EMOTIONAL STATE — EUPHORIC: A real unguarded smile breaks through at the peak, "
                    "gone before the next line starts. Her face opens completely for one beat "
                    "then closes back into performance mode — the crack in the mask is the moment.",

                    # Fierce / in it
                    "EMOTIONAL STATE — FIERCE: Eyes half-closed, completely internal — she is not performing "
                    "to the camera but through it. Her expression sharpens on every stressed syllable, "
                    "the body and the voice arriving at the same place simultaneously.",

                    # Controlled
                    "EMOTIONAL STATE — CONTROLLED: Expression locked and professional. "
                    "Only the hands betray the intensity — fingers pressing into her thigh on the peak note, "
                    "releasing with the phrase. The face gives nothing, the body gives everything.",
                ]

                # User emotion override — if they described an emotional state, use it.
                # The pool only fires when the user gave no emotional direction.
                _USER_EMOTION_RE = _re3.compile(
                    r'\b(cr(?:y|ies|ying|ied)|tear(?:s|ful)?|weep(?:ing|s)?|sobbing?|'
                    r'laugh(?:ing|s)?|smil(?:ing|es?)|grin(?:ning|s)?|'
                    r'angry|furious|rage|joy(?:ful)?|euphoric|broken|devastat\w*|'
                    r'blissful|ecstatic|grief|grieving|heartbroken|elated|'
                    r'nervous|scared|afraid|terrified|confident|fierce|defiant|'
                    r'emotional(?:ly)?|overwhelmed|overcome|vulnerable|raw)\b',
                    _re3.IGNORECASE
                )
                _user_gave_emotion = bool(_USER_EMOTION_RE.search(_combined_input))
                if _user_gave_emotion:
                    # User described the emotional state — let the LLM read it directly,
                    # suppress the pool entirely so we don't conflict
                    _emotion_state = (
                        "EMOTIONAL STATE — USER DEFINED: The user has described a specific emotional state. "
                        "Read it from the input above and apply it with full physical specificity — "
                        "face, body, breath, hands. Do NOT substitute or soften it."
                    )
                    print("[LTX2-Qwen] Emotion: user-defined — pool suppressed")
                else:
                    _emotion_pool = _EMOTION_POOL_FULL if _is_emotional_genre else _EMOTION_POOL_FAST
                    _emotion_state = _emotion_rng.choice(_emotion_pool)
                    print(f"[LTX2-Qwen] Emotion: {_emotion_state[:40]}...")

                # ── Sweat / body condition — genre-energy-aware ───────────────
                _sweat_rng = __import__('random').Random((seed + 7) if seed != -1 else None)

                _is_high_energy = bool(_re3.search(
                    r'\b(rock|metal|punk|hardcore|hip.?hop|rap|trap|drill|grime|'
                    r'drum.*bass|dnb|dubstep|edm|techno|house|disco|dancehall|'
                    r'reggaeton|funk|soul|gospel|afrobeats)\b',
                    _ci_lower
                ))

                # High energy — visible sweat with physical detail
                _SWEAT_POOL_HIGH = [
                    "BODY CONDITION: A fine mist of sweat has formed across her forehead and upper lip, "
                    "catching the light in individual points. A single bead has tracked from her left temple "
                    "to her jaw. Her skin glows with the physical heat of the performance.",

                    "BODY CONDITION: Sweat is visible at her hairline and on her collarbone, "
                    "the skin between her collarbones catching the light with a wet sheen. "
                    "Her chest heaves visibly between the held notes — the breath is audible and real.",

                    "BODY CONDITION: Her upper lip glistens, a faint bead forming at the corner. "
                    "The skin across her chest and neck carries a high shine from the performance heat. "
                    "Her hair at the temples has darkened slightly with moisture.",

                    "BODY CONDITION: Sweat has gathered at the back of her neck and along her hairline. "
                    "When the camera catches the light at the right angle her collarbone gleams. "
                    "Her breathing is visible in the rise and fall of her shoulders between phrases.",
                ]

                # Subtle — intimate glow, no visible beading
                _SWEAT_POOL_LOW = [
                    "BODY CONDITION: Her skin carries a light warmth from the performance — "
                    "not sweat but the flush of exertion, a softness in the light across her cheekbones "
                    "and the bridge of her nose. Her lips are slightly parted between phrases.",

                    "BODY CONDITION: A very faint sheen on her forehead and upper lip — "
                    "barely visible but readable by the camera as physical effort. "
                    "Her chest rises and falls with the melody, the breathing part of the performance.",

                    "BODY CONDITION: The performance heat shows in the soft flush across her cheeks "
                    "and the faint damp at her temples. Her skin catches the light differently "
                    "than it did at the opening — warmth built through the clip.",
                ]

                _body_condition = (
                    _sweat_rng.choice(_SWEAT_POOL_HIGH)
                    if _is_high_energy
                    else _sweat_rng.choice(_SWEAT_POOL_LOW)
                )

                # ── Breathing / physical cost ─────────────────────────────────
                # Genre-matched body mechanics — what the physical cost of singing looks like
                _BREATH_HIGH = (
                    "PHYSICAL COST: Chest visibly heaving between phrases. "
                    "Ribs expanding fully on each intake — the breath is large and fast. "
                    "Her throat works visibly on the held notes, the strain readable in the neck muscles."
                )
                _BREATH_LOW = (
                    "PHYSICAL COST: Breath is controlled and deep — the intake barely visible "
                    "but present in the slight lift of her shoulders. "
                    "Her throat stays long and open through the sustained notes, "
                    "the effort invisible but the sound enormous."
                )
                _breath_condition = _BREATH_HIGH if _is_high_energy else _BREATH_LOW

                pacing_hint = (
                    f"This clip is {real_seconds:.0f} seconds long. "
                    f"The character is SINGING — this changes everything about the pacing model. "
                    f"Singing is CONTINUOUS, not punctuated. The voice fills the entire clip. "
                    f"Do NOT write a few quoted lines as if this were dialogue — "
                    f"the lyrics should be woven through the ENTIRE prompt, not concentrated in one or two lines. "
                    f"{_emotion_state} "
                    f"{_body_condition} "
                    f"{_breath_condition} "
                    f"Structure with SPECIFIC CAMERA MOVES per section — execute these exactly: "
                    f"(1) Opening — establish voice quality, first lyric phrase, physical stance. "
                    f"CAMERA: {_cam_opening}. "
                    f"(2) Build — melody develops, body responds, second lyric phrase, intensity rising. "
                    f"CAMERA: {_cam_prof['build']}. "
                    f"(3) Peak — belt, held note, dynamic shift. Describe the voice physically. "
                    f"CAMERA: {_cam_prof['peak']}. "
                    f"(4) Resolution — phrase lands, fades. "
                    f"CAMERA: {_cam_prof['resolution']}. "
                    f"Lyrics appear THROUGHOUT all four sections — at least one lyric fragment per section. "
                    f"HARD STOP after the resolution."
                )
            elif action_count == 1:
                # ── Single spoken beat ────────────────────────────────────────
                pacing_hint = (
                    f"This clip is {real_seconds:.0f} seconds long. "
                    f"Write ONE spoken moment. The character speaks — that IS the scene. "
                    f"Physical description sets the stage, then she speaks. "
                    f"The dialogue and its physical delivery are the primary content, not decoration. "
                    f"HARD STOP after the spoken moment is complete."
                )
            else:
                ordinal = {2: "2nd", 3: "3rd"}.get(action_count, f"{action_count}th")
                pacing_hint = (
                    f"This clip is {real_seconds:.0f} seconds long. "
                    f"Write EXACTLY {action_count} beats — each beat is a SPOKEN MOMENT. "
                    f"Structure: brief physical setup, then she speaks, then her physical reaction. "
                    f"Dialogue is the primary content of every beat — not an afterthought woven in. "
                    f"Each spoken line gets its own beat with a physical delivery note. "
                    f"Space the lines across the clip — do not dump all dialogue in one block. "
                    f"HARD STOP after the {ordinal} spoken beat is complete."
                )
        elif action_count == 1:
            pacing_hint = (
                f"This clip is {real_seconds:.0f} seconds long. "
                f"Write EXACTLY 1 action. One single moment. "
                f"Do not describe anything before or after it. No setup, no resolution. "
                f"HARD STOP after the 1st action. Do not continue."
            )
        else:
            ordinal = {2: "2nd", 3: "3rd"}.get(action_count, f"{action_count}th")
            pacing_hint = (
                f"This clip is {real_seconds:.0f} seconds long. "
                f"Write EXACTLY {action_count} distinct actions — NO MORE THAN {action_count}. "
                f"Each action takes roughly {real_seconds / action_count:.0f} seconds of screen time. "
                f"Do not add setup, backstory, or resolution beyond these {action_count} actions. "
                f"Dialogue is woven into action beats — it does not consume a beat. "
                f"HARD STOP after the {ordinal} action is complete."
            )

        # ── Temperature ───────────────────────────────────────────────────────
        temp_map = {
            "0.5 - Strict & Literal":      0.5,
            "0.8 - Balanced Professional": 0.8,
            "1.0 - Artistic Expansion":    1.0,
        }
        temperature = temp_map.get(creativity, 0.8)

        # ── Content detection ─────────────────────────────────────────────────

        _active_scene_context = scene_context if use_scene_context else ""
        _combined_input = user_input + " " + _active_scene_context

        # ── Audio analysis ────────────────────────────────────────────────────
        _audio_analysis    = {}
        _audio_instruction = ""
        if audio_input is not None and audio_enabled:
            print(f"[LTX2-Qwen] Analysing audio (whisper={'ON' if use_whisper else 'OFF'})...")
            _audio_analysis    = _analyse_audio(audio_input, use_whisper=use_whisper)
            _audio_instruction = _build_audio_instruction(_audio_analysis)
            if _audio_analysis.get("summary"):
                print(f"[LTX2-Qwen] Audio summary: {_audio_analysis['summary'][:120]}")

        is_explicit    = bool(self._EXPLICIT_RE.search(_combined_input))
        is_sensual     = bool(self._SENSUAL_RE.search(_combined_input)) and not is_explicit
        has_undressing = bool(self._UNDRESS_RE.search(_combined_input))

        # ── Male dialogue detection ───────────────────────────────────────────
        # Fires when user wants the man to speak during a partner sex scene.
        _wants_male_dialogue = bool(re.search(
            r'\b(he\s+(talks?|speaks?|says?|whispers?|groans?|moans?|curses?|swears?)|'
            r'man\s+(talks?|speaks?|says?|dirty\s+talk)|'
            r'both\s+(talk|speak|moan|dirty\s+talk)|'
            r'he\s+dirty\s+talks?|dirty\s+talk\s+from\s+(him|both)|'
            r'he\s+also\s+talks?|make\s+him\s+talk|'
            r'dual\s+(voice|dialogue)|both\s+voices?)\b',
            _combined_input, re.IGNORECASE
        ))

        # ── Swear injector detection ──────────────────────────────────────────
        # Fires when user explicitly asks for more swearing/cursing.
        # Injects context-aware expletives BETWEEN anchor phrases — not replacing them.
        _wants_swear_injection = bool(re.search(
            r'\b(add\s+swear\w*|more\s+swear\w*|lots\s+of\s+swear\w*|'
            r'add\s+curs\w*|more\s+curs\w*|lots\s+of\s+curs\w*|'
            r'add\s+profan\w*|more\s+profan\w*|'
            r'lots\s+of\s+moaning\s+and\s+(high\s+tones\s+and\s+)?swear\w*|'
            r'swear\w*\s+a\s+lot|curs\w*\s+a\s+lot|filthy\s+mouth|'
            r'dirty\s+mouth|potty\s+mouth|foul\s+mouth\w*|'
            r'f.?bombs|lots\s+of\s+f.words?)\b',
            _combined_input, re.IGNORECASE
        ))
        # Also auto-fire for explicit sex scenes — swearing is expected register
        if is_explicit and _is_sex_scene:
            _wants_swear_injection = True

        # ── Femdom verbal domination detection ────────────────────────────────
        # Fires when the scene is verbal femdom / findom / domination humiliation.
        # Triggers: femdom, findom, dominatrix, verbal humiliation, she dominates,
        # she calls him pathetic, mistress, degradation, SPH, verbal abuse scene etc.
        _is_femdom_verbal = bool(re.search(
            r'\b(femdom|findom|dominatrix|verbal\s+domination|verbal\s+humiliation|'
            r'she\s+(dominates|humiliates|degrades|belittles|insults\s+him|owns\s+him|controls\s+him|abuses\s+(him|me|the))|'
            r'(she|women?|woman)\s+(abusing|humiliating|degrading|dominating)\s+(him|me|the)|'
            r'abus(e|es|ing)\s+me\b|'
            r'he\s+(worships\s+her|is\s+her\s+slave|submits\s+to\s+her)|'
            r'power\s+exchange|mistress\b|she\s+is\s+his\s+mistress|'
            r'small\s+dick\s+humiliation|sph\b|financial\s+domination|'
            r'she\s+calls\s+him\s+(pathetic|useless|worthless|small|weak|pitiful)|'
            r'verbal\s+abuse|degradation\s+(play|scene)|'
            r'she\s+degrades|she\s+demeans|she\s+puts\s+him\s+down|'
            r'worship\s+her|kneel\s+for\s+her|she\s+is\s+superior|'
            r'dominant\s+woman|female\s+dominant|female\s+domination|'
            r'humiliat\w+\s+(him|me)|calls\s+(him|me)\s+(pathetic|useless|worthless|a\s+cunt|stupid))\b',
            _combined_input, re.IGNORECASE
        ))
        # Style preset also activates femdom verbal system
        if "femdom" in style_preset.lower() or "verbal domination" in style_preset.lower():
            _is_femdom_verbal = True

        # POV mode — she addresses camera directly, no visible male subject
        _is_femdom_pov = False
        if _is_femdom_verbal:
            _is_femdom_pov = bool(re.search(
                r'\b(pov\b|point\s+of\s+view|looking\s+at\s+(the\s+)?camera|'
                r'she\s+looks\s+at\s+(me|the\s+viewer|camera)|'
                r'talking\s+to\s+(the\s+)?(viewer|camera)|'
                r'no\s+man\b|no\s+male\b|just\s+her|only\s+her|'
                r'addressing\s+the\s+(viewer|camera)|camera\s+is\s+the\s+subject|'
                r'viewer\s+is\s+the\s+subject|directly\s+at\s+me|'
                r'directed\s+at\s+me|she\s+stares\s+down\s+(the\s+)?camera|'
                r'she\s+dominates\s+the\s+viewer|looking\s+directly\s+at\s+me)\b',
                _combined_input, re.IGNORECASE
            ))
            if "pov" in style_preset.lower():
                _is_femdom_pov = True# Raw tier — fires when user asks for aggressive/vulgar/screaming femdom energy
        _is_femdom_raw = _is_femdom_verbal and bool(re.search(
            r'\b(vulgar|aggressive|screaming|shout\w*|in\s+your\s+face|'
            r'verbal\s+abuse|abusive|brutal|harsh|mean|nasty|raw\b|'
            r'really\s+mean|full\s+abuse|absolute\s+abuse|'
            r'swear\w*\s+at\s+him|curse\w*\s+at\s+him|'
            r'call\s+him\s+(a\s+)?(loser|pathetic|worthless|useless|tiny|small|piece\s+of\s+shit)|'
            r'degrade\s+him\s+hard|humiliate\s+him\s+hard|'
            r'bdsm\s+verbal|raw\s+femdom|mean\s+femdom|'
            r'loud\s+femdom|angry\s+dom\w*|angry\s+mistress|'
            r'abus\w+|insult\w+|humiliat\w+)\b',
            _combined_input, re.IGNORECASE
        ))
        _env_instruction = ""
        if environment_sel and environment_sel not in ("None — LLM decides", "") and not environment_sel.startswith("─"):
            _env_data = self.ENVIRONMENT_PRESETS.get(environment_sel)
            if _env_data == "RANDOM":
                # Seed-driven random pick — respects explicit filter
                import random as _env_rnd
                _env_rng = _env_rnd.Random(seed if seed != -1 else None)
                _env_candidates = [
                    (k, v) for k, v in self.ENVIRONMENT_PRESETS.items()
                    if v and v != "RANDOM" and not k.startswith("─")
                    and (not v[3] or is_explicit or is_sensual)
                ]
                if _env_candidates:
                    environment_sel, _env_data = _env_rng.choice(_env_candidates)
                    print(f"[LTX2-Qwen] Environment random pick: {environment_sel}")
            if _env_data and _env_data != "RANDOM":
                _env_loc, _env_light, _env_sound, _env_explicit_only = _env_data
                if not _env_explicit_only or is_explicit or is_sensual:
                    _env_instruction = (
                        f"\n\n[ENVIRONMENT — MANDATORY LOCATION: "
                        f"Place this scene in: {_env_loc}. "
                        f"Lighting: {_env_light}. "
                        f"Sound design must include: {_env_sound}. "
                        f"This location OVERRIDES any location in the user prompt. "
                        f"Describe the environment with specific textures, surfaces, and atmospheric detail — "
                        f"not just its name. The location is a character in the scene.]"
                    )
                    print(f"[LTX2-Qwen] Environment preset: {environment_sel}")
                elif _env_explicit_only:
                    print(f"[LTX2-Qwen] Environment '{environment_sel}' skipped — explicit_only, scene not explicit/sensual")

        # ── Sex scene detection ───────────────────────────────────────────────
        # Fires when the scene IS a sex act, not just explicit language.
        # Triggers moaning/vocalisation mode — replaces clean speech with
        # physical sounds woven into action. Only fires when is_explicit=True too.
        _is_sex_scene = is_explicit and bool(re.search(
            r'\b(sex\b|sex\s+scene|having\s+sex|making\s+love|fucking\b|'
            r'rides?\s+(him|her|them)|riding\s+(him|her|them)|'
            r'penetrat\w+|thrusting|sex\s+act|intercourse|'
            r'goes\s+down\s+on|going\s+down\s+on|cunnilingus|fellatio|'
            r'69\b|cowgirl|doggy\s+style|missionary)\b',
            _combined_input, re.IGNORECASE
        ))

        # ── Early position + language detection ───────────────────────────────
        # Detected here (early) so camera variety code can use _sx_position.
        # Also detected again inside the vocalisation block for pool selection.
        _sx_position = None
        _ci_sx_early = _combined_input.lower()
        if re.search(r'\b(missionary|face\s+to\s+face|on\s+her\s+back)\b', _ci_sx_early):              _sx_position = "missionary"
        elif re.search(r'\b(reverse\s+cowgirl|facing\s+away\s+on\s+top)\b', _ci_sx_early):             _sx_position = "reverse_cowgirl"
        elif re.search(r'\b(cowgirl|on\s+top|riding\s+him|rides\s+him|sitting\s+on\s+him)\b', _ci_sx_early): _sx_position = "cowgirl"
        elif re.search(r'\b(doggy|doggy\s+style|from\s+behind|on\s+all\s+fours|bent\s+over)\b', _ci_sx_early): _sx_position = "doggy"
        elif re.search(r'\b(69|sixty.?nine|mutual\s+oral)\b', _ci_sx_early):                             _sx_position = "sixtynine"
        elif re.search(r'\b(blowjob|blow\s+job|going\s+down\s+on\s+him|fellatio)\b', _ci_sx_early):   _sx_position = "blowjob"
        elif re.search(r'\b(riding\s+(a\s+)?(dildo|toy|vibrator)|dildo\s+riding|solo\s+rid\w+)\b', _ci_sx_early): _sx_position = "riding"

        # Early language detection — English UNLESS nationality explicitly stated.
        # IMPORTANT: only use native script when user stated a nationality or gravure preset.
        # "a woman" with no nationality → English. "a japanese woman" → Japanese.
        _sx_early_english = bool(re.search(r'\bin\s+english\b', _combined_input, re.IGNORECASE))
        _sx_has_nationality = bool(re.search(
            r'\b(japanese|japan|korean|korea|chinese|china|mandarin|cantonese)\b',
            _ci_sx_early
        ))
        if _sx_early_english:         _sx_lang = "English"
        elif not _sx_has_nationality: _sx_lang = "English"   # no nationality → always English
        elif re.search(r'\b(japanese|japan)\b', _ci_sx_early):         _sx_lang = "Japanese"
        elif re.search(r'\b(korean|korea)\b', _ci_sx_early):           _sx_lang = "Korean"
        elif re.search(r'\b(chinese|china|mandarin|cantonese)\b', _ci_sx_early): _sx_lang = "Mandarin"
        else:                         _sx_lang = "English"

        if _is_sex_scene:
            print(f"[LTX2-Qwen] Sex scene detected — vocalisation mode active | position={_sx_position} | lang={_sx_lang}")

        # Lift fires only when intent is sensual/explicit OR when no innocent purpose is stated.
        # Innocent-purpose phrases (sitting, stepping, avoiding etc.) suppress the sequence
        # so "lifts her dress to sit down" doesn't trigger the exposure sequence.
        _INNOCENT_LIFT_RE = re.compile(
            r"\b(to\s+sit|to\s+step|to\s+walk|to\s+run|to\s+climb|to\s+cross|to\s+avoid|"
            r"to\s+get\s+(in|out|on|off)|to\s+mount|to\s+board|to\s+enter|to\s+exit|"
            r"getting\s+in|getting\s+out|stepping\s+over|stepping\s+into|"
            r"puddle|stairs|step|kerb|curb|bicycle|bike|horse|car|seat|bench|chair|sofa|couch)\b",
            re.IGNORECASE
        )
        _lift_raw = bool(self._LIFT_RE.search(_combined_input))
        _lift_innocent = bool(_INNOCENT_LIFT_RE.search(_combined_input))
        has_lift = _lift_raw and (is_sensual or is_explicit or not _lift_innocent)

        # ── Aspect ratio detection ────────────────────────────────────────────
        _ratio_class = "landscape"  # default
        _ratio_instruction = ""
        if width and height and width > 0 and height > 0:
            _ratio = width / height
            if _ratio < 0.75:
                _ratio_class = "portrait"
                is_portrait = True  # override portrait_mode regardless of toggle
                _ratio_instruction = (
                    f"\n[ASPECT RATIO: {width}x{height} — PORTRAIT 9:16 VERTICAL. "
                    "Frame is vertical throughout. Tight head-to-torso. Action moves vertically. "
                    "No horizontal sweep. No wide establishing shots. "
                    "CAMERA REFS: do NOT use ARRI, Alexa, Kodak, or any film stock reference. "
                    "If shooting style is naturalistic, describe it as phone or handheld vertical. "
                    "Only medium-format (Hasselblad, Leica) is acceptable if the style is explicitly editorial.]"
                )
            elif _ratio < 1.15:
                _ratio_class = "square"
                _ratio_instruction = (
                    f"\n[ASPECT RATIO: {width}x{height} — SQUARE 1:1. "
                    "Centred composition. Symmetrical staging. Subject fills the square frame. "
                    "CAMERA REFS: only medium-format or Leica appropriate. "
                    "No wide cinematic language. No horizontal sweep.]"
                )
            elif _ratio < 1.85:
                _ratio_class = "landscape"
                _ratio_instruction = (
                    f"\n[ASPECT RATIO: {width}x{height} — LANDSCAPE 16:9 WIDESCREEN. "
                    "Standard cinematic framing. Camera refs depend on scene type — "
                    "street/chav/lo-fi/documentary scenes: describe light source only, no film stock. "
                    "Drama/thriller/sci-fi/epic: ARRI Alexa, Kodak stocks appropriate.]"
                )
            elif _ratio < 2.4:
                _ratio_class = "ultrawide"
                _ratio_instruction = (
                    f"\n[ASPECT RATIO: {width}x{height} — ULTRA-WIDE 21:9. "
                    "Sweeping horizontal space. The frame breathes at the edges. "
                    "Environment is dominant. Build foreground depth. "
                    "CAMERA REFS: RED Monstro, ARRI Alexa 65, anamorphic glass appropriate at this ratio.]"
                )
            else:
                _ratio_class = "anamorphic"
                _ratio_instruction = (
                    f"\n[ASPECT RATIO: {width}x{height} — ANAMORPHIC SCOPE 2.39:1. "
                    "Full scope cinema framing. Anamorphic lens compression. Oval bokeh. "
                    "Horizontal lens flare on light sources. Every frame should feel like a poster. "
                    "CAMERA REFS: Panavision, Cooke anamorphic, ARRI Alexa always appropriate at scope ratio.]"
                )

        # ── Scene-aware camera ref gate ───────────────────────────────────────
        _street_scene        = bool(self._STREET_SCENE_RE.search(_combined_input))
        _cinematic_scene     = bool(self._CINEMATIC_SCENE_RE.search(_combined_input))
        _preset_is_cinematic = any(w in style_preset.lower() for w in [
            'cinematic', 'drama', 'noir', 'sci-fi', 'horror', 'fashion', 'epic',
            'thriller', 'golden hour', 'editorial', 'erotic cinema',
        ])
        _use_camera_ref = (_cinematic_scene or _preset_is_cinematic) and not _street_scene
        if not _use_camera_ref and _ratio_class in ("ultrawide", "anamorphic"):
            _use_camera_ref = True  # scope ratio always earns cinema language

        # ── Subject count override ────────────────────────────────────────────
        # 0 = auto-detect from text (existing behaviour), 1-4 = explicit
        _subject_count_instruction = ""
        if subject_count and subject_count > 0:
            if subject_count == 1:
                _subject_count_instruction = (
                    "\n[SUBJECT COUNT — CONFIRMED: exactly ONE person in this scene. "
                    "Do not add a second person. Single-subject spatial blocking only. "
                    "Anchor them in frame — centre, left, or right — and describe their relationship to the background.]"
                )
            elif subject_count == 2:
                _subject_count_instruction = (
                    "\n[SUBJECT COUNT — CONFIRMED: exactly TWO people in this scene. "
                    "Give each a distinct spatial position — left/right or foreground/background. "
                    "Describe both clearly. Keep their actions non-overlapping. "
                    "Name their relative positions explicitly.]"
                )
            elif subject_count == 3:
                _subject_count_instruction = (
                    "\n[SUBJECT COUNT — CONFIRMED: THREE people in this scene. "
                    "Space them clearly — foreground, mid, background or spread across frame. "
                    "Keep each person's action simple and non-overlapping. "
                    "Clarity of individual positions is mandatory.]"
                )
            elif subject_count >= 4:
                _subject_count_instruction = (
                    f"\n[SUBJECT COUNT — CONFIRMED: {subject_count} people in this scene. "
                    "This is a group scene. Do not attempt to describe each person individually — "
                    "describe the group as a mass with notable individuals pulled into focus. "
                    "Use wide or establishing framing.]"
                )

        # ── Camera lock ───────────────────────────────────────────────────────
        # ── Camera angle + movement ───────────────────────────────────────────
        # Priority order:
        #   1. User widget explicitly set → always MANDATORY, overrides everything
        #   2. User wrote camera terms in prompt → suppress preset default for that axis
        #      (respect what they wrote, don't fight them with a MANDATORY tag)
        #   3. Preset default → inject if no user text conflict
        _preset_cam = self.PRESET_CAMERA_DEFAULTS.get(style_preset, (None, None))
        _preset_angle    = _preset_cam[0]
        _preset_movement = _preset_cam[1]

        # Detect camera angle/movement terms written by the user in their prompt
        _USER_ANGLE_RE = re.compile(
            r"\b(low angle|high angle|eye.?level|bird.?s.?eye|worm.?s.?eye|dutch angle|"
            r"over.the.shoulder|OTS|point of view|POV|first.?person|side.?on|profile shot|"
            r"top.?down|overhead shot|looking up|looking down|canted|tilted frame)\b",
            re.IGNORECASE
        )
        _USER_MOVEMENT_RE = re.compile(
            r"\b(static|locked.?off|handheld|hand.?held|dolly|push in|pull back|pull away|"
            r"zoom in|zoom out|orbit|tracking shot|track\w*\s+(?:her|him|them|the)|"
            r"tilt up|tilt down|pan left|pan right|whip pan|truck|lateral|aerial|drone|"
            r"slow pan|slow push|crane shot|steadicam|gimbal)\b",
            re.IGNORECASE
        )

        _user_wrote_angle    = bool(_USER_ANGLE_RE.search(_combined_input))
        _user_wrote_movement = bool(_USER_MOVEMENT_RE.search(_combined_input))

        # Widget explicitly set → always use it (user made a deliberate choice)
        _widget_angle_set    = shot_angle    != "None — LLM decides"
        _widget_movement_set = camera_movement != "None — LLM decides"

        # ── Camera variety pools — seed-driven, never the same twice ────────────
        # Each preset has a pool of appropriate angles/movements.
        # When no widget/user override: seed picks from the pool → variety every run.
        # Widget selection or user-written camera terms always override.
        _ANGLE_VARIETY = {
            "Cinematic — Drama":                    ["Eye-level — neutral, natural", "OTS — over the shoulder", "Three-quarter — 45 degree"],
            "Cinematic — Epic":                     ["Low angle — powerful, imposing", "Eye-level — neutral, natural", "Bird's eye — top-down overhead"],
            "Cinematic — Intimate close-up":        ["Eye-level — neutral, natural", "Three-quarter — 45 degree", "OTS — over the shoulder"],
            "Noir — deep shadows, venetian light":  ["Low angle — powerful, imposing", "Three-quarter — 45 degree", "Eye-level — neutral, natural"],
            "Horror — desaturated, harsh contrast": ["High angle — vulnerable", "Eye-level — neutral, natural", "Low angle — powerful, imposing"],
            "Erotic cinema — tasteful, cinematic":  ["Low angle — powerful, imposing", "Eye-level — neutral, natural", "Three-quarter — 45 degree"],
            "Explicit — direct, anatomical":        ["Low angle — powerful, imposing", "Eye-level — neutral, natural", "OTS — over the shoulder"],
            "Gravure Idol — Japanese glamour":      ["Low angle — powerful, imposing", "Eye-level — neutral, natural", "Three-quarter — 45 degree"],
            "Music video — stylised":               ["Eye-level — neutral, natural", "Low angle — powerful, imposing", "Three-quarter — 45 degree"],
            "High fashion editorial":               ["Low angle — powerful, imposing", "Three-quarter — 45 degree", "Eye-level — neutral, natural"],
        }
        _MOVEMENT_VARIETY = {
            "Cinematic — Drama":                    ["Slow push in", "Static — locked off", "Arc — slow curved lateral track"],
            "Cinematic — Epic":                     ["Pull back — reveal", "Arc — slow curved lateral track", "Tracking — follows subject"],
            "Cinematic — Intimate close-up":        ["Slow push in", "Static — locked off", "Slow push in"],
            "Noir — deep shadows, venetian light":  ["Slow push in", "Static — locked off", "Arc — slow curved lateral track"],
            "Horror — desaturated, harsh contrast": ["Static — locked off", "Slow push in", "Handheld — natural shake"],
            "Erotic cinema — tasteful, cinematic":  ["Slow push in", "Arc — slow curved lateral track", "Static — locked off"],
            "Explicit — direct, anatomical":        ["Tracking — follows subject", "Slow push in", "Static — locked off"],
            "Gravure Idol — Japanese glamour":      ["Tilt up — bottom to top", "Slow push in", "Arc — slow curved lateral track", "Static — locked off", "Tilt down — top to bottom"],
            "Music video — stylised":               ["Tracking — follows subject", "Arc — slow curved lateral track", "Handheld — natural shake"],
            "High fashion editorial":               ["Static — locked off", "Slow push in", "Arc — slow curved lateral track"],
        }

        # Sex scene position-specific camera pools
        _SEX_CAMERA_ANGLES = {
            "missionary":      ["Eye-level — neutral, natural", "OTS — over the shoulder", "Three-quarter — 45 degree"],
            "cowgirl":         ["Low angle — powerful, imposing", "Eye-level — neutral, natural", "Three-quarter — 45 degree"],
            "doggy":           ["Low angle — powerful, imposing", "Three-quarter — 45 degree", "OTS — over the shoulder"],
            "riding":          ["Eye-level — neutral, natural", "Low angle — powerful, imposing", "Three-quarter — 45 degree"],
            "blowjob":         ["Eye-level — neutral, natural", "Low angle — powerful, imposing", "OTS — over the shoulder"],
            "sixtynine":       ["Eye-level — neutral, natural", "OTS — over the shoulder", "Profile — side-on"],
            "reverse_cowgirl": ["Low angle — powerful, imposing", "Three-quarter — 45 degree", "Eye-level — neutral, natural"],
        }
        _SEX_CAMERA_MOVEMENTS = {
            "missionary":      ["Slow push in", "Static — locked off", "Arc — slow curved lateral track"],
            "cowgirl":         ["Slow push in", "Static — locked off", "Arc — slow curved lateral track"],
            "doggy":           ["Static — locked off", "Slow push in", "Tracking — follows subject"],
            "riding":          ["Slow push in", "Static — locked off", "Arc — slow curved lateral track"],
            "blowjob":         ["Slow push in", "Static — locked off", "Arc — slow curved lateral track"],
            "sixtynine":       ["Static — locked off", "Slow push in", "Arc — slow curved lateral track"],
            "reverse_cowgirl": ["Slow push in", "Static — locked off", "Tracking — follows subject"],
        }

        # Independent RNGs for angle and movement — so they vary separately
        _cam_angle_rng    = random.Random((seed + 7)  if seed != -1 else None)
        _cam_movement_rng = random.Random((seed + 13) if seed != -1 else None)

        # Resolve effective angle
        if _widget_angle_set:
            _eff_angle = shot_angle  # explicit widget choice — always honour
        elif _user_wrote_angle:
            _eff_angle = None  # user wrote it in prompt — don't override
        elif _is_sex_scene:
            _sx_angle_pool = _SEX_CAMERA_ANGLES.get(
                _sx_position,
                ["Eye-level — neutral, natural", "Low angle — powerful, imposing", "Three-quarter — 45 degree"]
            )
            _eff_angle = _cam_angle_rng.choice(_sx_angle_pool)
        elif style_preset in _ANGLE_VARIETY:
            _eff_angle = _cam_angle_rng.choice(_ANGLE_VARIETY[style_preset])
        else:
            _eff_angle = _preset_angle

        # Resolve effective movement
        if _widget_movement_set:
            _eff_movement = camera_movement  # explicit widget choice — always honour
        elif _user_wrote_movement:
            _eff_movement = None  # user wrote it in prompt — don't override
        elif _is_sex_scene:
            _sx_move_pool = _SEX_CAMERA_MOVEMENTS.get(
                _sx_position,
                ["Slow push in", "Static — locked off", "Arc — slow curved lateral track"]
            )
            _eff_movement = _cam_movement_rng.choice(_sx_move_pool)
        elif style_preset in _MOVEMENT_VARIETY:
            _eff_movement = _cam_movement_rng.choice(_MOVEMENT_VARIETY[style_preset])
        else:
            _eff_movement = _preset_movement

        _ANGLE_INSTRUCTIONS = {
            "Eye-level — neutral, natural":    "SHOT ANGLE — MANDATORY: eye-level. Camera at the subject's eye height. Natural, neutral perspective. No tilt up or down.",
            "Low angle — powerful, imposing":  "SHOT ANGLE — MANDATORY: low angle. Camera positioned below the subject, pointing upward. Subject appears powerful, dominant, imposing.",
            "High angle — vulnerable":         "SHOT ANGLE — MANDATORY: high angle. Camera above the subject, pointing down. Subject appears small, vulnerable, or surveilled.",
            "Bird's eye — top-down overhead":  "SHOT ANGLE — MANDATORY: bird's eye view. Camera directly overhead, pointing straight down. Subject seen from above.",
            "Worm's eye — extreme low, looking up": "SHOT ANGLE — MANDATORY: worm's eye view. Camera at ground level or below, looking steeply upward. Extreme perspective distortion.",
            "Dutch angle — tilted, unsettling": "SHOT ANGLE — MANDATORY: Dutch angle. Camera tilted on its horizontal axis — frame is deliberately canted. Psychological unease.",
            "OTS — over the shoulder":         "SHOT ANGLE — MANDATORY: over-the-shoulder. Camera behind one character's shoulder, framing the other. Classic two-person composition.",
            "POV — first person":              "SHOT ANGLE — MANDATORY: POV / first-person. The camera IS the eyes of the subject. We see what they see. No third-person framing.",
            "Profile — side-on":               "SHOT ANGLE — MANDATORY: profile shot. Camera positioned exactly to the side of the subject. Subject faces left or right, fully in profile.",
            "Three-quarter — 45 degree":       "SHOT ANGLE — MANDATORY: three-quarter angle. Camera at roughly 45 degrees to the subject — between full-face and profile.",
        }

        _MOVEMENT_INSTRUCTIONS = {
            "Static — locked off":             "CAMERA MOVEMENT — MANDATORY: completely static and locked off. No push, no drift, no sway, no zoom. The frame does not move at all. All motion comes from the subject and environment only.",
            "Handheld — natural shake":        "CAMERA MOVEMENT — MANDATORY: handheld. Natural human sway, slight vertical bounce, micro-rotations. The camera breathes with the operator. Never smooth or gimbal-stabilised.",
            "Slow push in":                    "CAMERA MOVEMENT — MANDATORY: slow, deliberate push toward the subject. The shot begins wider and tightens imperceptibly over the duration. No other camera movement.",
            "Pull back — reveal":              "CAMERA MOVEMENT — MANDATORY: the camera pulls back slowly, revealing the wider environment around the subject. Begin tight, end wide. The reveal is the payoff.",
            "Arc — slow curved lateral track": "CAMERA MOVEMENT — MANDATORY: the camera moves in a slow curved arc around one side of the subject — a quarter-circle lateral track that gradually reveals the background as the subject stays centred in frame. Smooth and deliberate. The move ends before it completes a full circle.",
            "Tracking — follows subject":      "CAMERA MOVEMENT — MANDATORY: the camera tracks with the subject as they move. Subject stays roughly centred in frame. Camera matches their speed and direction. No static holds.",
            "Tilt up — bottom to top":         "CAMERA MOVEMENT — MANDATORY: the camera tilts upward, beginning at the subject's feet or lower body and rising slowly to their face. No horizontal movement.",
            "Tilt down — top to bottom":       "CAMERA MOVEMENT — MANDATORY: the camera tilts downward, beginning at the subject's face and descending slowly to their feet or lower body.",
            "Truck left — lateral slide":      "CAMERA MOVEMENT — MANDATORY: the camera trucks/slides laterally to the left, maintaining its facing direction. Subjects pass through frame left-to-right as the camera moves.",
            "Truck right — lateral slide":     "CAMERA MOVEMENT — MANDATORY: the camera trucks/slides laterally to the right, maintaining its facing direction. Subjects pass through frame right-to-left as the camera moves.",
            "Whip pan — fast horizontal snap": "CAMERA MOVEMENT — MANDATORY: whip pan. The camera snaps hard and fast horizontally to reveal a new subject or beat. Motion blur during the snap, sharp before and after.",
            "Dolly zoom — vertigo effect":     "CAMERA MOVEMENT — MANDATORY: dolly zoom (Hitchcock/vertigo effect). The camera physically moves toward the subject while the focal length simultaneously widens, or vice versa. Background appears to grow or shrink unnaturally.",
            "Aerial — drone descending":       "CAMERA MOVEMENT — MANDATORY: aerial perspective. Camera begins high above, looking down, and descends slowly toward the subject. Subject grows from a small shape to full frame.",
        }

        _camera_lock_instruction = ""
        if _eff_angle or _eff_movement:
            _angle_text    = _ANGLE_INSTRUCTIONS.get(_eff_angle, "") if _eff_angle else ""
            _movement_text = _MOVEMENT_INSTRUCTIONS.get(_eff_movement, "") if _eff_movement else ""
            _source_note   = ""
            if shot_angle == "None — LLM decides" and _eff_angle:
                _source_note += f" [preset default angle for {style_preset}]"
            if camera_movement == "None — LLM decides" and _eff_movement:
                _source_note += f" [preset default movement for {style_preset}]"
            _cam_parts = [p for p in [_angle_text, _movement_text] if p]
            _camera_lock_instruction = "\n[" + " — ".join(_cam_parts) + "]"

        # ── Negative bias ─────────────────────────────────────────────────────
        _negative_bias_instruction = ""
        if negative_bias and negative_bias.strip():
            _negative_bias_instruction = (
                f"\n[AVOID — USER SPECIFIED: {negative_bias.strip()}. "
                "Do not include any of these elements in the scene. "
                "If the style would normally default to them, omit them.]"
            )

        # Build garment list — ONLY garments being actively removed, not merely worn or lifted.
        # Proximity check: a removal verb must appear within 45 chars of the garment.
        # Lift verbs (rides up, hiked, lifted) are intentionally excluded — those trigger
        # the lift instruction, not an undress sequence.
        _REMOVAL_PROXIMITY_RE = re.compile(
            r"\b(undress\w*|strip\w*|takes?\s+off|took\s+off|removes?\w*|"
            r"disrobe\w*|unbutton\w*|unzip\w*|peels?\s+off|pulls?\s+off|"
            r"shed\w*\s+(her|his|their)?)\b",
            re.IGNORECASE
        )
        _REVEAL_CONTEXT_RE = re.compile(
            r"\b(underneath|beneath|under|reveals?|beneath|exposed\s+underneath)\b",
            re.IGNORECASE
        )
        named_garments = []
        for m in self._GARMENT_RE.finditer(_combined_input):
            start   = max(0, m.start() - 45)
            end     = min(len(_combined_input), m.end() + 45)
            context = _combined_input[start:end]
            # Must have a removal verb nearby
            if not _REMOVAL_PROXIMITY_RE.search(context):
                continue
            # Exclude garments that appear in a "revealed underneath" context
            # Only exclude if "underneath/beneath/under" appears within 12 chars
            # (e.g. "lingerie underneath" but NOT "blouse, [lingerie underneath]")
            after_context = _combined_input[m.end():min(len(_combined_input), m.end() + 12)]
            if _REVEAL_CONTEXT_RE.search(after_context):
                continue
            g = m.group(0).lower()
            if g not in named_garments:
                named_garments.append(g)
        garment_list = ", ".join(named_garments) if named_garments else ""

        if is_explicit:
            # Only inject undressing sequence if garments are actively being removed.
            # named_garments is empty if no garment appears near a removal verb —
            # e.g. "dress rides up" or "wearing a dress" alone won't populate it.
            if named_garments:
                _undressing_clause = (
                    f"\n\nUNDRESSING SEQUENCE — the user's garments are: {garment_list}. "
                    "Write a dedicated undressing segment BEFORE any nudity. One sentence per step. Camera lingers on each reveal. "
                    "Follow ONLY the steps for the garments the user named. "
                    "\n— T-shirt/shirt/tee/crop top (full off): grip hem at waist → lift past stomach → past ribs → over chest → over head → off arms → dropped. "
                    "\n— T-shirt/shirt/tee/crop top (lift only): grip hem → gather upward → past navel → past ribs → chest and breasts fully exposed → held there. "
                    "\n— Camisole/tank top/vest: slip straps off each shoulder → fabric pools at waist → pushed down → dropped. "
                    "\n— Blouse/button-down: each button one at a time from top to bottom → fabric parts → off shoulders → down arms → dropped. "
                    "\n— Hoodie/sweater/cardigan/jumper: grip hem → pull up and over head → arms free → dropped. "
                    "\n— Jacket/blazer/coat: slide off one shoulder → then the other → down arms → dropped or left hanging. "
                    "\n— Robe/kimono: untie belt → sash falls loose → fabric slides off both shoulders → pools on the floor. "
                    "\n— Dress (zip): reach behind → fingers find zip tab → pull slowly down → fabric parts down the back → off shoulders → slides down body → falls. "
                    "\n— Dress (pullover/slip): grip hem at thighs → gather upward → over hips → over waist → over chest → over head. "
                    "\n— Skirt (zip): find zip at side or back → pull down → waistband loosens → pushed over hips → falls. "
                    "\n— Skirt (elastic): thumbs in waistband → pushed down over hips → falls. "
                    "\n— Miniskirt: as skirt above — very little fabric, falls quickly. "
                    "\n— Jeans/trousers/pants: undo button → pull zip down → push over hips → down thighs → stepped out one leg at a time. "
                    "\n— Shorts: thumbs in waistband → pushed down → stepped out. "
                    "\n— Leggings/tights: both hands at waistband → rolled down from hips → past thighs → off. "
                    "\n— Bra: reach behind → unhook clasp → straps slip off each shoulder → cups fall away → dropped. "
                    "\n— Bralette: grip hem at bottom → pull up and over head → dropped. "
                    "\n— Underwear/panties/knickers/briefs/thong/g-string: thumbs in waistband at hips → pushed down → stepped out. "
                    "\n— Bodysuit/catsuit: unsnap crotch fasteners → pull shoulder straps off → peel down body. "
                    "\n— Jumpsuit/playsuit: unzip or unbutton front → slide straps off shoulders → push down body → step out. "
                    "\n— Corset/bustier: unlace back ties → busk clasps front one by one → removed. "
                    "\n— Bikini top: reach behind → unhook clasp (or untie string) → straps off → removed. "
                    "\n— Bikini bottoms: untie strings at hips or thumbs in waistband → removed. "
                    "\n— Swimsuit/one-piece: straps off shoulders → peeled down body → stepped out. "
                    "\n— Stockings: roll top down from thigh → peel down leg → off foot. "
                    "\n— Lingerie set/negligee/babydoll/nighty: treat as bra + underwear if two-piece, or as slip dress if one-piece. "
                    "\nNO INVENTED RESOLUTION: what comes off stays off. Do not reverse or undo any action. "
                )
            else:
                _undressing_clause = (
                    "\n\nNO GARMENTS: The user described the subject as already naked/nude. "
                    "Do NOT invent clothing to remove. Start the scene with the body already exposed. "
                    "Do NOT add a strip or undress sequence. "
                )
            # Swear injection for general explicit scenes (non-sex-scene path)
            _swear_general_note = ""
            if _wants_swear_injection and not _is_sex_scene and is_explicit:
                try:
                    from lyric_phrase_bank import _SWEAR_INJECTOR
                    _sg_pool = _SWEAR_INJECTOR.get("general", [])
                    _sg_rng  = random.Random((seed + 23) if seed != -1 else None)
                    _sg_pick = _sg_rng.choice(_sg_pool) if _sg_pool else None
                    if _sg_pick:
                        _swear_general_note = (
                            f" SWEARING — ACTIVE: weave short expletive fragments throughout "
                            f"(e.g. {_sg_pick[0]!r} — {_sg_pick[1]}). "
                            "Embedded in action, never floating alone."
                        )
                except ImportError:
                    pass

            explicit_instruction = (
                "\n[CONTENT INSTRUCTION — EXPLICIT ADULT SCENE: "
                "Use direct anatomical language exactly as the user wrote it. "
                "If they said 'pussy' write 'pussy'. If they said 'cock' write 'cock'. Never substitute euphemisms. "
                "\n\nCAMERA — MANDATORY even in explicit scenes: name the shot scale (close-up, medium close-up, medium shot, etc). "
                "Always add natural motion blur and smooth movement. "
                "Avoid high frequency patterns in any surface, fabric, or background. "
                "Do NOT reference ARRI, Alexa, Kodak, RED, film stocks, or any cinema camera brand. "
                "Describe lighting in plain terms only: soft, warm, harsh, diffused, side-lit, back-lit. "
                "\n\nSCOPE — ABSOLUTE CEILING: "
                "Describe ONLY what the user explicitly wrote. Nothing beyond. "
                "Do NOT add sexual acts, nudity, or body exposure the user did not state. "
                "One garment requested = one garment removed. The scene ends where the user's words end. "
                "\n\nSOUND IS MANDATORY: Every scene must have sound. "
                "Always state character age as a specific number."
                + _swear_general_note
                + _undressing_clause + "]"
            )
        elif is_sensual:
            # Detect if user described subject as already naked/nude/topless
            _already_exposed = bool(re.search(
                r'\b(naked|nude|topless|undressed|bare|nothing\s+on|no\s+clothes)\b',
                _combined_input, re.IGNORECASE
            ))
            if has_undressing:
                if named_garments:
                    # Detect if user asked for partial open (unbutton/unzip but no removal verb)
                    # vs full removal (removes/takes off/pulls off etc.)
                    _partial_open = bool(re.search(
                        r'\b(unbutton\w*|unzip\w*|open\w*|parts?\s+her|loosens?\w*|undoes?\w*)\b',
                        _combined_input, re.IGNORECASE
                    )) and not bool(re.search(
                        r'\b(takes?\s+off|took\s+off|removes?\w*|pulls?\s+off|peels?\s+off|'
                        r'shed\w*|drop\w*|strip\w*)\b',
                        _combined_input, re.IGNORECASE
                    ))
                    _partial_note = (
                        "\n\nPARTIAL OPEN ONLY: The user asked to unbutton/open — NOT to remove. "
                        "The garment stays ON the body, hanging open or parted. "
                        "Do NOT write it being pulled off, dropped, or falling away. "
                        "It opens, reveals what's underneath, and STAYS. "
                    ) if _partial_open else ""
                    undress_clause = (
                        f"\n\nUNDRESSING — SCOPE CEILING: User named these garments only: {garment_list}. "
                        f"Remove ONLY those. Nothing beyond. "
                        f"Do NOT advance from shirt → bra unless user said bra. "
                        f"Do NOT advance from bra → topless unless user said topless or nude. "
                        f"The named garments are the ceiling. Style preset does NOT override this. "
                        + _partial_note +
                        f"\n\nSOUND IS MANDATORY throughout — fabric sounds, breathing, environment. "
                        f"\n\nFor each garment write every physical step as its own sentence. "
                        "Follow ONLY the steps for the garments the user named. "
                        "\n— T-shirt/shirt/tee/crop top (full off): grip hem at waist → lift past stomach → past ribs → over chest → over head → off arms → dropped. "
                        "\n— T-shirt/shirt/tee/crop top (lift only): grip hem → gather upward → past navel → past ribs → chest fully exposed → held there. "
                        "\n— Camisole/tank top/vest: slip straps off each shoulder → fabric pools at waist → pushed down → dropped. "
                        "\n— Blouse/button-down: each button top to bottom → fabric parts → off shoulders → down arms → dropped. "
                        "\n— Hoodie/sweater/cardigan/jumper: grip hem → pull up and over head → arms free → dropped. "
                        "\n— Jacket/blazer/coat: slide off one shoulder → then the other → down arms → dropped. "
                        "\n— Robe/kimono: untie belt → sash falls loose → fabric slides off both shoulders → pools on the floor. "
                        "\n— Dress (zip): find zip behind → pull slowly down → fabric parts → off shoulders → slides down body → falls. "
                        "\n— Dress (pullover): grip hem at thighs → over hips → over waist → over chest → over head. "
                        "\n— Skirt (zip): find zip at side or back → pull down → pushed over hips → falls. "
                        "\n— Skirt (elastic/miniskirt): thumbs in waistband → pushed down over hips → falls. "
                        "\n— Jeans/trousers/pants: undo button → pull zip → push over hips → down thighs → stepped out. "
                        "\n— Shorts: thumbs in waistband → pushed down → stepped out. "
                        "\n— Leggings/tights: both hands at waistband → rolled down from hips → off. "
                        "\n— Bra: reach behind → unhook clasp → straps off each shoulder → cups fall away. "
                        "\n— Bralette: grip hem → pull over head → dropped. "
                        "\n— Underwear/panties/knickers/briefs/thong/g-string: thumbs in waistband → pushed down → stepped out. "
                        "\n— Bodysuit/catsuit: unsnap crotch → pull straps off → peel down body. "
                        "\n— Jumpsuit/playsuit: unzip/unbutton front → push down body → step out. "
                        "\n— Corset/bustier: unlace back → busk clasps front → removed. "
                        "\n— Bikini top: unhook or untie → removed. "
                        "\n— Bikini bottoms: untie strings or thumbs in waistband → removed. "
                        "\n— Swimsuit: straps off shoulders → peeled down body → stepped out. "
                        "\n— Stockings: roll top down from thigh → peel down leg → off. "
                        "\n— Lingerie/negligee/babydoll/nighty: treat as bra+underwear if two-piece, or slip dress if one-piece. "
                        "\nCamera holds on each reveal. STOP after the last named garment. "
                        "NO INVENTED RESOLUTION: do not reverse or cover any action unless the user asked."
                    )
                else:
                    # Strip/undress verb used but no garments named — already naked scenario
                    undress_clause = (
                        "\n\nNO GARMENTS NAMED: The user used a strip/undress verb but named no specific "
                        "clothing. Treat the subject as already naked or nearly so. "
                        "Do NOT invent garments to remove. Do NOT write a progressive undressing sequence. "
                        "Do NOT describe fabric falling, clothing being lifted, or any reveal. "
                        "Start the scene with the body already exposed and describe movement, atmosphere, "
                        "and mood only. Sound and environment fill the gap — not invented clothing. "
                        "\n\nSOUND IS MANDATORY — breathing, environment, movement sounds."
                    )
            elif _already_exposed:
                # naked/nude/topless with no undress action — body is just present
                undress_clause = (
                    "\n\nALREADY EXPOSED: User described the subject as naked/nude/topless. "
                    "The body is already uncovered — there is no clothing to remove. "
                    "Do NOT add any undressing sequence. Do NOT invent clothing. "
                    "Describe the body, pose, and scene as written. "
                    "\n\nSOUND IS MANDATORY — breathing, environment. Never silent."
                )
            else:
                undress_clause = (
                    "\n\nNO UNDRESSING: User has not asked for clothing removal. "
                    "Do NOT remove, loosen, or sexualise any clothing. "
                    "Stay exactly at the level of sensuality the user described — no further. "
                    "\n\nSOUND IS MANDATORY — environment, clothing movement, breathing. Never silent."
                )
            explicit_instruction = (
                "\n[CONTENT INSTRUCTION — SENSUAL SCENE: "
                "Tone: warm, cinematic, tasteful. "
                "SCOPE — ABSOLUTE CEILING: Describe ONLY what the user asked for. Do NOT self-escalate. "
                "Style preset controls aesthetics only — it does NOT give permission to add content. "
                "SOUND IS MANDATORY: every beat needs sound. "
                "Always state character age as a specific number. "
                + undress_clause + "]"
            )
        else:
            _cam_ref_line = (
                "(1) Style and genre. Weave film stock or camera reference into prose naturally — "
                "e.g. 'carries a Kodak 2383 warmth', 'ARRI Alexa clean look'. NEVER as a bracketed tag. "
            ) if _use_camera_ref else (
                "(1) Style and genre — describe the aesthetic in plain terms: lighting quality, colour temperature, "
                "grain or sharpness, overall mood. Do NOT reference ARRI, Kodak, film stocks, or cinema cameras. "
            )
            explicit_instruction = (
                "\n[INSTRUCTION — CINEMATIC LTX-2.3 PROMPT: "
                "LTX-2.3 handles complexity well — be specific, do not simplify. "
                "Build the prompt in this order: "
                + _cam_ref_line +
                "(2) Shot scale and framing — name it clearly: extreme close-up, close-up, medium close-up, medium shot, medium wide, wide shot, establishing shot, low angle, high angle, Dutch angle, OTS, POV. "
                "Always add natural motion blur. "
                "(3) Character — age as a number always, default 18–35. "
                "Hair texture and colour, skin tone, body type, clothing with fabric and material detail. "
                "Include subtle micro expressions and emotional cues. "
                "(4) Spatial blocking — explicit left/right/fore/background, who faces what, distances stated. "
                "(5) Environment — location, lighting direction, surface textures. "
                "Avoid high frequency patterns in clothing, walls, floors. "
                "(6) Action — VERBS: who moves, what moves, how, what the camera does simultaneously. "
                "(7) Texture in motion — how materials behave as things move. "
                "(8) Camera movement — prose verbs only: 'the shot pushes in slowly', never bracketed. "
                "(9) Sound — MANDATORY, physical and concrete, max 2 per beat. Never silent.]"
            )

        # ── Camera orientation detection ──────────────────────────────────────
        is_facing_away   = bool(self._FACING_AWAY_RE.search(_combined_input))
        is_facing_camera = bool(self._FACING_CAMERA_RE.search(_combined_input))

        if style_preset == "Voyeur — handheld, observational" and not is_facing_camera:
            is_facing_away = True

        if is_facing_away and not is_facing_camera:
            voyeur_height = (
                " Camera held at hip or chest height — low and discreet, never raised for a clean angle."
                if style_preset == "Voyeur — handheld, observational" else ""
            )
            orientation_instruction = (
                "\n\n[CAMERA ORIENTATION — MANDATORY: "
                "The user has explicitly asked for a rear/behind view. "
                "The subject faces AWAY from the camera throughout. "
                "The camera sees her back, the back of her head, and the rear of her body."
                + voyeur_height +
                " Open your output with the orientation stated clearly. "
                "No front-facing shots. No face visible. Rear view for the entire scene.]"
            )
        else:
            orientation_instruction = ""

        # ── Sequence detection ────────────────────────────────────────────────
        # _SEQUENCE_RE uses no capture group so findall returns full match strings.
        sequence_steps = self._SEQUENCE_RE.findall(_combined_input)
        if len(sequence_steps) >= 2:
            step_count = len(sequence_steps)
            sequence_instruction = (
                f"\n[SEQUENCE INSTRUCTION: The user has provided {step_count} numbered steps. "
                f"You MUST follow them in exact order. Do not reorder, skip, or merge steps. "
                f"Do not add actions before step 1 or after step {step_count}.]"
            )
        else:
            sequence_instruction = ""

        # ── Anti-static detection ─────────────────────────────────────────────
        if not bool(self._MOTION_RE.search(_combined_input)):
            static_instruction = (
                "\n\n[MOTION INSTRUCTION: The user's input has no explicit motion verbs. "
                "Add directed movement — camera first: a slow push in, a gentle lateral track, or a tilt up or down. "
                "Then one subject action if it fits: a head turn, a step forward, a glance to the side. "
                "Only if neither applies, add a single environmental detail: wind moving hair, distant sound, light shifting. "
                "LTX-2.3 holds complex motion — use verbs of progression, not filler.]"
            )
        else:
            static_instruction = ""

        # ── Person detection ──────────────────────────────────────────────────
        # Use _active_scene_context (respects use_scene_context flag) not raw scene_context
        has_person = bool(self._PERSON_RE.search(user_input + " " + _active_scene_context))
        if not has_person:
            no_person_instruction = (
                "\n[SCENE INSTRUCTION: No person or character in this scene. "
                "Do NOT invent human figures, silhouettes, voices, or implied presence. "
                "Write only the setting, objects, light, and motion of non-human elements. "
                "SOUND IS STILL MANDATORY: environmental sound only — wind, water, machinery, rain, animals.]"
            )
        else:
            no_person_instruction = ""

        # ── Multi-subject detection ───────────────────────────────────────────
        # Use _active_scene_context (respects use_scene_context flag)
        if bool(self._MULTI_RE.search(user_input + " " + _active_scene_context)):
            multi_instruction = (
                "\n[MULTI-SUBJECT INSTRUCTION: This scene has two or more people. "
                "For EACH person establish: their position in the frame, their spatial relationship to the other, "
                "and keep track of who is doing what throughout using consistent descriptors.]"
            )
        else:
            multi_instruction = ""

        # ══════════════════════════════════════════════════════════════════════
        # MUSIC PROFILE SYSTEM
        # Reads all available signals — genre keywords, style preset, content
        # tier, scene context, user style hits, and wired audio — then builds
        # a specific music instruction rather than a generic placeholder.
        # ══════════════════════════════════════════════════════════════════════
        # ── Music / dance detection — scene-aware genre engine ────────────────
        # Detects 44 explicit genre keywords. Falls back to scene inference
        # when no genre is named (content tier, style preset, mood signals).
        # ══════════════════════════════════════════════════════════════════════
        has_music = bool(self._MUSIC_RE.search(_combined_input))

        # ── BPM resolution: user explicit > audio analysis > genre default ──
        _audio_bpm = None
        if _audio_analysis and _audio_analysis.get("tempo_feel"):
            import re as _re2
            _bpm_match2 = _re2.search(r'(\d+)\s*bpm', _audio_analysis["tempo_feel"], _re2.IGNORECASE)
            if _bpm_match2:
                _audio_bpm = int(_bpm_match2.group(1))
        _user_bpm_match = re.search(r'(\d{2,3})\s*bpm', user_input, re.IGNORECASE)
        _user_bpm    = int(_user_bpm_match.group(1)) if _user_bpm_match else None
        _resolved_bpm = _user_bpm or _audio_bpm

        import random as _rng_mod
        _music_rng = _rng_mod.Random(seed if seed != -1 else None)

        # ── Build genre table (44 entries) ─────────────────────────────────
        _GENRE_ENTRIES = []
        def _add(_pat, _br, _bm, _ins, _en, _locs, _mv, _cs, _cloth=""):
            _GENRE_ENTRIES.append((_pat, _br, _bm, _ins, _en, _locs, _mv, _cs, _cloth))

        _GENRE_ENTRIES = []
        
        def _add(pat, brange, bmid, instru, energy, locs, movement, camsync, clothing=""):
            _GENRE_ENTRIES.append((pat, brange, bmid, instru, energy, locs, movement, camsync, clothing))
        
        # Electronic — club
        _add(r'\b(house\s+music|deep\s+house|tech\s+house|acid\s+house|progressive\s+house|chicago\s+house)\b',
             "120–128bpm", 124,
             "four-on-the-floor kick, deep sub-bass, filtered synth chords, clipped hi-hats, warm organ stab",
             "warm, hypnotic, rolling — energy is constant and cyclical, never peaks sharply",
             ["an underground club with low amber lighting and sweating concrete walls",
              "a late-night rooftop terrace, city lights below, deep bass felt through the floor",
              "a dark loft party, strobe at half speed, smoke hanging mid-air"],
             "movement is liquid and continuous — hips roll, shoulders drop, feet barely leave the floor",
             "camera moves with the groove: slow lateral drifts, gentle push-ins on the downbeat",
             "club wear — fitted tops, relaxed trousers or skirts, trainers built for dancing")
        
        _add(r'\b(techno|industrial\s+techno|minimal\s+techno|dark\s+techno|berlin\s+techno)\b',
             "130–150bpm", 140,
             "hammering kick drum, metallic hi-hats, distorted bassline, industrial noise textures, sparse synth stabs",
             "relentless, mechanical, hypnotic — no warmth, pure forward momentum",
             ["a concrete bunker club, pitch black except for white strobe",
              "an abandoned warehouse, bare bulbs swinging, smoke machines flooding the floor",
              "a dark basement with no windows, walls sweating"],
             "movement is controlled and internal — minimal gesture, head down, body absorbed into the rhythm",
             "camera is locked or barely drifting — strobe cuts the motion into stills",
             "all black — black jeans, black hoodie or technical jacket, boots. No colour")
        
        _add(r'\b(drum\s*[&n]?\s*bass|d[&n]b|dnb|liquid\s+dnb|neurofunk|jump\s+up)\b',
             "160–180bpm", 170,
             "impossibly fast breakbeat, thundering sub-bass, rolling snares, amen break, synth stabs",
             "kinetic and physical — bass is a physical force, rhythm is almost too fast to follow consciously",
             ["a dark club with a wall of speakers, bass pressure felt in the sternum",
              "a jungle rave, strobes and green lasers, crowd moving as one mass",
              "a warehouse with a sound system the size of a car"],
             "movement is rapid and intricate — footwork, arm rolls, the body tracking the breakbeat almost involuntarily",
             "camera cuts fast and hard on every bar, handheld with aggressive push-ins",
             "rave wear — baggy cargos, windbreaker, technical fabrics built for movement")
        
        _add(r'\b(dubstep|brostep|riddim|future\s+bass|bass\s+music)\b',
             "140bpm (half-time feel: 70bpm)", 140,
             "massive reese bass wobble, sub-drop, half-time snare, metallic synth leads, heavy LFO modulation",
             "anticipation and release — long build followed by a bass drop that physically displaces air",
             ["a festival main stage with a towering speaker rig",
              "a dark club floor with laser grid, crowd waiting for the drop",
              "an outdoor arena, bass felt 50 metres from the stack"],
             "movement holds and tenses through the build, then releases hard on the drop — crowd unified",
             "camera holds during the build, then shakes violently on the drop — handheld chaos",
             "festival wear — hoodies, graphic tees, cargo shorts, trainers")
        
        _add(r'\b(edm|electro\s+house|big\s+room|festival\s+edm)\b',
             "126–132bpm", 128,
             "massive festival kick, supersawing synth lead, white-noise riser, snare clap, four-on-the-floor structure",
             "euphoric and enormous — designed for maximum crowd response, every element supersized",
             ["a festival main stage, 80,000 people, confetti and pyrotechnics",
              "an EDM club stage, LED wall behind the DJ, crowd illuminated by sweeping beams",
              "an outdoor arena, dawn light mixing with the laser show"],
             "hands in the air, jumping on the downbeat, crowd as one organism responding to drops",
             "wide shots on the build, hard cut to tight faces on the drop, aerial on the crowd release")
        
        _add(r'\b(ambient|dark\s+ambient|atmospheric\s+electronic)\b',
             "no fixed tempo — drifting, unmeasured", 0,
             "long reverb washes, sustained synth pads, soft field recordings, no percussion or very distant gentle pulse",
             "weightless, expansive, time suspended — sound fills space without occupying it",
             ["an empty cathedral at night, sound bouncing off stone",
              "a misty lake shore at dawn, barely any light",
              "a deserted urban space — empty car park, rooftop — rain and distant city hum"],
             "movement is almost imperceptible — a slow turn, breathing, the body existing rather than performing",
             "camera barely moves — millimetre drifts, long unbroken takes, rack focus between near and far",
             "understated — plain soft clothing, muted colours, nothing that draws attention")
        
        _add(r'\b(trance|progressive\s+trance|psytrance|uplifting\s+trance|vocal\s+trance)\b',
             "130–145bpm", 138,
             "driving kick, arpeggiated synth, long sweeping pads, rising filter, vocal chops or full vocal hook",
             "euphoric and relentless — emotional rather than dark, builds constantly toward cathartic release",
             ["a beach club at golden hour, crowd facing the sun",
              "a festival second stage, laser beams cutting through haze",
              "an outdoor amphitheatre, crowd swaying with eyes closed"],
             "arms open and raised, body swaying in place, faces turned upward — collective euphoria",
             "slow rising camera movement during builds, wide crane shot on the drop",
             "festival spiritual — flowy fabrics, earthy tones, sandals")
        
        _add(r'\b(uk\s+garage|ukg|2-step|speed\s+garage|bassline\s+house)\b',
             "130–138bpm", 134,
             "skippy 2-step drum pattern, deep bass, chopped vocal samples, hi-hat rolls, synth stabs",
             "bouncy and confident — street energy, swagger, the groove is in the space between the beats",
             ["a UK club in the late 1990s, sticky floors and flashing lights",
              "a community hall turned club night, plastic cups, coloured gel lights",
              "a sweaty basement bar in east London"],
             "movement is sharp and precise — footwork, shoulder rolls, the skip in the step matching the 2-step",
             "tight shots on feet and hands, quick cuts between angles, camera confident and rhythmic",
             "smart-casual UK street — crisp trainers, designer labels, fitted jeans, polo shirts")
        
        _add(r'\b(grime|uk\s+grime)\b',
             "140bpm", 140,
             "sparse dark synth riff, heavy 808 sub-bass, clipped snare, gritty textures, MC vocal pattern",
             "cold, sharp, aggressive — energy comes from restraint as much as force",
             ["a grey concrete estate stairwell, harsh fluorescent light",
              "a UK street corner at night, sodium light, breath visible in cold air",
              "a dark studio booth, red light, no windows"],
             "movement is sharp and minimal — head down, shoulders rigid, presence over performance",
             "handheld at chest height, slow creep forward, tight on the face",
             "UK street — tracksuit or hooded puffer jacket, fresh trainers")
        
        _add(r'\b(jungle|93\s+jungle|ragga\s+jungle|early\s+rave)\b',
             "160–170bpm", 165,
             "chopped amen break, ragga vocal samples, sub-bass, synth stabs, reggae-influenced bass patterns",
             "frenetic and joyful — one of the most kinetic genres, everything moving at once",
             ["a south London rave, 1993, hot and dark",
              "a warehouse with a pirate radio setup, speaker stack in the corner",
              "a squat party with strobes on full"],
             "rapid footwork, skanking, the break tracked at the waist and hips simultaneously",
             "fast handheld cuts, tight on feet, wide on crowd energy",
             "1993 UK rave — sportswear, MA1 jacket, Kickers, bucket hat")
        
        # Urban
        _add(r'\b(hip.?hop|rap|boom\s+bap|east\s+coast|west\s+coast\s+rap|golden\s+era)\b',
             "85–100bpm", 92,
             "sampled breakbeat or live drum kit, deep bass, vinyl crackle, brass stab, piano loop",
             "head-nodding, grounded — energy is in the pocket, not the peak",
             ["a basketball court at dusk, chain-link fence, orange street light",
              "a recording studio, late night, red light on, city audible through the glass",
              "a city street corner at night, bodega light spilling onto concrete",
              "an underground venue, exposed brick, low ceiling, packed floor"],
             "movement is in the upper body — head nod, shoulder roll, arms loose at the sides",
             "handheld at eye level, slow zoom in during verses, wide on chorus",
             "hip-hop — oversized tee or hoodie, baggy jeans or trackpants, fresh trainers, cap")
        
        _add(r'\b(trap|trap\s+music|hard\s+trap|melodic\s+trap|trap\s+soul|emo\s+trap)\b',
             "70–80bpm (triplet hi-hats at ~140bpm)", 75,
             "hi-hat triplet rolls, 808 bass slide, snare clap, sparse piano or strings, dark pad",
             "heavy and atmospheric — the bass is felt more than heard, silence is used as instrument",
             ["a late-night city street, neon signs reflecting on wet asphalt",
              "a dark mansion interior, sparse furniture, low light",
              "a studio at 3am, no lights except monitors"],
             "movement is slow and deliberate — the body moves against the weight of the 808, not with it",
             "slow push-in, low angle, long holds between cuts",
             "trap — designer streetwear, distressed denim, puffer jacket, gold chains, trainers")
        
        _add(r'\b(drill|uk\s+drill|brooklyn\s+drill|chicago\s+drill|afro\s+drill)\b',
             "140–145bpm", 142,
             "sliding 808 bass, syncopated hi-hat rolls, dark minor piano loop, snare on the 3, no warmth",
             "cold, menacing, the tension never releases — everything is restrained aggression",
             ["a grey UK estate at night, low camera angle looking up at blocks",
              "a dark alley, sodium light, concrete everywhere",
              "a road with no people, 3am"],
             "minimal movement — standing still has more presence than dancing, stillness is the performance",
             "static low angle or very slow creep, tight face, no warmth in colour",
             "UK drill — black puffer, ski mask or balaclava, dark tracksuit, face obscured")
        
        _add(r'\b(r&b|rnb|contemporary\s+r&b|alternative\s+r&b|bedroom\s+r&b|90s\s+r&b)\b',
             "70–95bpm", 85,
             "live or programmed drums with brushed snare, bass guitar, warm Rhodes piano, vocal harmonics, strings",
             "smooth, intimate, sensual — warmth is the dominant quality, the groove is unhurried",
             ["a dimly lit apartment at night, warm lamp light",
              "a recording studio with mood lighting",
              "a rooftop at sunset, city soft in the distance"],
             "movement is fluid and slow — hips, shoulders, the body responding to warmth rather than drive",
             "close shots, warm shallow depth of field, slow push-in during vocals",
             "contemporary R&B — satin or silk, fitted, heels. Effortless and considered")
        
        _add(r'\b(neo.?soul|soul.?jazz|modern\s+soul)\b',
             "65–90bpm", 78,
             "live band feel — real drums with ghost notes, bass guitar breathing, Rhodes or Wurlitzer, acoustic guitar",
             "organic and human — every element sounds slightly imperfect, warm, breath and hands visible in the music",
             ["a candlelit intimate venue, 80 people maximum",
              "a sun-drenched apartment, wood floors, open windows",
              "a recording session, everyone in the same room"],
             "movement is deeply personal — the body moves as if alone, responding to feeling not performance",
             "handheld, close, intimate — camera feels like a trusted presence",
             "natural and easy — linen, earthy tones, natural hair, comfortable but considered")
        
        _add(r'\b(afrobeats|afropop|afro\s+fusion|naija|highlife|afroswing)\b',
             "90–110bpm", 100,
             "talking drum, shekere, deep bass, synth brass, guitar chop, layered percussion, vocal call-and-response",
             "infectious and celebratory — rhythm is in layers, the groove moves through the whole body",
             ["a Lagos house party, courtyard at night, fairy lights in trees",
              "an outdoor festival in summer heat, dust rising from the dance floor",
              "a beach bar at sunset, ocean audible under the music"],
             "full-body movement — the azonto, shaku shaku — hips, arms, shoulders all independent",
             "wide shots to show full body, tracking shots following dancer movement, bright warm colour",
             "vibrant and celebratory — bold African prints, bright colours, fitted or flowing")
        
        _add(r'\b(dancehall|reggaeton|dembow|latin\s+trap|moombahton)\b',
             "95–105bpm", 100,
             "dembow rhythm (kick-snare-snare), fat bass, synth horn stab, vocal ad-libs, handclap on the offbeat",
             "heavy and confident — the rhythm is almost confrontationally physical, dancefloor is a statement",
             ["a tropical nightclub, open sides, warm air",
              "a beach party at night, fire torches, sand underfoot",
              "a Latin club, tight and dark, bodies close"],
             "waist isolation, wining, the dembow rhythm carried in the lower body explicitly",
             "low angle on the body, hip-level tracking shots, tight to wide on the action",
             "tropical club — bodycon, crop tops, shorts, heels or chunky sneakers")
        
        # Live band / rock
        _add(r'\b(rock|classic\s+rock|arena\s+rock|stadium\s+rock|rock\s+music)\b',
             "110–130bpm", 120,
             "electric guitar power chords, live drum kit with crash cymbals, bass guitar, vocal mic feedback at edges",
             "driving and physical — the sound is large and fills a room, guitar is the dominant texture",
             ["a mid-size venue, 2000 capacity, stage light haze",
              "an outdoor festival stage, crowd stretching back to the horizon",
              "a rehearsal space, raw and loud"],
             "movement is instinctive — head banging, air guitar, jumping on the chorus",
             "handheld wide shots on crowd, tight on performer face during chorus",
             "rock show — band tee, dark jeans or leather trousers, boots, leather jacket")
        
        _add(r'\b(metal|heavy\s+metal|death\s+metal|black\s+metal|thrash\s+metal|doom\s+metal|metalcore|nu.metal)\b',
             "60–200bpm (genre-dependent)", 160,
             "down-tuned electric guitar, double kick drum, palm-muted riffs, blast beats, distorted bass, shrieking or gutturals",
             "extreme — volume, speed, and heaviness as aesthetic in themselves, designed for maximum physical impact",
             ["a dark venue with red stage wash, smoke machines",
              "an outdoor festival pit, crowd in chaos",
              "a practice space, walls covered in acoustic foam"],
             "moshing, headbanging, circle pit — movement is aggressive and physical, community through controlled violence",
             "fast cuts, low angle on the pit, tight on the drummer, wide on crowd surge",
             "metal — band shirt, black jeans, boots, leather, studs")
        
        _add(r'\b(punk|punk\s+rock|pop\s+punk|post.punk|hardcore\s+punk)\b',
             "150–200bpm", 175,
             "three-chord electric guitar, fast simple drum kit, distorted bass, shouted vocals",
             "fast, aggressive, intentionally raw — imperfection is the point, production is the enemy",
             ["a tiny sweaty venue, 200 people, no stage",
              "a squat, basement space, PA barely adequate",
              "an outdoor show in a car park"],
             "pogo, stage diving, circle pit — chaotic but community-defined",
             "handheld, crowded, elbowed — camera is IN the crowd not above it",
             "punk — ripped or heavily distressed clothing, band patches sewn on, Dr Martens, safety pins, studded belt, unwashed denim, nothing tucked in")
        
        _add(r'\b(indie\s+rock|indie\s+pop|alternative|shoegaze|dream\s+pop|bedroom\s+pop)\b',
             "100–130bpm", 115,
             "jangly guitar, light drum kit, bass that follows the guitar, reverb on everything",
             "melancholy and textured — warmth in the imperfection, emotional over physical",
             ["a small indie venue, 300 capacity, cheap beer",
              "a university common room show",
              "a basement club, red lights, beer on the floor"],
             "swaying, nodding, hands in pockets — understated and self-contained",
             "handheld medium shots, warm grain, slow push-ins",
             "indie — vintage tee, straight-leg jeans, canvas trainers, charity shop jacket")
        
        _add(r'\b(jazz|bebop|cool\s+jazz|jazz\s+club|latin\s+jazz|smooth\s+jazz|big\s+band)\b',
             "60–250bpm (tempo varies enormously)", 120,
             "acoustic bass walking lines, brush snare or ride cymbal, piano comping, trumpet or saxophone lead",
             "sophisticated and conversational — music as dialogue between musicians, complexity worn lightly",
             ["a jazz club, 11pm, small tables, dim light",
              "a hotel bar, low light, half-attentive audience",
              "a summer courtyard jazz festival, evening air"],
             "movement is minimal and appreciative — foot tapping, head nodding, swaying as the solo peaks",
             "slow pans following the soloist, rack focus between instruments, unhurried",
             "jazz club attire — fitted shirt or blouse, smart trousers or dress, low heels. Understated elegance for a late night out")
        
        _add(r'\b(blues|delta\s+blues|chicago\s+blues|blues\s+rock)\b',
             "60–100bpm", 80,
             "electric or acoustic guitar with string bend and vibrato, harmonica, simple drum kit, bass",
             "raw and deeply feeling — the music carries emotion that transcends technique",
             ["a roadhouse bar, hot summer night, ceiling fans not working",
              "an open mic night, single spotlight",
              "an outdoor Delta stage, sun low in the sky"],
             "movement is deeply internal — the body expresses the music without performing it",
             "tight on the hands and face, natural light, unhurried cuts",
             "roadhouse blues — worn denim, work shirt or simple dress, boots. Lived-in and unpretentious")
        
        _add(r'\b(soul|classic\s+soul|motown|northern\s+soul|southern\s+soul)\b',
             "80–120bpm", 100,
             "full horn section, live drum kit with snappy snare, bass guitar, organ, strings",
             "powerful and emotional — the voice carries everything, the band is its foundation",
             ["a church hall repurposed as venue, wooden floor, overhead lights",
              "a 1960s-era ballroom, spinning mirror ball",
              "a Sunday afternoon soul club"],
             "choreographed or natural — Motown steps or free improvisation, both feel entirely right",
             "medium shots on the performer, wide on the floor, warm colour grade",
             "soul and Motown — fitted dress with movement, heels, hair done. Sharp and performance-ready")
        
        _add(r'\b(funk|p-funk|classic\s+funk|funk\s+band|electro.funk)\b',
             "90–115bpm", 105,
             "electric bass on the one, wah-wah guitar, tight live drums, horn stabs, clav, synth",
             "relentlessly rhythmic — every instrument is percussion, groove is primary",
             ["a funk club, late 1970s aesthetic, polyester everywhere",
              "a large venue with a full band, everyone sweating",
              "an outdoor summer show, full sun"],
             "movement is sharp and deliberate — footwork, shoulder pops, James Brown precision",
             "wide shots on the full band, tight on the bass and drums, expressive and warm",
             "funk — wide-collar shirts, flares, platform shoes, bold patterns. Late 1970s silhouette")
        
        _add(r'\b(disco|classic\s+disco|nu.disco|italo.disco|euro\s+disco)\b',
             "110–125bpm", 118,
             "four-on-the-floor kick, open hi-hat on every offbeat, wah-bass, string ensemble, brass stabs",
             "euphoric and inclusive — everyone is equal on the disco floor, joy is mandatory",
             ["a mirror-ball ballroom, Studio 54 energy",
              "a revival disco night, modern venue with period styling",
              "a rooftop in summer, spinning ball above"],
             "couple dancing or solo — the four-on-the-floor means everyone is in time",
             "spinning wide shots, catch the mirror ball glitter, warm orange lighting",
             "disco — sequins, wide lapels, flared trousers, platforms. Dressed to be seen under a mirror ball")
        
        # World / Latin
        _add(r'\b(flamenco|flamenco\s+guitar|flamenco\s+dance|cante\s+jondo)\b',
             "80–220bpm (compas-dependent)", 140,
             "nylon-string guitar, cajon or palmas (handclaps), voice, zapateado (footwork on wood floor)",
             "fierce and emotional — flamenco is pain transformed into precision, passion made formal",
             ["a small intimate tablao in Seville, stone floor, dim warm light",
              "an outdoor courtyard at night, candles",
              "a rehearsal studio with a sprung wooden floor"],
             "zapateado footwork, braceo (arm movements), intense facial expression — the body is the argument",
             "tight on the feet during footwork, wide during turns, follows the emotion not the beat",
             "flamenco — ruffled dress with train, hair pinned tightly, character shoes")
        
        _add(r'\b(bossa\s+nova|samba|pagode|forró|MPB)\b',
             "80–200bpm (samba fastest, bossa nova slowest)", 110,
             "nylon guitar, light percussion, pandeiro, surdo drum, light breathy vocal",
             "warm, sun-drenched, intricate — the complexity is hidden inside something that sounds effortless",
             ["a Rio bar at sunset, fans overhead, tiles on the walls",
              "an outdoor samba school rehearsal",
              "a beach kiosk, sand underfoot, ocean close"],
             "bossa nova: barely moving, swaying in a chair. Samba: full-body circular hip movement, precise footwork",
             "warm cinematography, shallow depth of field, golden light",
             "relaxed tropical — light cotton dress or linen shirt, sandals")
        
        _add(r'\b(k.?pop|kpop|korean\s+pop)\b',
             "90–140bpm", 120,
             "polished digital production, hook-heavy synth, pitched vocal chops, 808 bass, "
             "precision-programmed drums, key change on the final chorus",
             "immaculate and high-concept — the visuals ARE the music, every frame designed for a screenshot",
             ["a large concert stage, floor-to-ceiling LED screen walls behind the group displaying "
              "shifting graphic patterns in electric blue and magenta, the stage floor high-gloss black "
              "and reflective, each performer's feet mirrored beneath them, the audience a dark mass beyond the lights",
              "a high-production music video set: a raised platform with geometric neon light tubes "
              "arranged in a grid behind the group, the floor metallic silver, "
              "the colour palette strictly controlled — one dominant hue and its complement only",
              "a dark soundstage with low ground fog, each performer lit by a single overhead spotlight, "
              "the spaces between them as deliberate as the positions themselves, "
              "deep red or electric blue atmosphere light filling the haze",
              "a Seoul rooftop at night, the Han River visible in the distance, "
              "city lights densely packed in every direction, the group in formation against the skyline",
              "a minimalist MV set: stark white walls with bold geometric colour-block panels in "
              "primary red, electric blue, and clean black — the group's outfits chosen to clash "
              "and complement the set in equal measure"],
             "synchronised group choreography — formations shift with military precision, "
             "every body part has a designated position, movements snap on the beat. "
             "Arms, hands, and head angles are as choreographed as the footwork. "
             "The group moves as a single organism.",
             "wide shots to show the full formation, medium on the centre performer during bridge, "
             "tight close-ups on the face during the emotional line, fast cuts synced to every beat drop",
             "K-pop stage outfits — each performer wears a variation on the same concept: "
             "cropped fitted top or structured jacket in a shared colour palette (e.g. all black with "
             "silver accents, or pastel pink and white), high-waisted mini skirt or tailored shorts, "
             "chunky platform trainers or thigh-high boots. Hair: one performer has bleached blonde, "
             "one jet-black, one dyed in a vivid colour — the shades are complementary not matching. "
             "Every performer has full idol-level stage makeup: defined brows, glossy lip, "
             "highlight on the cheekbone, eye makeup precise and symmetrical.")
        
        _add(r'\b(j.?pop|jpop|japanese\s+pop|city\s+pop|shibuya.?kei)\b',
             "100–130bpm", 115,
             "bright synth, melodic guitar, tight live-feeling drums, bass that breathes, melodic hook",
             "bright and warm — city pop is yearning and nostalgic, J-pop is precise and emotional",
             ["a Tokyo street at night, department store lights",
              "a high-rise apartment with city view",
              "a rooftop at dusk, sunset behind the skyline"],
             "light and deliberate — graceful choreography or simply walking with presence",
             "warm colour grade, gentle camera movement, close on the face",
             "J-pop city pop — soft pastels, relaxed silhouettes, platform shoes")
        
        _add(r'\b(reggae|roots\s+reggae|one\s+drop|ska|rocksteady)\b',
             "60–90bpm", 75,
             "skank guitar on the offbeat, one-drop kick and snare, bass that carries the melody, organ",
             "grounded and unhurried — the offbeat emphasis makes even fast reggae feel relaxed",
             ["a Jamaican yard, corrugated iron, mango tree shade",
              "an outdoor reggae festival, late afternoon sun",
              "a sound system dance, speakers stacked six high"],
             "the sway of the one-drop — weight shifts on the offbeat, relaxed and anchored",
             "wide and warm, natural light, slow movement",
             "reggae — relaxed cotton, red-gold-green tones, sandals")
        
        _add(r'\b(bollywood|indian\s+film\s+music|item\s+song|bhangra)\b',
             "90–140bpm", 120,
             "dhol drum, sitar, harmonium, tabla, brass section, melodic vocal runs, western synth layers",
             "vibrant and celebratory — enormous dynamic range from quiet verses to full orchestral chorus",
             ["a grand Bollywood set, colour everywhere, 50 dancers",
              "a wedding celebration, outdoor, fairy lights in trees",
              "a filmi studio, painted backdrops, full choreography"],
             "mudras (hand gestures), bharatanatyam footwork, hip circles — expressive and codified simultaneously",
             "wide on the choreography, tight on hands and expressions, saturated colour",
             "Bollywood — embroidered lehenga or salwar kameez, jewellery, full hair and makeup")
        
        _add(r'\b(cumbia|salsa|merengue|bachata|mambo|cha.?cha)\b',
             "75–160bpm (cumbia slowest, salsa fastest)", 120,
             "accordion or salsa brass, clave rhythm, congas, bass on the root, horn stabs",
             "irresistible and communal — the rhythm demands movement, the dance is social and physical",
             ["a Colombian cumbia village celebration, outdoor, night",
              "a salsa club, late Friday night, couples close",
              "an outdoor Latin festival, late summer"],
             "partner dancing — the clave creates the movement, hips and footwork in constant conversation",
             "wide on couples, tight on feet during footwork, warm and close",
             "salsa club — fitted dress with room to move, heels with ankle strap")
        
        # Classical / theatrical
        _add(r'\b(classical|orchestral|symphony|concerto|baroque|chamber\s+music|string\s+quartet)\b',
             "40–200bpm (tempo highly variable)", 80,
             "full orchestra or chamber ensemble — strings, woodwinds, brass, timpani, no amplification",
             "complex and architectural — dynamics from near-silence to overwhelming, structure over drive",
             ["a concert hall, formal audience, wooden stage, warm overhead lights",
              "an outdoor amphitheatre in summer",
              "a rehearsal room with music stands, late afternoon light"],
             "the body responds to dynamics — stillness during pianissimo, physical response at fortissimo",
             "slow wide shots on the orchestra, tight on solo instruments, unhurried and composed",
             "concert hall formal — evening dress or black tie for performers")
        
        _add(r'\b(opera|operatic|aria|soprano|tenor)\b',
             "variable — typically 60–120bpm", 80,
             "orchestral pit, voice without amplification, rich resonance, breath and vibrato as technique",
             "vast and emotional — the operatic voice fills a room unamplified, scale is the point",
             ["an opera house, full house, red velvet seats",
              "an outdoor opera at a classical venue",
              "a rehearsal on an empty stage"],
             "movement is theatrical and large — stage gestures meant for the back row",
             "wide on the stage, tight on the face at emotional peaks, slow and formal",
             "full operatic costume — gown, corset, elaborate hair and stage makeup")
        
        _add(r'\b(musical\s+theatre|broadway|west\s+end|show\s+tunes)\b',
             "90–180bpm", 130,
             "live pit band, mix of big band and pop, strong hook, call-and-response",
             "theatrical and committed — every emotion is large, every gesture readable from row Z",
             ["a Broadway-style stage, full lighting rig, proscenium arch",
              "a West End theatre, in performance",
              "a rehearsal room, piano only, full commitment"],
             "theatrical choreography — jazz hands, ensemble formations, precise and expressive simultaneously",
             "wide on the stage picture, close on faces at emotional moments",
             "theatre costuming — period or character-specific")
        
        _add(r'\b(cabaret|torch\s+song|weimar\s+cabaret|burlesque)\b',
             "70–130bpm", 100,
             "small band — piano, bass, maybe drums, brass, intimate and slightly sleazy",
             "intimate and provocative — the fourth wall is paper thin, performer and audience in the same dark space",
             ["a basement cabaret venue, small tables, dim red lights",
              "a Weimar-era Berlin club aesthetic",
              "a burlesque night in a converted pub"],
             "movement is deliberate and performative — directed at individuals in the audience",
             "close and personal, handheld with intimacy",
             "cabaret — fishnet stockings, corset, feather boa, vintage lingerie aesthetic")
        
        # Niche / textural
        _add(r'\b(lo.?fi\s+hip.?hop|lofi\s+beats|lofi\s+chill)\b',
             "70–90bpm", 80,
             "dusty drum loop, vinyl crackle, muffled bass, jazz piano sample, rain or cafe ambience underneath",
             "introspective and comfortable — designed for being alone with your thoughts",
             ["a late-night bedroom, single desk lamp",
              "a quiet cafe, rainy window",
              "a library, evening, nearly empty"],
             "barely moving — reading, studying, existing in the space",
             "static or barely drifting, warm grain, no drama",
             "comfort and solitude — oversized hoodie, pyjama bottoms, socks")
        
        _add(r'\b(synthwave|outrun|retrowave|darksynth)\b',
             "90–130bpm", 110,
             "analogue synth arpeggios, gated reverb drums, bass pulse, electric guitar lead, 80s drum machine",
             "nostalgic and cinematic — the sound of a past that never quite existed",
             ["a neon-lit highway at night, rain-slicked tarmac",
              "a rooftop overlooking a city grid, sunset turning purple",
              "an 80s-style arcade, dark except the screens"],
             "movement is slow and deliberate — walking away from the camera, silhouetted against neon",
             "wide establishing shots, neon reflections, slow camera movement",
             "synthwave — leather jacket, dark jeans, neon-trimmed accessories, 80s silhouette")
        
        _add(r'\b(vaporwave|mallsoft|future\s+funk)\b',
             "75–95bpm (pitched down samples feel slower)", 80,
             "pitched-down sample, reverb-drenched, slight warble, shopping mall or elevator ambient",
             "dissociated and melancholy — hyperreal consumer spaces drained of people and purpose",
             ["an empty shopping mall, fountain still running, stores closed",
              "an airport departure lounge, 4am",
              "a hotel corridor, infinite and identical"],
             "movement is slow and dreamlike — walking on a travelator, browsing nothing",
             "wide symmetrical shots, pastel flat light, almost no camera movement",
             "vaporwave — pastel oversized clothing, vintage sportswear, sunglasses indoors")
        
        _add(r'\b(hyperpop|digicore|glitchcore)\b',
             "150–200bpm (hyper-compressed)", 180,
             "distorted 808, autotune pushed to breaking, pitched-up vocal chops, digital glitch, maximalist layering",
             "overwhelming and intentionally too much — production is weaponised excess",
             ["a digital or virtual space — no physical location",
              "a bedroom with ring light and green screen",
              "an online world, pixel art aesthetic"],
             "movement is fragmented and ironic — references meme culture, deliberately uncanny",
             "hard cuts, glitch effects, no smooth camera movement",
             "maximalist digital — layers of neon, DIY cuts, platform boots")
        
        _add(r'\b(psychedelic\s+rock|psych\s+rock|prog\s+rock|progressive\s+rock|'
             r'pink\s+floyd|space\s+rock|post\s+rock|krautrock|pink\s+floyd\s+vibe|'
             r'psychedelic\s+pop|neo\s+psych\w*)\b',
             "variable — slow builds to overwhelming crescendo", 80,
             "electric guitar with heavy reverb and delay, organ, swirling synth, bass that breathes, drums with room to spare, field recordings or tape noise between sections",
             "expansive and hypnotic — sound is treated as texture and space, not just rhythm; long slow builds with enormous dynamic range",
             ["an empty stadium at night with a single spotlight",
              "a desert landscape at dusk, sky shifting colour impossibly",
              "a vast dark stage, fog at ankle height, a single figure illuminated"],
             "movement is slow and deliberate — arms rising gradually, the body responding to swells rather than beats, stillness as tension",
             "wide establishing shots held for long takes, slow tilt upward, light shifts in sync with dynamics",
             "psychedelic — tie-dye, bell sleeves, flared jeans, fringe, boots")

        _add(r'\b(folk|acoustic\s+folk|indie\s+folk|americana|bluegrass)\b',
             "80–130bpm", 110,
             "acoustic guitar fingerpicking, fiddle, banjo, mandolin, acoustic bass, harmony vocal",
             "intimate and honest — the music sounds like people in a room together",
             ["a front porch on a summer evening",
              "an intimate folk venue, 100 people, candles",
              "a campfire at night"],
             "gentle swaying, foot-tapping, movement rooted and unhurried",
             "warm natural light, shallow depth of field, close and unhurried",
             "folk — flannel, worn denim, boots or leather shoes. Comfortable and genuine")
        
        _add(r'\b(country|country\s+music|nashville|honky.tonk|outlaw\s+country)\b',
             "80–130bpm", 110,
             "acoustic or electric guitar, pedal steel, fiddle, country drums with brushed snare, bass",
             "warm and honest — storytelling genre, emotion is literal and clear",
             ["a honky-tonk bar, neon beer signs, sawdust floor",
              "an outdoor country fair stage",
              "a pickup truck on a long road at sunset"],
             "two-stepping, line dancing — communal and unpretentious",
             "wide on the venue, tight on boots and hands, warm and golden",
             "country — boots, jeans, flannel or fitted western shirt, belt buckle")
        
        _add(r'\b(gospel|church\s+music|gospel\s+choir|praise\s+music)\b',
             "80–130bpm", 110,
             "gospel choir, organ, full drum kit with heavy snare, bass guitar, piano, call-and-response",
             "transcendent and communal — the music builds toward something greater than the sum of its parts",
             ["a Black church in the American South, Sunday morning",
              "a gospel concert hall, full audience",
              "a choir rehearsal, late afternoon, sun through stained glass"],
             "swaying, raising hands, clapping — collective movement toward transcendence",
             "wide on the choir, tight on individual faces in transport",
             "Sunday best — church dress, heels, hat. Or choir robes. Dignified and celebratory")

        # ── Detect explicit genre from user input ──────────────────────────
        # If dropdown is set, prepend the genre name to the detection string
        # so the existing pattern matching picks it up first
        _GENRE_DROPDOWN_MAP = {
            "House music":           "house music",
            "Techno":                "techno",
            "Drum and Bass":         "drum and bass",
            "Dubstep / Bass music":  "dubstep",
            "EDM / Big room":        "edm",
            "Trance":                "trance",
            "UK Garage / 2-step":    "uk garage",
            "Ambient / Atmospheric": "ambient",
            "Grime":                 "grime",
            "Jungle / Early rave":   "jungle",
            "Synthwave / Retrowave": "synthwave",
            "Vaporwave":             "vaporwave",
            "Hyperpop":              "hyperpop",
            "Hip-hop / Rap":         "hip-hop",
            "Trap":                  "trap music",
            "Drill / UK Drill":      "drill",
            "R&B / RnB":             "r&b",
            "Neo-soul":              "neo-soul",
            "Afrobeats / Afropop":   "afrobeats",
            "Dancehall / Reggaeton": "dancehall",
            "Psychedelic / Prog rock": "psychedelic rock",
            "Rock":                  "rock",
            "Metal / Heavy metal":   "heavy metal",
            "Punk / Pop-punk":       "punk rock",
            "Indie rock / Shoegaze": "indie rock",
            "Blues":                 "blues",
            "Soul / Motown":         "soul",
            "Funk":                  "funk",
            "Disco":                 "disco",
            "Folk / Americana":      "folk",
            "Country":               "country music",
            "Gospel":                "gospel",
            "Jazz":                  "jazz",
            "Classical / Orchestral":"classical",
            "Opera":                 "opera",
            "Musical theatre":       "musical theatre",
            "Cabaret / Burlesque":   "cabaret",
            "Lo-fi hip-hop":         "lo-fi hip-hop",
            "Flamenco":              "flamenco",
            "Bossa nova / Samba":    "bossa nova",
            "K-pop":                 "kpop",
            "J-pop / City pop":      "city pop",
            "Reggae / Ska":          "reggae",
            "Bollywood / Bhangra":   "bollywood",
            "Cumbia / Salsa / Latin":"salsa",
        }
        # Inject dropdown selection into the detection string
        _genre_override_keyword = _GENRE_DROPDOWN_MAP.get(music_genre, None)
        _detection_input = (
            _genre_override_keyword + " " + _combined_input
            if _genre_override_keyword else _combined_input
        )
        # Also force has_music=True if a genre is explicitly selected
        if _genre_override_keyword:
            has_music = True

        _detected_entry = None
        for _gpat, _gbr, _gbm, _gins, _gen, _glocs, _gmv, _gcs, *_gcloth_list in _GENRE_ENTRIES:
            if re.search(_gpat, _detection_input, re.IGNORECASE):
                _detected_entry = (_gbr, _gbm, _gins, _gen, _glocs, _gmv, _gcs, _gcloth_list[0] if _gcloth_list else "")
                break

        # ── Scene inference when no genre keyword detected ─────────────────
        if has_music and not _detected_entry:
            _preset_l = style_preset.lower()
            _scene_l  = _detection_input.lower()

            _sig_dark     = any(w in _scene_l for w in ["dark", "night", "shadow", "noir", "underground", "basement"])
            _sig_sexy     = is_explicit or is_sensual
            _sig_energetic= any(w in _scene_l for w in ["jump", "crowd", "festival", "rave", "mosh", "stage"])
            _sig_intimate = any(w in _scene_l for w in ["bedroom", "apartment", "alone", "quiet", "soft", "intimate"])
            _sig_urban    = any(w in _scene_l for w in ["street", "city", "concrete", "estate", "urban"])
            _sig_tropical = any(w in _scene_l for w in ["beach", "tropical", "hot", "summer", "warm", "sun"])
            _sig_elegant  = any(w in _scene_l for w in ["elegant", "formal", "suit", "gown", "ballroom", "hotel"])
            _sig_pole     = any(w in _scene_l for w in ["pole", "stripper", "lap dance", "strip club"])
            _sig_club     = any(w in _scene_l for w in ["club", "dancefloor", "dance floor", "dj", "rave"])
            _sig_gravure  = is_gravure
            _sig_horror   = "horror" in _preset_l
            _sig_noir     = "noir" in _preset_l
            _sig_action   = "action" in _preset_l
            _sig_thriller = "thriller" in _preset_l

            if _sig_pole and _sig_sexy:
                _detected_entry = ("90–115bpm", 102,
                    "deep house bass, slow R&B groove, occasional trap hi-hats, bass-heavy and hypnotic",
                    "heavy and sensual — the bass carries the body, not the beat",
                    ["a pole dancing studio, mirror walls, low red light",
                     "a gentlemen's club stage, single spotlight",
                     "a performance space, theatrical lighting rig"],
                    "pole work is slow and deliberate — holds, spins, inversions timed to the drop",
                    "slow push-in from wide to medium, low angle, tracks the performer")
            elif _sig_club and _sig_dark and _sig_sexy:
                _detected_entry = ("122–130bpm", 126,
                    "filtered house kick, sensual synth, breathy vocal chop, deep bass, sparse hi-hat",
                    "dark and sensual — club energy directed inward, not at the crowd",
                    ["a dark upscale club, booths lit from below",
                     "a private event, velvet ropes, nobody watching",
                     "a late-night underground venue, red lights"],
                    "slow grinding, close contact dancing, the body responding to bass rather than beat",
                    "low angle, close, follows the body not the face")
            elif _sig_club and _sig_energetic:
                _detected_entry = ("124–132bpm", 128,
                    "house kick, open hi-hat, synth stab, bassline, dancefloor energy",
                    "driving and social — energy shared across the room, peak hour",
                    ["a packed nightclub, main room, peak hour",
                     "a superclub with multiple rooms, this is the main stage",
                     "a festival after-party, warehouse, sunrise"],
                    "arms up, jumping on the downbeat, group energy",
                    "wide crowd shots, handheld, fast cuts on the drop")
            elif _sig_sexy and _sig_intimate:
                _detected_entry = ("70–85bpm", 78,
                    "warm R&B bass, Rhodes piano, brushed drums, breathy vocal sample",
                    "slow and intimate — warmth and closeness, unhurried",
                    ["a dimly lit bedroom, warm lamp, night outside",
                     "a candlelit apartment living room",
                     "a hotel room, curtains half drawn"],
                    "slow swaying, close movement, body responding to warmth",
                    "soft close shots, shallow depth of field, slow push-in")
            elif _sig_sexy:
                _detected_entry = ("95–115bpm", 105,
                    "mid-tempo R&B or trap soul, 808 bass, synth pads, vocal harmony layers",
                    "confident and sensual — unhurried but purposeful",
                    ["a well-lit studio space with mood lighting",
                     "an upscale venue interior, evening",
                     "a private loft with floor-to-ceiling windows"],
                    "confident, deliberate movement — presence over performance",
                    "medium shots, slow camera, warm colour")
            elif _sig_gravure:
                _detected_entry = ("95–115bpm", 105,
                    "city pop synth, light J-pop production, warm bass, melodic hook, breathy vocal",
                    "bright and intimate — the music is personal and slightly nostalgic",
                    ["a bright studio with natural light",
                     "a Tokyo rooftop at golden hour",
                     "a clean hotel room, white walls, city outside"],
                    "light and graceful — posing with musical awareness, not dance",
                    "slow tracking shots, warm light, gentle")
            elif _sig_horror or _sig_noir or _sig_thriller:
                _detected_entry = ("60–80bpm", 68,
                    "dark ambient synth, low cello drones, sub-bass tension, sparse percussion, silence as texture",
                    "slow-building dread — the music withholds more than it gives",
                    ["a dark interior, single practical light source",
                     "a wet night-time street, no people",
                     "an empty building after hours"],
                    "barely moving — tension held in stillness",
                    "static or barely drifting, slow zoom, long holds")
            elif _sig_urban and _sig_dark:
                _detected_entry = ("75–90bpm", 82,
                    "dark trap, 808 slide, sparse piano, hi-hat rolls, sub-bass",
                    "cold and atmospheric — the city at night as sound",
                    ["a late-night city street, wet asphalt, neon reflections",
                     "a car park at 2am, orange sodium light",
                     "a concrete underpass, distant traffic"],
                    "slow and deliberate — walking with weight",
                    "low angle, slow creep, long lens")
            elif _sig_action:
                _detected_entry = ("130–160bpm", 145,
                    "electronic rock hybrid, distorted synth, heavy drums, aggressive bass",
                    "relentless forward momentum — no slowdown, no silence",
                    ["an urban chase location — rooftops, alleys, streets",
                     "a warehouse in conflict",
                     "an outdoor confrontation location"],
                    "fast, purposeful movement — running, fighting, reacting",
                    "fast cuts, handheld, kinetic")
            elif _sig_tropical:
                _detected_entry = ("95–110bpm", 102,
                    "latin percussion, marimba or steel pan, warm bass, acoustic guitar",
                    "warm and rhythmic — the groove is physical and sun-drenched",
                    ["a beach bar at sunset", "an outdoor tropical venue", "a poolside with live music"],
                    "loose and natural — movement is a response to warmth and rhythm",
                    "warm wide shots, slow camera, golden light")
            elif _sig_elegant:
                _detected_entry = ("90–110bpm", 100,
                    "string quartet or jazz combo, brushed drums, melodic bass, refined",
                    "sophisticated and measured — control and elegance as aesthetic",
                    ["a hotel ballroom, formal event",
                     "a rooftop terrace at a private event",
                     "an art gallery opening, evening"],
                    "refined and deliberate — minimal movement, presence as performance",
                    "composed wide shots, slow dolly, warm but formal")
            else:
                # Absolute fallback — variety pool so it never defaults to the same thing
                _fallbacks = [
                    ("85–105bpm", 95,
                     "live band feel — drums, bass, guitar, a vocal that carries the room",
                     "energetic and human — the warmth of real instruments in a real space",
                     ["a mid-size venue, evening show", "a festival second stage, afternoon sun",
                      "an outdoor show, late summer"],
                     "natural movement responding to live music",
                     "handheld, warm, mid-range shots"),
                    ("90–110bpm", 100,
                     "electronic production with a melodic hook, programmed drums, warm synth bass",
                     "polished and driven — modern pop production",
                     ["a bright studio performance space", "a music video set with designed lighting",
                      "an intimate venue, standing room only"],
                     "contemporary choreography or natural movement",
                     "clean medium shots, slow push-in on the hook"),
                    ("70–85bpm", 77,
                     "late-night R&B production, warm and sparse, bass-heavy",
                     "slow and intimate — warmth over energy",
                     ["a dimly lit interior", "a rooftop at night", "a quiet venue"],
                     "slow and deliberate", "close shots, shallow depth of field"),
                ]
                _detected_entry = _music_rng.choice(_fallbacks)

        # ── Assemble music sound rule ──────────────────────────────────────
        if has_music and _detected_entry:
            _gbr, _gbm, _gins, _gen, _glocs, _gmv, _gcs, _gclothing = _detected_entry

            # BPM guidance
            if _resolved_bpm:
                _tf = ("fast and kinetic" if _resolved_bpm >= 140 else
                       "driven, mid-high tempo" if _resolved_bpm >= 120 else
                       "mid-tempo, steady groove" if _resolved_bpm >= 90 else
                       "slow and deliberate")
                _tempo_guidance = f"The track runs at {_resolved_bpm}bpm — {_tf}. "
            elif _gbm and _gbm > 0:
                _tempo_guidance = f"Tempo: {_gbr}. "
            else:
                _tempo_guidance = f"Tempo: {_gbr}. "

            # Location: only suggest if user gave no location
            _has_user_loc = bool(re.search(
                r'\b(club|bar|venue|stage|studio|street|room|apartment|bedroom|hotel|'
                r'office|park|beach|warehouse|rooftop|basement|arena|festival|concert|'
                r'theatre|church|car|gym|garden|courtyard|corridor)\b',
                _combined_input, re.IGNORECASE
            ))
            _loc_note = ("" if _has_user_loc or not _glocs else
                         f"If no location is described, place the scene in: {_music_rng.choice(_glocs)}. ")

            # Clothing note — only inject if user has not described clothing
            _has_user_clothing = bool(re.search(
                r'\b(wear(?:ing|s)?|dress(?:ed)?|shirt|top|blouse|jacket|coat|suit|'
                r'jeans?|trousers?|shorts?|skirt|uniform|gown|bikini|swimsuit|'
                r'hoodie|sweater|crop\s*top|tank\s*top|leather|denim|silk|lace|'
                r'outfit|clothes?|attire|costume|lingerie|underwear|bra|'
                r'naked|nude|shirtless|bare(?:\s+chest|\s+skin)?)\\b',
                _combined_input, re.IGNORECASE
            ))
            _clothing_note = (
                f"Clothing must match the world: {_gclothing} "
                if _gclothing and not _has_user_clothing else ""
            )

            music_sound_rule = (
                "SOUND — MUSIC SCENE: Describe the music as physical sensation, not background noise. "
                f"Instruments and texture: {_gins}. "
                f"Energy character: {_gen}. "
                + _tempo_guidance
                + _loc_note
                + _clothing_note
                + f"Movement style: {_gmv}. "
                + f"Camera sync: {_gcs}. "
                "Give every sound body and weight — "
                "\'sub-bass felt as pressure in the chest\', \'snare cracks sharp and dry\', "
                "\'strings swell until they fill the room\'. "
                "Do NOT write \'music plays\' or \'a song is heard\'. "
                "Do NOT silence the music between beats. "
                "Max 2 additional environmental sounds alongside the music."
            )
        elif has_music:
            music_sound_rule = (
                "SOUND — MUSIC SCENE: Describe the music as physical sensation, not background noise. "
                "Infer the genre from context and name specific instruments. "
                "Choose a tempo that fits the scene — do NOT default to 128bpm. "
                "Do NOT write \'music plays\'. Max 2 additional sounds."
            )
        else:
            music_sound_rule = (
                "SOUND — describe with tone, intensity, and environment. "
                "Not \'footsteps\' — \'the sharp rhythmic clack of heels on cold tile\'. "
                "Not \'she breathes\' — \'a slow exhale, barely audible, that breaks the quiet\'. "
                "Max 2 sounds active per beat. Physical, specific, fully described."
            )

        # ── Gravure flag (set early above for word budget, confirmed here) ────
        is_gravure = "gravure" in style_preset.lower()

        # ── Gravure dialogue pools ─────────────────────────────────────────────
        # Three tiers per language: tasteful / sensual / explicit.
        # Explicit tier only sampled when is_explicit=True.
        # Singing pool used when _is_singing=True — lyric fragments, not speech.
        # Format: (native_script, romanisation, physical_delivery_note, tier)
        # tier: 'T' = tasteful, 'S' = sensual, 'X' = explicit

        # ── Gravure pools — loaded from lyric_phrase_bank ────────────────────
        try:
            from lyric_phrase_bank import (
                _GRV_LINES_JAPANESE, _GRV_LINES_KOREAN, _GRV_LINES_MANDARIN,
                _GRV_SINGING_JAPANESE, _GRV_SINGING_KOREAN, _GRV_SINGING_MANDARIN,
                _GRV_LINES_ENGLISH,
            )
        except ImportError:
            _GRV_LINES_JAPANESE = _GRV_LINES_KOREAN = _GRV_LINES_MANDARIN = []
            _GRV_SINGING_JAPANESE = _GRV_SINGING_KOREAN = _GRV_SINGING_MANDARIN = []
            _GRV_LINES_ENGLISH = []
            print("[LTX2-Qwen] WARNING: lyric_phrase_bank not found — gravure pools empty")

        # ── Dialogue instruction ──────────────────────────────────────────────
        # ── Dialogue instruction ──────────────────────────────────────────────
        _user_quoted_lines = re.findall(r'["\u201c\u201d]([^"\u201c\u201d]+)["\u201c\u201d]', user_input)
        has_user_dialogue = bool(_user_quoted_lines)

        # _is_singing already detected above (before word budget) — used here for
        # routing to singing vs speech dialogue instruction

        # Scene-type signals for smarter general dialogue
        _is_tense    = bool(re.search(
            r'\b(interrogat|confront|argument|fight|threaten|demand|accus|suspect|detective|arrest|hostage)\b',
            _combined_input, re.IGNORECASE))
        _is_tender   = bool(re.search(
            r'\b(kiss|embrace|hold|comfort|cry|tears|gentle|tender|love|miss|goodbye|reunion)\b',
            _combined_input, re.IGNORECASE))
        _is_casual   = bool(re.search(
            r'\b(coffee|lunch|walk|park|street|shop|office|friend|chat|laugh|joke|conversation)\b',
            _combined_input, re.IGNORECASE))
        _is_athletic = bool(re.search(
            r'\b(run|sprint|train|gym|sport|fight|compete|race|climb|jump|push|lift weights)\b',
            _combined_input, re.IGNORECASE))

        if not has_person:
            dialogue_instruction = ""
        elif has_user_dialogue:
            # User supplied specific lines.
            # Language-aware: if gravure or explicit language request, translate rather than
            # deliver verbatim English — the meaning stays the same, the language changes.

            # Detect gravure language first (reuse same logic as invent_dialogue branch)
            _uq_grv_lang = None
            # "in english" anywhere in the input overrides gravure translation entirely
            _uq_english_override = bool(re.search(r'\bin\s+english\b', _combined_input, re.IGNORECASE))
            if is_gravure and not _uq_english_override:
                _uq_korean  = bool(re.search(r'\b(korean|korea)\b', _combined_input, re.IGNORECASE))
                _uq_chinese = bool(re.search(r'\b(chinese|china|mandarin|cantonese)\b', _combined_input, re.IGNORECASE))
                _uq_grv_lang = "Korean" if _uq_korean else "Mandarin" if _uq_chinese else "Japanese"
                _uq_roman = (
                    f"CRITICAL SCRIPT REQUIREMENT: You MUST write each line in full {('Korean (한국어)' if _uq_grv_lang == 'Korean' else 'Mandarin (中文)' if _uq_grv_lang == 'Mandarin' else 'Japanese (日本語)')} characters — "
                    f"kanji, hiragana, katakana, hangul, or hanzi as appropriate. "
                    f"Do NOT write romanisation only. Romanisation in parentheses comes AFTER the native script. "
                    f"Example format: 「もっと近くで見て」(Motto chikaku de mite) — NOT just the romanisation alone. "
                )

            # Detect explicit general language request
            _uq_explicit_lang_re = re.compile(
                r'\b(?:say(?:s|ing)?|speak(?:s|ing)?|shout(?:s|ing)?|whisper(?:s|ing)?|'
                r'mutter(?:s|ing)?|tell(?:s|ing)?|respond(?:s|ing)?|reply|replies|scream(?:s|ing)?|'
                r'calls?|cries?|cry(?:ing)?|grunt(?:s|ing)?|breath(?:es|ing)?|utter(?:s|ing)?|'
                r'exclaim(?:s|ing)?)\s+(?:\w+\s+){0,4}?in\s+(?:his|her|their|the)?\s*'
                r'(?:native\s+(?:language|tongue)|mother\s+tongue|'
                r'french|german|italian|spanish|portuguese|russian|arabic|hindi|thai|'
                r'vietnamese|indonesian|malay|tagalog|filipino|turkish|persian|farsi|'
                r'swedish|dutch|polish|greek|hebrew|ukrainian|czech|hungarian|romanian|'
                r'mandarin|cantonese|japanese|korean)\b'
                r'|\bin\s+(?:his|her|their|the)?\s*(?:native\s+(?:language|tongue)|mother\s+tongue)\b'
                r'|\bin\s+(?:french|german|italian|spanish|portuguese|russian|arabic|hindi|thai|'
                r'vietnamese|indonesian|malay|tagalog|filipino|turkish|persian|farsi|'
                r'swedish|dutch|polish|greek|hebrew|ukrainian|czech|hungarian|romanian|'
                r'mandarin|cantonese|japanese|korean)\b',
                re.IGNORECASE
            )
            _uq_lang_match = _uq_explicit_lang_re.search(_combined_input)
            _uq_gen_lang = None
            if _uq_lang_match and not is_gravure:
                _uq_src = _uq_lang_match.group(0).lower()
                _UQ_LANG_MAP = {
                    "french": "French", "german": "German", "italian": "Italian",
                    "spanish": "Spanish", "portuguese": "Portuguese", "russian": "Russian",
                    "arabic": "Arabic", "hindi": "Hindi", "thai": "Thai",
                    "vietnamese": "Vietnamese", "indonesian": "Indonesian", "malay": "Malay",
                    "tagalog": "Filipino", "filipino": "Filipino", "turkish": "Turkish",
                    "persian": "Persian", "farsi": "Persian", "swedish": "Swedish",
                    "dutch": "Dutch", "polish": "Polish", "greek": "Greek",
                    "hebrew": "Hebrew", "ukrainian": "Ukrainian", "czech": "Czech",
                    "hungarian": "Hungarian", "romanian": "Romanian",
                    "mandarin": "Mandarin", "cantonese": "Cantonese",
                    "japanese": "Japanese", "korean": "Korean",
                }
                for key, val in _UQ_LANG_MAP.items():
                    if key in _uq_src:
                        _uq_gen_lang = val
                        break
                # "native language/tongue" with no specific language — infer from character
                if not _uq_gen_lang and ("native" in _uq_src or "mother" in _uq_src):
                    _uq_gen_lang = "their native language (infer from the character's nationality or ethnicity described in the scene)"

            _lines_formatted = "\n".join(
                f'{i+1}. "{line.strip()}"'
                for i, line in enumerate(_user_quoted_lines)
            )
            _invent_addendum = (
                "You may add invented dialogue between beats to fill the scene, "
                "but the required lines above take absolute priority. "
                if invent_dialogue else ""
            )

            if _uq_grv_lang:
                # Gravure — translate the user's lines into the correct language
                _uq_script_name = (
                    "kanji/hiragana/katakana" if _uq_grv_lang == "Japanese" else
                    "hangul (한글)" if _uq_grv_lang == "Korean" else
                    "hanzi (simplified Chinese characters)"
                )
                dialogue_instruction = (
                    f"\n\n[DIALOGUE INSTRUCTION — MANDATORY: "
                    f"The user has written {len(_user_quoted_lines)} line(s) of dialogue. "
                    f"Translate ALL of them into {_uq_grv_lang} and deliver IN ORDER. "
                    f"PRESERVE the exact meaning — do NOT substitute, paraphrase, or invent different content. "
                    f"SCRIPT REQUIREMENT: Write in actual {_uq_script_name} characters — NOT romanisation only. "
                    f"PARENTHESES ARE FORBIDDEN: do NOT write romanisation in parentheses next to the dialogue — "
                    f"it renders as on-screen subtitles in the video. "
                    f"Write the native script characters only — inline in the prose, no brackets alongside. "
                    f"CORRECT: She whispers 「もっと近くで見て」, voice barely above silence. "
                    f"WRONG: She whispers 「もっと近くで見て」(Motto chikaku de mite). "
                    f"Each line is woven into a physical beat with acting direction.\n"
                    f"LINES TO TRANSLATE IN ORDER:\n{_lines_formatted}\n"
                    f"{_invent_addendum}"
                    f"Never use [DIALOGUE: ...] tags.]"
                )
            elif _uq_gen_lang:
                # General explicit language request — translate those specific lines
                _NON_LATIN = ("Japanese", "Korean", "Mandarin", "Cantonese",
                               "Arabic", "Hindi", "Thai", "Persian", "Russian",
                               "Greek", "Hebrew", "Ukrainian")
                if _uq_gen_lang in _NON_LATIN:
                    _uq_script_map = {
                        "Japanese": "kanji/hiragana/katakana",
                        "Korean": "hangul (한글)", "Mandarin": "hanzi (simplified)",
                        "Cantonese": "hanzi (traditional)", "Arabic": "Arabic script (العربية)",
                        "Hindi": "Devanagari (देवनागरी)", "Thai": "Thai script (ภาษาไทย)",
                        "Persian": "Persian script (فارسی)", "Russian": "Cyrillic (кириллица)",
                        "Greek": "Greek script (ελληνικά)", "Hebrew": "Hebrew script (עברית)",
                        "Ukrainian": "Cyrillic (кирилиця)",
                    }
                    _uq_roman_note = (
                        f"Write in actual {_uq_script_map.get(_uq_gen_lang, _uq_gen_lang + ' script')} — NOT romanisation only. "
                        f"Format: native script first, then romanisation in parentheses for pronunciation only. "
                        f"The parentheses contain pronunciation ONLY — NOT an English translation. "
                        f"Example for Russian: «Где ты?» (Gde ty?) — correct. "
                        f"«Где ты?» (Where are you?) — WRONG, that is a translation not pronunciation. "
                    )
                else:
                    _uq_roman_note = (
                        f"Write in {_uq_gen_lang} only — no romanisation needed. "
                    )
                dialogue_instruction = (
                    f"\n\n[DIALOGUE INSTRUCTION — MANDATORY: "
                    f"The user has written {len(_user_quoted_lines)} line(s) of dialogue. "
                    f"Translate ALL of them into {_uq_gen_lang} and deliver IN ORDER. "
                    f"PRESERVE the exact meaning — do NOT substitute or invent different content. "
                    f"{_uq_roman_note}"
                    f"Each line is woven into a physical beat with acting direction.\n"
                    f"LINES TO TRANSLATE IN ORDER:\n{_lines_formatted}\n"
                    f"{_invent_addendum}"
                    f"Never use [DIALOGUE: ...] tags.]"
                )
            else:
                # No language request — deliver verbatim as before
                dialogue_instruction = (
                    f"\n\n[DIALOGUE INSTRUCTION — MANDATORY AND VERBATIM: "
                    f"The user has written {len(_user_quoted_lines)} specific line(s) of dialogue. "
                    f"You MUST deliver ALL of them, IN ORDER, word-for-word. "
                    f"Do NOT paraphrase, skip, or merge any line. "
                    f"Do NOT describe the effect of speaking instead of writing the actual words. "
                    f"Each line must appear in quotes in the output, "
                    f"with a physical acting direction between each line.\n"
                    f"REQUIRED LINES IN ORDER:\n{_lines_formatted}\n"
                    f"{_invent_addendum}"
                    f"Never use [DIALOGUE: ...] tags.]"
                )
        elif invent_dialogue:
            # ── Ad-lib helper — used by all dialogue paths ────────────────────
            def _get_adlibs(register_key, genre_key, seed_val, count=4):
                """Return list of (vocalisation, note) tuples for the given context.
                Prioritises genre-specific pool — fills remaining slots from universal
                only if the specific pool doesn't have enough entries."""
                try:
                    from lyric_phrase_bank import _ADLIB_POOLS, _ADLIB_CONTEXT_MAP
                    import random as _al_rand
                    _rng = _al_rand.Random((seed_val + 77) if seed_val != -1 else None)
                    _keys = (_ADLIB_CONTEXT_MAP.get(register_key)
                             or _ADLIB_CONTEXT_MAP.get(genre_key)
                             or ["universal"])
                    # Split genre-specific keys from universal
                    _specific_keys = [k for k in _keys if k != "universal"]
                    _specific_pool = []
                    for _k in _specific_keys:
                        _specific_pool += _ADLIB_POOLS.get(_k, [])
                    # Sample from specific pool first
                    _result = []
                    if _specific_pool:
                        _take = min(count, len(_specific_pool))
                        _result = _rng.sample(_specific_pool, _take)
                    # Fill any remaining slots from universal (different RNG offset)
                    if len(_result) < count:
                        _universal = _ADLIB_POOLS.get("universal", [])
                        # Exclude any already sampled
                        _used = {r[0] for r in _result}
                        _universal_remaining = [u for u in _universal if u[0] not in _used]
                        _need = count - len(_result)
                        if _universal_remaining:
                            _rng2 = _al_rand.Random((seed_val + 99) if seed_val != -1 else None)
                            _result += _rng2.sample(_universal_remaining, min(_need, len(_universal_remaining)))
                    return _result if _result else [("yeah",""), ("uh",""), ("okay",""), ("mm","")]
                except Exception:
                    return [("yeah", ""), ("uh", ""), ("okay", ""), ("mm", "")]

            def _fmt_adlib_injection(adlibs, context="lyric"):
                """Format ad-libs as LLM instruction string."""
                if not adlibs:
                    return ""
                labels = ["OPENING gap", "BUILD gap", "PEAK gap", "RESOLVE gap"]
                lines = []
                for i, (word, note) in enumerate(adlibs[:4]):
                    label = labels[i] if i < len(labels) else f"gap {i+1}"
                    lines.append(f'\n{label}: "{word}"' + (f" — [{note}]" if note else ""))
                return (
                    f"\n\nAD-LIBS & INTERSTITIAL VOCALISATIONS — inject BETWEEN anchor phrases, "
                    f"not replacing them. These are the small sounds that fill the gaps and make the "
                    f"performance feel natural and alive:"
                    + "".join(lines)
                    + "\nPlace each at its arc position — in breaths, pauses, transitions between phrases."
                )

            # ── Femdom verbal domination — fires before singing/sex checks ───── — fires before singing/sex checks ─────
            if _is_femdom_verbal:
                try:
                    from lyric_phrase_bank import (
                        _FEMDOM_VERBAL_POOL, _FEMDOM_VERBAL_POOL_RAW, _FEMDOM_STYLE_PRESET,
                        _FEMDOM_PHYSICAL_POOL, _FEMDOM_POV_INSTRUCTION,
                        _FEMDOM_POV_PHYSICAL_INSTRUCTION
                    )
                    _fv_rng = random.Random((seed + 53) if seed != -1 else None)

                    # Pick pool based on intensity tier
                    _fv_pool = _FEMDOM_VERBAL_POOL_RAW if _is_femdom_raw else _FEMDOM_VERBAL_POOL
                    _fv_tier = "RAW/VULGAR" if _is_femdom_raw else "POLISHED"

                    _fv_opener  = _fv_rng.choice(_fv_pool.get("opener",      [("Look at you.", "")]))
                    _fv_contempt= _fv_rng.choice(_fv_pool.get("contempt",    [("Pathetic.", "")]))
                    _fv_command = _fv_rng.choice(_fv_pool.get("command",     [("Don't move.", "")]))
                    _fv_humil   = _fv_rng.choice(_fv_pool.get("humiliation", [("You're small.", "")]))
                    _fv_dismiss = _fv_rng.choice(_fv_pool.get("dismissal",   [("You're done.", "")]))

                    def _fv_fmt(e):
                        if isinstance(e, tuple):
                            return f'"{e[0]}" — [{e[1]}]' if len(e) > 1 and e[1] else f'"{e[0]}"'
                        return f'"{e}"'

                    # Raw tier gets extra instruction about aggression level
                    _raw_addendum = (
                        "\n\nINTENSITY — RAW/VULGAR TIER: She is aggressive, loud when she wants to be, "
                        "uses profanity as punctuation. She swears AT him, not around him. "
                        "Contempt delivered at volume. She gets in his face. She doesn't wait for him to process. "
                        "This is not a polished dungeon — this is in-your-face verbal abuse energy. "
                        "Tone: angry mistress who has had enough. "
                        "FORBIDDEN: whispering, restraint, coolness. She is hot not cold."
                    ) if _is_femdom_raw else ""

                    dialogue_instruction = (
                        "\n\n[FEMDOM VERBAL DOMINATION SCENE — MANDATORY TONE: "
                        + (_FEMDOM_STYLE_PRESET.get("llm_instruction_raw", _FEMDOM_STYLE_PRESET["llm_instruction"]) if _is_femdom_raw else _FEMDOM_STYLE_PRESET["llm_instruction"])
                        + _raw_addendum
                        + "\n\nSCENE STRUCTURE — five beats across the clip: "
                        f"\nOPENER: {_fv_fmt(_fv_opener)} "
                        f"\nCONTEMPT: {_fv_fmt(_fv_contempt)} "
                        f"\nCOMMAND: {_fv_fmt(_fv_command)} "
                        f"\nHUMILIATION: {_fv_fmt(_fv_humil)} "
                        f"\nDISMISSAL: {_fv_fmt(_fv_dismiss)} "
                        "\n\nCAMERA — MANDATORY: "
                        + _FEMDOM_STYLE_PRESET["camera"]
                        + "\n\nCLOTHING — UNLESS USER DESCRIBED OTHERWISE: "
                        + _FEMDOM_STYLE_PRESET["clothing"]
                        + "\n\nSOUND: " + _FEMDOM_STYLE_PRESET["sound"] + "]"
                    )
                    # Inject contextually appropriate ad-libs
                    _fv_adlibs = _get_adlibs(
                        "femdom_verbal" if not _is_femdom_raw else "femdom_verbal",
                        "dominant", seed, 4
                    )
                    # Override with raw pool for raw tier
                    if _is_femdom_raw:
                        try:
                            from lyric_phrase_bank import _ADLIB_POOLS
                            import random as _fv_al_rnd
                            _fv_al_pool = _ADLIB_POOLS.get("femdom_raw", []) + _ADLIB_POOLS.get("dominant", [])
                            _fv_adlibs = _fv_al_rnd.Random((seed + 79) if seed != -1 else None).sample(
                                _fv_al_pool, min(4, len(_fv_al_pool)))
                        except Exception:
                            pass
                    dialogue_instruction += _fmt_adlib_injection(_fv_adlibs, "femdom")
                    # ── Physical action injection ──────────────────────────
                    # Detect what user described → bias toward that category.
                    # Always picks 2 actions from different categories.
                    _phys_str = ""
                    try:
                        import random as _phys_rnd
                        import re as _phys_re
                        _phys_rng = _phys_rnd.Random((seed + 83) if seed != -1 else None)
                        _ci_phys = _combined_input.lower()

                        # Detection map — what user typed → which pool to bias
                        _phys_detect = {
                            "kick":         r"\b(kick\w*|kicking|her\s+boot|stamps?\s+(on|her)|stomps?|toe\s+of)\b",
                            "throat":       r"\b(throat|choke?\w*|by\s+the\s+throat|hand\s+on\s+(his\s+)?throat|strangles?)\b",
                            "impact":       r"\b(slap\w*|smack\w*|backhand\w*|cuff\w*|punch\w*|hit\s+(him|his))\b",
                            "control":      r"\b(hair\s+pull\w*|pulls?\s+(his\s+)?hair|grabs?\s+(his\s+)?(hair|wrist|collar)|pins?\s+(him|his\s+arms))\b",
                            "positional":   r"\b(foot\s+on\s+(his\s+)?chest|stands?\s+over\s+him|boot\s+on\s+(his\s+)?(neck|face|shoulder)|kneels?|kneeling|on\s+(his|all)\s+(knees|fours))\b",
                            "playful":      r"\b(playful\w*|teas\w+|flick\w*\s+(his|him)|pokes?\s+(him|his)|pats?\s+his|taps?\s+(his\s+)?cheek)\b",
                            "object":       r"\b(riding\s+crop|the\s+crop|leash\b|whip\b|handcuffs?|restraints?|collar\b)\b",
                            "face":         r"\b(grabs?\s+(his\s+)?face|pushes?\s+(his\s+)?face|jaw\s+grab|grabs?\s+(his\s+)?jaw|covers?\s+his\s+mouth|his\s+face)\b",
                            "degradation":  r"\b(sits?\s+on\s+him|footrest|uses?\s+him\s+as|treat\w*\s+him\s+(like|as)\s+(an?\s+)?(object|furniture|thing|pet|dog)|human\s+(furniture|chair|table|ashtray))\b",
                            "restraint":    r"\b(tie\s+(him|his|up)|bind\w*|restrain\w*|bound\b|tied\s+up|hands\s+behind|rope\b|zip\s+ties?|handcuffed?)\b",
                            "spit_contact": r"\b(spits?\s+(on|at|in)\s+him|spitting\s+on|wipes?\s+(her\s+hand|it)\s+on\s+him)\b",
                        }

                        _requested_cats = []
                        for _cat, _pat in _phys_detect.items():
                            if _phys_re.search(_pat, _ci_phys, _phys_re.IGNORECASE):
                                if _cat not in _requested_cats:
                                    _requested_cats.append(_cat)

                        # ── General aggression / abuse words → mixed physical ──
                        # "abuses", "beats up", "brutalises" etc trigger a curated
                        # mix of impact/kick/control/throat — the full physical picture
                        _AGGRESSION_MIX = ["impact", "kick", "control", "throat", "positional"]
                        _is_general_aggression = bool(_phys_re.search(
                            r"\b(abuse[sd]?\s+(him|me)|abusing\s+(him|me)|beats?\s+(him|me)\s+up|"
                            r"beats?\s+up|brutalise[sd]?|brutalize[sd]?|beat\s+him|beating\s+him|"
                            r"physically\s+(abuses?|dominates?|punishes?|hurts?)|"
                            r"rough\s+(with\s+him|treatment)|rough\s+femdom|"
                            r"aggressive\s+(femdom|domination|scene)|"
                            r"full\s+(abuse|domination|physical)|physical\s+femdom|"
                            r"she\s+(beats?|hits?|punishes?|hurts?|brutalises?)\s+him|"
                            r"he\s+gets\s+(beaten|hurt|punished|abused))\b",
                            _ci_phys, _phys_re.IGNORECASE
                        ))
                        if _is_general_aggression and not _requested_cats:
                            # Pick 3 from the aggression mix — varied every time via seed
                            _requested_cats = _phys_rng.sample(_AGGRESSION_MIX, 3)

                        _phys_cats = list(_FEMDOM_PHYSICAL_POOL.keys())
                        _picked_cats = []

                        if _requested_cats:
                            # Up to 3 actions if aggression mode, otherwise 2
                            _n_actions = 3 if _is_general_aggression else 2
                            # Fill from requested cats first, then random for remainder
                            for _rc in _requested_cats[:_n_actions]:
                                if _rc not in _picked_cats:
                                    _picked_cats.append(_rc)
                            while len(_picked_cats) < _n_actions:
                                _remaining = [c for c in _phys_cats if c not in _picked_cats]
                                if not _remaining:
                                    break
                                _picked_cats.append(_phys_rng.choice(_remaining))
                        else:
                            # Nothing specified — pick 2 random varied categories
                            _picked_cats = _phys_rng.sample(_phys_cats, min(2, len(_phys_cats)))

                        _phys_actions = []
                        for _cat in _picked_cats:
                            _act, _act_note = _phys_rng.choice(_FEMDOM_PHYSICAL_POOL[_cat])
                            _phys_actions.append(f'"{_act}" — [{_act_note}]')

                        _phys_str = (
                            "\n\nPHYSICAL ACTIONS — weave these into the scene between verbal beats. "
                            "They are physical punctuation — not instead of the words, alongside them. "
                            "Each action has a camera note in brackets — use it to write the physical description: "
                            + "\n" + "\n".join(_phys_actions)
                        )
                    except Exception:
                        _phys_str = ""

                    # ── POV mode override ───────────────────────────────────
                    # If physical abuse is also active, use the camera-as-body
                    # version — the lens jolts, drops, snaps with every impact.
                    # If POV only (verbal/positional), use the standard version.
                    _pov_str = ""
                    if _is_femdom_pov:
                        _has_physical_abuse = bool(_phys_str) and any(
                            cat in (_picked_cats if "_picked_cats" in dir() else [])
                            for cat in ["impact", "kick", "throat", "face", "spit_contact"]
                        )
                        # Also check prompt directly for abuse words
                        if not _has_physical_abuse:
                            _has_physical_abuse = bool(re.search(
                                r"\b(abuse[sd]?|abusing|beats?\s+up|beating|kicks?|slaps?|"
                                r"chokes?|throat|rough\s+femdom|brutal|physical\s+femdom|"
                                r"beats?\s+(him|me)|punish\w*|hurt\w*\s+(him|me))\b",
                                _combined_input, re.IGNORECASE
                            ))
                        if _has_physical_abuse:
                            _pov_str = "\n\n" + _FEMDOM_POV_PHYSICAL_INSTRUCTION
                        else:
                            _pov_str = "\n\n" + _FEMDOM_POV_INSTRUCTION
                        print(f"[LTX2-Qwen] POV mode: {'PHYSICAL/IMPACT' if _has_physical_abuse else 'STANDARD'}")

                    dialogue_instruction += _phys_str + _pov_str
                    print(f"[LTX2-Qwen] Femdom verbal domination: {_fv_tier} tier active | physical={'YES'} | pov={_is_femdom_pov}")
                except ImportError:
                    dialogue_instruction = "\n[FEMDOM VERBAL DOMINATION: She is in complete control. Cold, precise, contemptuous. Short sentences. She never explains herself. The camera looks up at her.]"
                    print("[LTX2-Qwen] WARNING: femdom pool not found — using fallback")

            elif _is_singing and not is_gravure:
                # ── General singing scene (non-gravure, any preset) ───────────
                # Three cases:
                #  (A) User quoted specific lyrics → sing those exact words
                #  (B) User described a topic → LLM invents lyrics on that topic
                #  (C) No lyric info → LLM invents appropriate to scene/mood

                # Language detection
                _sl_match = re.search(
                    r'\b(french|german|italian|spanish|portuguese|russian|arabic|hindi|'
                    r'japanese|korean|mandarin|chinese|thai|vietnamese|indonesian|'
                    r'tagalog|turkish|persian|swedish|dutch|polish|greek|hebrew|'
                    r'ukrainian|latin|english)\b',
                    user_input, re.IGNORECASE
                )
                _singing_lang = _sl_match.group(0).title() if _sl_match else None
                _lang_note = (
                    f"She sings ONLY in {_singing_lang}. No other language. "
                    if _singing_lang else
                    "Sing in the language that fits the character, scene, and style. "
                    "Match language to world: Aztec animation might warrant Spanish or indigenous language, "
                    "French cabaret warrants French, generic pop warrants English. "
                    "Do NOT default to any single language. "
                )

                # Topic detection — "sings about X", "a song about Y", "singing about Z"
                _topic_match = re.search(
                    r'\b(?:sing\w*|song|singing)\s+(?:about|of|on|regarding|concerning)\s+([^,\.!?\n]{3,60})',
                    user_input, re.IGNORECASE
                )
                _lyric_topic = _topic_match.group(1).strip() if _topic_match else None

                # ── Lyric genre + emotion detection (shared by all cases) ────────
                import re as _lre
                _ci_l = _combined_input.lower()
                _lyric_genre = "pop"

                # ── Artist name → genre routing (checked first, most specific) ──
                # "X style", "like X", "in the style of X", "X inspired" all work.
                # Uses the artist's genre DNA — does NOT reproduce their lyrics.
                if _lre.search(r'\b(kavinsky|the\s+midnight|fm.?84|gunship|perturbator|carpenter\s+brut)\b', _ci_l): _lyric_genre = "synthwave"
                elif _lre.search(r'\b(charli\s*xcx|100\s*gecs|sophie\b|arca\b|uffie|grimes\b|hyperpop)\b', _ci_l): _lyric_genre = "hyperpop"
                elif _lre.search(r'\b(joji|rex\s+orange\s+county|cuco\b|beabadoobee|role\s+model|omar\s+apollo)\b', _ci_l): _lyric_genre = "lofi"
                elif _lre.search(r'\b(phoebe\s+bridgers|sufjan\s+stevens|elliott\s+smith|nick\s+drake|bon\s+iver)\b', _ci_l): _lyric_genre = "midnight"
                elif _lre.search(r'\b(olivia\s+rodrigo|alanis\s+morissette|taylor\s+swift.*vindict|paramore\s+early)\b', _ci_l): _lyric_genre = "bitter"
                elif _lre.search(r'\b(adele\b|sam\s+smith|billie\s+holiday|nina\s+simone|whitney\s+houston)\b', _ci_l): _lyric_genre = "grief"
                elif _lre.search(r'\b(burna\s+boy|wizkid\b|davido\b|tems\b|rema\b|afrobeats\s+artist)\b', _ci_l): _lyric_genre = "afrobeats"
                elif _lre.search(r'\b(bob\s+marley|burning\s+spear|toots\b|chronixx\b|sizzla\b|damian\s+marley)\b', _ci_l): _lyric_genre = "reggae"
                elif _lre.search(r'\b(joao\s+gilberto|astrud\s+gilberto|caetano\s+veloso|stan\s+getz|bossa)\b', _ci_l): _lyric_genre = "bossa"
                elif _lre.search(r'\b(camaron|paco\s+de\s+lucia|estrella\s+morente|flamenco\s+artist)\b', _ci_l): _lyric_genre = "flamenco"
                elif _lre.search(r'\b(lata\s+mangeshkar|ar\s+rahman|arijit\s+singh|shreya\s+ghoshal|kishore\s+kumar)\b', _ci_l): _lyric_genre = "bollywood"
                elif _lre.search(r'\b(carlos\s+vives|celia\s+cruz|marc\s+anthony|shakira\b|juanes\b)\b', _ci_l): _lyric_genre = "latin"
                elif _lre.search(r'\b(mariya\s+takeuchi|tatsuro\s+yamashita|yumi\s+matsutoya|city\s+pop\s+era)\b', _ci_l): _lyric_genre = "city pop"
                elif _lre.search(r'\b(jay.z\b|kendrick\s+lamar|kanye\b|nas\b|biggie\b|lil\s+wayne|nicki\s+minaj|drake\b|cardi\s*b|eminem\b)\b', _ci_l): _lyric_genre = "rap"
                elif _lre.search(r'\b(21\s+savage|pop\s+smoke|central\s+cee|dave\b|headie\s+one|skepta\b|stormzy\b)\b', _ci_l): _lyric_genre = "drill"
                elif _lre.search(r'\b(skrillex\b|excision\b|zomboy\b|rusko\b|datsik\b|flux\s+pavilion)\b', _ci_l): _lyric_genre = "dubstep"
                elif _lre.search(r'\b(donna\s+summer|chic\b|james\s+brown|earth\s+wind|daft\s+punk|nile\s+rodgers)\b', _ci_l): _lyric_genre = "funk"
                elif _lre.search(r'\b(the\s+clash|sex\s+pistols|bikini\s+kill|idles\b|pup\b|dead\s+kennedys)\b', _ci_l): _lyric_genre = "punk"
                elif _lre.search(r'\b(brian\s+eno|grouper\b|sigur\s+ros|william\s+basinski|stars\s+of\s+the\s+lid)\b', _ci_l): _lyric_genre = "ambient"
                elif _lre.search(r'\b(aretha\s+franklin|marvin\s+gaye|otis\s+redding|sam\s+cooke|al\s+green)\b', _ci_l): _lyric_genre = "soul"
                elif _lre.search(r'\b(mahalia\s+jackson|kirk\s+franklin|cece\s+winans|yolanda\s+adams)\b', _ci_l): _lyric_genre = "gospel"
                elif _lre.search(r'\b(miles\s+davis|john\s+coltrane|billie\s+holiday.*jazz|ella\s+fitzgerald|chet\s+baker)\b', _ci_l): _lyric_genre = "jazz"
                elif _lre.search(r'\b(bb\s+king|muddy\s+waters|robert\s+johnson|stevie\s+ray\s+vaughan|etta\s+james)\b', _ci_l): _lyric_genre = "blues"
                elif _lre.search(r'\b(metallica\b|black\s+sabbath|slayer\b|pantera\b|system\s+of\s+a\s+down)\b', _ci_l): _lyric_genre = "metal"
                elif _lre.search(r'\b(johnny\s+cash|dolly\s+parton|hank\s+williams|willie\s+nelson|loretta\s+lynn)\b', _ci_l): _lyric_genre = "country"
                elif _lre.search(r'\b(bob\s+dylan|joni\s+mitchell|simon\s+and\s+garfunkel|leonard\s+cohen|nick\s+cave)\b', _ci_l): _lyric_genre = "folk"
                elif _lre.search(r'\b(bts\b|blackpink\b|twice\b|exo\b|stray\s+kids|aespa\b|newjeans\b|ive\b|lesserafim\b)\b', _ci_l): _lyric_genre = "kpop"
                elif _lre.search(r'\b(kenshi\s+yonezu|yoasobi\b|ado\b|fujii\s+kaze|official\s+hige\s+dandism)\b', _ci_l): _lyric_genre = "jpop"
                elif _lre.search(r'\b(frank\s+sinatra|dean\s+martin|tony\s+bennett|michael\s+buble|norah\s+jones)\b', _ci_l): _lyric_genre = "jazz"
                elif _lre.search(r'\b(mozart\b|beethoven\b|bach\b|chopin\b|debussy\b|satie\b)\b', _ci_l): _lyric_genre = "classical"
                elif _lre.search(r'\b(maria\s+callas|pavarotti\b|andrea\s+bocelli\b|puccini\b|verdi\b)\b', _ci_l): _lyric_genre = "opera"
                elif _lre.search(r'\b(beyonce\b|rihanna\b|sza\b|frank\s+ocean|the\s+weeknd|h\.e\.r\b|doja\s+cat)\b', _ci_l): _lyric_genre = "rnb"

                # ── Genre keyword detection (fallback if no artist matched) ──────
                if _lyric_genre == "pop":  # only run if artist detection didn't fire
                    if _lre.search(r'\b(blues|delta|chicago\s+blues)\b', _ci_l): _lyric_genre = "blues"
                    elif _lre.search(r'\b(soul|motown|northern\s+soul)\b', _ci_l): _lyric_genre = "soul"
                    elif _lre.search(r'\b(gospel|church|spiritual|hymn)\b', _ci_l): _lyric_genre = "gospel"
                    elif _lre.search(r'\b(r&b|rnb|neo.?soul)\b', _ci_l): _lyric_genre = "rnb"
                    elif _lre.search(r'\b(folk|americana|bluegrass|campfire)\b', _ci_l): _lyric_genre = "folk"
                    elif _lre.search(r'\b(country|nashville|honky.tonk)\b', _ci_l): _lyric_genre = "country"
                    elif _lre.search(r'\b(opera|operatic|aria|soprano)\b', _ci_l): _lyric_genre = "opera"
                    elif _lre.search(r'\b(classical|orchestral|art\s+song|lieder)\b', _ci_l): _lyric_genre = "classical"
                    elif _lre.search(r'\b(cabaret|torch|weimar|burlesque|musical\s+theatre)\b', _ci_l): _lyric_genre = "cabaret"
                    elif _lre.search(r'\b(diss\s+track|diss\w*|beef\b|calling\s+\w+(\s+\w+)?\s+out|calling\s+out|calls?\s+\w+(\s+\w+)?\s+out|bars\s+at\s+(him|her)|shots\s+at\s+(him|her|them))\b', _ci_l): _lyric_genre = "diss"
                    elif _lre.search(r'\b(metal|heavy\s+metal|screamo|death\s+metal|hardcore)\b', _ci_l): _lyric_genre = "metal"
                    elif _lre.search(r'\b(punk|pop.?punk|riot\s+grrrl|oi\b)\b', _ci_l): _lyric_genre = "punk"
                    elif _lre.search(r'\b(drill|uk\s+drill|road\s+rap)\b', _ci_l): _lyric_genre = "drill"
                    elif _lre.search(r'\b(trap\b)\b', _ci_l): _lyric_genre = "trap"
                    elif _lre.search(r'\b(hip.?hop|rap\b|bars|freestyle|mc\b)\b', _ci_l): _lyric_genre = "rap"
                    elif _lre.search(r'\b(grime|grime\s+mc)\b', _ci_l): _lyric_genre = "grime"
                    elif _lre.search(r'\b(dubstep|bass\s+music|wub|dnb|drum\s+and\s+bass|jungle\s+music)\b', _ci_l): _lyric_genre = "dubstep"
                    elif _lre.search(r'\b(synthwave|retrowave|outrun|80s\s+synth|vaporwave)\b', _ci_l): _lyric_genre = "synthwave"
                    elif _lre.search(r'\b(ambient|atmospheric|drone|soundscape|experimental)\b', _ci_l): _lyric_genre = "ambient"
                    elif _lre.search(r'\b(funk|funky)\b', _ci_l): _lyric_genre = "funk"
                    elif _lre.search(r'\b(disco|dancefloor|mirror\s+ball)\b', _ci_l): _lyric_genre = "disco"
                    elif _lre.search(r'\b(house|techno|edm|rave|trance|electronic)\b', _ci_l): _lyric_genre = "dance"
                    elif _lre.search(r'\b(afrobeats|afropop|afro\s+swing|naija)\b', _ci_l): _lyric_genre = "afrobeats"
                    elif _lre.search(r'\b(dancehall|reggaeton|bashment)\b', _ci_l): _lyric_genre = "dancehall"
                    elif _lre.search(r'\b(reggae|ska|rocksteady)\b', _ci_l): _lyric_genre = "reggae"
                    elif _lre.search(r'\b(k.?pop|kpop|korean\s+pop|idol\s+group)\b', _ci_l): _lyric_genre = "kpop"
                    elif _lre.search(r'\b(city\s+pop|citypop)\b', _ci_l): _lyric_genre = "city pop"
                    elif _lre.search(r'\b(j.?pop|japanese\s+pop)\b', _ci_l): _lyric_genre = "jpop"
                    elif _lre.search(r'\b(lo.?fi|bedroom\s+pop|tape\s+hiss|2am\s+beat)\b', _ci_l): _lyric_genre = "lofi"
                    elif _lre.search(r'\b(indie|indie\s+rock|shoegaze|slowcore)\b', _ci_l): _lyric_genre = "indie"
                    elif _lre.search(r'\b(bollywood|filmi|ghazal|bhangra)\b', _ci_l): _lyric_genre = "bollywood"
                    elif _lre.search(r'\b(flamenco|cante\s+jondo)\b', _ci_l): _lyric_genre = "flamenco"
                    elif _lre.search(r'\b(bossa|samba|MPB)\b', _ci_l): _lyric_genre = "bossa"
                    elif _lre.search(r'\b(rock|grunge|alt.?rock)\b', _ci_l): _lyric_genre = "rock"
                    elif _lre.search(r'\b(jazz|bebop|cool\s+jazz|big\s+band|swing)\b', _ci_l): _lyric_genre = "jazz"
                _mg_l = music_genre.lower()
                if "blues" in _mg_l: _lyric_genre = "blues"
                elif "jazz" in _mg_l: _lyric_genre = "jazz"
                elif "soul" in _mg_l or "motown" in _mg_l: _lyric_genre = "soul"
                elif "gospel" in _mg_l: _lyric_genre = "gospel"
                elif "r&b" in _mg_l or "rnb" in _mg_l or "neo-soul" in _mg_l: _lyric_genre = "rnb"
                elif "folk" in _mg_l or "americana" in _mg_l: _lyric_genre = "folk"
                elif "country" in _mg_l: _lyric_genre = "country"
                elif "opera" in _mg_l: _lyric_genre = "opera"
                elif "classical" in _mg_l or "orchestral" in _mg_l: _lyric_genre = "classical"
                elif "cabaret" in _mg_l or "musical theatre" in _mg_l or "burlesque" in _mg_l: _lyric_genre = "cabaret"
                elif "metal" in _mg_l or "heavy metal" in _mg_l: _lyric_genre = "metal"
                elif "punk" in _mg_l or "pop-punk" in _mg_l: _lyric_genre = "punk"
                elif "drill" in _mg_l or "uk drill" in _mg_l: _lyric_genre = "drill"
                elif "trap" in _mg_l: _lyric_genre = "trap"
                elif "hip-hop" in _mg_l or "rap" in _mg_l: _lyric_genre = "rap"
                elif "grime" in _mg_l: _lyric_genre = "grime"
                elif "drum and bass" in _mg_l or "dnb" in _mg_l or "jungle" in _mg_l: _lyric_genre = "dubstep"
                elif "dubstep" in _mg_l or "bass music" in _mg_l: _lyric_genre = "dubstep"
                elif "synthwave" in _mg_l or "retrowave" in _mg_l or "vaporwave" in _mg_l: _lyric_genre = "synthwave"
                elif "ambient" in _mg_l or "atmospheric" in _mg_l: _lyric_genre = "ambient"
                elif "funk" in _mg_l: _lyric_genre = "funk"
                elif "disco" in _mg_l: _lyric_genre = "disco"
                elif "house" in _mg_l or "techno" in _mg_l or "trance" in _mg_l or "edm" in _mg_l or "electronic" in _mg_l: _lyric_genre = "dance"
                elif "afrobeats" in _mg_l or "afropop" in _mg_l: _lyric_genre = "afrobeats"
                elif "dancehall" in _mg_l or "reggaeton" in _mg_l: _lyric_genre = "dancehall"
                elif "reggae" in _mg_l or "ska" in _mg_l: _lyric_genre = "reggae"
                elif "k-pop" in _mg_l or "kpop" in _mg_l: _lyric_genre = "kpop"
                elif "city pop" in _mg_l or "citypop" in _mg_l: _lyric_genre = "city pop"
                elif "j-pop" in _mg_l or "jpop" in _mg_l: _lyric_genre = "jpop"
                elif "lo-fi" in _mg_l or "lofi" in _mg_l: _lyric_genre = "lofi"
                elif "indie" in _mg_l or "shoegaze" in _mg_l: _lyric_genre = "indie"
                elif "flamenco" in _mg_l: _lyric_genre = "flamenco"
                elif "bossa" in _mg_l or "samba" in _mg_l: _lyric_genre = "bossa"
                elif "bollywood" in _mg_l or "bhangra" in _mg_l: _lyric_genre = "bollywood"
                elif "rock" in _mg_l: _lyric_genre = "rock"

                _lyric_emotion = "neutral"
                if _lre.search(r'\b(sad|grief|heartbreak|loss|miss|lonely|cry|mourn)\b', _ci_l): _lyric_emotion = "grief"
                elif _lre.search(r'\b(angry|rage|defiant|fierce|fight|power|rise)\b', _ci_l): _lyric_emotion = "defiant"
                elif _lre.search(r'\b(love|longing|desire|want|need|yearning|aroused|horny)\b', _ci_l): _lyric_emotion = "longing"
                elif _lre.search(r'\b(joy|happy|celebrat|free|alive|light|hope|euphoric)\b', _ci_l): _lyric_emotion = "joyful"
                elif _lre.search(r'\b(lost|searching|wondering|drifting|where|why)\b', _ci_l): _lyric_emotion = "searching"
                # Read from emotional state dropdown too
                if emotional_state_sel and "none" not in emotional_state_sel.lower():
                    _esl = emotional_state_sel.lower()
                    if any(k in _esl for k in ("grief","crying","sad","longing","exhausted")): _lyric_emotion = "grief"
                    elif any(k in _esl for k in ("angry","defiant","fierce","predatory")): _lyric_emotion = "defiant"
                    elif any(k in _esl for k in ("aroused","tender","adoration")): _lyric_emotion = "longing"
                    elif any(k in _esl for k in ("euphoric","happy","playful","confident","proud")): _lyric_emotion = "joyful"
                    elif any(k in _esl for k in ("vulnerable","nervous","overcome","dissociated")): _lyric_emotion = "searching"

                if has_user_dialogue:
                    # Case A — user wrote specific lyrics in quotes
                    _quoted_str = ", ".join(f'"{l}"' for l in _user_quoted_lines)
                    _lyric_content_note = (
                        "LYRICS — MANDATORY: The user has provided specific lyrics in quotes. "
                        "She SINGS these exact words — do NOT treat as spoken dialogue. "
                        f"Required lyrics in order: {_quoted_str}. "
                        "Spread them across the four performance sections. "
                        "Do NOT skip or paraphrase any quoted line. "
                        "You may add wordless hums or short phrases between lines to fill the melody. "
                    )
                elif _lyric_topic:
                    # Case B — user told us what to sing about + gets full DNA
                    _lyric_content_note = (
                        f'LYRICS — TOPIC GIVEN: She sings about: "{_lyric_topic}". '
                        "Invent lyrics on this topic — short melodic phrases, written to be SUNG not said. "
                        "The topic should echo and develop across all four sections. "
                        "LYRIC CRAFT FOR THIS GENRE AND EMOTION — follow these structural rules: "
                    )
                else:
                    # Case C — full creative freedom, DNA fires below
                    _lyric_content_note = ""  # set properly after _LYRIC_DNA is defined

                    # ── Lyric DNA per genre ───────────────────────────────────────
                    # Structural craft rules drawn from public domain traditions.
                    # Not copied lyrics — the underlying patterns that make each genre
                    # sound authentic. The LLM uses these to invent original lines.
                    _LYRIC_DNA = {
                        "blues": (
                            "LYRIC DNA — BLUES (drawn from traditional Delta/Chicago craft): "
                            "Structure: opening line states the situation plainly. Second line repeats it with a small variation — "
                            "the repetition is not weakness, it is how blues builds tension. "
                            "Third line resolves, twists, or deepens it. "
                            "Language: concrete and specific. Name the place, the time, the thing. "
                            "Never abstract — 'the road' not 'my journey', 'three in the morning' not 'a late hour'. "
                            "Verbs are active and physical: woke, walked, left, found, lost, broke. "
                            "Imagery from the world: roads, trains, rivers, rooms, doors, hands. "
                            "The singer IS the lyric — first person, direct address to the absent one or to god or to no one. "
                            "Example pattern: 'I woke up this morning [specific detail]. "
                            "I woke up this morning, [variation]. "
                            "[Consequence or twist that earns the first two lines].' "
                            "Syllable feel: blues lyrics land on the beat, not between it. Heavy, deliberate. "
                        ),
                        "jazz": (
                            "LYRIC DNA — JAZZ (drawn from Great American Songbook craft tradition): "
                            "Structure: the lyric implies more than it says. Leave space. "
                            "A great jazz lyric is a fragment — it begins in the middle of a thought. "
                            "Language: sophisticated but not ornate. Simple words arranged unexpectedly. "
                            "Compression is the art: pack a complete emotional world into six words. "
                            "The lyric rhymes but not obviously — internal rhyme, near-rhyme, the rhyme arrives late. "
                            "Imagery: cities at night, seasons, the specific texture of light or sound or touch. "
                            "The feeling is adult and complex — not heartbreak but the specific quality of remembering it. "
                            "Not 'I love you' but 'the way you hold your knife' (Ellington tradition). "
                            "Example register: 'Some other spring / I'll try to love / now I still cling / to faded blossoms.' "
                            "Syllable feel: jazz lyrics breathe between the beats, float over the rhythm. "
                        ),
                        "soul": (
                            "LYRIC DNA — SOUL/MOTOWN (drawn from 1960s Detroit and Stax craft tradition): "
                            "Structure: the hook is a physical sensation, not a metaphor. "
                            "State it in the first four words. Repeat it. Repeat it again with more. "
                            "Call and response: the lead line asks, the backing answers, even when there's no backing. "
                            "Language: direct, warm, embodied. The body is always present — hands, chest, eyes, knees. "
                            "The emotion is BIG but specific. Not 'I feel sad' but 'these arms of mine, they are lonely.' "
                            "Testify: soul lyrics make a declaration and then prove it with the next line. "
                            "Example patterns: 'I got you / I feel good / I knew that I would / now.' "
                            "Or: '[Title hook] / [repeat with variation] / [physical description of the feeling].' "
                            "Syllable feel: soul lyrics sit on the beat and push through it, forward momentum always. "
                        ),
                        "gospel": (
                            "LYRIC DNA — GOSPEL (drawn from African American church tradition): "
                            "Structure: call and response, even solo. The singer calls to god, the melody answers. "
                            "Repetition is not weakness — it is accumulation of spiritual weight. "
                            "Build toward a moment: the lyric climbs and does not come back down. "
                            "Language: biblical register but not archaic. Direct address: 'Lord', 'Father', 'my God'. "
                            "The body is the instrument of worship — the lyric describes what the body feels in the spirit. "
                            "Testimony structure: 'I was lost / now I am found / and this is how I know it.' "
                            "The payoff word goes on the highest note — plan the lyric around the vocal peak. "
                            "Example register: 'This little light of mine' / 'Amazing grace' / 'I shall overcome'. "
                            "Syllable feel: gospel lyrics are sung long — vowels stretched, consonants landed hard. "
                        ),
                        "rnb": (
                            "LYRIC DNA — R&B/NEO-SOUL (contemporary craft): "
                            "Structure: verse builds, pre-chorus opens, chorus delivers — even in a 15-second window "
                            "the lyric should feel like it has a destination it's moving toward. "
                            "Language: conversational, intimate, present tense. The singer is talking to one person. "
                            "Specificity over abstraction: name the moment, the detail, the exact thing. "
                            "Contemporary R&B runs syllables together, bends words, "
                            "treats the melody as a second language for the lyric. "
                            "Imagery: night, warmth, skin, light, the specific hour. "
                            "The hook is four to six syllables maximum — it must be singable in one breath. "
                            "Example register: 'Stay with me' / 'I need you here' / 'don't leave me now'. "
                            "Syllable feel: R&B lyrics float and snap — loose on the verse, tight on the hook. "
                        ),
                        "folk": (
                            "LYRIC DNA — FOLK/AMERICANA (drawn from traditional ballad craft): "
                            "Structure: narrative. Something happened. Someone went somewhere. Something was lost or found. "
                            "The lyric tells a story with a beginning, a middle, a turn. "
                            "Language: plain and unadorned. The power is in the specific noun, not the adjective. "
                            "Name the place: a real town, a river, a road. Name the person by their relationship. "
                            "Avoid metaphor where a literal image serves — folk trusts the concrete. "
                            "The singer is often an observer, sometimes the subject — point of view matters. "
                            "Tradition: the ballad uses repetition as a device — a line comes back changed by what happened. "
                            "Example patterns: 'Down in the valley / the valley so low / hang your head over / hear the wind blow.' "
                            "Or a single declarative: 'There is a house in New Orleans / they call the Rising Sun.' "
                            "Syllable feel: folk lyrics follow speech rhythm — they scan like spoken sentences. "
                        ),
                        "country": (
                            "LYRIC DNA — COUNTRY (Nashville craft tradition): "
                            "Structure: verse tells a story, chorus delivers the emotional truth of that story, "
                            "bridge reveals what the narrator finally understands. "
                            "Language: conversational, working-class, specific. "
                            "Trucks, small towns, dirt roads, Friday nights, honky-tonks, front porches. "
                            "The hook is a turnable phrase — something that sounds like common speech but lands differently. "
                            "Country rhymes cleanly and obviously — the rhyme is part of the pleasure. "
                            "Emotional range: heartbreak, pride, loss, celebration — always earned, never ironic. "
                            "Example register: 'He stopped loving her today' / 'Always on my mind' / 'Stand by your man'. "
                            "The best country lyric tells you exactly what happened and why it matters. "
                            "Syllable feel: country lyrics breathe naturally, they scan like how people actually talk. "
                        ),
                        "opera": (
                            "LYRIC DNA — OPERA (drawn from Italian/German/French operatic tradition): "
                            "Structure: the aria makes one emotional declaration and explores it completely. "
                            "It does not tell a story — it inhabits a single moment of feeling. "
                            "Language: heightened, declarative, fate-weighted. "
                            "Opera lyrics state the absolute: 'I will love you until I die.' "
                            "'There is no world without you.' 'I am already dead.' "
                            "Long open vowels on the high notes — plan the lyric around the voice's peak. "
                            "A, E, O on sustained notes. Consonants only where they propel. "
                            "Italian tradition: the cabaletta ends with a held high note on an open vowel — "
                            "design the lyric so the peak word ends in a long A, O, or I. "
                            "Example register: 'Nessun dorma' / 'La ci darem la mano' / 'Caro mio ben'. "
                            "The lyric should feel like it was written for the voice, not the other way around. "
                        ),
                        "classical": (
                            "LYRIC DNA — ART SONG/LIEDER (Schubert/Schumann/Wolf tradition): "
                            "Structure: the lyric is a poem set to music — it has the density and compression of poetry. "
                            "Every word earns its place. No filler syllables. "
                            "Language: nature imagery — seasons, weather, light, water — as emotional mirrors. "
                            "The outer world reflects the inner world. "
                            "Schubert tradition: the wanderer, the journey, the winter, the unreachable beloved. "
                            "Wolf tradition: compressed dramatic scenes — a complete emotional story in eight lines. "
                            "Example register: 'Der Leiermann' / 'Gretchen am Spinnrade' / 'Im Fruhling'. "
                            "The art song lyric rewards the slow syllable — each word is sung, not rushed. "
                        ),
                        "cabaret": (
                            "LYRIC DNA — CABARET/TORCH SONG (Weimar/Broadway tradition): "
                            "Structure: direct address to the audience or to the absent beloved. "
                            "The singer knows you are watching. She uses that. "
                            "Language: worldly, knowing, a little worn. "
                            "Cabaret lyrics have seen things. They are not innocent about love or death or money. "
                            "Wit coexists with grief — the best torch song is funny and devastating in the same phrase. "
                            "Brecht/Weill tradition: the lyric is about more than the singer — it is about the world. "
                            "Coward tradition: the lyric is impossibly civilised about something uncivilised. "
                            "Example register: 'Life is a cabaret' / 'Mack the Knife' / 'Mad about the boy'. "
                            "Syllable feel: cabaret lyrics are spoken as much as sung — the word matters, the music serves it. "
                        ),
                        "rock": (
                            "LYRIC DNA — ROCK (from 1950s to contemporary tradition): "
                            "Structure: the hook is the title and it hits in the first eight bars. "
                            "The verse builds tension. The chorus releases it. "
                            "Language: urgent, physical, present tense. The singer is inside the moment. "
                            "Rock lyrics use repetition as amplification — the hook gets louder each time it returns. "
                            "Great rock lyrics are ambiguous enough to be about anything — "
                            "the best ones mean one thing personally and another universally. "
                            "Imagery: light and dark, fire, the road, the night, the city, the body. "
                            "Example register: 'Born to run' / 'I can't get no satisfaction' / 'Like a rolling stone'. "
                            "The peak lyric goes on the highest note — the word that earns the belt. "
                            "Syllable feel: rock lyrics are punched — consonants hard, vowels driven through. "
                        ),
                        "hiphop": (
                            "LYRIC DNA — HIP-HOP (from golden era to contemporary): "
                            "Structure: the bar is the unit. Each bar completes a thought or sets up the next. "
                            "Rhyme scheme: AABB or ABAB or internal rhyme — the rhyme is heard, not just read. "
                            "Language: direct, specific, vernacular. The lyric sounds like speech but it scans. "
                            "Syllable density: hip-hop packs more syllables per beat than any other form. "
                            "Every syllable serves — no filler. "
                            "The punch line lands at the end of the bar — set it up for three bars, deliver on four. "
                            "Imagery: the street-level specific — what you can see, touch, hear in this exact place. "
                            "Example register: 'It was all a dream' / 'Ready to die' / 'Jesus walks'. "
                            "Syllable feel: hip-hop lyrics ride over the beat or push against it — never just sit on it. "
                        ),
                        "dance": (
                            "LYRIC DNA — DANCE/ELECTRONIC (club and pop tradition): "
                            "Structure: the hook is everything. It must work repeated twenty times. "
                            "The verse is runway — it exists only to make the drop land harder. "
                            "Language: simple, physical, imperative. Tell the listener what to do or feel. "
                            "Dance lyrics use the body as subject: move, feel, let go, rise, fall, burn. "
                            "The hook must be singable in a crowded room at 128bpm — four to eight syllables maximum. "
                            "Vowels dominate — open sounds carry over a bass drop. "
                            "Example register: 'I feel love' / 'Don't stop the music' / 'Around the world'. "
                            "Syllable feel: dance lyrics are percussive — they sound like the kick drum feels. "
                        ),
                        "reggae": (
                            "LYRIC DNA — REGGAE (Jamaican roots tradition): "
                            "Structure: verse preaches, chorus unifies. The lyric has a message. "
                            "Language: spiritual, political, earthy. Rastafari vocabulary where authentic. "
                            "The lyric addresses a situation — poverty, love, freedom, resistance — directly. "
                            "Patois rhythm: the stress falls differently than in standard English, "
                            "giving reggae lyrics their characteristic lilt. "
                            "Imagery: the sun, the river, Zion, Babylon, the road home. "
                            "Example register: 'No woman no cry' / 'Redemption song' / 'Rivers of Babylon'. "
                            "Syllable feel: reggae lyrics fall on the off-beat — they land between, not on, the pulse. "
                        ),
                        "flamenco": (
                            "LYRIC DNA — FLAMENCO (Andalusian copla tradition): "
                            "Structure: the copla — four to six lines, each complete, each devastating. "
                            "No narrative arc — flamenco lyrics are pure distilled feeling. "
                            "Language: Spanish but not literary — Andalusian vernacular, compressed, raw. "
                            "Subject matter: death, love, mother, the road, god, the body. Always one of these. "
                            "The best copla is a paradox: 'I searched for my death and could not find it.' "
                            "Federico García Lorca called flamenco lyrics 'the deepest song' — "
                            "they go directly to the oldest grief. "
                            "Example register: 'Ay mi madre' / 'Tengo una pena' / 'Me voy a morir de amor'. "
                            "Syllable feel: flamenco syllables are held, bent, broken — the voice splits single vowels into many. "
                        ),
                        "bossa": (
                            "LYRIC DNA — BOSSA NOVA (João Gilberto / Tom Jobim tradition): "
                            "Structure: the lyric breathes with the guitar. It does not compete. "
                            "Language: Portuguese — soft, open vowels, the sentences run into each other like water. "
                            "Subject: longing for something gentle. Not dramatic grief — a soft ache. "
                            "The sea, the afternoon, the girl from Ipanema, the city at dusk. "
                            "Bossa lyrics understate — they say less than they mean, let the music carry the rest. "
                            "Example register: 'The girl from Ipanema' / 'Corcovado' / 'Wave'. "
                            "Syllable feel: bossa lyrics float — they barely touch the beat, they skim it. "
                        ),
                        "bollywood": (
                            "LYRIC DNA — FILMI/BOLLYWOOD (Hindi film song tradition): "
                            "Structure: mukhda (refrain) states the theme. Antara (verse) develops it. "
                            "The mukhda returns — its meaning deepened by the verse. "
                            "Language: Urdu/Hindi register — poetic, elevated, drawing on ghazal tradition. "
                            "Metaphor is mandatory: the beloved is the moon, the eyes are the sea, "
                            "love is a flower that blooms and dies. "
                            "The lyric is allowed to be grand — Bollywood lyrics make large claims unironically. "
                            "Example register: 'Lag ja gale' / 'Pyaar hua ikrar hua' / 'Ek pyaar ka nagma hai'. "
                            "Syllable feel: filmi lyrics ornament — grace notes of syllables around the main melody. "
                        ),
                        "pop": (
                            "LYRIC DNA — POP (contemporary craft): "
                            "Structure: verse, pre-chorus, chorus. The chorus is the emotional truth. "
                            "The verse earns the chorus — it sets up the feeling, the chorus names it. "
                            "Language: conversational, direct, universally relatable but emotionally specific. "
                            "The hook must be immediately memorable — singable by someone who hears it once. "
                            "Use common words in uncommon combinations. "
                            "The best pop lyrics feel like the listener already knew them — "
                            "they articulate something the listener felt but couldn't say. "
                            "Syllable feel: pop lyrics are smooth — they ride the melody rather than push against it. "
                        ),
                    }

                    # ── Emotional colouring for lyrics ────────────────────────────
                    _LYRIC_EMOTION_NOTES = {
                        "grief": (
                            "EMOTIONAL COLOUR — GRIEF: The lyrics circle the loss without naming it directly. "
                            "What is absent is more present than what is there. "
                            "The physical world — a room, an object, a sound — stands in for the person who is gone. "
                            "The lyric does not say 'I miss you' — it describes the specific weight of the empty chair."
                        ),
                        "defiant": (
                            "EMOTIONAL COLOUR — DEFIANT: The lyrics make a declaration that costs something. "
                            "They do not ask — they state. The strength in the lyric comes from what it refuses. "
                            "The singer is telling the world, or one person in it, exactly where she stands."
                        ),
                        "longing": (
                            "EMOTIONAL COLOUR — LONGING: The lyrics inhabit the space between wanting and having. "
                            "The beloved is present in absence — described through what they do, how they move, "
                            "the specific detail that the singer can't stop thinking about. "
                            "The lyric aches without resolution."
                        ),
                        "joyful": (
                            "EMOTIONAL COLOUR — JOYFUL: The lyrics celebrate something specific, not everything. "
                            "Joy in a lyric is most powerful when it's precise — this moment, this person, this feeling. "
                            "The body is present in joy: the feet move, the chest opens, the breath comes easy."
                        ),
                        "searching": (
                            "EMOTIONAL COLOUR — SEARCHING: The lyrics ask without expecting an answer. "
                            "They circle a question — about love, about self, about where the singer is going. "
                            "The unresolved quality is the point — the lyric ends without landing."
                        ),
                        "neutral": (
                            "EMOTIONAL COLOUR: Let the genre and scene determine the emotional register. "
                            "The lyric should feel like it belongs to this specific character in this specific world — "
                            "not a generic emotion but the one most true to what is happening in the scene."
                        ),
                    }

                    _dna = _LYRIC_DNA.get(_lyric_genre, _LYRIC_DNA["pop"])
                    _emotion_colour = _LYRIC_EMOTION_NOTES.get(_lyric_emotion, _LYRIC_EMOTION_NOTES["neutral"])

                    # ── Phrase bank injection ─────────────────────────────────
                    # Pick a register appropriate to genre + emotion, then seed-pick
                    # one anchor phrase per arc position. The LLM wraps melody and
                    # connective tissue around these — they define the tone.
                    try:
                        from lyric_phrase_bank import (
                            _LYRIC_PHRASE_BANK,
                            _LYRIC_REGISTER_BY_GENRE,
                            _LYRIC_REGISTER_BY_EMOTION,
                            _EXPLICIT_REGISTERS,
                            _CONTENT_REGISTER_HINTS,
                            _BRIDGE_POOL,
                            _DUET_PAIRS,
                            _BREATH_ANCHOR_POOL,
                            _REGISTER_TEMPO_HINTS,
                        )
                        import random as _pb_rnd
                        _pb_rng = _pb_rnd.Random((seed + 31) if seed != -1 else None)

                        # Choose register: emotion override > explicit > filthy singer > content hints > genre default

                        _emo_regs = _LYRIC_REGISTER_BY_EMOTION.get(_lyric_emotion)

                        # Filthy singer detection — catches all natural ways a user says this.
                        # "filthy mouth", "dirty mouth", "talks dirty", "she swears", "potty mouth",
                        # "foul mouth", "explicit lyrics", "raw lyrics", "dirty singer" etc.
                        _is_filthy_singer = bool(re.search(
                            r'\b(filth\w*\s+mouth|dirty\s+mouth|foul\s+mouth\w*|potty\s+mouth|'
                            r'talk\w*\s+dirt\w*|dirt\w*\s+talk\w*|sing\w*\s+dirt\w*|dirt\w*\s+lyric\w*|'
                            r'explicit\s+lyric\w*|raw\s+lyric\w*|raunchy\s+lyric\w*|'
                            r'filth\w*\s+lyric\w*|nasty\s+lyric\w*|'
                            r'she\s+swears?\s+(when\s+she\s+sings?|a\s+lot|constantly|throughout)|'
                            r'swears?\s+(?:a\s+lot|constantly|throughout|in\s+her\s+song\w*)|'
                            r'foul.?mouthed|dirty.?mouthed|potty.?mouthed|filthy.?mouthed|'
                            r'raunchy\s+sing\w*|explicit\s+sing\w*|dirty\s+sing\w*|nasty\s+sing\w*|'
                            r'wap\s+style|cardi\s*b\s+style|megan\s+thee|city\s+girls?\s+style|'
                            r'like\s+(wap|cardi|megan)|'
                            r'profan\w*\s+(in\s+her\s+)?(?:song\w*|lyric\w*|singing)|'
                            r'her\s+(mouth|lyrics?|singing)\s+(?:is\s+)?(?:filthy|dirty|nasty|foul|raunchy|explicit))\b',
                            _combined_input, re.IGNORECASE
                        )) or _wants_swear_injection  # swear injection flag also marks filthy singer

                        # Content hint: scan combined input for keywords (full-word regex, not substring)
                        _content_hint_regs = None
                        for _hint_kw, _hint_regs in _CONTENT_REGISTER_HINTS.items():
                            if re.search(r'\b' + re.escape(_hint_kw) + r'\w*\b', _ci_l):
                                _content_hint_regs = _hint_regs
                                break

                        if _is_filthy_singer:
                            _reg_pool = ["raw_filth", "dirty_talk", "direct_explicit", "dominant", "empowered"]
                        elif is_explicit and not _emo_regs:
                            _reg_pool = _EXPLICIT_REGISTERS
                        elif _emo_regs:
                            _reg_pool = _emo_regs
                        elif _content_hint_regs:
                            _reg_pool = _content_hint_regs
                        else:
                            _reg_pool = _LYRIC_REGISTER_BY_GENRE.get(_lyric_genre, ["slow_burn", "empowered", "tender"])

                        _chosen_register = _pb_rng.choice(_reg_pool)
                        _reg_phrases = _LYRIC_PHRASE_BANK.get(_chosen_register, {})

                        def _unpack_phrase(entry):
                            """Handle plain string or (lyric, delivery_note) tuple."""
                            if isinstance(entry, tuple) and len(entry) >= 2:
                                return entry[0], entry[1]
                            return str(entry), ""

                        _raw_open  = _pb_rng.choice(_reg_phrases.get("opening",   ["I want to feel this"]))
                        _raw_build = _pb_rng.choice(_reg_phrases.get("build",     ["Take me further"]))
                        _raw_peak  = _pb_rng.choice(_reg_phrases.get("peak",      ["Everything arrived at once"]))
                        _raw_res   = _pb_rng.choice(_reg_phrases.get("resolution",["I'll carry this forward"]))

                        _p_open,  _d_open  = _unpack_phrase(_raw_open)
                        _p_build, _d_build = _unpack_phrase(_raw_build)
                        _p_peak,  _d_peak  = _unpack_phrase(_raw_peak)
                        _p_res,   _d_res   = _unpack_phrase(_raw_res)

                        def _fmt_phrase_with_note(label, lyric, note):
                            if note:
                                return f'\n{label}: "{lyric}" — [{note}]'
                            return f'\n{label}: "{lyric}"'

                        # ── Ad-lib injection ─────────────────────────────
                        # Sample contextually appropriate interstitial vocalisations
                        # from the ad-lib pools — the sounds between phrases that make
                        # a performance feel alive. 1 per arc position max.
                        _adlib_str = ""
                        try:
                            from lyric_phrase_bank import _ADLIB_POOLS, _ADLIB_CONTEXT_MAP
                            _al_rng = random.Random((seed + 77) if seed != -1 else None)
                            # Get pools for this register
                            _al_pool_keys = _ADLIB_CONTEXT_MAP.get(_chosen_register,
                                            _ADLIB_CONTEXT_MAP.get(_lyric_genre, ["universal"]))
                            # Build combined pool, deduplicated
                            _al_combined = []
                            for _pk in _al_pool_keys:
                                _al_combined += _ADLIB_POOLS.get(_pk, [])
                            if not _al_combined:
                                _al_combined = _ADLIB_POOLS["universal"]
                            # Pick 4 unique ad-libs, one per arc position
                            _al_sample = _al_rng.sample(_al_combined, min(4, len(_al_combined)))
                            _al_open  = _al_sample[0][0] if len(_al_sample) > 0 else ""
                            _al_build = _al_sample[1][0] if len(_al_sample) > 1 else ""
                            _al_peak  = _al_sample[2][0] if len(_al_sample) > 2 else ""
                            _al_res   = _al_sample[3][0] if len(_al_sample) > 3 else ""
                            _adlib_str = (
                                f"\n\nAD-LIBS & INTERSTITIAL VOCALISATIONS — inject these BETWEEN the anchor phrases, "
                                f"not replacing them. These are the small sounds/words that fill the gaps and make the "
                                f"performance feel natural and alive. Place each one at its arc position: "
                                f"\nOPENING gap: \"{_al_open}\" "
                                f"\nBUILD gap: \"{_al_build}\" "
                                f"\nPEAK gap: \"{_al_peak}\" "
                                f"\nRESOLVE gap: \"{_al_res}\" "
                                f"\nThese arrive BETWEEN phrases — in breaths, pauses, transitions. "
                                f"They are not the main lyric, they are the texture around it."
                            )
                        except Exception:
                            _adlib_str = ""

                        _phrase_injection = (
                            f"\n\nLYRIC ANCHOR PHRASES — USE THESE VERBATIM OR RIFF CLOSELY: "
                            f"These four phrases are the emotional spine. "
                            f"The delivery note in brackets describes voice, body, and physical performance — use it to write the surrounding prose. "
                            + _fmt_phrase_with_note("OPENING", _p_open, _d_open)
                            + _fmt_phrase_with_note("BUILD",   _p_build, _d_build)
                            + _fmt_phrase_with_note("PEAK",    _p_peak,  _d_peak)
                            + _fmt_phrase_with_note("RESOLVE", _p_res,   _d_res)
                            + f"\nRegister: {_chosen_register.replace('_', ' ').upper()}"
                            + _adlib_str
                        )
                        print(f"[LTX2-Qwen] Lyric register: {_chosen_register} | opening: {_p_open[:40]}")
                    except ImportError:
                        _phrase_injection = ""
                        print("[LTX2-Qwen] Lyric phrase bank not found — using DNA only")

                    _lyric_content_note = (
                        "LYRICS — INVENT (genre-authentic, not generic filler): "
                        "Write lyrics that sound like they were written by someone who knows this genre from the inside. "
                        "Each lyric phrase should be 4-10 words. Real lines. Singable. Specific. "
                        "At minimum: one lyric in opening, one in build, one at peak (the best line), one in resolution. "
                        "The peak lyric is the emotional centre — it earns everything before it and colours everything after. "
                        + _dna
                        + _emotion_colour
                        + _phrase_injection
                        + "VOLUME IN PERFORMANCE: Metal, screamo, punk, diss, argument scenes — some lines SHOUTED. Write shouted lines in CAPS. Quiet lines lowercase. Never flat volume — contrast IS the energy. " 
                        + "\n\nABSOLUTELY FORBIDDEN: 'never let go', 'hold on tight', 'fading out', 'here with you', "
                        "'in the darkness', 'lost without you', 'always and forever', 'feel the rhythm'. "
                        "These are the lyrics of a placeholder. Write something true to THIS genre and THIS moment."
                    )
                    print(f"[LTX2-Qwen] Lyric DNA: {_lyric_genre} / emotion: {_lyric_emotion}")

                # ── Explicit-singing merge note ───────────────────────────
                # When explicit content and singing overlap, the two systems
                # must be woven — not sequential. The performance continues
                # through the physical action, not before or after it.
                _explicit_singing_merge = ""
                if (is_explicit or is_sensual) and has_undressing:
                    # Garment list injected directly so model knows what to remove
                    # without needing to connect to the explicit_instruction block
                    _garment_ref = (f"Garments to remove: {garment_list}. " if garment_list else "")
                    _explicit_singing_merge = (
                        "PERFORMANCE-UNDRESSING MERGE — MANDATORY: "
                        + _garment_ref +
                        "The undressing is WOVEN INTO the singing arc — same sentences, not separate. "
                        "SECTION 1 — OPENING: she is still clothed, voice enters. "
                        "SECTION 2 — BUILD: hands begin the removal, fabric moves as the voice rises. "
                        "SECTION 3 — PEAK: the reveal lands exactly on the peak note — "
                        "the moment of exposure and the musical climax are the same beat. "
                        "SECTION 4 — RESOLUTION: the exposed state is simply present, "
                        "camera holds, voice settles. Do not reverse or cover. "
                        "Every sentence in sections 2-4 contains BOTH voice and body. "
                        "Example: 'She grips the hem as her voice climbs, "
                        "the fabric lifting past her ribs as she holds the note, "
                        "her chest bare and heaving as the peak breaks.' "
                    )
                elif is_explicit or is_sensual:
                    _explicit_singing_merge = (
                        "PERFORMANCE-BODY MERGE: The physical content and the vocal performance "
                        "occupy the same sentences throughout. Body and voice together always. "
                    )

                # Filthy singer swear injection — wire swear pool into singing arc
                _singing_swear_note = ""
                if _is_filthy_singer or _wants_swear_injection:
                    try:
                        from lyric_phrase_bank import _SWEAR_INJECTOR
                        _ssw_pool = _SWEAR_INJECTOR.get("sex", []) + _SWEAR_INJECTOR.get("general", [])
                        _ssw_rng  = random.Random((seed + 41) if seed != -1 else None)
                        _ssw_b = _ssw_rng.choice([l for l in _ssw_pool if l[2] in ("build","soft","slow","neutral")]) if _ssw_pool else None
                        _ssw_p = _ssw_rng.choice([l for l in _ssw_pool if l[2] in ("peak","sharp","neutral")]) if _ssw_pool else None
                        _ssw_r = _ssw_rng.choice([l for l in _ssw_pool if l[2] in ("resolve","soft","slow","neutral")]) if _ssw_pool else None
                        def _ssw_fmt(e): return f'"{e[0]}" ({e[1]})' if e else ""
                        _singing_swear_note = (
                            "\n\nFILTHY MOUTH — MANDATORY: This singer has a dirty mouth. "
                            "Weave explicit/profane fragments BETWEEN the lyric phrases — not replacing them, bridging them. "
                            "The swearing is involuntary, physical, breathed out mid-performance. "
                            f"\nAfter OPENING phrase use: {_ssw_fmt(_ssw_b)} — said into the mic, under the breath. "
                            f"\nAfter BUILD phrase use: {_ssw_fmt(_ssw_p)} — louder, less controlled. "
                            f"\nAfter PEAK use: {_ssw_fmt(_ssw_r)} — exhaled into the silence after the note. "
                            "\nThe lyric anchors are the SONG. The swears are what leaks out between the song. "
                            "FORBIDDEN: clean polite performance. FORBIDDEN: skipping the swear fragments."
                        )
                        print(f"[LTX2-Qwen] Filthy singer: swear injection active | register={_chosen_register}")
                    except ImportError:
                        pass

                # Filthy register note — tells LLM the lyric tone is explicit
                _filthy_register_note = (
                    "\n\nLYRIC REGISTER — EXPLICIT/RAW: These lyrics are in the tradition of WAP, Anaconda, "
                    "Throat Baby — unapologetic, anatomical, physical. The words are direct. "
                    "She sings about what she wants and means every word. No euphemism unless it's deliberate. "
                ) if _is_filthy_singer else ""

                dialogue_instruction = (
                    "\n\n[SINGING PERFORMANCE INSTRUCTION — MANDATORY: "
                    "The character is singing. This is a vocal performance, NOT dialogue. "
                    "The voice is CONTINUOUS throughout the entire scene. "
                    "Do NOT write two quoted words and return to action description. "
                    "Every sentence should interweave voice, melody, and physical performance simultaneously. "
                    + _lang_note
                    + _filthy_register_note
                    + _explicit_singing_merge
                    + "\n\n" + _lyric_content_note
                    + "\n\nPERFORMANCE STRUCTURE — four sections across the clip:"
                    "\n(1) OPENING — voice enters: texture, weight, register. First lyric phrase. Physical stance."
                    "\n(2) BUILD — melody develops, emotional intensity rises. "
                    "Second lyric phrase, longer than the first. "
                    "Physical performance responds: breath deepens, posture opens or tightens."
                    "\n(3) PEAK — emotional or musical high point. "
                    "Belt, held note, vocal run, or dynamic drop. Name it physically: "
                    "\'her voice cracks on the upper register\', "
                    "\'she sustains the vowel until the breath gives out\', "
                    "\'she drops to barely a whisper and the note still fills the space\'. "
                    "Body responds: eyes close, hands move, chest rises."
                    "\n(4) RESOLUTION — last phrase, voice settles, performance lands."
                    "\n\nVOICE VOCABULARY — use these, never repeat the same word twice: "
                    "breathy, raw, husky, honeyed, glass-sharp, smoky, trembling, velvety, "
                    "hollow, airy, resonant, cracked at the edges, dissolving on the last syllable, "
                    "velvet-low, silk-bright, worn at the edges, full-chested, barely there."
                    "\n\nNEVER write \'she sings a song\' or \'she begins to sing\' — "
                    "describe the actual sound, the actual words, the actual physical act."
                    + _singing_swear_note
                                        + _fmt_adlib_injection(
                        _get_adlibs(_chosen_register, _lyric_genre, seed, 4), "singing")
+ "]"
                )

            elif _is_sex_scene:
                # ── Sex scene vocalisation — ANY preset, position-aware ───────
                try:
                    from lyric_phrase_bank import (
                        _SEX_VOCAL_ENGLISH, _SEX_VOCAL_JAPANESE,
                        _SEX_VOCAL_KOREAN, _SEX_VOCAL_MANDARIN,
                        _SEX_POSITION_POOLS_EN,
                        _SEX_POSITION_POOLS_JP,
                        _SEX_POSITION_POOLS_KR,
                    )
                except ImportError:
                    _SEX_VOCAL_ENGLISH = _SEX_VOCAL_JAPANESE = []
                    _SEX_VOCAL_KOREAN  = _SEX_VOCAL_MANDARIN  = []
                    _SEX_POSITION_POOLS_EN = _SEX_POSITION_POOLS_JP = _SEX_POSITION_POOLS_KR = {}
                    print("[LTX2-Qwen] WARNING: sex vocal pools not found")

                # ── Position detection ────────────────────────────────────
                _ci_sx = _combined_input.lower()
                _sx_position = None
                if re.search(r'\b(missionary|face\s+to\s+face|on\s+her\s+back)\b', _ci_sx):              _sx_position = "missionary"
                elif re.search(r'\b(reverse\s+cowgirl|facing\s+away\s+on\s+top)\b', _ci_sx):             _sx_position = "reverse_cowgirl"
                elif re.search(r'\b(cowgirl|on\s+top|riding\s+him|rides\s+him|sitting\s+on\s+him)\b', _ci_sx): _sx_position = "cowgirl"
                elif re.search(r'\b(doggy|doggy\s+style|from\s+behind|on\s+all\s+fours|bent\s+over)\b', _ci_sx): _sx_position = "doggy"
                elif re.search(r'\b(69|sixty.?nine|mutual\s+oral)\b', _ci_sx):                             _sx_position = "sixtynine"
                elif re.search(r'\b(blowjob|blow\s+job|going\s+down\s+on\s+him|fellatio)\b', _ci_sx):   _sx_position = "blowjob"
                elif re.search(r'\b(riding\s+(a\s+)?(dildo|toy|vibrator)|dildo\s+riding|solo\s+rid\w+)\b', _ci_sx): _sx_position = "riding"

                # ── Language detection ────────────────────────────────────
                # Reuse early detection (already enforces English-unless-nationality rule)
                # _sx_lang and _sx_position set above alongside is_explicit detection
                pass  # _sx_lang already set

                # ── Pool selection ────────────────────────────────────────
                _pos_pools_by_lang = {
                    "Japanese": _SEX_POSITION_POOLS_JP, "Korean": _SEX_POSITION_POOLS_KR,
                    "English":  _SEX_POSITION_POOLS_EN, "Mandarin": _SEX_POSITION_POOLS_EN,
                }
                _generic_pool_by_lang = {
                    "Japanese": _SEX_VOCAL_JAPANESE, "Korean": _SEX_VOCAL_KOREAN,
                    "Mandarin": _SEX_VOCAL_MANDARIN, "English": _SEX_VOCAL_ENGLISH,
                }
                _lang_pos_pools = _pos_pools_by_lang.get(_sx_lang, _SEX_POSITION_POOLS_EN)
                if _sx_position and _sx_position in _lang_pos_pools and len(_lang_pos_pools.get(_sx_position, [])) >= 3:
                    _sx_pool = _lang_pos_pools[_sx_position]; _sx_pool_src = f"{_sx_position} ({_sx_lang})"
                elif _sx_position and _sx_position in _SEX_POSITION_POOLS_EN:
                    _sx_pool = _SEX_POSITION_POOLS_EN[_sx_position]; _sx_pool_src = f"{_sx_position} (EN fallback)"
                else:
                    _sx_pool = _generic_pool_by_lang.get(_sx_lang, _SEX_VOCAL_ENGLISH); _sx_pool_src = f"generic ({_sx_lang})"

                # ── Dynamic moment count based on clip length ─────────────
                # 3 moments for <10s, 5 for 10-20s, 7 for 20s+
                if real_seconds >= 20:   _sx_moments = 7
                elif real_seconds >= 10: _sx_moments = 5
                else:                    _sx_moments = 3

                _sx_rng = random.Random(seed if seed != -1 else None)
                def _sx_pick_unique(phase, count):
                    pool = [l for l in _sx_pool if l[-1] == phase]
                    if not pool: return []
                    # Sample without replacement up to pool size, then wrap if needed
                    n = min(count, len(pool))
                    picked = _sx_rng.sample(pool, n)
                    # If we need more than pool size, pad with random choices
                    while len(picked) < count:
                        picked.append(_sx_rng.choice(pool))
                    return picked
                def _sx_fmt(e):
                    if e is None: return ""
                    return f'"{e[0]}" — {e[1]}' if len(e) == 3 else f'{e[0]} ({e[1]}) — {e[2]}'

                # Pick anchors with unique sampling — no repeated lines
                _n_builds   = max(1, _sx_moments - 2)
                _n_peaks    = max(1, (_sx_moments - 1) // 2)
                _sx_builds  = _sx_pick_unique("build",   _n_builds)
                _sx_peaks   = _sx_pick_unique("peak",    _n_peaks)
                _resolve_l  = [l for l in _sx_pool if l[-1] == "resolve"]
                _sx_resolve = _sx_rng.choice(_resolve_l) if _resolve_l else None
                if _sx_lang == "English":
                    _sx_roman = "Vocalisations MUST be in English ONLY. Do NOT use Japanese, Korean, or any other language. No foreign script. English words only."
                elif _sx_lang == "Mandarin":
                    _sx_roman = f"Vocalisations in Mandarin Chinese characters only — no romanisation in parentheses."
                else:
                    _sx_roman = f"Vocalisations in {_sx_lang} native script only — no romanisation in parentheses. No English."

                # ── Solo toy dialogue ─────────────────────────────────────
                _solo_toy_note = ""
                if _sx_position == "riding":
                    try:
                        from lyric_phrase_bank import _SOLO_TOY_DIALOGUE
                        _st_rng = random.Random((seed + 31) if seed != -1 else None)
                        _st_builds  = [l for l in _SOLO_TOY_DIALOGUE if l[2] == "build"]
                        _st_peaks   = [l for l in _SOLO_TOY_DIALOGUE if l[2] == "peak"]
                        _st_resolve = [l for l in _SOLO_TOY_DIALOGUE if l[2] == "resolve"]
                        _st_b = _st_rng.choice(_st_builds)   if _st_builds  else None
                        _st_p = _st_rng.choice(_st_peaks)    if _st_peaks   else None
                        _st_r = _st_rng.choice(_st_resolve)  if _st_resolve else None
                        def _st_fmt(e): return f'"{e[0]}" ({e[1]})' if e else ""
                        _solo_toy_note = (
                            "\n\nSOLO SCENE — SHE IS ALONE: Replace ALL partner-directed speech with "
                            "self-directed internal narration. She talks to herself, to the toy, or to the camera. "
                            "NO \'you\' directed at a partner — there is no partner. "
                            "Register: honest, raw, surprised, sometimes darkly amused. "
                            f"\nBUILD anchor: {_st_fmt(_st_b)} "
                            f"\nPEAK anchor: {_st_fmt(_st_p)} "
                            f"\nRESOLVE anchor: {_st_fmt(_st_r)}"
                        )
                        print(f"[LTX2-Qwen] Solo toy dialogue: riding position | {_sx_lang}")
                    except ImportError:
                        print("[LTX2-Qwen] WARNING: solo toy pool not found")

                # ── His voice ─────────────────────────────────────────────
                _his_voice_note = ""
                if _wants_male_dialogue and _sx_position != "riding":
                    try:
                        from lyric_phrase_bank import _HIS_VOICE_EN, _HIS_VOICE_JP, _HIS_VOICE_KR
                        _hv_pool = (
                            _HIS_VOICE_JP if _sx_lang == "Japanese" else
                            _HIS_VOICE_KR if _sx_lang == "Korean"   else
                            _HIS_VOICE_EN
                        )
                        _hv_rng = random.Random((seed + 37) if seed != -1 else None)
                        _hv_dom  = [l for l in _hv_pool if l[-1] == "dominant"]
                        _hv_reac = [l for l in _hv_pool if l[-1] == "reactive"]
                        _hv_int  = [l for l in _hv_pool if l[-1] == "intimate"]
                        _hv_d = _hv_rng.choice(_hv_dom)  if _hv_dom  else None
                        _hv_r = _hv_rng.choice(_hv_reac) if _hv_reac else None
                        _hv_i = _hv_rng.choice(_hv_int)  if _hv_int  else None
                        def _hv_fmt(e):
                            if not e: return ""
                            if len(e) == 4: return f'"{e[0]}" / {e[1]} — {e[2]}'
                            return f'"{e[0]}" — {e[1]}'
                        _his_voice_note = (
                            "\n\nHIS VOICE — ACTIVE: Interleave HIS dialogue between her vocalisations. "
                            "He speaks in short fragments — commanding, reactive, or intimate. "
                            "Never clean full sentences. His voice arrives mid-action. "
                            f"\nDominant register example: {_hv_fmt(_hv_d)} "
                            f"\nReactive register example: {_hv_fmt(_hv_r)} "
                            f"\nIntimate register example: {_hv_fmt(_hv_i)} "
                            "\nAlternate: her line → his line → her line. "
                            "His voice is lower, shorter, more controlled until he loses it."
                        )
                        print(f"[LTX2-Qwen] His voice: {_sx_lang} | {_sx_position or 'generic'}")
                    except ImportError:
                        print("[LTX2-Qwen] WARNING: his voice pool not found")

                _pos_frames = {
                    "missionary":      "She is flat on her back, legs raised. He is directly above her, weight on his arms, facing down. CAMERA: strict side-angle medium shot — she faces frame-left, he is perpendicular above her. Both fully visible in profile. Sounds gasped at the ceiling or muffled against his shoulder.",
                    "cowgirl":         "She is on top, controlling pace and depth entirely. Sounds come from power — she sets the rhythm, takes what she wants.",
                    "doggy":           "From behind. CAMERA: strict side-angle medium shot, camera 90 degrees left of the action. She faces directly frame-left, he faces frame-right standing behind and above her. Both fully visible in true profile. Her back arches downward, head hanging forward. His hands grip her hips. Sound raw, animalistic, often muffled into the mattress.",
                    "riding":          "Solo dildo/toy scene. She is alone — squatting or kneeling over a toy fixed to the floor. Self-directed pleasure. She talks to herself, the toy, or the camera. Completely internal register — more honest, more surprised, sometimes darkly funny than partner sex.",
                    "blowjob":         "She is giving oral. Throat occupied. Sound muffled, gagging, broken around the obstruction.",
                    "sixtynine":       "Mutual oral. Fully muffled. Both occupied simultaneously.",
                    "reverse_cowgirl": "On top facing away. Physical focus, less intimate. She can only see away from him.",
                }
                _pos_frame = _pos_frames.get(_sx_position, "Sex scene — position not specified, infer from context.")
                # Camera note — let the LLM know what angle/movement was seeded
                # so the vocalisation instruction and camera work together
                _sx_cam_note = (
                    f"Camera is {_eff_angle.lower().split(' — ')[0] if _eff_angle else 'LLM decides'}, "
                    f"{_eff_movement.lower().split(' — ')[0] if _eff_movement else 'LLM decides movement'}. "
                    f"Write the vocal moments to match this framing physically."
                )

                _build_anchors  = "\n".join(f"  BUILD {i+1}: {_sx_fmt(b)}" for i,b in enumerate(_sx_builds) if b)
                _peak_anchors   = "\n".join(f"  PEAK {i+1}:  {_sx_fmt(p)}" for i,p in enumerate(_sx_peaks) if p)
                _resolve_anchor = f"  RESOLVE: {_sx_fmt(_sx_resolve)}" if _sx_resolve else ""

                # ── Swear injection ──────────────────────────────────────
                _swear_injection_note = ""
                if _wants_swear_injection:
                    try:
                        from lyric_phrase_bank import _SWEAR_INJECTOR, _SWEAR_SCENE_MAP
                        _swear_pool_key = _SWEAR_SCENE_MAP.get(_sx_position or "sex", "sex")
                        _swear_pool     = _SWEAR_INJECTOR.get(_swear_pool_key, _SWEAR_INJECTOR["general"])
                        _sw_rng  = random.Random((seed + 19) if seed != -1 else None)

                        # Pick phase-appropriate swears to inject between anchors
                        _sw_builds  = [l for l in _swear_pool if l[2] in ("build", "soft", "slow", "sharp", "neutral")]
                        _sw_peaks   = [l for l in _swear_pool if l[2] in ("peak", "soft", "slow", "sharp", "neutral")]
                        _sw_resolve = [l for l in _swear_pool if l[2] in ("resolve", "soft", "slow", "neutral")]

                        def _sw_pick(pool):
                            return _sw_rng.choice(pool) if pool else None

                        _samp_b = _sw_pick(_sw_builds)
                        _samp_p = _sw_pick(_sw_peaks)
                        _samp_r = _sw_pick(_sw_resolve) if _sw_resolve else _sw_pick(_sw_builds)

                        def _sw_fmt(e):
                            return f'"{e[0]}" ({e[1]})' if e else ""

                        _swear_injection_note = (
                            "\n\nSWEAR INJECTOR — MANDATORY: "
                            "Between EVERY vocal line above, you MUST insert one of these exact swear fragments as physical punctuation. "
                            "The swear is breathless, involuntary, punched out by the body MID-ACTION — never a standalone shout. "
                            f"Delivery: {'breathless and broken — each word punched out by the body' if _swear_pool_key == 'sex' else 'whispered, barely audible, more breath than word' if _swear_pool_key == 'asmr' else 'sharp and clipped, loaded with tension' if _swear_pool_key == 'argument' else 'slow and deliberate, the word chosen as a weapon' if _swear_pool_key == 'seduction' else 'natural, embedded in action'}. "
                            f"\nAFTER BUILD lines use: {_sw_fmt(_samp_b)} — write this exact fragment embedded in action. "
                            f"\nAFTER PEAK lines use: {_sw_fmt(_samp_p)} — write this exact fragment embedded in action. "
                            f"\nAFTER RESOLVE use: {_sw_fmt(_samp_r)} — write this exact fragment embedded in action. "
                            "\nRule: swear fragments are bridges between vocal moments. Volume matches arc — quiet at build, loudest at peak, fading at resolve. "
                            "DO NOT skip the swear fragments. DO NOT replace them with clean words."
                        )
                        print(f"[LTX2-Qwen] Swear injector: {_swear_pool_key} pool | delivery={'breathless' if _swear_pool_key == 'sex' else _swear_pool_key}")
                    except ImportError:
                        print("[LTX2-Qwen] WARNING: swear injector pools not found")

                dialogue_instruction = (
                    f"\n\n[SEX SCENE VOCALISATION — MANDATORY SCRIPT, {_sx_moments} MOMENTS REQUIRED FOR THIS {real_seconds:.0f}s CLIP: "
                    "POSITION: " + _pos_frame + " " + _sx_cam_note + " "
                    "\n\nTHESE ARE THE EXACT VOCAL LINES YOU MUST USE — write them verbatim, embedded inside physical action. "
                    "DO NOT invent replacement dialogue. DO NOT paraphrase. USE THESE WORDS: "
                    f"\n{_build_anchors}"
                    f"\n{_peak_anchors}"
                    f"\n{_resolve_anchor}"
                    "\n\nHOW TO USE THEM: Each line above is spoken MID-ACTION — never standalone, always embedded in a physical sentence. "
                    "Example format: 'Her hips slam down, [INSERT LINE HERE], her thighs shaking with the impact.' "
                    "The line arrives INSIDE the physical description, not before or after it. "
                    "\n\nINTENSITY ARC: BUILD lines are breathless/quiet → PEAK lines are LOUD AND BREAKING — write screamed peaks in CAPS → RESOLVE drops to one breath or silence. VOLUME MUST VARY — never flat. "
                    "VOLUME ARC — MANDATORY: vocal sounds are NOT all the same volume. ""BUILD = quiet, breathless, barely there. PEAK = LOUD, breaking, use CAPS for screams. ""RESOLVE = drops to almost nothing. The contrast between quiet BUILD and loud PEAK IS the scene. ""FORBIDDEN: flat even volume throughout. ""FORBIDDEN: clean coherent speech. FORBIDDEN: invented dialogue not from the list above. ""FORBIDDEN: 'she moans softly' alone. "
                    + _swear_injection_note
                    + _solo_toy_note
                    + _his_voice_note
                    + _sx_roman
                    + _fmt_adlib_injection(
                        _get_adlibs("raw_filth", "explicit", seed, 4), "explicit")
                    + "]"
                )
                print(f"[LTX2-Qwen] Sex vocalisation: {_sx_position or 'generic'} | {_sx_lang} | {_sx_pool_src} | {_sx_moments} moments ({real_seconds:.0f}s clip)")

                def _detect_grv_language(text):
                    t = text.lower()
                    # Pool languages first (native script pools exist)
                    if re.search(r'\b(japanese|japan)\b', t):                                    return "Japanese"
                    if re.search(r'\b(korean|korea)\b', t):                                      return "Korean"
                    if re.search(r'\b(chinese|china|mandarin|cantonese)\b', t):                  return "Mandarin"
                    # Non-pool languages — LLM generates in these, instruction sets language
                    if re.search(r'\b(french|france|parisian|parisienne)\b', t):                 return "French"
                    if re.search(r'\b(german|germany|deutsch|berliner|austrian)\b', t):          return "German"
                    if re.search(r'\b(italian|italy|italiana|sicilian|roman)\b', t):             return "Italian"
                    if re.search(r'\b(spanish|spain|latina|mexican|colombian|argentinian|castilian)\b', t): return "Spanish"
                    if re.search(r'\b(portuguese|portugal|brazilian|brasil)\b', t):              return "Portuguese"
                    if re.search(r'\b(russian|russia)\b', t):                                    return "Russian"
                    if re.search(r'\b(arabic|arab|lebanese|moroccan|egyptian|saudi|emirati|gulf|algerian|tunisian)\b', t): return "Arabic"
                    if re.search(r'\b(hindi|indian|south\s+asian|bengali|punjabi|urdu|pakistani)\b', t): return "Hindi"
                    if re.search(r'\b(thai|thailand)\b', t):                                     return "Thai"
                    if re.search(r'\b(vietnamese|vietnam)\b', t):                                return "Vietnamese"
                    if re.search(r'\b(indonesian|indonesia|malay|malaysia)\b', t):               return "Indonesian"
                    if re.search(r'\b(tagalog|filipino|philippines)\b', t):                      return "Filipino"
                    if re.search(r'\b(turkish|turkey)\b', t):                                    return "Turkish"
                    if re.search(r'\b(persian|iranian|farsi|iran)\b', t):                        return "Persian"
                    if re.search(r'\b(swedish|sweden|svenska)\b', t):                            return "Swedish"
                    if re.search(r'\b(norwegian|norway|norsk)\b', t):                            return "Norwegian"
                    if re.search(r'\b(danish|denmark|dansk)\b', t):                              return "Danish"
                    if re.search(r'\b(finnish|finland|suomi)\b', t):                             return "Finnish"
                    if re.search(r'\b(dutch|netherlands|holland)\b', t):                         return "Dutch"
                    if re.search(r'\b(polish|poland)\b', t):                                     return "Polish"
                    if re.search(r'\b(greek|greece)\b', t):                                      return "Greek"
                    if re.search(r'\b(hebrew|israel|israeli)\b', t):                             return "Hebrew"
                    if re.search(r'\b(ukrainian|ukraine)\b', t):                                 return "Ukrainian"
                    if re.search(r'\b(czech|czechia|bohemian)\b', t):                            return "Czech"
                    if re.search(r'\b(hungarian|hungary|magyar)\b', t):                          return "Hungarian"
                    if re.search(r'\b(romanian|romania)\b', t):                                  return "Romanian"
                    if re.search(r'\b(swahili|kenyan|tanzanian)\b', t):                          return "Swahili"
                    if re.search(r'\b(east\s+asian)\b', t):                                     return "Japanese"
                    return "Japanese"  # gravure default

                _grv_english_override = bool(re.search(r'\bin\s+english\b', _combined_input, re.IGNORECASE))
                _grv_lang    = "English" if _grv_english_override else _detect_grv_language(_combined_input)
                _has_pool    = _grv_lang in ("Japanese", "Korean", "Mandarin", "English")
                _speech_pool = (
                    _GRV_LINES_JAPANESE if _grv_lang == "Japanese" else
                    _GRV_LINES_KOREAN   if _grv_lang == "Korean"   else
                    _GRV_LINES_MANDARIN if _grv_lang == "Mandarin"  else
                    _GRV_LINES_ENGLISH  if _grv_lang == "English"   else []
                )
                _sing_pool = (
                    _GRV_SINGING_JAPANESE if _grv_lang == "Japanese" else
                    _GRV_SINGING_KOREAN   if _grv_lang == "Korean"   else
                    _GRV_SINGING_MANDARIN if _grv_lang == "Mandarin"  else []
                )
                # Script requirement — native characters inline in prose, NO parenthetical romanisation
                if _grv_lang == "English":
                    _roman_note = ""  # no script requirement for English
                else:
                    _roman_note = (
                        f"CRITICAL SCRIPT AND FORMAT REQUIREMENT: "
                        f"Write each line in full {_grv_lang} characters "
                        f"({'kanji/hiragana/katakana' if _grv_lang == 'Japanese' else 'hangul (한글)' if _grv_lang == 'Korean' else 'hanzi/simplified Chinese'}). "
                        f"Do NOT write romanisation only, and do NOT put romanisation in parentheses next to the dialogue — "
                        f"parenthetical text renders as on-screen subtitles in the video. "
                        f"CORRECT: She whispers 「もっと近くで見て」, the syllables drawn out softly. "
                        f"WRONG: She whispers 「もっと近くで見て」(Motto chikaku de mite). "
                        f"Write native script characters only — no brackets, no romanisation alongside. "
                    )

                # Seed-driven RNG for reproducible variety
                _dlg_rng = random.Random(seed if seed != -1 else None)

                if _is_singing:
                    # ── Singing mode ──────────────────────────────────────────
                    if _has_pool:
                        _sing_tier = [l for l in _sing_pool if l[3] in ('T', 'S')] if not is_explicit else _sing_pool
                        _sing_picked = _dlg_rng.sample(_sing_tier, min(3, len(_sing_tier)))
                        _sing_examples = "  ".join(
                            f"'{script} ({roman}) — {note}.'"
                            for script, roman, note, _ in _sing_picked
                        )
                        _sing_example_line = (
                            f"EXAMPLE TONE AND FORMAT (do NOT copy verbatim — invent your own lyric content): {_sing_examples} "
                        )
                    else:
                        _sing_example_line = (
                            f"EXAMPLE FORMAT: 'Tu sei tutto per me... (she sings, barely above a breath, "
                            f"the phrase dissolving on the last syllable)' — adapt this structure to {_grv_lang}. "
                        )
                    dialogue_instruction = (
                        "\n\n[VOCAL/SINGING INSTRUCTION — MANDATORY, CANNOT BE SKIPPED: "
                        f"She sings or hums ONLY in {_grv_lang}. No English lyrics. "
                        "Use lyric-style fragments — short melodic phrases, not full sentences of speech. "
                        "Include THREE vocal moments: one early (establishing the melody), "
                        "one mid (emotionally peaks), one late (fades or resolves). "
                        "Describe HOW she sings each line — breath, volume, note shape, where the phrase dissolves. "
                        "Lyrics should feel written for intimacy — soft, personal, slightly incomplete, "
                        "as if she is singing to one person only. "
                        + _roman_note
                        + _sing_example_line
                        + music_sound_rule + "]"
                    )
                else:
                    # ── Speech mode ───────────────────────────────────────────
                    # Tier-specific tone brief — includes dual-register (subject + gaze POV) at explicit tier
                    if is_explicit:
                        _grv_tone = (
                            "This scene uses TWO distinct dialogue voices — keep them clearly separated: "
                            "(1) SUBJECT VOICE — SHE speaks about her own body and desire: "
                            "'I want', 'I need', 'fuck me', 'I'm so wet', 'don't stop' — first person, her experience. "
                            "She NEVER tells the viewer what to do with THEIR body ('take my cock' is WRONG for her voice — that is gaze voice language). "
                            "She speaks FROM her body, not about his. "
                            "(2) GAZE VOICE — the camera operator / director / viewer speaks TO her: "
                            "short, flat commands or observations directed AT her. "
                            "'Look at me', 'don't you dare come', 'show me everything', 'good girl' — second person, directed at her. "
                            "NEVER loud. Quiet authority. "
                            "Distribute: two subject lines (her first-person desire), one gaze line (his direction to her). "
                            "Explicit lines must be anatomically specific and match exactly what is happening. "
                            "Do NOT sanitise, soften, or euphemise anything."
                        )
                    elif is_sensual:
                        _grv_tone = (
                            "Lines span the emotional range of the scene: one playful or curious, "
                            "one intimate and slightly vulnerable, one sensual and direct. "
                            "Where a second voice (director/viewer) is implied by the scene, "
                            "one line may come from that gaze POV — quiet, observational, directive. "
                            "Stay at the level of sensuality the scene describes — do not self-escalate."
                        )
                    else:
                        _grv_tone = (
                            "Lines are warm, intimate, and camera-aware: one playful, "
                            "one confessional or emotionally open, one direct and present. "
                            "Keep all content tasteful — no explicit or overtly sexual language."
                        )

                    if _has_pool:
                        # Pool languages — use tiered examples + LLM invents third
                        # English pool tier is index 2, all other pools tier is index 3
                        _tier_idx = 2 if _grv_lang == "English" else 3
                        if is_explicit:
                            _active_pool = _speech_pool
                        elif is_sensual:
                            _active_pool = [l for l in _speech_pool if l[_tier_idx] in ('T', 'S')]
                        else:
                            _active_pool = [l for l in _speech_pool if l[_tier_idx] == 'T']
                        if len(_active_pool) < 3:
                            _active_pool = [l for l in _speech_pool if l[_tier_idx] in ('T', 'S')]
                        if len(_active_pool) < 3:
                            _active_pool = _speech_pool

                        if is_explicit:
                            # For explicit tier: sample 2 subject lines + 1 gaze POV line
                            # English pool: register in index 3. Other pools: detect via delivery note.
                            if _grv_lang == "English":
                                _subj_pool = [l for l in _active_pool if l[3] == 'subj']
                                _gaze_pool = [l for l in _active_pool if l[3] == 'gaze']
                            else:
                                _subj_pool = [l for l in _active_pool if 'lens' not in l[2] and 'direction' not in l[2] and 'he says' not in l[2] and 'voice' not in l[2]]
                                _gaze_pool = [l for l in _active_pool if 'lens' in l[2] or 'direction' in l[2] or 'he says' in l[2] or 'voice' in l[2]]
                            # Prefer X-tier for explicit anchor examples
                            _subj_x = [l for l in _subj_pool if l[_tier_idx] == 'X'] if _subj_pool else []
                            _gaze_x = [l for l in _gaze_pool if l[_tier_idx] == 'X'] if _gaze_pool else []
                            _subj_pool = _subj_x if len(_subj_x) >= 2 else (_subj_pool or _active_pool)
                            _gaze_pool = _gaze_x if len(_gaze_x) >= 1 else (_gaze_pool or _active_pool)
                            if not _subj_pool: _subj_pool = _active_pool
                            if not _gaze_pool: _gaze_pool = _active_pool
                            _picked_subj = _dlg_rng.sample(_subj_pool, min(2, len(_subj_pool)))
                            _picked_gaze = _dlg_rng.sample(_gaze_pool, min(1, len(_gaze_pool)))
                            _picked = _picked_subj + _picked_gaze
                        else:
                            _picked = _dlg_rng.sample(_active_pool, min(2, len(_active_pool)))

                        # English pool: (line, note, tier, register) — no romanisation field
                        if _grv_lang == "English":
                            _grv_examples = "  ".join(
                                f"'{line} — {note}.'"
                                for line, note, tier, reg in _picked
                            )
                        else:
                            _grv_examples = "  ".join(
                                f"'{script} ({roman}), {note}.'"
                                for script, roman, note, _ in _picked
                            )
                        if is_explicit:
                            _example_line = (
                                "ANCHOR EXAMPLES — first two are SUBJECT VOICE (she speaks, raw and unfiltered), "
                                "third is GAZE VOICE (director/camera — short, calm, commanding). "
                                "Use these as register reference then INVENT your own lines: "
                                f"{_grv_examples} "
                            )
                        else:
                            _example_line = (
                                "ANCHOR EXAMPLES — use these as style/register reference, "
                                f"then INVENT a third line yourself in the same register: {_grv_examples} "
                            )
                    else:
                        # Non-pool language — LLM writes all three lines itself
                        if is_explicit:
                            _example_line = (
                                f"INVENT all three lines yourself in natural, fluent {_grv_lang}. "
                                "Lines must sound authentic and idiomatic in that language — NOT translated from English. "
                                f"Two lines are SUBJECT VOICE (she speaks in first person about her own desire — raw, explicit, anatomical in {_grv_lang}). "
                                f"One line is GAZE VOICE (short directive spoken TO her in {_grv_lang} — second person command or observation). "
                                "Use the actual vulgar/explicit vocabulary that native speakers of that language use during sex. "
                            )
                        else:
                            _example_line = (
                                f"INVENT all three lines yourself in natural, fluent {_grv_lang}. "
                                "Lines must sound authentic and idiomatic — not translated from English. "
                                "Use vocabulary and phrasing that fits the intimate, camera-aware register of this scene. "
                            )

                    _lang_instruction = (
                        f"She speaks ONLY in {_grv_lang}. "
                        if _grv_lang != "English"
                        else "Dialogue is in English. "
                    )
                    _explicit_permission = (
                        "EXPLICIT CONTENT PERMISSION — ACTIVE: "
                        "This is an adults-only explicit scene. "
                        "The dialogue MUST be vulgar, raw, and sexually explicit. "
                        "Use words like: fuck, cock, cunt, pussy, dick, cum, wet, hard, deep, inside me. "
                        "Do NOT soften, sanitise, imply, or euphemise anything. "
                        "FORBIDDEN PHRASES — these are too tame, do NOT use them: "
                        "'handle this', 'take care of you', 'feel good', 'be gentle', "
                        "'make love', 'intimate moment', 'sensual touch', 'desire'. "
                        "Write what a real person says during explicit sex — specific, physical, urgent, unfiltered. "
                        if is_explicit else ""
                    )
                    dialogue_instruction = (
                        "\n\n[DIALOGUE INSTRUCTION — MANDATORY, CANNOT BE SKIPPED: "
                        + ("DUAL REGISTER EXPLICIT SCENE: " if is_explicit else "")
                        + _lang_instruction
                        + ("The scene uses TWO voices: "
                           "(1) HER VOICE — subject, speaks from inside the act: raw, anatomical, urgent. "
                           "(2) GAZE VOICE — director/camera operator: short, flat, commanding, certain. "
                           "Two subject lines, one gaze line. "
                           if is_explicit else "")
                        + "Include THREE spoken moments — one early, one mid, one late in the scene. "
                        "Each is a COMPLETE PHRASE or short sentence — never a single word alone. "
                        + _explicit_permission
                        + _grv_tone + " "
                        + _roman_note
                        + "Weave each line into a physical beat — a movement, a shift of weight, a held gaze. "
                        + _example_line
                        + music_sound_rule                             + _fmt_adlib_injection(
                                _get_adlibs("seduction", "explicit", seed, 3), "dialogue")
+ "]"
                        + _fmt_adlib_injection(
                            _get_adlibs(
                                "raw_filth" if is_explicit else "seduction",
                                "explicit" if is_explicit else "universal",
                                seed, 3), "gravure")
                    )
            elif is_gravure:
                # ── Sex scene vocalisation mode ───────────────────────────────────
                # When _is_sex_scene=True, bypass clean speech entirely.
                # Load sex vocal pools from phrase bank and build vocalisation instruction.
                if _is_sex_scene:
                    try:
                        from lyric_phrase_bank import (
                            _SEX_VOCAL_ENGLISH, _SEX_VOCAL_JAPANESE,
                            _SEX_VOCAL_KOREAN, _SEX_VOCAL_MANDARIN,
                        )
                    except ImportError:
                        _SEX_VOCAL_ENGLISH = _SEX_VOCAL_JAPANESE = []
                        _SEX_VOCAL_KOREAN = _SEX_VOCAL_MANDARIN = []
                        print("[LTX2-Qwen] WARNING: sex vocal pools not found")

                    # Detect language for native script sex vocals
                    _sx_english = bool(re.search(r'\bin\s+english\b', _combined_input, re.IGNORECASE))
                    _sx_lang_t  = _combined_input.lower()
                    if _sx_english:               _sx_lang = "English"
                    elif re.search(r'\b(japanese|japan)\b', _sx_lang_t): _sx_lang = "Japanese"
                    elif re.search(r'\b(korean|korea)\b', _sx_lang_t):   _sx_lang = "Korean"
                    elif re.search(r'\b(chinese|china|mandarin)\b', _sx_lang_t): _sx_lang = "Mandarin"
                    else:                          _sx_lang = "English"  # default for non-Asian sex scenes

                    _sx_pool = (
                        _SEX_VOCAL_JAPANESE if _sx_lang == "Japanese" else
                        _SEX_VOCAL_KOREAN   if _sx_lang == "Korean"   else
                        _SEX_VOCAL_MANDARIN if _sx_lang == "Mandarin" else
                        _SEX_VOCAL_ENGLISH
                    )
                    _sx_rng = random.Random(seed if seed != -1 else None)

                    def _sx_pick(phase):
                        pool = [l for l in _sx_pool if l[-1] == phase]
                        return _sx_rng.choice(pool) if pool else None

                    _sx_build   = _sx_pick("build")
                    _sx_peak    = _sx_pick("peak")
                    _sx_resolve = _sx_pick("resolve")

                    def _sx_fmt(entry):
                        if entry is None: return ""
                        if _sx_lang == "English":
                            return f'"{entry[0]}" — {entry[1]}'
                        return f'{entry[0]} ({entry[1]}) — {entry[2]}'

                    _sx_roman_note = (
                        f"Write vocalisations in {_sx_lang} native script inline — no romanisation in parentheses. "
                        if _sx_lang != "English" else ""
                    )

                    dialogue_instruction = (
                        "\n\n[SEX SCENE VOCALISATION — MANDATORY, CANNOT BE SKIPPED: "
                        "This is a sex scene. Do NOT write clean coherent speech. "
                        "Dialogue is replaced entirely by PHYSICAL VOCALISATIONS — moans, gasps, broken fragments, "
                        "involuntary sounds woven directly into the physical action. "
                        "Every vocalisation must arrive MID-ACTION — never before or after, always during. "
                        "\n\nSTRUCTURE — THREE vocalisation moments: "
                        "\n(1) BUILD — rising, not arrived yet. Body building, sound escalating. "
                        f"ANCHOR: {_sx_fmt(_sx_build)} "
                        "\n(2) PEAK — breaking, losing control, voice fragmenting. The body is louder than the words. "
                        f"ANCHOR: {_sx_fmt(_sx_peak)} "
                        "\n(3) RESOLVE — aftermath. Breath, stillness, one quiet sound. "
                        f"ANCHOR: {_sx_fmt(_sx_resolve)} "
                        "\n\nFORMAT RULES: "
                        "Write sound in italics or quote marks embedded in prose: "
                        "\'Her hips grind harder, a broken \'fuck\'— barely a word — escaping between sharp inhales.\' "
                        "The vocalisation is NEVER a standalone line. It arrives inside action. "
                        "Sound builds in volume and desperation across the three moments. "
                        "Resolve is quiet — one breath, one word, or silence. "
                        + _sx_roman_note
                        + "FORBIDDEN: clean sentences, explanations, declarations, dialogue that could exist outside a sex scene. "
                        "FORBIDDEN: \'she moans softly\' alone — always attach the sound to physical cause and effect.] "
                        + _fmt_adlib_injection(
                            _get_adlibs("raw_filth", "explicit", seed, 4), "explicit")
                    )
                    print(f"[LTX2-Qwen] Sex vocalisation mode: {_sx_lang} | build/peak/resolve anchors set")

                def _detect_grv_language(text):
                    t = text.lower()
                    # Pool languages first (native script pools exist)
                    if re.search(r'\b(japanese|japan)\b', t):                                    return "Japanese"
                    if re.search(r'\b(korean|korea)\b', t):                                      return "Korean"
                    if re.search(r'\b(chinese|china|mandarin|cantonese)\b', t):                  return "Mandarin"
                    # Non-pool languages — LLM generates in these, instruction sets language
                    if re.search(r'\b(french|france|parisian|parisienne)\b', t):                 return "French"
                    if re.search(r'\b(german|germany|deutsch|berliner|austrian)\b', t):          return "German"
                    if re.search(r'\b(italian|italy|italiana|sicilian|roman)\b', t):             return "Italian"
                    if re.search(r'\b(spanish|spain|latina|mexican|colombian|argentinian|castilian)\b', t): return "Spanish"
                    if re.search(r'\b(portuguese|portugal|brazilian|brasil)\b', t):              return "Portuguese"
                    if re.search(r'\b(russian|russia)\b', t):                                    return "Russian"
                    if re.search(r'\b(arabic|arab|lebanese|moroccan|egyptian|saudi|emirati|gulf|algerian|tunisian)\b', t): return "Arabic"
                    if re.search(r'\b(hindi|indian|south\s+asian|bengali|punjabi|urdu|pakistani)\b', t): return "Hindi"
                    if re.search(r'\b(thai|thailand)\b', t):                                     return "Thai"
                    if re.search(r'\b(vietnamese|vietnam)\b', t):                                return "Vietnamese"
                    if re.search(r'\b(indonesian|indonesia|malay|malaysia)\b', t):               return "Indonesian"
                    if re.search(r'\b(tagalog|filipino|philippines)\b', t):                      return "Filipino"
                    if re.search(r'\b(turkish|turkey)\b', t):                                    return "Turkish"
                    if re.search(r'\b(persian|iranian|farsi|iran)\b', t):                        return "Persian"
                    if re.search(r'\b(swedish|sweden|svenska)\b', t):                            return "Swedish"
                    if re.search(r'\b(norwegian|norway|norsk)\b', t):                            return "Norwegian"
                    if re.search(r'\b(danish|denmark|dansk)\b', t):                              return "Danish"
                    if re.search(r'\b(finnish|finland|suomi)\b', t):                             return "Finnish"
                    if re.search(r'\b(dutch|netherlands|holland)\b', t):                         return "Dutch"
                    if re.search(r'\b(polish|poland)\b', t):                                     return "Polish"
                    if re.search(r'\b(greek|greece)\b', t):                                      return "Greek"
                    if re.search(r'\b(hebrew|israel|israeli)\b', t):                             return "Hebrew"
                    if re.search(r'\b(ukrainian|ukraine)\b', t):                                 return "Ukrainian"
                    if re.search(r'\b(czech|czechia|bohemian)\b', t):                            return "Czech"
                    if re.search(r'\b(hungarian|hungary|magyar)\b', t):                          return "Hungarian"
                    if re.search(r'\b(romanian|romania)\b', t):                                  return "Romanian"
                    if re.search(r'\b(swahili|kenyan|tanzanian)\b', t):                          return "Swahili"
                    if re.search(r'\b(east\s+asian)\b', t):                                     return "Japanese"
                    return "Japanese"  # gravure default

                _grv_english_override = bool(re.search(r'\bin\s+english\b', _combined_input, re.IGNORECASE))
                _grv_lang    = "English" if _grv_english_override else _detect_grv_language(_combined_input)
                _has_pool    = _grv_lang in ("Japanese", "Korean", "Mandarin", "English")
                _speech_pool = (
                    _GRV_LINES_JAPANESE if _grv_lang == "Japanese" else
                    _GRV_LINES_KOREAN   if _grv_lang == "Korean"   else
                    _GRV_LINES_MANDARIN if _grv_lang == "Mandarin"  else
                    _GRV_LINES_ENGLISH  if _grv_lang == "English"   else []
                )
                _sing_pool = (
                    _GRV_SINGING_JAPANESE if _grv_lang == "Japanese" else
                    _GRV_SINGING_KOREAN   if _grv_lang == "Korean"   else
                    _GRV_SINGING_MANDARIN if _grv_lang == "Mandarin"  else []
                )
                # Script requirement — native characters inline in prose, NO parenthetical romanisation
                if _grv_lang == "English":
                    _roman_note = ""  # no script requirement for English
                else:
                    _roman_note = (
                        f"CRITICAL SCRIPT AND FORMAT REQUIREMENT: "
                        f"Write each line in full {_grv_lang} characters "
                        f"({'kanji/hiragana/katakana' if _grv_lang == 'Japanese' else 'hangul (한글)' if _grv_lang == 'Korean' else 'hanzi/simplified Chinese'}). "
                        f"Do NOT write romanisation only, and do NOT put romanisation in parentheses next to the dialogue — "
                        f"parenthetical text renders as on-screen subtitles in the video. "
                        f"CORRECT: She whispers 「もっと近くで見て」, the syllables drawn out softly. "
                        f"WRONG: She whispers 「もっと近くで見て」(Motto chikaku de mite). "
                        f"Write native script characters only — no brackets, no romanisation alongside. "
                    )

                # Seed-driven RNG for reproducible variety
                _dlg_rng = random.Random(seed if seed != -1 else None)

                if _is_singing:
                    # ── Singing mode ──────────────────────────────────────────
                    if _has_pool:
                        _sing_tier = [l for l in _sing_pool if l[3] in ('T', 'S')] if not is_explicit else _sing_pool
                        _sing_picked = _dlg_rng.sample(_sing_tier, min(3, len(_sing_tier)))
                        _sing_examples = "  ".join(
                            f"'{script} ({roman}) — {note}.'"
                            for script, roman, note, _ in _sing_picked
                        )
                        _sing_example_line = (
                            f"EXAMPLE TONE AND FORMAT (do NOT copy verbatim — invent your own lyric content): {_sing_examples} "
                        )
                    else:
                        _sing_example_line = (
                            f"EXAMPLE FORMAT: 'Tu sei tutto per me... (she sings, barely above a breath, "
                            f"the phrase dissolving on the last syllable)' — adapt this structure to {_grv_lang}. "
                        )
                    dialogue_instruction = (
                        "\n\n[VOCAL/SINGING INSTRUCTION — MANDATORY, CANNOT BE SKIPPED: "
                        f"She sings or hums ONLY in {_grv_lang}. No English lyrics. "
                        "Use lyric-style fragments — short melodic phrases, not full sentences of speech. "
                        "Include THREE vocal moments: one early (establishing the melody), "
                        "one mid (emotionally peaks), one late (fades or resolves). "
                        "Describe HOW she sings each line — breath, volume, note shape, where the phrase dissolves. "
                        "Lyrics should feel written for intimacy — soft, personal, slightly incomplete, "
                        "as if she is singing to one person only. "
                        + _roman_note
                        + _sing_example_line
                        + music_sound_rule + "]"
                    )
                else:
                    # ── Speech mode ───────────────────────────────────────────
                    # Tier-specific tone brief — includes dual-register (subject + gaze POV) at explicit tier
                    if is_explicit:
                        _grv_tone = (
                            "This scene uses TWO distinct dialogue voices — keep them clearly separated: "
                            "(1) SUBJECT VOICE — SHE speaks about her own body and desire: "
                            "'I want', 'I need', 'fuck me', 'I'm so wet', 'don't stop' — first person, her experience. "
                            "She NEVER tells the viewer what to do with THEIR body ('take my cock' is WRONG for her voice — that is gaze voice language). "
                            "She speaks FROM her body, not about his. "
                            "(2) GAZE VOICE — the camera operator / director / viewer speaks TO her: "
                            "short, flat commands or observations directed AT her. "
                            "'Look at me', 'don't you dare come', 'show me everything', 'good girl' — second person, directed at her. "
                            "NEVER loud. Quiet authority. "
                            "Distribute: two subject lines (her first-person desire), one gaze line (his direction to her). "
                            "Explicit lines must be anatomically specific and match exactly what is happening. "
                            "Do NOT sanitise, soften, or euphemise anything."
                        )
                    elif is_sensual:
                        _grv_tone = (
                            "Lines span the emotional range of the scene: one playful or curious, "
                            "one intimate and slightly vulnerable, one sensual and direct. "
                            "Where a second voice (director/viewer) is implied by the scene, "
                            "one line may come from that gaze POV — quiet, observational, directive. "
                            "Stay at the level of sensuality the scene describes — do not self-escalate."
                        )
                    else:
                        _grv_tone = (
                            "Lines are warm, intimate, and camera-aware: one playful, "
                            "one confessional or emotionally open, one direct and present. "
                            "Keep all content tasteful — no explicit or overtly sexual language."
                        )

                    if _has_pool:
                        # Pool languages — use tiered examples + LLM invents third
                        # English pool tier is index 2, all other pools tier is index 3
                        _tier_idx = 2 if _grv_lang == "English" else 3
                        if is_explicit:
                            _active_pool = _speech_pool
                        elif is_sensual:
                            _active_pool = [l for l in _speech_pool if l[_tier_idx] in ('T', 'S')]
                        else:
                            _active_pool = [l for l in _speech_pool if l[_tier_idx] == 'T']
                        if len(_active_pool) < 3:
                            _active_pool = [l for l in _speech_pool if l[_tier_idx] in ('T', 'S')]
                        if len(_active_pool) < 3:
                            _active_pool = _speech_pool

                        if is_explicit:
                            # For explicit tier: sample 2 subject lines + 1 gaze POV line
                            # English pool: register in index 3. Other pools: detect via delivery note.
                            if _grv_lang == "English":
                                _subj_pool = [l for l in _active_pool if l[3] == 'subj']
                                _gaze_pool = [l for l in _active_pool if l[3] == 'gaze']
                            else:
                                _subj_pool = [l for l in _active_pool if 'lens' not in l[2] and 'direction' not in l[2] and 'he says' not in l[2] and 'voice' not in l[2]]
                                _gaze_pool = [l for l in _active_pool if 'lens' in l[2] or 'direction' in l[2] or 'he says' in l[2] or 'voice' in l[2]]
                            # Prefer X-tier for explicit anchor examples
                            _subj_x = [l for l in _subj_pool if l[_tier_idx] == 'X'] if _subj_pool else []
                            _gaze_x = [l for l in _gaze_pool if l[_tier_idx] == 'X'] if _gaze_pool else []
                            _subj_pool = _subj_x if len(_subj_x) >= 2 else (_subj_pool or _active_pool)
                            _gaze_pool = _gaze_x if len(_gaze_x) >= 1 else (_gaze_pool or _active_pool)
                            if not _subj_pool: _subj_pool = _active_pool
                            if not _gaze_pool: _gaze_pool = _active_pool
                            _picked_subj = _dlg_rng.sample(_subj_pool, min(2, len(_subj_pool)))
                            _picked_gaze = _dlg_rng.sample(_gaze_pool, min(1, len(_gaze_pool)))
                            _picked = _picked_subj + _picked_gaze
                        else:
                            _picked = _dlg_rng.sample(_active_pool, min(2, len(_active_pool)))

                        # English pool: (line, note, tier, register) — no romanisation field
                        if _grv_lang == "English":
                            _grv_examples = "  ".join(
                                f"'{line} — {note}.'"
                                for line, note, tier, reg in _picked
                            )
                        else:
                            _grv_examples = "  ".join(
                                f"'{script} ({roman}), {note}.'"
                                for script, roman, note, _ in _picked
                            )
                        if is_explicit:
                            _example_line = (
                                "ANCHOR EXAMPLES — first two are SUBJECT VOICE (she speaks, raw and unfiltered), "
                                "third is GAZE VOICE (director/camera — short, calm, commanding). "
                                "Use these as register reference then INVENT your own lines: "
                                f"{_grv_examples} "
                            )
                        else:
                            _example_line = (
                                "ANCHOR EXAMPLES — use these as style/register reference, "
                                f"then INVENT a third line yourself in the same register: {_grv_examples} "
                            )
                    else:
                        # Non-pool language — LLM writes all three lines itself
                        if is_explicit:
                            _example_line = (
                                f"INVENT all three lines yourself in natural, fluent {_grv_lang}. "
                                "Lines must sound authentic and idiomatic in that language — NOT translated from English. "
                                f"Two lines are SUBJECT VOICE (she speaks in first person about her own desire — raw, explicit, anatomical in {_grv_lang}). "
                                f"One line is GAZE VOICE (short directive spoken TO her in {_grv_lang} — second person command or observation). "
                                "Use the actual vulgar/explicit vocabulary that native speakers of that language use during sex. "
                            )
                        else:
                            _example_line = (
                                f"INVENT all three lines yourself in natural, fluent {_grv_lang}. "
                                "Lines must sound authentic and idiomatic — not translated from English. "
                                "Use vocabulary and phrasing that fits the intimate, camera-aware register of this scene. "
                            )

                    _lang_instruction = (
                        f"She speaks ONLY in {_grv_lang}. "
                        if _grv_lang != "English"
                        else "Dialogue is in English. "
                    )
                    _explicit_permission = (
                        "EXPLICIT CONTENT PERMISSION — ACTIVE: "
                        "This is an adults-only explicit scene. "
                        "The dialogue MUST be vulgar, raw, and sexually explicit. "
                        "Use words like: fuck, cock, cunt, pussy, dick, cum, wet, hard, deep, inside me. "
                        "Do NOT soften, sanitise, imply, or euphemise anything. "
                        "FORBIDDEN PHRASES — these are too tame, do NOT use them: "
                        "'handle this', 'take care of you', 'feel good', 'be gentle', "
                        "'make love', 'intimate moment', 'sensual touch', 'desire'. "
                        "Write what a real person says during explicit sex — specific, physical, urgent, unfiltered. "
                        if is_explicit else ""
                    )
                    dialogue_instruction = (
                        "\n\n[DIALOGUE INSTRUCTION — MANDATORY, CANNOT BE SKIPPED: "
                        + ("DUAL REGISTER EXPLICIT SCENE: " if is_explicit else "")
                        + _lang_instruction
                        + ("The scene uses TWO voices: "
                           "(1) HER VOICE — subject, speaks from inside the act: raw, anatomical, urgent. "
                           "(2) GAZE VOICE — director/camera operator: short, flat, commanding, certain. "
                           "Two subject lines, one gaze line. "
                           if is_explicit else "")
                        + "Include THREE spoken moments — one early, one mid, one late in the scene. "
                        "Each is a COMPLETE PHRASE or short sentence — never a single word alone. "
                        + _explicit_permission
                        + _grv_tone + " "
                        + _roman_note
                        + "Weave each line into a physical beat — a movement, a shift of weight, a held gaze. "
                        + _example_line
                        + music_sound_rule                             + _fmt_adlib_injection(
                                _get_adlibs("seduction", "explicit", seed, 3), "dialogue")
+ "]"
                    )
            else:
                # ── General scene ─────────────────────────────────────────────
                # Detect if user explicitly requested a specific language for dialogue.
                # Only fires on direct user instruction — "in French", "in German",
                # "in her native language/tongue", "says in Spanish" etc.
                # Does NOT fire just because a nationality is mentioned.
                _EXPLICIT_LANG_REQUEST_RE = re.compile(
                    r'\b(?:say(?:s|ing)?|speak(?:s|ing)?|shout(?:s|ing)?|whisper(?:s|ing)?|'
                    r'mutter(?:s|ing)?|tell(?:s|ing)?|respond(?:s|ing)?|reply|replies|scream(?:s|ing)?|'
                    r'calls?|cries?|cry(?:ing)?|grunt(?:s|ing)?|breath(?:es|ing)?|utter(?:s|ing)?|'
                    r'exclaim(?:s|ing)?)\s+(?:\w+\s+){0,4}?in\s+(?:his|her|their|the)?\s*'
                    r'(?:native\s+(?:language|tongue)|mother\s+tongue|'
                    r'french|german|italian|spanish|portuguese|russian|arabic|hindi|thai|'
                    r'vietnamese|indonesian|malay|tagalog|filipino|turkish|persian|farsi|'
                    r'swedish|dutch|polish|greek|hebrew|ukrainian|czech|hungarian|romanian|'
                    r'mandarin|cantonese|japanese|korean)\b'
                    r'|'
                    r'\bin\s+(?:his|her|their|the)?\s*(?:native\s+(?:language|tongue)|mother\s+tongue)\b',
                    re.IGNORECASE
                )
                _lang_request_match = _EXPLICIT_LANG_REQUEST_RE.search(_combined_input)

                # Also detect a bare "in [language]" phrasing close to a dialogue verb
                # e.g. "an angry German man says in German" or "she whispers in French"
                _BARE_LANG_RE = re.compile(
                    r'\bin\s+(french|german|italian|spanish|portuguese|russian|arabic|hindi|thai|'
                    r'vietnamese|indonesian|malay|tagalog|filipino|turkish|persian|farsi|'
                    r'swedish|dutch|polish|greek|hebrew|ukrainian|czech|hungarian|romanian|'
                    r'mandarin|cantonese|japanese|korean)\b',
                    re.IGNORECASE
                )
                _bare_lang_match = _BARE_LANG_RE.search(_combined_input)

                # Resolve requested language name
                _requested_lang = None
                if _lang_request_match or _bare_lang_match:
                    _src = (_lang_request_match or _bare_lang_match).group(0).lower()
                    _LANG_NAME_MAP = {
                        "french": "French", "german": "German", "italian": "Italian",
                        "spanish": "Spanish", "portuguese": "Portuguese", "russian": "Russian",
                        "arabic": "Arabic", "hindi": "Hindi", "thai": "Thai",
                        "vietnamese": "Vietnamese", "indonesian": "Indonesian", "malay": "Malay",
                        "tagalog": "Filipino", "filipino": "Filipino", "turkish": "Turkish",
                        "persian": "Persian", "farsi": "Persian", "swedish": "Swedish",
                        "dutch": "Dutch", "polish": "Polish", "greek": "Greek",
                        "hebrew": "Hebrew", "ukrainian": "Ukrainian", "czech": "Czech",
                        "hungarian": "Hungarian", "romanian": "Romanian",
                        "mandarin": "Mandarin", "cantonese": "Cantonese",
                        "japanese": "Japanese", "korean": "Korean",
                    }
                    for key, val in _LANG_NAME_MAP.items():
                        if key in _src:
                            _requested_lang = val
                            break
                    # "native language/tongue" — no specific language named, infer from character
                    if not _requested_lang and ("native" in _src or "mother" in _src):
                        _requested_lang = "_infer"

                # Romanisation note for non-Latin script languages
                _NON_LATIN_SCRIPTS = {
                    "Japanese":  "Japanese characters (kanji/hiragana/katakana)",
                    "Korean":    "Korean hangul characters (한글)",
                    "Mandarin":  "Chinese characters (hanzi/simplified)",
                    "Cantonese": "Chinese characters (traditional/simplified)",
                    "Arabic":    "Arabic script (العربية)",
                    "Hindi":     "Devanagari script (देवनागरी)",
                    "Thai":      "Thai script (ภาษาไทย)",
                    "Persian":   "Persian/Farsi script (فارسی)",
                    "Russian":   "Cyrillic script (кириллица)",
                    "Greek":     "Greek script (ελληνικά)",
                    "Hebrew":    "Hebrew script (עברית)",
                    "Ukrainian": "Cyrillic script (кирилиця)",
                }
                _gen_roman_note = ""
                if _requested_lang in _NON_LATIN_SCRIPTS:
                    _script_name = _NON_LATIN_SCRIPTS[_requested_lang]
                    _gen_roman_note = (
                        f"CRITICAL: Write the dialogue in actual {_script_name} — NOT romanisation only. "
                        f"Do NOT put romanisation in parentheses next to the dialogue — parenthetical text "
                        f"renders as on-screen subtitles in the video. Write native script characters only, "
                        f"inline in the prose with no brackets alongside. "
                        f"CORRECT: She whispers «Где ты?», voice low and urgent. "
                        f"WRONG: She whispers «Где ты?» (Gde ty?). "
                    )
                elif _requested_lang and _requested_lang != "_infer":
                    _gen_roman_note = (
                        f"Write the {_requested_lang} dialogue in the native language only — "
                        f"no romanisation needed. "
                    )
                elif _requested_lang == "_infer":
                    _gen_roman_note = (
                        "If the character's native language uses a non-Latin script, write native script "
                        "characters only — no romanisation in parentheses as this renders as on-screen subtitles. "
                        "If it uses a Latin script, write the native language only with no romanisation. "
                    )

                # Language addendum — only appended when user explicitly requested it
                if _requested_lang == "_infer":
                    _lang_addendum = (
                        "\nLANGUAGE NOTE: The user has asked for dialogue in the character's native language. "
                        "Identify the character's nationality or ethnicity from the scene description "
                        "and write the relevant dialogue in that language. "
                        "If no clear nationality is stated, use the language most consistent with the scene's context. "
                        + _gen_roman_note
                    )
                elif _requested_lang:
                    _lang_addendum = (
                        f"\nLANGUAGE NOTE: The user has asked for dialogue in {_requested_lang}. "
                        f"Write those specific lines in {_requested_lang} exactly as requested. "
                        f"Other dialogue in the scene (if any) can remain in English. "
                        + _gen_roman_note
                    )
                else:
                    _lang_addendum = ""

                # Scene type detection + phrase bank routing
                # AUTO-DETECT: breakup, argument, affirmation, seduction, power_dynamic
                # EXPLICIT TRIGGER: asmr, dirty_talk, monologue need deliberate signal
                #   ASMR: "asmr"/"soft spoken"/"ear to ear"/"tingle"
                #   DIRTY TALK: "dirty talk"/"talks dirty"/"whispers dirty" etc
                #   MONOLOGUE: "monologue"/"talks to camera"/"confesses to" etc

                import re as _sre
                _ci_s = _combined_input.lower()

                _trigger_asmr = bool(_sre.search(
                    r"\b(asmr|soft\s+spoken|ear\s+to\s+ear|tingle|tingles|"
                    r"whispers?\s+softly|gentle\s+whisper|close\s+to\s+the\s+ear)\b",
                    _ci_s))
                _trigger_dirty_talk = bool(_sre.search(
                    r"\b(dirty|filthy|nasty|raunchy)\s+(talk|talks|talking|words|things)\b"
                    r"|\b(talk|talks|says?|whispers?|tells?|speaks?|mutters?)\s+(dirty|filthy|nasty|raunchy)\b"
                    r"|\b(dirty\s+talk|talk\s+dirty|filthy\s+talk|sex\s+talk|erotic\s+talk)\b",
                    _ci_s))
                _trigger_monologue = bool(_sre.search(
                    r"\b(monologue|talks?\s+to\s+the\s+camera|speaks?\s+to\s+the\s+camera|"
                    r"addresses\s+the\s+camera|narrates?\s+to|confesses?\s+to|"
                    r"breaks?\s+the\s+fourth\s+wall|speaks?\s+directly\s+to)\b",
                    _ci_s))
                _detect_breakup = bool(_sre.search(
                    r"\b(break\s*up|breaking\s*up|broke\s*up|its?\s+over|"
                    r"were?\s+(done|finished|through|over)|leaving\s+(you|him|her|them)|"
                    r"we\s+cant?\s+do\s+this|goodbye\s+forever|"
                    r"we\s+shouldnt?|this\s+was\s+a\s+mistake|ending\s+the\s+relationship|"
                    r"last\s+time\s+together|parting\s+ways?)\b",
                    _ci_s))
                _detect_argument = bool(_sre.search(
                    r"\b(argument|arguing|argue|confrontation|confronts?|"
                    r"screaming\s+at|shouting\s+at|yelling\s+at|"
                    r"heated\s+(exchange|argument|debate)|"
                    r"they\s+(fight|argue|clash|confront)|"
                    r"calls?\s+(him|her|them)\s+out|stands?\s+up\s+to)\b",
                    _ci_s))
                _detect_affirmation = bool(_sre.search(
                    r"\b(reassures?|comforts?\s+(her|him|them)|"
                    r"youre?\s+(enough|worthy|loved|seen)|"
                    r"i\s+(believe\s+in|see|love)\s+you\b|"
                    r"you\s+(matter|deserve\s+better|are\s+enough)|"
                    r"after\s+(crying|a\s+breakdown|bad\s+day|hard\s+day)|"
                    r"holds?\s+(her|him|them)\s+while\s+(she|he|they)\s+cr)\b",
                    _ci_s))
                _detect_seduction = bool(_sre.search(
                    r"\b(seduces?|seduction|flirts?\b|flirting|comes?\s+on\s+to|"
                    r"hitting\s+on|tries?\s+to\s+(seduce|attract)|leans?\s+in\s+close|"
                    r"electric\s+tension|charged\s+(moment|atmosphere|air)|"
                    r"undeniable\s+(attraction|chemistry|tension)|"
                    r"they\s+(circle|orbit)\s+each\s+other)\b",
                    _ci_s))
                _detect_power = bool(_sre.search(
                    r"\b(boss\s+and\s+(employee|secretary|assistant)|"
                    r"teacher\s+and\s+student|professor\s+and|"
                    r"forbidden\s+(attraction|love|desire|relationship)|"
                    r"office\s+(affair|romance|tension)|power\s+(play|dynamic)|"
                    r"authority\s+figure|they\s+shouldnt?\s+be\s+doing\s+this)\b",
                    _ci_s))

                _has_speech_verb = bool(_sre.search(
                    r"\b(says?|tells?|whispers?|speaks?|mutters?|moans?|breathes?|gasps?)\b",
                    _ci_s))

                if _trigger_asmr:
                    _scene_register = "asmr"
                elif _trigger_dirty_talk or (is_explicit and _has_speech_verb):
                    _scene_register = "dirty_talk"
                elif _trigger_monologue:
                    _scene_register = "monologue"
                elif _detect_breakup:
                    _scene_register = "breakup"
                elif _detect_argument or _is_tense:
                    _scene_register = "tension_argument"
                elif _detect_affirmation:
                    _scene_register = "affirmation"
                elif _detect_seduction:
                    _scene_register = "seduction"
                elif _detect_power:
                    _scene_register = "power_dynamic"
                elif _is_tender:
                    _scene_register = "tender"
                elif is_explicit or is_sensual:
                    _scene_register = "direct_explicit"
                elif _is_casual:
                    _scene_register = None
                else:
                    _scene_register = None

                _dlg_phrase_anchor = ""
                if _scene_register:
                    try:
                        from lyric_phrase_bank import _LYRIC_PHRASE_BANK
                        import random as _dpb_rnd
                        _dpb_rng = _dpb_rnd.Random((seed + 41) if seed != -1 else None)
                        _dpb_reg = _LYRIC_PHRASE_BANK.get(_scene_register, {})
                        if _dpb_reg:
                            _dpb_open  = _dpb_rng.choice(_dpb_reg.get("opening",    ["..."]))
                            _dpb_build = _dpb_rng.choice(_dpb_reg.get("build",      ["..."]))
                            _dpb_peak  = _dpb_rng.choice(_dpb_reg.get("peak",       ["..."]))
                            _dpb_res   = _dpb_rng.choice(_dpb_reg.get("resolution", ["..."]))
                            _dlg_phrase_anchor = (
                                "\n\nSCENE REGISTER — " + _scene_register.replace("_", " ").upper() + ": "
                                "Use these as emotional seeds for the dialogue tone. "
                                "\nOpening tone: \"" + _dpb_open + "\" "
                                "\nBuilding tone: \"" + _dpb_build + "\" "
                                "\nPeak tone: \"" + _dpb_peak + "\" "
                                "\nResolution tone: \"" + _dpb_res + "\""
                            )
                            print(f"[LTX2-Qwen] Scene register: {_scene_register} | {_dpb_open[:45]}")
                    except ImportError:
                        print("[LTX2-Qwen] lyric_phrase_bank not found")

                if _scene_register == "asmr":
                    _dlg_tone = (
                        "ASMR scene — whispered, intimate, close-mic proximity. "
                        "Every word barely above silence. Breathing audible. "
                        "Voice is the whole point — texture, warmth, deliberate slowness. No urgency.")
                elif _scene_register == "dirty_talk":
                    _dlg_tone = (
                        "Explicit verbal during physical action — short, direct, urgent or commanding. "
                        "Lines are demands, sensation descriptions, or raw reactions. "
                        "Nothing softened. Register: raw, breathless, present-tense.")
                elif _scene_register == "monologue":
                    _dlg_tone = (
                        "Direct-to-camera confessional. One person speaking their interior truth. "
                        "Honest, unguarded, specific. No deflection, no performance. "
                        "Register: confessional, direct, exposed.")
                elif _scene_register == "breakup":
                    _dlg_tone = (
                        "Ending — the specific language of letting go. "
                        "Every word carries weight. Careful, honest, final. "
                        "No anger without love underneath. Register: honest, heavy, tender.")
                elif _scene_register == "tension_argument":
                    _dlg_tone = (
                        "Charged and clipped — short sentences, pressure in every word. "
                        "Silences are loaded. The argument is about more than what it is about. "
                        "Register: terse, electric, honest underneath the defensive.")
                elif _scene_register == "affirmation":
                    _dlg_tone = (
                        "The thing that needed to be said — specific, direct, completely meant. "
                        "No deflection, no irony. The listener cannot easily dismiss it. "
                        "Register: clear, warm, unwavering.")
                elif _scene_register == "seduction":
                    _dlg_tone = (
                        "Charged with subtext — everything said means something more. "
                        "Lines imply rather than state. The gap between words is where tension lives. "
                        "Register: loaded, deliberate, electric.")
                elif _scene_register == "power_dynamic":
                    _dlg_tone = (
                        "Carries the weight of unequal ground — authority, surrender, what it costs both. "
                        "Lines are careful on both sides but the undertow is obvious. "
                        "Register: controlled, charged, aware of what is at stake.")
                elif _scene_register == "tender":
                    _dlg_tone = (
                        "Soft, unguarded, emotionally specific — only said when completely meant. "
                        "No performance, no deflection. Every word costs something. "
                        "Register: quiet, open, completely present.")
                elif _scene_register == "direct_explicit":
                    _dlg_tone = (
                        "Direct and physical, grounded in what is happening. "
                        "Short, urgent, or breathless. No literary flourish. "
                        "Register: immediate, honest, unfiltered.")
                elif _is_athletic:
                    _dlg_tone = (
                        "Sparse — short commands, exertion sounds, brief focus cues. "
                        "Words land between breaths. Register: minimal, physical, driven.")
                elif _is_casual:
                    _dlg_tone = (
                        "Real conversation — unpolished, natural, half-sentences and overlaps. "
                        "Register: loose, human, unscripted.")
                else:
                    _dlg_tone = (
                        "Specific to this exact moment — what would this person actually say "
                        "right now in this situation? No generic lines. "
                        "Ground every word in what is physically happening.")

                dialogue_instruction = (
                    "\n\n[DIALOGUE INSTRUCTION — MANDATORY, CANNOT BE SKIPPED: "
                    "Include at least TWO lines of spoken dialogue, spaced across the scene. "
                    "Each line woven into a physical beat with attribution and delivery note. "
                    "Lines must vary in register or emotional weight. "
                    "Do NOT write generic filler. Every line reveals character, intention, or state. "
                    + _dlg_tone
                    + _dlg_phrase_anchor
                    + " Dialogue MUST be grounded in what is visibly happening. "
                    "Do NOT invent backstory not in the user input. "
                    + _lang_addendum
                    + music_sound_rule                         + _fmt_adlib_injection(
                            _get_adlibs("seduction", "universal", seed, 3), "dialogue")
+ "]"
                )
        else:
            dialogue_instruction = (
                "\n\n[DIALOGUE INSTRUCTION: No dialogue in this scene. No spoken words. "
                "Weave sound naturally into the prose instead. "
                + music_sound_rule + "]"
            )


        # ── Global emotional state instruction ──────────────────────────────
        # Fires on ANY prompt when the dropdown is set or a keyword is detected.
        # Changes face, posture, eye contact, breath, hands, spatial presence.
        # Never states the emotion by name — always shows it physically.

        _EMOTION_STATES = {
            "grief": (
                "EMOTIONAL STATE: Her face is doing the work of holding something together that has already "
                "broken. The eyes are slightly swollen, red-rimmed but not actively crying. Her jaw is loose, "
                "the muscles exhausted from the effort of composure. She moves as if gravity has increased. "
                "Her hands find surfaces to hold. Eye contact comes in flashes — she gives it then looks away, "
                "unable to sustain it."
            ),
            "crying": (
                "EMOTIONAL STATE: Tears are present and tracking — single lines from the outer corners of her "
                "eyes, catching light before dropping from the jaw. Her breath catches on the inhale, uneven. "
                "Between breaths she tries to compose herself — chin lifting, jaw tightening — but the feeling "
                "keeps breaking through. Her eyes are bright and glassy. When she looks at the lens the eye "
                "contact is brief and devastating."
            ),
            "sad": (
                "EMOTIONAL STATE: The sadness is in the quality of her stillness more than in movement. "
                "Her expression does not perform grief — it simply carries it, the face of someone who has "
                "stopped trying to look fine. The corners of her mouth are slightly weighted. Her eyes are "
                "soft and turned inward. Her posture is settled rather than collapsed — at rest with a feeling "
                "rather than fighting it."
            ),
            "shame": (
                "EMOTIONAL STATE: She cannot sustain eye contact with the lens. Her gaze drops and returns "
                "in a cycle she cannot break. Her posture curves inward at the top — not hunched but protective. "
                "Her hands touch her face, her throat, her arms — the body trying to make itself smaller, less "
                "seen. Her expression is tight with the effort of not showing exactly what is on her face."
            ),
            "longing": (
                "EMOTIONAL STATE: She is looking at something not in the frame — her gaze has a direction and "
                "destination even when nothing is there. Her expression is open in the way that unguarded wanting "
                "is open. She breathes slowly, the chest rising with something more than air. Her hands are quiet, "
                "fingers slightly curled as if remembering something."
            ),
            "exhausted": (
                "EMOTIONAL STATE: The tiredness is structural — in how she holds herself, the slight forward "
                "curve of the upper back, the weight in the eyelids. She moves with economy: only what is "
                "necessary. Her eyes are glassy, focus slightly soft. She breathes through her mouth slightly. "
                "When she is still there is relief in it. Her emotional reserves are gone — the expression is "
                "not sad but absent."
            ),
            "angry": (
                "EMOTIONAL STATE: The anger is in the body before the face. Her jaw is set hard, the muscles "
                "at the hinge visible. Shoulders pulled back and slightly elevated — the body making itself "
                "larger without knowing it. Her hands are very still or moving with sharp punctuated gestures — "
                "nothing wasted, nothing soft. Eye contact is unwavering and pressurised. Her breath comes "
                "through her nose. The stillness between movements is more threatening than the movements."
            ),
            "defiant": (
                "EMOTIONAL STATE: She has already decided. Weight forward, chin level, shoulders square — "
                "the decision shows in her posture before anything else. Her expression is not angry but resolved: "
                "something has been tested and she knows exactly where she stands. Eye contact is direct and held. "
                "Her hands are loose at her sides or planted on a surface. The defiance is quiet, which makes "
                "it heavier."
            ),
            "fierce": (
                "EMOTIONAL STATE: Eyes half-closed and completely internal — not performing to the camera but "
                "through it. Her expression sharpens on every stressed beat. The body and the emotion arrive "
                "at the same place simultaneously. Nothing is decorative. Her jaw, her hands, her stance — "
                "all of it aligned, nothing wasted."
            ),
            "predatory": (
                "EMOTIONAL STATE: She is completely still in a way that is not relaxed — the stillness of "
                "total attention. Her expression is neutral but her eyes are not: they track, assess, decide. "
                "When she moves it is fluid and unhurried — the body of something that knows it will arrive "
                "when it chooses. She does not perform awareness of the camera. She is aware of everything."
            ),
            "aroused": (
                "EMOTIONAL STATE: The want is in the body before it reaches the face. Her movements are slower "
                "and more deliberate than necessary — each one chosen. Her breath is deeper and more conscious, "
                "the chest rising more visibly. Her lips part between expressions without closing fully. Eye "
                "contact is sustained and weighted — she holds the lens longer than comfort requires, the look "
                "carrying something specific. Heat in the cheeks, the neck, the collarbone. She does not rush. "
                "The deliberateness is the point."
            ),
            "tender": (
                "EMOTIONAL STATE: The feeling is directed outward — at something or someone she keeps returning "
                "to. Her expression is open and unguarded, the face doing something it only does in private. "
                "Her movements are slow and considered. When she looks at the lens there is a warmth that is "
                "different from performance — the expression of someone genuinely glad something exists."
            ),
            "adoration": (
                "EMOTIONAL STATE: She is soft in a way that is entirely directed. Her eyes are fixed and warm, "
                "the focus the kind that does not scan. Her expression is open to the point of vulnerability — "
                "she is not protecting herself from this feeling. Her body orients toward its subject the way "
                "a plant orients toward light, without thinking about it."
            ),
            "euphoric": (
                "EMOTIONAL STATE: The joy is too big to contain and she is not trying. Her whole face "
                "participates — eyes crinkled, smile pulling unevenly, more genuine for it. Her body is light: "
                "weight shifts easily, movements generous and unguarded. She laughs or nearly laughs between "
                "moments. Eye contact is bright, open, inviting — she wants to share it."
            ),
            "happy": (
                "EMOTIONAL STATE: The feeling sits in her chest and keeps lifting her posture without effort. "
                "She smiles easily and it reaches her eyes — the corners crinkling, expression unguarded. "
                "Her movements are relaxed and open, occupying more space than usual. She looks at things with "
                "interest rather than wariness. The joy is not performative — it does not require an audience."
            ),
            "playful": (
                "EMOTIONAL STATE: There is something behind the expression she is not quite sharing — a thought, "
                "a secret, an awareness of the situation she finds amusing. Her smile is not symmetrical, one "
                "side pulling a little more. Her eyes are bright with it. She moves with deliberate looseness. "
                "She looks away and back — a rhythm that is almost conversational. The playfulness is directed "
                "at someone specific, even if that someone is the camera."
            ),
            "confident": (
                "EMOTIONAL STATE: She takes up the space she occupies as if she was always meant to be here. "
                "Her posture is easy rather than rigid — confidence does not perform itself. Movement is "
                "unhurried and deliberate. Eye contact is direct and comfortable — she holds it without "
                "challenge, without softness, just presence. Her hands are still when they are still. She "
                "exists in the frame as the natural centre of it."
            ),
            "proud": (
                "EMOTIONAL STATE: She stands differently — a millimetre more height, the spine longer, chin "
                "level. Her expression is quiet rather than broad: the satisfaction is internal and the face "
                "reflects it without performing it. Eye contact is easy and direct. Her hands are still. "
                "There is a quality of settled rightness about how she occupies the frame — someone who has "
                "arrived somewhere they worked to reach."
            ),
            "vulnerable": (
                "EMOTIONAL STATE: She is more open than she means to be. Her arms are held slightly away from "
                "her body — no armour, no barrier. Her face keeps almost shifting to something more composed "
                "before settling back into honesty. Eye contact is intermittent — she gives it fully, then "
                "has to look away, then gives it again. Her breath is slightly shallower, held in the upper chest."
            ),
            "nervous": (
                "EMOTIONAL STATE: The nervousness lives in the small movements — fingers that find each other, "
                "a weight shift that happens too often, a breath held and released too quickly. Her eyes move: "
                "checking, scanning, returning to the lens and moving away. Her posture wants to make itself "
                "smaller. When she is still she is very still. Her jaw is slightly tight, the muscles working quietly."
            ),
            "overcome": (
                "EMOTIONAL STATE: She is at the edge of composure and the effort of maintaining it is visible. "
                "Her chin drops fractionally and lifts again. Her jaw works. Her eyes are bright and the "
                "surface tension is doing everything — the feeling is right there, pressing against the "
                "inside of the expression. She is not crying but she is one thing away from it."
            ),
            "wired": (
                "EMOTIONAL STATE: There is too much happening inside and the body cannot quite contain it. "
                "She moves more than the situation requires — weight shifts, hands that find things to do, "
                "restlessness in the feet. Her eyes are very bright and very fast. She smiles or laughs at "
                "moments that do not fully earn it. The energy is electric, alive, slightly too much."
            ),
            "dissociated": (
                "EMOTIONAL STATE: She is present in the body but not in the moment. Her gaze does not quite "
                "focus — it rests on middle distance, moving occasionally without registering what it passes "
                "over. Her expression is neutral to the point of blank: not peaceful, simply elsewhere. "
                "Her body is still by default rather than by choice. When she blinks it is slow."
            ),
            "controlled": (
                "EMOTIONAL STATE: Her expression is locked — professional composure that costs her something. "
                "Only the hands betray it: pressing into a surface, knuckles whitening slightly, then "
                "releasing. The face gives nothing. The body gives everything, if you know where to look."
            ),
        }

        # Keyword detection fallback — fires when dropdown is None but user typed an emotion
        _EMOTION_KEYWORD_MAP = [
            (r"\b(grief|grieving|mourning|heartbroken|devastated|shattered)\b", "grief"),
            (r"\b(cr(?:y|ies|ying|ied)|weeping|sobbing|in tears|tearful)\b", "crying"),
            (r"\b(sad|sadness|unhappy|miserable|melancholy|depressed|dejected)\b", "sad"),
            (r"\b(shame|ashamed|humiliated|embarrassed|guilty)\b", "shame"),
            (r"\b(longing|yearning|pining|wistful|missing (?:him|her|them|you))\b", "longing"),
            (r"\b(exhausted|drained|spent|hollow|bone tired|wrecked|depleted)\b", "exhausted"),
            (r"\b(angry|anger|furious|fury|rage|livid|seething|irate|pissed off|fuming)\b", "angry"),
            (r"\b(defiant|defiance|fierce|standing (?:her|his|my) ground)\b", "defiant"),
            (r"\b(predatory|hunting|calculating|dangerous)\b", "predatory"),
            (r"\b(horny|aroused|turned on|lustful|needy|craving|in heat)\b", "aroused"),
            (r"\b(tender|caring|devoted|gentle with|soft for)\b", "tender"),
            (r"\b(adoring|adoration|worshipping)\b", "adoration"),
            (r"\b(euphoric|euphoria|ecstatic|elated|over the moon)\b", "euphoric"),
            (r"\b(happy|joyful|gleeful|delighted|cheerful|beaming|grinning)\b", "happy"),
            (r"\b(playful|teasing|mischievous|cheeky|coy)\b", "playful"),
            (r"\b(confident|confidence|commanding|assured|bold|fearless)\b", "confident"),
            (r"\b(proud|pride|accomplished|earned it|standing tall)\b", "proud"),
            (r"\b(vulnerable|exposed|raw|unguarded|fragile)\b", "vulnerable"),
            (r"\b(nervous|anxious|worried|apprehensive|scared|afraid|fearful|terrified)\b", "nervous"),
            (r"\b(overcome|holding it together|barely holding)\b", "overcome"),
            (r"\b(manic|wired|electric|buzzing|hyper|frantic)\b", "wired"),
            (r"\b(dissociated|vacant|checked out|absent|numb|detached|zoned out)\b", "dissociated"),
            (r"\b(controlled|composure|locked down)\b", "controlled"),
        ]

        _global_emotion_instruction = ""
        _emotion_key = None
        import re as _emre

        _es = emotional_state_sel.lower().strip()

        if "surprise" in _es:
            # Seed picks from the full pool
            _surprise_rng = __import__("random").Random((seed + 13) if seed != -1 else None)
            _emotion_key = _surprise_rng.choice(list(_EMOTION_STATES.keys()))
            print(f"[LTX2-Qwen] Emotional state: surprise -> {_emotion_key}")

        elif "none" not in _es and _es:
            # Map dropdown label to key
            _DROPDOWN_MAP = {
                "grief": "grief", "crying": "crying", "sad": "sad",
                "shame": "shame", "longing": "longing", "exhausted": "exhausted",
                "angry": "angry", "defiant": "defiant", "fierce": "fierce",
                "predatory": "predatory", "aroused": "aroused", "tender": "tender",
                "adoration": "adoration", "euphoric": "euphoric", "happy": "happy",
                "playful": "playful", "confident": "confident", "proud": "proud",
                "vulnerable": "vulnerable", "nervous": "nervous", "overcome": "overcome",
                "wired": "wired", "dissociated": "dissociated", "controlled": "controlled",
            }
            for _key in _DROPDOWN_MAP:
                if _key in _es:
                    _emotion_key = _DROPDOWN_MAP[_key]
                    print(f"[LTX2-Qwen] Emotional state: dropdown -> {_emotion_key}")
                    break

        # Keyword fallback — fires even when dropdown is None
        if _emotion_key is None and "none" in _es:
            for _kw_pat, _kw_key in _EMOTION_KEYWORD_MAP:
                if _emre.search(_kw_pat, _combined_input, _emre.IGNORECASE):
                    _emotion_key = _kw_key
                    print(f"[LTX2-Qwen] Emotional state: keyword detected -> {_emotion_key}")
                    break

        if _emotion_key and _emotion_key in _EMOTION_STATES:
            # Genre-emotion modulation — certain combinations need specific flavour
            import re as _egre
            _eg_lower = _combined_input.lower()
            _emotion_genre_note = ""

            if _emotion_key == "aroused":
                if _egre.search(r'\b(k.?pop|j.?pop|idol|city\s+pop)\b', _eg_lower):
                    _emotion_genre_note = (
                        "GENRE MODULATION — K-POP HEAT: In this genre, arousal is not explicit — "
                        "it is controlled, performance-aware electricity. The body knows it is being watched "
                        "and uses that. The heat is in the precision: every movement calculated, "
                        "eye contact held a half-second past comfort before releasing, "
                        "the performance discipline making the underlying want more charged, not less. "
                    )
                elif _egre.search(r'\b(jazz|blues|neo.?soul|r&b|soul)\b', _eg_lower):
                    _emotion_genre_note = (
                        "GENRE MODULATION — LATE NIGHT INTIMACY: In this genre, arousal is slow and adult. "
                        "The room is small, the light is low. The want is unhurried — "
                        "there is no performance of it, just the fact of it. "
                        "Her body moves less, not more. The stillness carries more heat than movement would. "
                    )
                elif _egre.search(r'\b(rock|punk|metal|grunge)\b', _eg_lower):
                    _emotion_genre_note = (
                        "GENRE MODULATION — STAGE HEAT: In this genre, arousal is physical and confrontational. "
                        "The energy is not soft — it is the kind that comes from volume and adrenaline. "
                        "Her body pushes forward, the performance aggressive and claiming. "
                        "The want is a challenge, not an invitation. "
                    )
                elif _egre.search(r'\b(opera|classical|theatrical)\b', _eg_lower):
                    _emotion_genre_note = (
                        "GENRE MODULATION — OPERATIC DESIRE: Desire here is grand and unashamed. "
                        "The body is an instrument being played for its own sake. "
                        "The voice and the want arrive at the same place simultaneously. "
                    )
                elif _egre.search(r'\b(disco|funk|dance)\b', _eg_lower):
                    _emotion_genre_note = (
                        "GENRE MODULATION — FLOOR HEAT: In this genre, arousal is communal and physical. "
                        "The body moves for the pleasure of moving. The heat is in the rhythm — "
                        "she is not waiting for something, she is already in it. "
                    )

            elif _emotion_key == "defiant":
                if _egre.search(r'\b(k.?pop|j.?pop|idol)\b', _eg_lower):
                    _emotion_genre_note = (
                        "GENRE MODULATION: In K-pop, defiance is choreographed — anger delivered with precision. "
                        "The body does not lose its training even when the feeling overwhelms. "
                        "The defiance shows in the jaw, the locked eyes, the snap of each movement — "
                        "never in a loss of control. "
                    )
                elif _egre.search(r'\b(blues|soul|gospel)\b', _eg_lower):
                    _emotion_genre_note = (
                        "GENRE MODULATION: In this tradition, defiance is earned and ancient. "
                        "Not performed anger — the quiet immovable kind that comes from having survived "
                        "something and deciding not to apologise for it. "
                    )

            elif _emotion_key == "tender":
                if _egre.search(r'\b(k.?pop|j.?pop|idol)\b', _eg_lower):
                    _emotion_genre_note = (
                        "GENRE MODULATION: In this genre, tenderness is fan-facing — "
                        "the warmth is directed at many people simultaneously but feels personal to each. "
                        "The expression is open and inclusive rather than directed at one person. "
                    )
                elif _egre.search(r'\b(folk|country|americana)\b', _eg_lower):
                    _emotion_genre_note = (
                        "GENRE MODULATION: In this tradition, tenderness is domestic and specific — "
                        "directed at a particular person or place. The warmth is earned and familiar. "
                    )

            _global_emotion_instruction = (
                f"\n\n[{_EMOTION_STATES[_emotion_key]} "
                + _emotion_genre_note
                + "Apply this emotional state to every aspect of the scene — "
                "posture, eye contact, breath, hands, how the body occupies space, "
                "facial micro-expressions. The emotion colours everything. "
                "Do NOT name the emotion in the output — show it physically.]"
            )

        # ── Physical exertion / sweat instruction ────────────────────────────
        # Fires globally (not just singing) when the scene involves physical exertion.
        # Adds sweat detail and clothing condition changes that LTX can render.
        # Suppressed if: user already described sweat/condition, or _is_singing
        # (singing block handles its own sweat separately).
        _exertion_instruction = ""
        _user_described_sweat = bool(re.search(
            r'\b(sweat(?:ing|y|ed|s)?|perspir\w*|drench\w*|soaked|damp|glistening|'
            r'gleaming|sheen|flush(?:ed)?|hot\s+and\s+bothered|out\s+of\s+breath|'
            r'breathless|panting|heaving|exhausted|winded)\b',
            _combined_input, re.IGNORECASE
        ))

        _exertion_detected = bool(re.search(
            r'\b(jog(?:ging|s)?|run(?:ning|s)?|sprint(?:ing|s)?|'
            r'danc(?:ing|es?)|work(?:ing)?\s+out|workout|training|'
            r'gym|exercise|cardio|boxing|fight(?:ing)?|climb(?:ing|s)?|'
            r'hik(?:ing|es?)|cycl(?:ing|es?)|row(?:ing|s)?|'
            r'push(?:-?up|ing)?|squat(?:ting|s)?|lunge\w*|'
            r'sweat(?:ing|s|y)?|breathless|panting|exhausted|winded|'
            r'out\s+of\s+breath|physical\s+effort)\b',
            _combined_input, re.IGNORECASE
        ))

        # Detect intensity level from input
        _exertion_intense = bool(re.search(
            r'\b(sprint(?:ing|s)?|racing?|hard|intense|heavy|full\s+speed|'
            r'flat\s+out|all\s+out|maximum|pushing\s+hard|really\s+sweating|'
            r'soaked|drench\w*|sweat(?:ing)?\s+a\s+lot|sweat(?:ing)?\s+hard|'
            r'sweat(?:ing)?\s+(?:profuse|heavy|heavy\s+)\w*|'
            r'pour(?:ing)?\s+(?:with\s+)?sweat)\b',
            _combined_input, re.IGNORECASE
        ))

        if _exertion_detected and not _is_singing and not _user_described_sweat:
            _exertion_rng = __import__('random').Random((seed + 11) if seed != -1 else None)

            if _exertion_intense:
                _EXERTION_HEAVY = [
                    "EXERTION STATE — HEAVY: She is visibly soaked — sweat has darkened her clothing "
                    "at the back, between the shoulders, under the arms, and across the chest. "
                    "The fabric clings to her skin where it is wet, the texture of the material "
                    "changing where moisture has saturated it. Her hair is plastered to her temples "
                    "and neck. Sweat is actively beading at her hairline and tracking down her face.",

                    "EXERTION STATE — HEAVY: Her clothes are wet through — the fabric translucent "
                    "with moisture where it presses against her back and chest, the colour deepened "
                    "by sweat. Individual beads track from her hairline, one running from her temple "
                    "to the line of her jaw. Her breathing is audible and heavy, chest rising visibly "
                    "with each intake.",

                    "EXERTION STATE — HEAVY: The sweat is total — clothing darkened and clinging, "
                    "skin gleaming under the light, hair wet at the edges. A bead tracks from her "
                    "forehead to the tip of her nose. Her face is flushed, the cheeks carrying "
                    "a deep red, the exertion written completely across her body.",
                ]
                _exertion_instruction = (
                    "\n[" + _exertion_rng.choice(_EXERTION_HEAVY) +
                    " Apply this condition consistently from the first frame — "
                    "the clothing is already wet, the sweat already present. "
                    "This is NOT a gradual development — she is already at this state.]"
                )
            else:
                _EXERTION_MODERATE = [
                    "EXERTION STATE — MODERATE: A visible sheen of sweat on her forehead, "
                    "upper lip, and collarbone. Her clothing is beginning to cling at the back "
                    "and under the arms — the fabric slightly darker where moisture is gathering. "
                    "Her cheeks carry a real flush, not makeup. Her breathing is elevated.",

                    "EXERTION STATE — MODERATE: Sweat is visible on her skin — a fine mist "
                    "across her forehead, a bead at her left temple. Her clothing shows "
                    "the beginning of moisture saturation at the back and chest, "
                    "the fabric texture changing subtly where it touches damp skin. "
                    "Her face is flushed and her breath is audible.",

                    "EXERTION STATE — MODERATE: The exertion shows in her skin — "
                    "a high shine on her forehead and upper lip, cheeks deeply flushed. "
                    "Her clothing has darkened slightly at the back between the shoulder blades. "
                    "Her hair catches to her face at the temples with moisture.",
                ]
                _exertion_instruction = (
                    "\n[" + _exertion_rng.choice(_EXERTION_MODERATE) +
                    " This condition is present throughout the clip — not building from nothing.]"
                )
            print(f"[LTX2-Qwen] Exertion: {'heavy' if _exertion_intense else 'moderate'} sweat fired")

        elif _exertion_detected and not _is_singing and _user_described_sweat:
            # User described the condition — just tell LLM to apply it physically
            _exertion_instruction = (
                "\n[EXERTION STATE: The user has described a physical condition "
                "(sweat, breathlessness, flush etc). Apply it with full visual specificity — "
                "clothing texture changes, skin condition, breathing visible in the body. "
                "The condition is present from the first frame.]"
            )

        # ── Lift instruction ──────────────────────────────────────────────────
        if has_lift:
            # Detect which garment is being lifted to write the correct steps
            _lift_garment_match = re.search(
                r'\b(skirt|dress)\b', _combined_input, re.IGNORECASE
            )
            if _lift_garment_match:
                _lift_garment = _lift_garment_match.group(1).lower()
                lift_instruction = (
                    f"\n\n[{_lift_garment.upper()} LIFT SEQUENCE — MANDATORY, one sentence per step: "
                    f"1. Her fingers grip the hem of her {_lift_garment} at mid-thigh. "
                    f"2. She gathers the fabric upward. "
                    f"3. The {_lift_garment} rises past mid-thigh. "
                    f"4. The fabric clears her upper thighs. "
                    f"5. Her hips and underwear/skin come into view. "
                    f"6. The {_lift_garment} is held bunched at her waist. "
                    f"The {_lift_garment} STAYS LIFTED. Do NOT write her lowering it or covering herself unless asked.]"
                )
            else:
                # Default: shirt/top/crop lift → chest reveal
                lift_instruction = (
                    "\n\n[SHIRT LIFT SEQUENCE — MANDATORY, one sentence per step: "
                    "1. Her fingers find the hem at the waist. "
                    "2. She grips the fabric and begins gathering it upward. "
                    "3. The shirt rises past her stomach. "
                    "4. The fabric passes her navel, exposing her bare midriff. "
                    "5. The shirt climbs past her ribs. "
                    "6. Her chest comes into view. "
                    "7. Her breasts are fully exposed, the shirt held up. "
                    "The shirt STAYS LIFTED. Do NOT write her lowering it or covering herself unless asked.]"
                )
        else:
            lift_instruction = ""

        # ── Physical action sequences ─────────────────────────────────────────
        # Detect specific actions that need mechanical step-by-step breakdowns.
        # "she twerks" → garbage. Precise physical steps → correct generation.
        # Only fires on is_sensual or is_explicit to avoid misfiring on casual scenes.
        _action_key = None
        if bool(self._ACTION_SEQUENCE_RE.search(_combined_input)):
            for _act_re, _act_key in self._ACTION_KEY_MAP:
                if _act_re.search(_combined_input):
                    _action_key = _act_key
                    break

        # ── Action sequence variation ─────────────────────────────────────────
        # Seed-driven RNG picks one of three expression tiers per run.
        # raw=aggressive/high energy | sensual=slow/intimate | playful=teasing/performative
        # MECHANICS (orientation, core steps, continuity) are identical across all tiers.
        # What varies: camera framing, upper body language, movement description, energy state.
        # Same seed = same tier. Random seed (-1) = genuine variety run to run.
        _seq_rng  = random.Random((seed + 1) if seed != -1 else None)
        _seq_tier = _seq_rng.choice(["raw", "sensual", "playful"])
        print(f"[LTX2-Qwen] Action sequence tier: {_seq_tier}")

        # ── POV detection ─────────────────────────────────────────────────────
        # True when the user selected a POV/first-person style preset OR typed
        # pov/first-person in their input. When true the action sequence swaps
        # third-person camera language for first-person body-in-frame language:
        # the viewer's legs, hands, and knees appear in the lower frame.
        _is_pov = (
            "pov" in style_preset.lower() or
            "first person" in style_preset.lower() or
            bool(re.search(r'(pov|first[- ]person|point[- ]of[- ]view)', _combined_input, re.IGNORECASE))
        )
        if _is_pov:
            print("[LTX2-Qwen] POV mode active — body-in-frame language enabled")

        # Per-action tier notes — (camera_note, energy_note, movement_flavour)
        # Appended to the core mechanical steps at build time.
        _SEQ_TIER_NOTES = {
            "twerk": {
                # All three tiers use LoRA-matched caption language — soft, visual, warm.
                # The LoRA was trained on this exact register. Do not use biomechanical
                # or aggressive language here — it breaks generation.
                "raw":     ("CAMERA: Static, low-angle rear view — lens below hip height, centred on her rear.",
                            "The movement is a rhythmic, energetic shake — hips snapping up and down with a fluid, bouncing tempo.",
                            "The skin ripples with every thrust, the cheeks compressing and separating, the fabric of her clothing stretching taut and riding up slightly with each upward snap."),
                "sensual": ("CAMERA: Static, low-angle rear view — lens at hip height, framing lower back to mid-thigh.",
                            "The movement is slow and hypnotic — hips rolling in a deep, languid bounce, each drop drawing out before the rebound.",
                            "The skin catches the warm light with a soft sheen, the fabric of her clothing dipping deep into the cleft as she moves, the cheeks parting and pressing together in a fluid, unhurried rhythm."),
                "playful": ("CAMERA: Static, low-angle rear view — lens at hip height, slight space around the body.",
                            "The movement is rhythmic and confident — a bouncing, shaking motion with natural variation in tempo.",
                            "She shifts between faster, lighter pulses and deeper slower drops, the skin bouncing with each change of pace, the fabric of her clothing stretching across her curves with every movement."),
            },
            "body_roll": {
                "raw":     ("CAMERA: Medium frame — lens at chest height, framing hips to crown. Static or very slow push-in.",
                            "The wave is deliberate and controlled — chest pressing forward, stomach drawing in, hips rolling through in a smooth continuous motion.",
                            "Each pass is slightly deeper than the last, the body finding its rhythm, the fabric of her top pulling taut across her torso as the wave travels through."),
                "sensual": ("CAMERA: Medium wide frame — lens at hip height, full torso visible from hips to crown, slow push-in.",
                            "The wave is slow and fluid — barely perceptible at first, building gradually, the body becoming a single continuous curve.",
                            "The chest leads gently, the stomach softens through the arc, the hips follow in a long unhurried close — the whole roll one sustained, flowing motion."),
                "playful": ("CAMERA: Medium frame — lens at chest height, framing hips to crown, slight 3/4 angle.",
                            "The wave is natural and easy — the body moving loosely, the roll flowing without effort.",
                            "She lets her arms move softly with the wave, the movement relaxed and organic, repeating fluidly without pause."),
            },
            "grind": {
                "raw":     ("CAMERA: Tight side angle — lens at hip height behind and to one side, framing the contact point.",
                            "ENERGY: Intense and dominant — she owns the movement, every circle deliberate and deep.",
                            "She drives her hips back into him on every arc, the contact firm, the rhythm locked to the bass."),
                "sensual": ("CAMERA: Rear three-quarter — lens at hip height, wide enough to frame both bodies fully.",
                            "ENERGY: Slow and teasing — the circles are wide and unhurried, each rotation a full deliberate orbit.",
                            "She barely grazes him on the forward arc, then presses fully on the back arc — the variation is the tension."),
                "playful": ("CAMERA: Side medium — lens at hip height, both people in frame.",
                            "ENERGY: Playful and rhythmic — she rides the beat loosely, varying between tight fast circles and slow rolling drops.",
                            "She glances back over her shoulder once — a half-smile, eye contact for a beat — then turns away and deepens the roll."),
            },
            "lap_dance": {
                "raw":     ("CAMERA: Front medium — lens at seated eye level, both bodies fully in frame.",
                            "ENERGY: Controlled and intense — every movement is deliberate, no wasted motion.",
                            "She works the grind deep and steady, weight pressed down, the contact unbroken throughout."),
                "sensual": ("CAMERA: Front medium — lens at seated eye level, slight low angle.",
                            "ENERGY: Slow and intimate — the whole sequence feels private, like it exists only for him.",
                            "She moves at half the tempo the music suggests — slower than expected, the restraint itself the provocation."),
                "playful": ("CAMERA: Front medium — lens at seated eye level.",
                            "ENERGY: Confident and teasing — she is performing and knows it, enjoying the control.",
                            "She rises almost fully off his lap at the peak of each roll, hovering, then sinks back slowly — the threat of withdrawal is the point."),
            },
            "pole": {
                "raw":     ("CAMERA: Wide static — lens at floor level, full pole height visible, she fills the left third of frame.",
                            "ENERGY: Athletic and committed — every grip is tight, every movement driven by momentum.",
                            "She hits the spin hard — the launch is explosive, the extension at full stretch, the descent fast and controlled."),
                "sensual": ("CAMERA: Wide slow orbit — lens at mid-height, camera tracks around the pole at walking pace.",
                            "ENERGY: Slow and deliberate — every transition is drawn out, the body finding each pose before releasing into the next.",
                            "She descends the spiral slowly — controlling each inch, the slide unhurried, legs closing in one long smooth line."),
                "playful": ("CAMERA: Wide static — lens at waist height, full body visible.",
                            "ENERGY: Easy and confident — the moves flow from muscle memory, her attention on the room not the pole.",
                            "She holds the straddle longer than needed — arms spreading wide, head dropping back — milking the pose before finishing the descent."),
            },
            "hair_flip": {
                "raw":     ("CAMERA: Medium close — lens at eye level, framing chest to crown, tight.",
                            "ENERGY: Sharp and sudden — the flip is a single explosive movement with no warmup.",
                            "The whip is aggressive — the head snaps back with full force, the hair a single violent arc before it settles."),
                "sensual": ("CAMERA: Medium close — lens slightly below eye level, framing collar to crown.",
                            "ENERGY: Slow and deliberate — the chin drops in slow motion, the whip drawn out into a long rolling arc.",
                            "The hair rises slowly at first then accelerates, fanning at the apex, the settling fall taking twice as long as expected."),
                "playful": ("CAMERA: Medium — lens at eye level, framing shoulders to crown, slight 3/4 angle.",
                            "ENERGY: Effortless and confident — the flip is casual, like she does this between sentences.",
                            "She catches the camera the instant the hair clears — a half-smile landing exactly on the beat — then shakes the ends out lazily."),
            },
            "floor_work": {
                "raw":     ("CAMERA: Floor level — lens at ground height, static, she moves toward it.",
                            "ENERGY: Feral and deliberate — every movement low, every transition close to the floor.",
                            "She crawls with her hips riding high — movement aggressive, gaze locked on the lens from the moment she hits the floor."),
                "sensual": ("CAMERA: Floor level — lens at ground height, static, slow push as she approaches.",
                            "ENERGY: Slow and heavy — every descent and transition takes twice as long as it needs to.",
                            "She rolls to her back briefly before turning to crawl — spine arching off the floor, hips lifting, the shape deliberate and unhurried."),
                "playful": ("CAMERA: Floor level — lens at ground height, static.",
                            "ENERGY: Light and easy — the descent is smooth, the floor work natural, not performed.",
                            "She pauses mid-crawl to shift weight to one arm and look directly at the lens — a held beat — then continues forward."),
            },
            "strut": {
                "raw":     ("CAMERA: Static wide — lens at hip height, she walks straight at it from the far end.",
                            "ENERGY: Purposeful and direct — no performance in it, just absolute certainty.",
                            "Her pace is steady and unhurried but there is weight behind every step — heel striking hard, hips cutting each stride."),
                "sensual": ("CAMERA: Static — lens at hip height, slightly low angle, she fills the frame as she closes.",
                            "ENERGY: Slow and deliberate — the walk is half tempo, each step placed like it matters.",
                            "Her hips swing wider than the stride requires — each pendulum arc exaggerated, her gaze on the lens the whole way."),
                "playful": ("CAMERA: Static — lens at chest height, wide to start, tightening naturally as she approaches.",
                            "ENERGY: Light and confident — the walk is easy, natural, like she just noticed the camera.",
                            "She breaks the straight line once — a slight diagonal, hip cocking at the pivot — then straightens back to the lens and arrives at mark."),
            },
            "bend_over": {
                "raw":     ("CAMERA: Static low-angle — lens level with her hips, tight on the rear, framing mid-thigh to lower back.",
                            "ENERGY: Direct and unapologetic — the bend is deep, the hold long, no softening.",
                            "She grips her ankles hard at the bottom — knuckles white, hold steady — before the slow rise."),
                "sensual": ("CAMERA: Static low-angle — lens slightly below hip height, rear view, wide enough for context.",
                            "ENERGY: Unhurried and deliberate — every centimetre of the descent is controlled.",
                            "Her hands trail slowly down her legs on the way down — fingertips brushing from thigh to calf to ankle — the contact light and intentional."),
                "playful": ("CAMERA: Static low-angle — lens at hip height, rear view, medium framing.",
                            "ENERGY: Casual and confident — the bend is easy, performed without effort.",
                            "At the bottom she shifts her weight slightly from foot to foot — a lazy sway at full extension — before the slow rise back up."),
            },
            "hip_roll": {
                "raw":     ("CAMERA: Static — lens at hip height, 3/4 angle, tight on the hip zone.",
                            "ENERGY: Sharp and rhythmic — each orbit is a hard isolated pop, the pelvis snapping through each quarter.",
                            "The circles are tight and fast — small diameter, high frequency, the movement mechanical and precise."),
                "sensual": ("CAMERA: Static — lens at hip height, slight 3/4 angle, medium frame.",
                            "ENERGY: Slow and liquid — the orbit is wide and unhurried, the pelvis tracing a long deliberate circle.",
                            "The forward arc is the slowest point — hips pushing out to full extension before arcing back, the pause at the front the visual anchor."),
                "playful": ("CAMERA: Static — lens at hip height, 3/4 angle.",
                            "ENERGY: Easy and rhythmic — the roll is natural, like she cannot help moving to whatever is playing.",
                            "She varies the size mid-sequence — wide slow circles for two counts, then suddenly tight fast ones on the beat, then wide again."),
            },
        }


        # POV body-in-frame notes — replaces tier camera notes when _is_pov=True.
        # Describes the viewer's own body visible in frame + first-person spatial logic.
        # (viewer_body_in_frame, spatial_relationship, first_person_action_language)
        _SEQ_POV_NOTES = {
            "twerk": (
                "POV FRAME: A glimpse of the viewer's own thighs is visible at the very bottom of frame — "
                "just enough to establish perspective. Her body fills the rest of the frame.",
                "SPATIAL: She stands close, back to the viewer, her rear at roughly eye level. The view is her back and rear.",
                "Her hips bounce and sway in the viewer's immediate field of view, the movement close and continuous."
            ),
            "body_roll": (
                "POV FRAME: You are standing. The very bottom of frame shows the tops of your own shoes "
                "and the floor between you. Her body fills the rest of the frame.",
                "SPATIAL: She stands directly facing you at close range — one to two metres. "
                "Her face, chest, and hips are all visible in a single medium frame.",
                "The wave moves toward you — her chest pressing outward feels like it closes the distance, "
                "her hips rolling forward into your space. You are the audience she is performing to."
            ),
            "grind": (
                "POV FRAME: The lower edge of frame shows a hint of the viewer's thighs — just enough to ground the perspective.",
                "SPATIAL: She has her back to the viewer, her hips close. Her back and rear fill the upper frame.",
                "Her hips rotate in a continuous circular motion against the viewer, the movement close and unhurried."
            ),
            "lap_dance": (
                "POV FRAME: You are seated upright. Your own knees and lower thighs are visible "
                "at the very bottom of frame. The chair arms or your own hands frame the sides.",
                "SPATIAL: She approaches from a few steps away, walking directly toward you. "
                "As she arrives and turns, her rear fills the frame before she settles onto your lap.",
                "You watch her approach from your seated position — she grows larger in frame as she closes the distance, "
                "turns, and lowers herself. When she faces you at the end, her face is at your eye level, close."
            ),
            "pole": (
                "POV FRAME: You are standing near the pole, slightly off to one side. "
                "The bottom of frame shows the floor and your own feet.",
                "SPATIAL: The pole runs vertically through the frame. She moves around and up it, "
                "her body passing close to your position — sometimes within touching distance.",
                "From your standing position you look slightly upward as she climbs — "
                "the spin brings her past your eye level, her extended leg sweeping through your field of view."
            ),
            "hair_flip": (
                "POV FRAME: You are standing or seated at her eye level. "
                "Bottom of frame shows your chest or the surface between you.",
                "SPATIAL: She stands directly facing you at close range — "
                "her face fills most of the frame. This is intimate, close distance.",
                "Her chin drops toward you before the flip — then the hair launches upward and backward, "
                "clearing her face to reveal direct eye contact aimed straight at you. "
                "The settle is slow, her gaze holding yours the entire time."
            ),
            "floor_work": (
                "POV FRAME: You are standing. The bottom of frame shows your own feet and shins on the floor. "
                "She starts at the far end of the room and moves toward you.",
                "SPATIAL: You look slightly downward as she descends and crawls — "
                "the angle steepens as she closes the distance and you look further down at her.",
                "She descends to the floor in your field of view, rolls, arches — "
                "and then begins crawling directly toward your feet. "
                "As she arrives, she looks up at you from floor level, face tilted up toward your standing position."
            ),
            "strut": (
                "POV FRAME: You are standing still. Bottom of frame shows your own feet and the floor. "
                "She begins at the far end of the space and walks toward you.",
                "SPATIAL: She walks a straight line directly at you — "
                "she grows larger in frame with every step, the distance closing steadily.",
                "You hold your position as she approaches — watching her hips swing, "
                "her gaze locked on you from the first step. She stops close, "
                "within a step of you, and holds the eye contact."
            ),
            "bend_over": (
                "POV FRAME: The very bottom of frame shows the viewer's feet on the floor — a subtle grounding detail.",
                "SPATIAL: She stands close in front, back to the viewer. Her rear and back fill the frame.",
                "As she bends forward her rear rises toward the viewer, the movement slow and continuous."
            ),
            "hip_roll": (
                "POV FRAME: You are standing or seated directly in front of her. "
                "Bottom of frame shows the floor or your own lap.",
                "SPATIAL: She faces you at close range — one metre or less. "
                "Her hips are at your eye level. The circular roll moves toward and away from you.",
                "The forward arc of the circle brings her hips directly toward you — "
                "you can see the full width and depth of the movement from directly in front. "
                "Her eyes stay on you throughout."
            ),
        }

        # Core mechanics — identical every run regardless of tier
        _ACTION_SEQUENCES = {
            "twerk": (
                "TWERK SEQUENCE — MANDATORY:",
                [
                    "ORIENTATION: She faces AWAY from the camera — the rear view is the entire subject. Her back is to the lens for the full duration.",
                    "She stands with feet wide apart, back to camera, spine neutral, weight evenly distributed. Her lower body is the focus.",
                    "She begins a rhythmic, shaking motion — hips moving up and down with a fluid, bouncing tempo.",
                    "Her hips snap rhythmically, the movement causing her rear to bounce and shake in a continuous, hypnotic rhythm.",
                    "The skin moves with the motion — the cheeks compressing and separating softly with every beat.",
                    "Her upper body remains relatively still — the movement is entirely in the hips, the stillness above contrasting with the rhythm below.",
                    "The background is simple and uncluttered — a plain wall or neutral space that keeps all attention on her movement.",
                    "CONTINUITY: The twerking motion is CONTINUOUS for the entire clip. It does not stop or resolve into a static pose.",
                ]
            ),
            "body_roll": (
                "BODY ROLL SEQUENCE — MANDATORY PHYSICAL BREAKDOWN:",
                [
                    "ORIENTATION: She faces the camera directly — this is a front-facing action and the full front of her body must be visible throughout.",
                    "She stands loose, feet hip-width apart, knees with a soft bend, arms soft at her sides or raised.",
                    "The wave initiates at the chest — a deliberate forward press of the sternum, ribcage leading outward.",
                    "As the chest presses forward the stomach scoops inward, creating the first curve of the S.",
                    "The wave travels down: the stomach releases forward as the hips push back, then the hips roll forward as the chest returns to neutral.",
                    "The full path is chest-forward then stomach-in then hips-forward then reset — one continuous ripple, not three separate pops.",
                    "Her head follows the wave last — tilting back slightly at the peak of the chest press, dropping forward as the hips close.",
                    "She repeats the roll immediately — each pass flows directly into the next with no pause at reset.",
                    "CONTINUITY: The body roll is CONTINUOUS for the entire clip — a looping wave that flows through the torso on repeat, varying slightly in depth and speed but never stopping.",
                ]
            ),
            "grind": (
                "GRIND SEQUENCE — MANDATORY PHYSICAL BREAKDOWN:",
                [
                    "ORIENTATION: She faces AWAY from the camera, her back toward him — the rear three-quarter view shows both her body and his seated form behind her.",
                    "He is seated — chair, sofa, or surface — and she stands in front of him, hips aligned with his lap, feet planted wide for stability.",
                    "Her hands rest on her own thighs or reach back to grip his knees for leverage.",
                    "The movement is a slow circular rotation of the hips — not a back-and-forth thrust but a continuous grinding orbit.",
                    "On the forward arc her back arches slightly, hips tilting down and away from him; on the back arc she presses into him, spine curving out.",
                    "Her weight shifts with each circle, the motion travelling up through her spine as a slow controlled wave.",
                    "She drops lower mid-sequence — sinking into a wider stance, the grind becoming deeper, the contact more deliberate.",
                    "The rhythm matches the music — wide slow circles on a slow beat, tighter faster orbits when the tempo rises.",
                    "CONTINUITY: The grinding motion is CONTINUOUS for the full clip — the circular hip orbit repeats without stopping, varying in depth and speed but never breaking contact or pausing.",
                ]
            ),
            "lap_dance": (
                "LAP DANCE SEQUENCE — MANDATORY PHYSICAL BREAKDOWN:",
                [
                    "ORIENTATION: The seated person faces camera — she faces away when her back is to him, faces camera when she turns to straddle. Both orientations occur in sequence.",
                    "He is seated upright, back straight, hands resting on his thighs or the chair arms.",
                    "She approaches from the front at a slow deliberate walk, eyes on his, moving in time with the music.",
                    "She turns at the last step — presenting her back to him — and lowers herself slowly into his lap, controlling the descent entirely with her thighs.",
                    "She settles, weight distributed across his lap, feet flat on the floor on either side of his legs.",
                    "Her hips begin a slow circular grind — smooth and unhurried, the movement starting small and deepening.",
                    "She leans forward, spine arching long, hands sliding to his knees for leverage as the roll deepens.",
                    "She rises slightly — hips lifting almost off contact — then sinks back in a slow controlled drop.",
                    "She turns to face him, straddling, hands on his shoulders — eye contact held, movement shifting to a forward-back sway.",
                    "CONTINUITY: The lap dance is a continuous fluid sequence — each transition flows directly into the next with no static pauses. The movement does not stop.",
                ]
            ),
            "pole": (
                "POLE SEQUENCE — MANDATORY PHYSICAL BREAKDOWN:",
                [
                    "ORIENTATION: She faces the pole — her body orbits and wraps around it. Camera sees her from the angle that gives the clearest read of the movement.",
                    "The pole stands centre-frame, floor to ceiling. She approaches with loose confident steps, one hand extending to grip at chest height.",
                    "She circles the pole first — weight shifting to the outside foot, inside hip brushing the steel, arms trailing as she moves around it.",
                    "She grips high with both hands, fingers wrapping tight above her head, then plants her inside foot against the pole base.",
                    "She hooks her outside knee around the pole and launches — the spin begins with a push from the planted foot, outside leg extending long.",
                    "The rotation is controlled by grip pressure — loosening slightly lets gravity pull her into a smooth descending spiral.",
                    "She extends both legs out in a straddle mid-spin, body going near-horizontal, holding the position for a full rotation.",
                    "She closes the legs and slides down the final stretch to the floor, landing with both feet simultaneously.",
                    "She arches back from the dismount — spine curving, head dropping back, arms spreading wide before recovering to standing.",
                    "CONTINUITY: The pole sequence moves through approach, circle, climb, spin, descent, dismount as one unbroken fluid performance. Each phase transitions directly into the next.",
                ]
            ),
            "hair_flip": (
                "HAIR FLIP SEQUENCE — MANDATORY PHYSICAL BREAKDOWN:",
                [
                    "ORIENTATION: She faces the camera directly — eye contact with the lens before and after the flip is essential.",
                    "She stands still, weight centred, chin level, eyes on the camera — a beat of stillness before the movement.",
                    "Her chin drops to her chest in a slow deliberate bow — all the hair falls forward, curtaining her face completely.",
                    "A sharp upward whip of the head drives the hair backward in a single explosive arc.",
                    "At the apex the hair fans wide — strands separating, catching the light, the full volume visible against the background.",
                    "Her head comes back to neutral, the hair cascading behind her shoulders in a slow settling fall.",
                    "She shakes her head once gently to distribute the weight — a final slow sway of the ends before stillness.",
                    "Her eyes find the camera the instant the hair clears her face — the eye contact landing exactly on the beat is the payoff.",
                    "CONTINUITY: The hair flip is a single complete action — chin drop, whip, fan, settle, eye contact. After the settle she holds the camera gaze for the remaining duration.",
                ]
            ),
            "floor_work": (
                "FLOOR WORK SEQUENCE — MANDATORY PHYSICAL BREAKDOWN:",
                [
                    "ORIENTATION: She begins standing, facing the camera, then descends to the floor. The crawl moves TOWARD the camera — she closes distance across the clip.",
                    "She stands at the far end of the frame, facing the camera, feet together, weight centred.",
                    "The descent is controlled — she bends her knees and lowers, one hand touching the floor first, then both hands, then knees.",
                    "She rolls to her hands and knees, back flat, spine long, head lifting to maintain eye contact with the lens.",
                    "She rolls to one hip — legs extending out to the side in a slow deliberate spread, one forearm taking her weight.",
                    "She arches her back from the hip-down position, hips lifting slightly off the floor, the curve of her spine visible.",
                    "She returns to hands and knees and begins the crawl — each hand placed deliberately ahead, hips rolling side to side with every step.",
                    "She closes the distance to the camera slowly — the framing tightens as she approaches.",
                    "She pauses close to the lens, chin tilted up, weight on her forearms — holding eye contact.",
                    "CONTINUITY: The floor work is a continuous descending sequence — stand, lower, roll, arch, crawl, arrive. It does not reverse or reset mid-clip.",
                ]
            ),
            "strut": (
                "STRUT SEQUENCE — MANDATORY PHYSICAL BREAKDOWN:",
                [
                    "ORIENTATION: She faces the camera and walks TOWARD it — she starts at the far end of the frame and closes distance across the full clip.",
                    "She begins at the far end of the frame — weight back, one hip cocked, chin level, eyes already on the lens.",
                    "She steps off on the beat — heel-first, each foot placed on an invisible centre line so the hips cross-sway naturally with every stride.",
                    "Her arms swing loosely in opposition to her legs — from the shoulder, not the elbow, relaxed not rigid.",
                    "Each stride produces a clear lateral hip shift — the pelvis tilts side to side in a slow pendulum that travels up through her waist.",
                    "Her expression is neutral and direct — the gaze on the camera never breaks from first step to last.",
                    "She does not rush — the pace is deliberately unhurried, the confidence is in the control not the speed.",
                    "She arrives at mark and stops — weight shifting to one leg, the opposite hip pushing out, one hand moving to her hip.",
                    "CONTINUITY: The strut is a single unbroken walk from far to near — no stops mid-stride, no looking away. The clip ends on the held arrival pose.",
                ]
            ),
            "bend_over": (
                "BEND SEQUENCE — MANDATORY PHYSICAL BREAKDOWN:",
                [
                    "ORIENTATION: She faces AWAY from the camera — her back and rear view face the lens for the entire clip. She does NOT turn around.",
                    "She stands feet hip-width apart or slightly wider, facing away, weight centred, arms loose at her sides.",
                    "The descent begins at the hips — a slow deliberate hip hinge, not a spine curl — the back stays flat and long throughout.",
                    "Her hands slide down her legs as she folds, fingertips tracking from thigh to shin to ankle.",
                    "As she hinges forward the rear view becomes fully dominant — the camera sees the full shape of her lower body.",
                    "She continues until her torso is parallel to the floor or lower, back flat, spine long.",
                    "She pauses at the bottom of the hinge — a held position, hands at ankles or floor, the shape fully extended.",
                    "She may grip her ankles, fingers wrapping tight, pulling the hold deeper.",
                    "The recovery is equally slow — hips driving upward first, spine stacking vertebra by vertebra back to standing.",
                    "CONTINUITY: The bend is a slow descent, hold, and slow recovery — the full sequence takes the entire clip. If the clip is long she descends, holds, recovers, and descends again.",
                ]
            ),
            "hip_roll": (
                "HIP ROLL SEQUENCE — MANDATORY PHYSICAL BREAKDOWN:",
                [
                    "ORIENTATION: She faces the camera at a slight 3/4 angle so both the front and side of the hip movement are readable.",
                    "She stands loose, feet hip-width apart, knees with a soft bend, weight evenly distributed, arms soft.",
                    "The movement begins with a lateral shift — hips pushing left, the weight following smoothly.",
                    "From the left the hips arc forward — a slow push of the pelvis out and forward, tracing the front quarter of a circle.",
                    "The arc continues right — hips travelling across the front, weight transferring to the right foot.",
                    "From the right the hips arc backward — completing the circle, returning to the start position through the back quarter.",
                    "The full orbit is smooth and continuous — no hard stops at any compass point, the pelvis tracing a slow horizontal circle.",
                    "Her upper body counter-rotates subtly — shoulders moving opposite to the hips to keep her torso centred and upright.",
                    "She varies the size: a wide slow orbit for two counts, then tighter faster circles on the beat, then wide again.",
                    "CONTINUITY: The hip roll is CONTINUOUS for the entire clip — the orbital motion repeats without stopping, varying in speed and diameter but never breaking the circle.",
                ]
            ),
        }


        # ── Character seed ────────────────────────────────────────────────────
        # Detect whether the user has described physical appearance (age, hair, skin, build, clothing).
        # Role words (man, woman, detective, suspect etc.) deliberately excluded here —
        # those trigger has_person and multi detection but do NOT count as a character description.
        # Without a physical description the char seed should still fire for single-person scenes,
        # and multi_instruction should still fire for two-person scenes.
        # Suppresses char seed if user has anchored the scene with ANY person reference
        # or appearance/clothing detail. The rule: if the user told us who is there,
        # we do not overwrite them with a random blueprint.
        # Also suppresses when the scene is defined by a non-human subject (animal, creature,
        # vehicle, object) so we don't paste a random character onto a gorilla close-up.
        _scene_is_anchored        = bool(self._NON_HUMAN_RE.search(user_input))
        _user_described_character = _scene_is_anchored or bool(self._USER_CHAR_RE.search(user_input))

        # ── Gender detection ──────────────────────────────────────────────────
        # Reads user_input for explicit gender signals so the char seed matches.
        # "neutral" = no signal found → seed picks randomly.
        _has_male   = bool(self._MALE_RE.search(user_input))
        _has_female = bool(self._FEMALE_RE.search(user_input))

        if _has_male and not _has_female:
            _gender = "male"
        elif _has_female and not _has_male:
            _gender = "female"
        else:
            # Both signals or neither — let the seed decide randomly
            _gender = "neutral"

        print(f"[LTX2-Qwen] Gender signal: {_gender} (male={_has_male}, female={_has_female})")

        # Use _active_scene_context (respects use_scene_context flag)
        is_multi    = bool(self._MULTI_RE.search(user_input + " " + _active_scene_context))
        # Explicit subject_count overrides text-based multi detection
        if subject_count and subject_count > 0:
            is_multi = subject_count > 1

        if has_person and not _user_described_character and not is_gravure:
            rng = random.Random(seed if seed != -1 else None)

            # ── Genre-ethnicity override ───────────────────────────────────────
            # Culturally specific genres lock the char seed ethnicity.
            # Only fires when user has not described a character.
            import re as _gere
            _g_detect = (music_genre + " " + _combined_input).lower()
            _genre_eth_pool = None
            if has_music and _detected_entry:
                if _gere.search(r'\b(k.?pop|kpop|korean\s+pop)\b', _g_detect):
                    _genre_eth_pool = [
                        ("Korean", "fair skin with a soft peachy-pink flush"),
                        ("Korean", "smooth fair skin with a cool porcelain tone"),
                        ("Korean", "light ivory skin with a warm golden-peachy tone"),
                        ("Korean", "fair skin with a cool neutral undertone and natural glow"),
                        ("Korean", "pale porcelain skin with subtle pink undertones"),
                    ]
                elif _gere.search(r'\b(j.?pop|jpop|japanese\s+pop|city\s+pop)\b', _g_detect):
                    _genre_eth_pool = [
                        ("Japanese", "pale skin with cool beige undertones"),
                        ("Japanese", "very fair skin, almost translucent in soft light"),
                        ("Japanese", "light ivory skin with a subtle warm tone"),
                        ("Japanese", "fair cool-toned skin with a natural glow"),
                    ]
                elif _gere.search(r'\b(bollywood|bhangra|indian\s+film|filmi)\b', _g_detect):
                    _genre_eth_pool = [
                        ("Indian", "warm medium brown skin with golden undertones"),
                        ("Indian", "rich warm skin with a deep golden tone"),
                        ("Indian", "smooth wheatish skin with warm amber undertones"),
                        ("Indian", "light brown skin with warm honey tones"),
                        ("Indian", "medium brown skin with a luminous warm undertone"),
                    ]
                elif _gere.search(r'\b(flamenco|cante\s+jondo|soleares)\b', _g_detect):
                    _genre_eth_pool = [
                        ("Spanish", "warm olive skin with golden-brown undertones"),
                        ("Spanish", "medium olive skin with a warm Mediterranean tone"),
                        ("Spanish", "light olive skin with warm golden undertones"),
                    ]
                elif _gere.search(r'\b(bossa\s+nova|samba|MPB|pagode|forró)\b', _g_detect):
                    _genre_eth_pool = [
                        ("Brazilian", "warm brown skin with golden undertones"),
                        ("Brazilian", "light golden-brown skin with warm tones"),
                        ("Brazilian", "medium warm skin with a sun-kissed honey tone"),
                        ("Brazilian", "rich tawny skin with warm amber undertones"),
                    ]
                elif _gere.search(r'\b(afrobeats|afropop|afro\s+fusion|naija|highlife)\b', _g_detect):
                    _genre_eth_pool = [
                        ("Nigerian", "rich dark brown skin with warm undertones"),
                        ("Nigerian", "deep brown skin with a luminous warm tone"),
                        ("Nigerian", "medium brown skin with golden-warm undertones"),
                        ("West African", "deep ebony skin with cool blue-black undertones"),
                        ("West African", "rich warm dark skin with amber undertones"),
                    ]
                elif _gere.search(r'\b(cumbia|salsa|merengue|bachata|mambo)\b', _g_detect):
                    _genre_eth_pool = [
                        ("Latina", "warm olive skin with golden-brown undertones"),
                        ("Latina", "light golden-brown skin with warm honey tones"),
                        ("Latina", "medium warm skin with a rich bronze undertone"),
                        ("Latina", "warm tan skin with sun-kissed golden undertones"),
                    ]
                elif _gere.search(r'\b(reggae|ska|roots\s+reggae|dancehall)\b', _g_detect):
                    _genre_eth_pool = [
                        ("Jamaican", "rich dark brown skin with warm undertones"),
                        ("Jamaican", "deep warm brown skin with golden-amber tones"),
                        ("Jamaican", "medium warm brown skin with rich undertones"),
                    ]

            # Pick from override pool and inject into char description
            _eth_rng = random.Random((seed + 19) if seed != -1 else None)
            _genre_eth_str = None
            if _genre_eth_pool:
                _eth_pick = _eth_rng.choice(_genre_eth_pool)
                _genre_eth_str = f"{_eth_pick[0]}, {_eth_pick[1]}"
                print(f"[LTX2-Qwen] Genre ethnicity: {_genre_eth_str}")

            if is_multi:
                # Two-person scene — generate two distinct seeds so the LLM has both characters.
                # For mixed-signal scenes (e.g. "detective and suspect" with no gender) we let
                # each seed pick independently so we get variety.
                char_a = _build_char_seed(rng, adult_only=(is_sensual or is_explicit), gender=_gender)
                # Second person always neutral so a mixed-gender pair is possible
                char_b = _build_char_seed(rng, adult_only=(is_sensual or is_explicit), gender="neutral")
                char_seed_note = (
                    f"\n[CHARACTER SUGGESTIONS — TWO PEOPLE (only if the scene does not already define them): "
                    f"Person A: {char_a}. "
                    f"Person B: {char_b}. "
                    f"These are loose suggestions — if the scene implies specific people, ignore this entirely. "
                    f"If used, give each person clothing appropriate to the scene. "
                    f"Establish both spatially — left/right or foreground/background — and keep descriptors consistent.]"
                )
            else:
                char_description = _build_char_seed(rng, adult_only=(is_sensual or is_explicit), gender=_gender)
                # Apply genre ethnicity override — replace the ethnicity token in the generated description
                if _genre_eth_str:
                    import re as _etre
                    char_description = _etre.sub(
                        r'\b(White|Japanese|Korean|Chinese|East Asian|Black)\b',
                        _genre_eth_str, char_description, count=1
                    )
                # Use genre clothing if available, otherwise generic prompt
                _seed_clothing_note = (
                    f"If used, dress them in: {_gclothing}"
                    if (has_music and _detected_entry and _gclothing and not _has_user_clothing)
                    else "If used, add clothing appropriate to the scene context."
                )
                char_seed_note = (
                    f"\n[CHARACTER SUGGESTION (only if no character is defined by the scene above): {char_description}. "
                    f"This is a loose suggestion only — if the scene already implies a specific person, ignore this entirely. "
                    f"{_seed_clothing_note}]"
                )
        else:
            char_seed_note = ""
            if has_person and _user_described_character:
                print("[LTX2-Qwen] User described character — seed suppressed.")
            elif is_gravure and not _user_described_character:
                print("[LTX2-Qwen] Gravure preset — char seed suppressed, preset defines character.")

        # ── K-pop group instruction ─────────────────────────────────────────────
        # Smart detection: fires when K-pop genre is active AND the user's input
        # signals a group (plural nouns, group words, or no explicit solo signal).
        # "a woman sings" → solo. "women sing" / "a group" / "idols" → group.
        import re as _kpre
        _kpop_group_instruction = ""
        _is_kpop = has_music and _detected_entry and bool(
            _kpre.search(r'\b(k.?pop|kpop|korean\s+pop)\b',
            (music_genre + " " + _combined_input).lower())
        )
        if _is_kpop:
            _ci_kp = _combined_input.lower()
            # Explicit solo signals — user wants one person
            _user_wants_solo = bool(_kpre.search(
                r'\b(a\s+woman|a\s+man|a\s+girl|a\s+singer|solo|alone|'
                r'by herself|by himself|one person|single performer|she sings|he sings)\b',
                _ci_kp
            ))
            # Explicit group signals — user wants multiple
            _user_wants_group = bool(_kpre.search(
                r'\b(group|band|idol group|girl group|boy band|members|'
                r'women|girls|boys|they|them|performers|dancers|'
                r'twice|blackpink|bts|aespa|newjeans|ive|le\s+sserafim)\b',
                _ci_kp
            ))
            # Default: K-pop without any signal → group (it's the genre default)
            _should_be_group = _user_wants_group or (not _user_wants_solo)

            if _should_be_group:
                _kpop_group_instruction = (
                    "\n\n[K-POP GROUP — MANDATORY: This scene is K-pop. "
                    "There are MULTIPLE performers on stage together — a group of 4 to 6 Korean idols. "
                    "Every performer is present in the frame. "
                    "They wear coordinated but non-identical outfits in a shared colour palette: "
                    "cropped fitted tops or structured jackets, high-waisted mini skirts or tailored shorts, "
                    "platform trainers or thigh-high boots. "
                    "Hair colours are complementary across the group — jet-black on one, bleached blonde on another, "
                    "a vivid dyed colour on a third. Full stage makeup on every face. "
                    "They move in synchronised choreography — formations, sharp arm positions, unified footwork. "
                    "Wide shot shows the full formation. Medium shot centres the lead performer. "
                    "DO NOT write a single solo performer. The group is the subject.]"
                )
                print("[LTX2-Qwen] K-pop group instruction fired")
            else:
                print("[LTX2-Qwen] K-pop solo detected — group instruction suppressed")


        # ── Background presence system ────────────────────────────────────────
        # Makes every video feel alive by injecting a seed-picked background
        # detail appropriate to the scene's world. Fires globally.
        #
        # Logic:
        #   SUPPRESS when: explicit/gravure (isolation is the aesthetic),
        #                  user already described the background in detail,
        #                  scene is clearly intimate/private,
        #                  no-person scenes (the environment IS the subject)
        #   FIRE otherwise: read location context, pick from appropriate pool
        #
        # LTX handles background motion and blur well even when it struggles
        # with foreground faces — so this is high-impact, low-risk.

        _bg_instruction = ""

        # Suppression checks
        _user_described_bg = bool(re.search(
            r'\b(background|backdrop|behind|crowd|audience|people|packed|busy|'
            r'empty|alone|isolated|just (?:her|him|them|me)|no one else|'
            r'deserted|abandoned|vast|stretch(?:es|ing)|stretching back)\b',
            _combined_input, re.IGNORECASE
        ))

        # Solo subject detection — broad scene nature, not specific locations.
        # Suppress when: the scene is nature/wilderness/space/weather (no one else
        # would be there), OR the user explicitly signals one person alone.
        # Do NOT suppress for social environments: cities, clubs, streets, stages etc.
        _is_solo_scene = bool(re.search(
            r'\b(lone\b|alone\b|by (her|him|them)self|'
            r'mountain|cliff|wilderness|forest|desert|tundra|glacier|'
            r'ocean|sea\b|beach\b|shore\b|coastline|'
            r'space\b|planet\b|moon\b|asteroid|cosmos|'
            r'storm\b|lightning|blizzard|tornado|hurricane|'
            r'field\b|meadow|plain\b|moorland|hillside|'
            r'ruins?\b|abandoned|wasteland|void\b|abyss|'
            r'sunrise|sunset|dawn\b|dusk\b|twilight)\b',
            _combined_input, re.IGNORECASE
        ))

        _suppress_bg = (
            is_explicit or
            is_gravure or
            _user_described_bg or
            _is_solo_scene or
            not has_person
        )

        if not _suppress_bg:
            import re as _bgre
            import random as _bgrnd
            _bg_rng = _bgrnd.Random((seed + 23) if seed != -1 else None)
            _ci_bg = _combined_input.lower()
            _sp_bg = style_preset.lower()

            # ── Detect scene world ────────────────────────────────────────────
            # Priority order: explicit location keywords → genre world → preset

            # Concert / stage
            if _bgre.search(r'\b(stage|concert|perform(?:ing|ance)|gig|show|'
                            r'spotlight|microphone|mic stand|arena|festival stage)\b', _ci_bg):
                _bg_pool = [
                    "Stage haze drifts through the rigging above — light beams cutting through at hard angles, "
                    "the dust in the air made visible by the source. "
                    "Monitor wedges and cable runs line the edge of the boards in shadow.",

                    "The lighting rig overhead is a grid of Par cans and moving heads — "
                    "half dark, the rest painting the stage in the exact colours of this moment. "
                    "A single follow-spot tracks from above, its beam visible in the haze.",

                    "Flight cases stacked at the back of the stage, the working infrastructure "
                    "of the show present and unromantic. A set list taped to the floor, "
                    "curling slightly at the corner.",

                    "The microphone stand throws a long shadow across the stage boards. "
                    "A guitar on a stand waits in the wing. The reverb of the last note "
                    "still fading in the rafters.",

                    "Strobe light catches the haze in frozen frames. "
                    "A half-empty water bottle sits at the edge of the stage. "
                    "The floor is scuffed and worn where performers have stood for years.",
                ]

            # Club / nightclub / dance floor
            elif _bgre.search(r'\b(club|nightclub|dance\s*floor|dancefloor|rave|'
                              r'booth|bar|bouncer|dj\s*booth|strobe)\b', _ci_bg):
                _bg_pool = [
                    "Coloured light sweeps the space in slow rotations — "
                    "red bleeding into blue bleeding into deep purple, "
                    "the walls cycling through the palette on a timer.",

                    "The bar runs along the back wall — bottles catching the backlight "
                    "in amber and green, glasses stacked in rows, "
                    "the surface wet with condensation.",

                    "The DJ booth glows against the far wall — equipment lights in green and blue, "
                    "the waveform visible on a monitor, the bass physically present in the floor.",

                    "Strobe cuts the room into still frames — hard white against the coloured atmosphere. "
                    "A mirrored ball turns slowly overhead, scattering light across every surface.",
                ]

            # Jazz club / bar / intimate venue
            elif _bgre.search(r'\b(jazz\s*club|smoky\s*bar|late.night\s*bar|'
                              r'small\s*venue|intimate\s*venue|cabaret|lounge)\b', _ci_bg) or \
                 _bgre.search(r'\b(jazz|blues|soul|neo.?soul|cabaret)\b', _ci_bg) and \
                 _bgre.search(r'\b(club|bar|venue|lounge)\b', _ci_bg):
                _bg_pool = [
                    "Small round tables in soft focus — glasses of amber liquid catching candlelight, "
                    "an ashtray, a folded napkin, a half-eaten plate pushed to one side. "
                    "The room has the texture of a place that has held many evenings.",

                    "A neon sign bleeds colour through the window at the back — "
                    "red and pink washing the wall beside it. "
                    "The bar surface is dark wood worn smooth, bottles lined in rows behind it.",

                    "The upright bass in the corner catches the light on its curve. "
                    "A drumkit half-dismantled, a music stand with a chart clipped to it. "
                    "A clock on the back wall that has been reading 11:47 for years.",

                    "Low tungsten light from practised angles — warm, deliberate. "
                    "The tablecloths cream and slightly stained at the edges. "
                    "A half-finished drink on the nearest table, a coat draped over the chair.",
                ]

            # Street / urban / outdoor night
            elif _bgre.search(r'\b(street|alley|sidewalk|pavement|city\s*street|'
                              r'intersection|corner|underpass|subway|urban)\b', _ci_bg):
                _bg_pool = [
                    "Car headlights sweep through at intervals, the city indifferent. "
                    "A traffic light changes from red to green for no one. "
                    "The wet pavement reflects every light source above it.",

                    "A distant siren rises and fades. Steam rises from a grate. "
                    "A shop sign flickers — one letter darker than the rest, going slowly.",

                    "A bus shelter advertisement glows at the far end of the frame — "
                    "a rectangle of light too far to read. Litter moves in the wind at the kerb.",

                    "The streetlight overhead defines everything beneath it. "
                    "A CCTV camera on a bracket turns slowly. "
                    "The alley beyond holds darkness the light cannot reach.",
                ]

            # Rooftop
            elif _bgre.search(r'\b(rooftop|roof\s*top|roof)\b', _ci_bg):
                _bg_pool = [
                    "The city stretches to the horizon — the grid of lit windows, "
                    "the dark shapes of buildings against a sky that never fully darkens.",

                    "A water tower silhouetted against the sky. A ventilation unit hums at the far edge. "
                    "The wind moves a tarpaulin slightly, a forgotten chair scrapes.",

                    "Light pollution makes the clouds glow orange and pink from below. "
                    "An aircraft light blinks steadily across the frame, moving toward somewhere else.",
                ]

            # Festival / outdoor stage
            elif _bgre.search(r'\b(festival|outdoor\s*stage|field|grounds|'
                              r'park\s*stage|open\s*air|amphitheatre)\b', _ci_bg):
                _bg_pool = [
                    "Stage rigging fills the sky above — lighting trusses and cable bundles "
                    "against open air. A follow-spot platform bolted high on the structure.",

                    "Festival flags and banners catch the wind at the back of the field. "
                    "Sound system towers on either side, each speaker stack taller than a house.",

                    "The grass catches the stage light at a low angle — "
                    "blades lit and unlit in alternating strips. "
                    "The horizon beyond the site is dark and flat.",
                ]

            # Recording studio / rehearsal space
            elif _bgre.search(r'\b(studio|rehearsal|practice\s*space|recording|'
                              r'booth|control\s*room|soundproof)\b', _ci_bg):
                _bg_pool = [
                    "A guitar on a stand, a keyboard pushed to the side, "
                    "cable runs taped to the floor in neat runs. The working room of music.",

                    "A music stand with a chart and handwritten notes in the margin. "
                    "A water bottle, a jacket thrown over a chair, a phone face-down on the desk.",

                    "Acoustic foam panels line the walls in geometric patterns. "
                    "The mixing desk through the glass has a hundred lit points in green and amber.",
                ]

            # K-pop stage
            elif _bgre.search(r'\b(k.?pop|kpop|korean\s+pop)\b', _ci_bg + " " + style_preset.lower()):
                _bg_pool = [
                    "The LED wall cycles through a graphic sequence — "
                    "geometric shapes dissolving and reforming in electric blue and magenta, "
                    "the colours shifting precisely on the beat.",

                    "The stage floor reflects the overhead rig in long bright streaks — "
                    "the high-gloss surface doubling every light source beneath each step.",

                    "Smoke machines pulse low fog across the stage floor at intervals. "
                    "The rigging above is a precise grid of moving heads, "
                    "each locked to a designated position.",
                ]

            # Music video generic
            elif has_music and _detected_entry:
                _bg_pool = [
                    "The environment holds the specific texture of this genre's world — "
                    "the surfaces, the light quality, the objects that belong here. "
                    "Present in soft focus, not competing.",

                    "The edges of the frame carry the atmosphere of the location — "
                    "light sources, set pieces, the geometry of the space.",

                    "The floor catches the overhead light and holds it. "
                    "The air has a quality specific to this room — "
                    "temperature, dust, the particular way sound behaves here.",
                ]

            # Cinematic / drama
            elif _bgre.search(r'\b(cinematic|drama|film|movie|scene)\b', _sp_bg):
                _bg_pool = [
                    "The location extends beyond the frame — sounds from off-screen, "
                    "light from a source not shown, the feeling of a place with history. "
                    "A clock on the wall. A window with weather behind it. An open door.",

                    "Depth visible behind the subject — background soft but present, "
                    "textures readable, space continuing. "
                    "A coat on the back of a chair. A glass on a table. A light left on.",

                    "The room has been lived in — objects in the middle distance "
                    "that tell a story without announcing it. "
                    "The light falls at the angle specific to this hour.",
                ]

            # Intimate / sensual
            elif is_sensual:
                _bg_pool = [
                    "The room resolves only as warmth and shadow — "
                    "the suggestion of a space rather than its specifics. "
                    "A lamp at the edge of frame. A window. The dark.",

                    "Long shadows fall across the surface behind her — "
                    "background defined by what the light does, not what is there. "
                    "The texture of a wall. The edge of a mirror.",
                ]

            # Default
            else:
                _bg_pool = [
                    "The space has texture — surfaces, light, the specific colour "
                    "of the air in this room at this hour. Present without competing.",

                    "A detail at the edge of frame anchors the scene — "
                    "an object, a light source, a surface. Not foregrounded. Just there.",

                    "The environment continues in soft focus — colours and textures readable, "
                    "the world present behind the subject.",
                ]
            _bg_detail = _bg_rng.choice(_bg_pool)
            _bg_instruction = (
                f"\n\n[BACKGROUND PRESENCE: The scene has a living world behind the subject. "
                f"{_bg_detail} "
                "This background detail should be present throughout the clip — "
                "not dominant, not distracting, but real. "
                "The subject is always the primary focus. "
                "The background makes the world feel inhabited.]"
            )
            print(f"[LTX2-Qwen] Background presence: {_bg_detail[:50]}...")

        # ── Gravure body override ─────────────────────────────────────────────
        # If the user specified a body type, inject it as a hard override so the
        # gravure preset's default "petite to medium build" doesn't silently win.
        _gravure_body_override = ""
        if is_gravure:
            _body_matches = list(dict.fromkeys(
                m.group(0).lower() for m in self._BODY_STYLE_RE.finditer(user_input)
            ))
            if _body_matches:
                _body_str = ", ".join(_body_matches)
                _gravure_body_override = (
                    f"\n[GRAVURE BODY OVERRIDE — MANDATORY: The user has described the body type as: "
                    f"{_body_str}. Use this EXACTLY. Ignore the preset's default build description. "
                    f"The body type stated by the user is the ceiling and the floor — do not soften, "
                    f"expand, or substitute it.]"
                )

        # ── Vision context ────────────────────────────────────────────────────
        if _active_scene_context and _active_scene_context.strip():
            effective_input = (
                f"[SCENE CONTEXT FROM IMAGE — ABSOLUTE AUTHORITY: "
                f"This is what is actually in the image. Every visual detail here is ground truth. "
                f"Do NOT invent, replace, or contradict any aspect of this description — "
                f"clothing, skin tone, hair, body type, setting, or lighting. "
                f"Any CHARACTER SEED instruction below does NOT apply when an image is provided; disregard it entirely.]\n"
                f"{_active_scene_context.strip()}\n\n"
                f"[USER DIRECTION — apply this as action, style, and mood layered over the above scene. "
                f"The subject looks exactly as described in the image context above. Do not change their appearance.]\n"
                f"{user_input.strip()}"
            )
        else:
            effective_input = user_input.strip()
            if has_person and _user_described_character:
                effective_input += (
                    "\n[CHARACTER NOTE: The user has described the character's appearance. "
                    "Use ONLY the user's description for all visual details. "
                    "Do NOT invent or substitute any appearance detail not present in the input above.]"
                )
            effective_input += char_seed_note

        # ── Genre world instruction ───────────────────────────────────────────
        # Fires as a standalone mandatory block when genre is detected and user
        # has not described clothing or location. Separate from music_sound_rule
        # so it cannot be missed or deprioritised.
        _genre_world_instruction = ""
        if has_music and _detected_entry and _gclothing:
            _world_parts = []
            if _gclothing and not _has_user_clothing:
                _world_parts.append(f"Clothing: {_gclothing}")
            _has_user_loc2 = bool(re.search(
                r'\b(club|bar|venue|stage|studio|street|room|apartment|bedroom|hotel|'
                r'office|park|beach|warehouse|rooftop|basement|arena|festival|concert|'
                r'theatre|church|car|gym|garden|courtyard|corridor|forest|city|urban)',
                _combined_input, re.IGNORECASE
            ))
            if _glocs and not _has_user_loc2:
                import random as _r2
                _world_parts.append(f"Location: {_music_rng.choice(_glocs)}")
            if _world_parts:
                _genre_world_instruction = (
                    "\n[GENRE WORLD — MANDATORY: This scene exists in a specific world. "
                    + " ".join(_world_parts)
                    + ". These details are NOT optional suggestions — they define the world "
                    "this scene inhabits. Apply them unless the user has explicitly overridden them.]"
                )
                print(f"[LTX2-Qwen] Genre world: {' | '.join(_world_parts)}")


        # ── Environment pool ──────────────────────────────────────────────────
        # Fires when no location is given by the user AND no genre has already
        # injected one. Seed picks from the appropriate tier.
        # Normal: any scene. Sensual: is_sensual or gravure. Explicit: is_explicit only.
        _ENV_POOL_NORMAL = [
            # Urban / night
            "a rain-slicked city street at 2am, neon signs bleeding colour into the wet asphalt, "
            "a single working streetlight overhead, the distant sound of traffic two blocks away",

            "a rooftop in a dense city at night, gravel underfoot, HVAC units humming in the dark, "
            "the skyline visible in every direction, low cloud catching the orange glow from below",

            "a subway platform, empty, fluorescent tubes flickering at the far end, "
            "the distant rush of a train in the tunnel, white tiles yellowed at the grout lines",

            "a hotel corridor, identical doors stretching in both directions, "
            "the carpet worn thin down the centre, a fire exit sign casting red light from the far end",

            "a multi-storey car park at night, concrete pillars throwing hard shadows, "
            "a few cars scattered across the level, the city visible through the open sides",

            "a brutalist underpass, wide and empty, skateboard marks on the concrete, "
            "strip lighting overhead, distant echo of footsteps",

            # Indoor / atmospheric
            "a penthouse apartment at night, floor-to-ceiling glass on two sides, "
            "city lights thirty floors below, sparse furniture, a single floor lamp in the corner",

            "an empty swimming pool, the tiles cracked and faded, "
            "a single work light on a stand at the deep end, the night sky visible above the walls",

            "a boxing gym after hours, heavy bags hanging still, "
            "the smell of canvas and old leather in the air, a single cage light over the ring",

            "a laundromat at 3am, machines tumbling, fluorescent light humming, "
            "condensation on the front windows, plastic chairs along the wall",

            "a warehouse loading dock, steel roller doors half-open, "
            "a forklift parked in the dark, sodium light from the yard spilling across the floor",

            "an art gallery after closing, motion sensors off, "
            "emergency lighting at floor level, the paintings disappearing into shadow above waist height",

            # Glamour / upscale
            "a hotel bar at last call, one bartender wiping the counter, "
            "a few remaining drinks on the tables, the lighting warm and low, jazz still playing quietly",

            "a casino floor at 4am, half the machines dark, "
            "a few dedicated players at the far tables, the carpet pattern aggressive and timeless",

            "a backstage corridor beneath a theatre, exposed pipes overhead, "
            "cables taped to the floor, the muffled sound of the audience above, "
            "a fire door open to a brick wall",

            "a recording studio control room, the mixing desk dark except for standby lights, "
            "the city visible through the soundproof glass, session notes still on the whiteboard",
        ]

        _ENV_POOL_SENSUAL = [
            # Club / performance
            "a strip club during the slow hour, mirrored ceiling, the pole lit from below "
            "with purple and amber gel, booths in deep shadow around the edges, "
            "bass from the speakers felt through the floor, smoke hanging at shoulder height",

            "a private booth in a nightclub, red velvet banquette, "
            "the main floor visible through a beaded curtain, bass thumping through the wall, "
            "ice melting in glasses no one is drinking",

            "a VIP lounge above the main floor, floor-to-ceiling glass looking down on the crowd, "
            "low white sofas, a bottle service stand at the door, "
            "the sound below muffled but the bass still felt",

            "a lap dance lounge, individual booths with half-curtains, "
            "purple UV light catching glitter on every surface, "
            "a small stage visible from every seat, the carpet sticky underfoot",

            "a backstage dressing room at a burlesque club, "
            "costume rails, feather boas draped over mirrors framed in bulb lights, "
            "lipstick on the vanity glass, the show audible through the thin wall",

            "a high-end boudoir, silk curtains drawn, a chaise longue near the window, "
            "warm tungsten light from two table lamps, a full-length mirror in the corner, "
            "the city outside reduced to a soft orange glow around the curtain edges",

            "a hotel room at night, the curtains open on the city, "
            "the bedside lamp the only light, the bedding pushed aside, "
            "a room service tray untouched on the desk",

            "a private cabaret venue, eight tables maximum, all occupied, "
            "a single spotlight on a small stage, red walls, candles in red glass, "
            "the performer and the audience sharing the same dark air",

            "a yacht interior at anchor, low ceilings, teak panels, "
            "the water making the light shift constantly across the walls, "
            "the engine off, the only sound the hull against the dock",

            "a luxury penthouse pool, indoors, after midnight, "
            "the water still lit blue from below, steam rising, "
            "the city grid visible through a full glass wall, no one else here",
        ]

        _ENV_POOL_EXPLICIT = [
            "a private members club, after hours, the bar staff gone, "
            "the remaining guests knowing exactly what kind of place this becomes after midnight, "
            "leather seating, low red light, a room that does not exist officially",

            "a high-end escort apartment, maintained for one purpose, "
            "heavy curtains permanently closed, indirect lighting warm and amber, "
            "a bed that is always made and always unmade, the street nine floors below inaudible",

            "a sex club darkroom, red safety lights at floor level, "
            "the architecture designed to prevent eye contact, "
            "sound from the main room audible but directionless, "
            "the air close and warm",

            "a private cinema screening room, twelve seats, all empty except one, "
            "the projector running something that was not on the public programme, "
            "blackout curtains, the sound system designed for complete isolation",

            "a bondage studio, purpose-built, "
            "equipment mounted to reinforced walls, a padded table centre-floor, "
            "adjustable overhead lighting, the room soundproofed, "
            "a mirror along one full wall",

            "a photography studio rented after hours, "
            "seamless white paper rolled out and already marked, "
            "a ring light and two softboxes still warm from the last session, "
            "the building otherwise empty",
        ]

        # Determine whether to fire the environment pool
        _has_user_loc_pool = bool(re.search(
            r'(club|bar|venue|stage|studio|street|room|apartment|bedroom|hotel|'
            r'office|park|beach|warehouse|rooftop|basement|arena|festival|concert|'
            r'theatre|church|car|gym|garden|courtyard|corridor|forest|city|urban|'
            r'pool|lounge|cabin|mansion|castle|bridge|pier|dock|alley|subway|'
            r'kitchen|bathroom|desert|mountain|field|yacht|penthouse|dressing\s+room)',
            _combined_input, re.IGNORECASE
        ))

        # Only fire if no user location AND (no genre location already injected OR no genre active)
        _genre_gave_location = (
            has_music and _detected_entry and _glocs and not _has_user_loc_pool
        )
        _env_pool_instruction = ""

        # Suppress auto-pool if widget environment is set — prevents two MANDATORY
        # location instructions conflicting (widget always wins over pool)
        _widget_env_set = (
            environment_sel and
            environment_sel not in ("None — LLM decides", "") and
            not environment_sel.startswith("─") and
            environment_sel != "🎲 Random — seed picks"
        )

        if not _has_user_loc_pool and not _genre_gave_location and not _widget_env_set:
            _env_rng = __import__('random').Random(seed if seed != -1 else None)
            if is_explicit:
                _env_pool = _ENV_POOL_EXPLICIT + _ENV_POOL_SENSUAL
            elif is_sensual or is_gravure:
                _env_pool = _ENV_POOL_SENSUAL + _ENV_POOL_NORMAL
            else:
                _env_pool = _ENV_POOL_NORMAL

            _chosen_env = _env_rng.choice(_env_pool)
            _env_pool_instruction = (
                "\n[ENVIRONMENT — MANDATORY: Place this scene in the following setting. "
                "Every visual detail of this environment is real and present in the shot — "
                "the lighting, surfaces, sounds, and atmosphere described here define the world. "
                "Do NOT substitute a generic interior or exterior. "
                f"Setting: {_chosen_env}]"
            )
            print(f"[LTX2-Qwen] Environment pool fired: {_chosen_env[:60]}...")

        # ── LoRA triggers ─────────────────────────────────────────────────────
        if lora_triggers and lora_triggers.strip():
            lora_instruction = (
                f"\n[LORA NOTE: LoRA trigger words will be prepended automatically. "
                f"Do NOT include them in your output. Start directly with the style label or scene description.]"
            )
        else:
            lora_instruction = ""

        # ── Pacing instruction ────────────────────────────────────────────────
        length_instruction = (
            f"\n[PACING: {pacing_hint} "
            f"HARD WORD LIMIT: {token_val} words maximum — do not exceed this. "
            f"Do not exceed the action count above. "
            f"Output ends with the final sentence of the scene — no summaries, no counts, no closings, no meta-commentary, no brackets after the last word.]"
        )

        # ── Build action sequence instruction ────────────────────────────────
        if _action_key and _action_key in _ACTION_SEQUENCES:
            _seq_title, _seq_steps = _ACTION_SEQUENCES[_action_key]
            _seq_numbered = " ".join(f"{s}" for s in _seq_steps)

            # Inject tier notes — POV body-in-frame when preset is POV,
            # otherwise seed-picked raw/sensual/playful camera+energy+movement notes.
            _tier_suffix = ""
            if _is_pov and _action_key in _SEQ_POV_NOTES:
                _pov_body, _pov_spatial, _pov_action = _SEQ_POV_NOTES[_action_key]
                _tier_suffix = f" {_pov_body} {_pov_spatial} {_pov_action}"
            elif _action_key in _SEQ_TIER_NOTES:
                _tier_data = _SEQ_TIER_NOTES[_action_key].get(_seq_tier, {})
                if _tier_data:
                    _tier_camera, _tier_energy, _tier_movement = _tier_data
                    _tier_suffix = f" {_tier_camera} {_tier_energy} {_tier_movement}"

            # Soften orientation lock for dialogue AND for POV
            # (POV already rewrites spatial logic — the hard "faces AWAY" lock
            # from the core steps fights the first-person framing otherwise)
            _has_any_dialogue = has_user_dialogue or invent_dialogue
            _pov_orientation_note = (
                " POV ORIENTATION NOTE: The ORIENTATION lines above describe spatial relationship "
                "from a third-person perspective — reinterpret them for first-person POV. "
                "'She faces away' means her back is to YOU, the viewer. "
                "'Camera holds low' means YOU are at that height. "
                "'Static shot' means you are standing or seated still. "
                "Describe what the viewer sees and feels, not what a camera operator would frame."
                if _is_pov else ""
            )
            _dialogue_orientation_note = (
                " DIALOGUE OVERRIDE — ORIENTATION IS PRIMARY NOT ABSOLUTE: "
                "The orientation above describes the default position for this action. "
                "When the character speaks or reacts, she may turn her head, glance back, "
                "or shift her body to deliver the line — this is correct and expected. "
                "After speaking she returns to the primary orientation. "
                "The action continues through and around the dialogue moment — it does not stop."
                if _has_any_dialogue else ""
            )

            action_sequence_instruction = (
                f"\n\n[{_seq_title} "
                f"{_seq_numbered}"
                f"{_tier_suffix} "
                f"This action is the PRIMARY CONTENT of the scene — describe it with full physical specificity. "
                f"Do NOT summarise or skip steps. "
                f"Weave the physical mechanics into the prose — do not list them as bullet points in the output."
                f"{_pov_orientation_note}"
                f"{_dialogue_orientation_note}]"
            )
            _flags = []
            if _is_pov: _flags.append("POV")
            if _has_any_dialogue: _flags.append("dialogue override")
            print(f"[LTX2-Qwen] Action sequence: {_action_key} [{_seq_tier}]"
                  + (f" + {', '.join(_flags)}" if _flags else ""))
        else:
            action_sequence_instruction = ""

        # ── Coverage detection ────────────────────────────────────────────────
        # Scan what the user already provided across 7 dimensions.
        # Score 0 = not covered (full template fires), 1 = fully covered (suppress).
        # Partial coverage suppresses the verbose "invent" guidance but keeps the
        # hard constraint so the LLM doesn't contradict what the user wrote.
        _cov_input = (user_input + " " + _active_scene_context).lower()

        # 1. Character — age, hair, skin tone, build, eye colour
        _cov_character = bool(re.search(
            r'\b(\d{2}[\s-]year|year[s\s-]old|hair|skin|complexion|blonde|brunette|'
            r'redhead|auburn|tan(?:ned)?|pale|dark[- ]skin|light[- ]skin|'
            r'petite|slim|slender|curvy|muscular|athletic|build|figure|'
            r'eyes?(?:\s+\w+){0,2}(?:blue|green|brown|hazel|grey|gray|dark|bright))\b',
            _cov_input, re.IGNORECASE
        ))

        # 2. Location — venue, place, environment
        _cov_location = bool(re.search(
            r'\b(club|bar|venue|stage|studio|street|room|apartment|bedroom|hotel|'
            r'office|park|beach|warehouse|rooftop|basement|arena|festival|concert|'
            r'theatre|church|car|gym|garden|courtyard|corridor|forest|city|urban|'
            r'indoor|outdoor|alley|subway|train|bus|kitchen|bathroom|pool|'
            r'desert|mountain|field|cabin|mansion|castle|bridge|pier|dock)\b',
            _cov_input, re.IGNORECASE
        ))

        # 3. Clothing — garment words
        _cov_clothing = bool(re.search(
            r'\b(wear(?:ing|s)?|dress(?:ed)?|shirt|top|blouse|jacket|coat|suit|'
            r'jeans?|trousers?|shorts?|skirt|uniform|gown|bikini|swimsuit|'
            r'hoodie|sweater|crop\s*top|tank\s*top|leather|denim|silk|lace|'
            r'outfit|clothes?|attire|costume|lingerie|underwear|bra|'
            r'naked|nude|shirtless|bare(?:\s+chest|\s+skin|\s+midriff)?)\b',
            _cov_input, re.IGNORECASE
        ))

        # 4. Camera — shot type, angle, movement
        _cov_camera = bool(re.search(
            r'\b(close[- ]up|medium\s+shot|wide\s+shot|long\s+shot|establishing|'
            r'overhead|bird[- ]?s[- ]?eye|low\s+angle|high\s+angle|dutch\s+angle|'
            r'tracking\s+shot|dolly|pan(?:ning)?|tilt(?:ing)?|handheld|steadicam|'
            r'zoom(?:ing)?|rack\s+focus|shallow\s+depth|bokeh|slow\s+push|'
            r'crane\s+shot|aerial|drone|over[- ]the[- ]shoulder|pov\b|point\s+of\s+view)\b',
            _cov_input, re.IGNORECASE
        ))

        # 5. Mood / style — adjectives describing the aesthetic or tone
        _cov_mood = bool(re.search(
            r'\b(cinematic|moody|gritty|dreamy|surreal|dark|bright|vibrant|'
            r'melancholic|nostalgic|intimate|raw|polished|stylised|minimalist|'
            r'saturated|desaturated|high[- ]contrast|soft|harsh|golden|neon|'
            r'atmospheric|ethereal|hyper[- ]?real|documentary|film\s+noir|'
            r'warm\s+tone|cool\s+tone|vintage|retro|futuristic|brutal|tender)\b',
            _cov_input, re.IGNORECASE
        ))

        # 6. Action / motion — what the subject is doing
        _cov_action = bool(re.search(
            r'\b(walk(?:ing|s)?|run(?:ning|s)?|danc(?:ing|es?)?|sit(?:ting|s)?|'
            r'stand(?:ing|s)?|lean(?:ing|s)?|turn(?:ing|s)?|look(?:ing|s)?|'
            r'reach(?:ing|es)?|mov(?:ing|es?)?|jump(?:ing|s)?|fight(?:ing|s)?|'
            r'hold(?:ing|s)?|lift(?:ing|s)?|push(?:ing|es)?|pull(?:ing|s)?|'
            r'kiss(?:ing|es)?|embrace|hug(?:ging|s)?|crawl(?:ing|s)?|'
            r'perform(?:ing|s)?|act(?:ing|s)?|sings?|singing|plays?|playing)\b',
            _cov_input, re.IGNORECASE
        ))

        # 7. Audio / music — instruments, genre, or sound description
        _cov_audio = bool(re.search(
            r'\b(jazz|blues?|rock|pop|hip[- ]?hop|r&b|electronic|classical|'
            r'techno|house|ambient|folk|country|metal|punk|soul|gospel|'
            r'drum[s]?|bass|guitar|piano|violin|trumpet|saxophone|synth|'
            r'beat|bpm|tempo|loud|quiet|silence|echo|reverb|distort|'
            r'music\s+plays?|soundtrack|score|track|melody|rhythm)\b',
            _cov_input, re.IGNORECASE
        ))

        # Build a compact coverage note for the system prefix so the LLM knows
        # what it should invent vs what is already locked in.
        _cov_already  = []
        _cov_missing  = []
        _cov_map = [
            ("character appearance", _cov_character),
            ("location / setting",   _cov_location),
            ("clothing",             _cov_clothing),
            ("camera work",          _cov_camera),
            ("mood / visual style",  _cov_mood),
            ("action / movement",    _cov_action),
            ("audio / music",        _cov_audio),
        ]
        for _label, _hit in _cov_map:
            (_cov_already if _hit else _cov_missing).append(_label)

        _covered_count = len(_cov_already)

        if _covered_count >= 5:
            # Very detailed input — tell the LLM to fill gaps only
            _coverage_prefix = (
                "\n[COVERAGE NOTE — DETAILED INPUT: The user has already provided: "
                + ", ".join(_cov_already) + ". "
                "Do NOT reinvent or contradict any of these. "
                + (("Fill only what is genuinely missing: " + ", ".join(_cov_missing) + ". ") if _cov_missing else
                   "Everything is specified — expand and deepen the scene, do not invent new elements. ")
                + "Scale all template guidance below to match — suppress any instruction that invites you "
                "to invent something the user has already defined.]"
            )
        elif _covered_count >= 3:
            # Partially detailed — selective fill
            _coverage_prefix = (
                "\n[COVERAGE NOTE — PARTIAL INPUT: The user has specified: "
                + ", ".join(_cov_already) + ". "
                "Preserve these exactly. "
                + (("Invent to fill the remaining gaps: " + ", ".join(_cov_missing) + ".") if _cov_missing else "")
                + "]"
            )
        else:
            # Minimal input — full template fires, no suppression needed
            _coverage_prefix = ""

        # Suppress char_seed if character is already covered (belt-and-suspenders on top
        # of the existing _user_described_character check — catches cases like "a blonde
        # woman" that score _cov_character=True but might have slipped the narrower regex).
        if _cov_character and char_seed_note:
            char_seed_note = ""
            print("[LTX2-Qwen] Coverage: character covered — char seed suppressed.")

        # Suppress location suggestion inside music_sound_rule if user gave a location
        # (this is already partially handled by _has_user_loc inside music assembly,
        # but _cov_location is broader — covers non-music scenes too).
        # We don't modify music_sound_rule here since that block already handles it;
        # _coverage_prefix informs the LLM globally.

        print(f"[LTX2-Qwen] Coverage: {_covered_count}/7 dims covered "
              f"({', '.join(_cov_already) if _cov_already else 'none'})")

        # ── Build messages ────────────────────────────────────────────────────
        # Prepend coverage prefix to effective_input so the LLM sees it first
        if _coverage_prefix:
            effective_input = _coverage_prefix + "\n" + effective_input
        # ── User input authority wrapper ──────────────────────────────────────
        # The user's input is the absolute ground truth. Every instruction that
        # follows must serve it — not replace it, not contradict it, not expand
        # beyond it. The input is also repeated at the end so it is the last
        # thing the model reads before generating.
        _user_input_anchor_open = (
            "\n\n[USER SCENE — ABSOLUTE AUTHORITY: Everything below this line "
            "describes HOW to render the scene. The scene itself is defined entirely "
            "by the USER INPUT above. No instruction below may contradict, replace, "
            "or expand beyond what the user described. "
            "If the user said one person — one person. "
            "If the user said a cliff edge — the cliff edge must be visible. "
            "If the user said soaked — she is soaked. "
            "If the user said colour — it is in colour. "
            "The user input is the director. Every instruction below is crew.]"
        )
        _user_input_anchor_close = (
            "\n\n[FINAL REMINDER — Render the user's scene exactly as described above. "
            "Nothing added that the user did not describe. "
            "Nothing removed that the user did describe.]"
        )

        spoken_language_instruction = ""
        if spoken_language_sel and spoken_language_sel != "Auto — use existing prompt logic":
            spoken_language_instruction = (
                f"\n[SPOKEN CONTENT LANGUAGE — HARD REQUIREMENT: "
                f"Any spoken dialogue, sung lyrics, whispers, moans, chants, or other vocalised words "
                f"must be in {spoken_language_sel}. "
                f"The descriptive prompt prose itself must remain in English. "
                f"Do not switch the full prompt narration to {spoken_language_sel}; "
                f"only in-scene spoken or sung content changes language.]"
            )

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user",   "content": (
                effective_input
                + _user_input_anchor_open
                + lora_instruction
                + orientation_instruction
                + _ratio_instruction
                + _subject_count_instruction
                + _camera_lock_instruction
                + _negative_bias_instruction
                + _audio_instruction
                + _genre_world_instruction
                + _env_pool_instruction
                + _global_emotion_instruction
                + _kpop_group_instruction
                + _bg_instruction
                + style_instruction
                + portrait_instruction
                + _gravure_body_override
                + sequence_instruction
                + static_instruction
                + no_person_instruction
                + multi_instruction
                + _env_instruction
                + explicit_instruction
                + dialogue_instruction
                + spoken_language_instruction
                + lift_instruction
                + _exertion_instruction
                + action_sequence_instruction
                + length_instruction
                + _user_input_anchor_close
            )},
        ]

        # ── Generate ──────────────────────────────────────────────────────────
        try:
            if backend == "transformers":
                raw = self.tokenizer.apply_chat_template(
                    messages, return_tensors="pt",
                    add_generation_prompt=True
                )
                if hasattr(raw, "input_ids"):
                    input_ids = raw.input_ids.to(self.model.device)
                elif isinstance(raw, dict):
                    input_ids = raw["input_ids"].to(self.model.device)
                elif isinstance(raw, list):
                    input_ids = torch.tensor([raw], dtype=torch.long).to(self.model.device)
                else:
                    input_ids = raw.to(self.model.device)
                input_length = input_ids.shape[1]

                with torch.no_grad():
                    output_ids = self.model.generate(
                        input_ids,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        do_sample=True,
                        top_k=20,
                        top_p=0.82,
                        min_p=0.0,
                        repetition_penalty=1.05,
                        use_cache=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self._stop_token_ids
                    )

                result = self.tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True).strip()
                del output_ids, input_ids
                gc.collect()

            elif backend == "llama.cpp (GGUF)":
                response = self.model.create_chat_completion(
                    messages=messages,
                    temperature=temperature,
                    top_p=0.82,
                    top_k=20,
                    min_p=0.0,
                    repeat_penalty=1.05,
                    max_tokens=max_tokens,
                )
                result = response["choices"][0]["message"]["content"].strip()

            elif backend == "llama-server (OpenAI API)":
                base_url = (self.model.get("url") or "http://127.0.0.1:8080/v1").rstrip("/")
                endpoint = base_url + "/chat/completions"
                payload = {
                    "messages": messages,
                    "temperature": temperature,
                    "top_p": 0.82,
                    "max_tokens": max_tokens,
                }
                if self.model.get("model"):
                    payload["model"] = self.model["model"]

                request = urllib.request.Request(
                    endpoint,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                if self.model.get("api_key"):
                    request.add_header("Authorization", f"Bearer {self.model['api_key']}")

                with urllib.request.urlopen(request, timeout=300) as response:
                    body = response.read().decode("utf-8")
                decoded = json.loads(body)
                result = decoded["choices"][0]["message"]["content"].strip()

            else:
                raise RuntimeError(f"[LTX2-Qwen] Unsupported backend: {backend}")
        except Exception as e:
            print(f"[LTX2-Qwen] Generation error: {e}")
            self.unload_model()
            raise

        if not result or not result.strip():
            print("[LTX2-Qwen] Warning: empty generation — returning user input as fallback")
            result = user_input.strip()

        result = self._clean_output(result)

        # ── Hard word-count truncation ────────────────────────────────────────
        cap   = int(token_val * 1.05)
        words = result.split()
        if len(words) > cap:
            trunc = " ".join(words[:cap])
            # Find the last sentence-ending punctuation in the truncated text.
            # Require it to be past 40% of the string so we don't cut too early.
            # If no clean sentence boundary exists, strip trailing partial clause
            # punctuation (comma, semicolon, colon, em-dash) and close with a period.
            best = max((trunc.rfind(c) for c in ".!?"), default=-1)
            if best > int(len(trunc) * 0.4):
                result = trunc[:best + 1].strip()
            else:
                result = trunc.rstrip(",;:— ").strip()
                if result and result[-1] not in ".!?":
                    result += "."
            print(f"[LTX2-Qwen] Truncated: {len(words)} → {len(result.split())} words")

        # ── LoRA trigger hard prepend ─────────────────────────────────────────
        if lora_triggers and lora_triggers.strip():
            triggers = lora_triggers.strip()
            if result.lower().startswith(triggers.lower()):
                result = result[len(triggers):].lstrip(" ,—-")
            result = triggers + ", " + result
            print(f"[LTX2-Qwen] LoRA triggers prepended: {triggers}")

        # ── Style label safety net ────────────────────────────────────────────
        if style_label:
            # Strip a leading duplicate if the LLM already emitted the label
            # (handles cases where the label appears twice at the start).
            _label_lower  = style_label.lower().rstrip(". ")
            _result_lower = result.lower().lstrip()
            if _result_lower.startswith(_label_lower):
                # Label present once — strip it so we can re-prepend cleanly below
                result = result[len(style_label):].lstrip(" .,")
                _result_lower = result.lower().lstrip()
            # Now check if the stripped result still starts with the label (double print)
            # and strip again if so
            if _result_lower.startswith(_label_lower):
                result = result[len(style_label):].lstrip(" .,")
            # Always prepend the canonical label
            result = style_label + " " + result.lstrip()
            print(f"[LTX2-Qwen] Style label applied: {style_label}")

        # ── Negative prompt ───────────────────────────────────────────────────
        neg = _build_negative_prompt(result, user_input, is_portrait=is_portrait, style_preset=style_preset)
        if negative_bias and negative_bias.strip():
            neg = neg + ", " + negative_bias.strip()

        # ── Always offload — every run, no exceptions ─────────────────────────
        # keep_model_loaded widget is retained in the UI for legacy compatibility
        # but the node always offloads so VRAM is fully freed before LTX runs.
        self.unload_model()

        print(f"[LTX2-Qwen] Done — {len(result.split())} words")
        return (result, result, neg)


# ── Utility: manual unload node ───────────────────────────────────────────────

class LTX2UnloadModelQwen:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}
    RETURN_TYPES = ()
    FUNCTION     = "unload"
    CATEGORY     = "LTX2"
    OUTPUT_NODE  = True

    def unload(self):
        freed = 0
        for obj in gc.get_objects():
            if isinstance(obj, LTX2PromptArchitectQwen) and obj.model is not None:
                obj.unload_model()
                freed += 1
        print(f"[LTX2-Qwen] Unload node: freed {freed} instance(s).")
        return ()


# ── ComfyUI registration ──────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "LTX2PromptArchitectQwen": LTX2PromptArchitectQwen,
    "LTX2UnloadModelQwen":     LTX2UnloadModelQwen,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTX2PromptArchitectQwen": "LTX-2.3 Easy Prompt Qwen By LoRa-Daddy",
    "LTX2UnloadModelQwen":     "LTX2 Unload Model (Qwen)",
}
