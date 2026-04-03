import gc
import os
import io
import json
import base64
import torch
import numpy as np
from PIL import Image
import urllib.request
import urllib.error
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# ── HuggingFace housekeeping ─────────────────────────────────────────────────
# Only disable telemetry at import time — safe, does not block downloads.
# Offline/online state is controlled per-run via the offline_mode toggle.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
# ─────────────────────────────────────────────────────────────────────────────


def comfy_tensor_to_pil(tensor) -> Image.Image:
    if tensor.ndim == 4:
        tensor = tensor[0]
    arr = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def pil_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


# One shared instance so VRAM is reused if node runs multiple times
_INSTANCE = {"processor": None, "model": None, "source": None, "backend": None, "device": None}

DESCRIBE_PROMPT = (
    "Describe this image in one paragraph of plain sentences, around 100-130 words. "
    "Start with 'Style: photorealistic' or 'Style: anime' or 'Style: 3D animation' etc. "
    "Then describe the person — your FIRST sentence about the person MUST explicitly state their ethnicity and skin tone using plain terms "
    "such as 'a Black man', 'a white woman', 'a South Asian man', 'an East Asian woman', 'a Latino man', 'a Middle Eastern woman' etc. "
    "Do not skip, soften, or omit this — it is mandatory. "
    "Then continue with their age, hair colour and style, body type, "
    "what they are wearing or doing, and any exposed body parts you can see — "
    "use plain words: breasts, nipples, pussy, penis, anus, buttocks. "
    "Describe their pose, what they are on or interacting with, "
    "the camera framing and angle, the lighting and time of day, and the setting. "
    "Write it as one flowing paragraph. Do not use bullet points, lists, or labels. "
    "If there is no person in the image, describe the scene instead — the environment, setting, lighting, time of day, mood, and any notable objects or details."
)



MODEL_OPTIONS = {
    "Qwen2.5-VL-3B — Fast (huihui abliterated)":    "huihui-ai/Qwen2.5-VL-3B-Instruct-abliterated",
    "Qwen2.5-VL-7B — Better NSFW (prithiv caption)": "prithivMLmods/Qwen2.5-VL-7B-Abliterated-Caption-it",
}


def _resolve_model_target(gpu_id: int):
    try:
        if isinstance(gpu_id, str):
            gpu_id = gpu_id.strip()
            gpu_id = int(gpu_id) if gpu_id else -1
    except Exception:
        gpu_id = -1
    if not torch.cuda.is_available():
        return "cpu", "cpu"
    if gpu_id is None or gpu_id < 0:
        current_idx = torch.cuda.current_device()
        return "auto", f"cuda:{current_idx}"
    cuda_count = torch.cuda.device_count()
    if gpu_id >= cuda_count:
        fallback_idx = torch.cuda.current_device()
        print(
            f"[VisionDescribe] Requested GPU {gpu_id} is unavailable. "
            f"Falling back to cuda:{fallback_idx}."
        )
        return f"cuda:{fallback_idx}", f"cuda:{fallback_idx}"
    return f"cuda:{gpu_id}", f"cuda:{gpu_id}"


def _get_torch_dtype_for_target(target_device: str):
    if target_device == "cpu" or not torch.cuda.is_available():
        return torch.float32
    if target_device == "auto":
        return torch.float16

    try:
        gpu_index = int(str(target_device).split(":")[-1])
    except Exception:
        gpu_index = torch.cuda.current_device()

    try:
        major, _minor = torch.cuda.get_device_capability(gpu_index)
        if major >= 8:
            return torch.bfloat16
    except Exception as e:
        print(f"[VisionDescribe] Could not inspect {target_device} capability: {e}")

    return torch.float16


def _build_model_load_kwargs(target_device: str, offline_mode: bool, dtype):
    load_kwargs = {
        "dtype": dtype,
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


class LTX2VisionDescribe:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Connect your starting image here. The vision model will analyse it and output a scene description for use with the Easy Prompt node."}),
                "🖼 use image vision?": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "When ON: runs the vision model and outputs a scene description. Turn OFF to skip the vision model and return an empty string without rewiring."
                }),
                "🧩 backend": (["local transformers", "openai compatible server"], {
                    "default": "local transformers",
                    "tooltip": "Choose whether to run the local Qwen2.5-VL model or call an OpenAI-compatible vision server such as llama.cpp."
                }),
                "model_name": (list(MODEL_OPTIONS.keys()), {
                    "default": "Qwen2.5-VL-3B — Fast (huihui abliterated)",
                    "tooltip": "3B is faster and uses ~6GB VRAM. 7B is slower but describes explicit content more accurately. Both download automatically on first run."
                }),
                "🌐 server url": ("STRING", {
                    "default": "http://127.0.0.1:8080/v1",
                    "multiline": False,
                    "placeholder": "http://127.0.0.1:8080/v1",
                    "tooltip": "Used only with the server backend. Base URL of the OpenAI-compatible multimodal server."
                }),
                "🌐 server model": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "optional model name",
                    "tooltip": "Used only with the server backend. Optional model value sent to /chat/completions."
                }),
                "🔑 server api key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "optional bearer token",
                    "tooltip": "Used only with the server backend. Optional bearer token for authenticated OpenAI-compatible endpoints."
                }),
                "offline_mode": ("BOOLEAN", {"default": False, "tooltip": "Turn ON if you have no internet connection. Uses locally cached models only. Leave OFF to allow automatic download from HuggingFace on first run."}),
                "local_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Optional: local snapshot path (overrides model dropdown)",
                    "tooltip": "Optional. Paste the full path to a locally downloaded model snapshot folder. This overrides the model dropdown above. Leave blank to use HuggingFace cache automatically."
                }),
                "gpu_id": ("INT", {
                    "default": -1, "min": -1, "max": 15, "step": 1,
                    "display": "number",
                    "tooltip": "Choose which CUDA GPU should load this vision model. Set -1 for automatic placement, or 0/1/2... to pin the node to a specific GPU."
                }),
            },
        }

    RETURN_TYPES  = ("STRING",)
    RETURN_NAMES  = ("scene_context",)
    FUNCTION      = "describe"
    CATEGORY      = "LTX2"

    def describe(self, image, **kwargs):
        use_image_vision = kwargs.get("🖼 use image vision?", True)
        backend       = kwargs.get("🧩 backend", "local transformers")
        model_name    = kwargs.get("model_name", list(MODEL_OPTIONS.keys())[0])
        server_url    = kwargs.get("🌐 server url", "http://127.0.0.1:8080/v1")
        server_model  = kwargs.get("🌐 server model", "")
        server_api_key = kwargs.get("🔑 server api key", "")
        offline_mode  = kwargs.get("offline_mode", False)
        local_path    = kwargs.get("local_path", "")
        gpu_id        = kwargs.get("gpu_id", -1)
        bypass = not use_image_vision
        if bypass:
            print("[VisionDescribe] Vision disabled — returning empty string.")
            return ("",)

        global _INSTANCE

        pil_image = comfy_tensor_to_pil(image)
        print(f"[VisionDescribe] Image: {pil_image.size}")

        if backend == "openai compatible server":
            base_url = (server_url or "http://127.0.0.1:8080/v1").rstrip("/")
            endpoint = base_url + "/chat/completions"
            payload = {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an image description tool for an AI video pipeline. "
                            "Describe exactly what you see in plain factual prose. "
                            "Be direct and accurate. Do not embellish or invent details."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": DESCRIBE_PROMPT},
                            {"type": "image_url", "image_url": {"url": pil_to_data_url(pil_image)}},
                        ],
                    },
                ],
                "max_tokens": 180,
                "temperature": 0.3,
                "top_p": 0.9,
            }
            if server_model and server_model.strip():
                payload["model"] = server_model.strip()

            request = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            if server_api_key and str(server_api_key).strip():
                request.add_header("Authorization", f"Bearer {str(server_api_key).strip()}")
            with urllib.request.urlopen(request, timeout=300) as response:
                body = response.read().decode("utf-8")
            decoded = json.loads(body)
            content = decoded["choices"][0]["message"]["content"]
            if isinstance(content, list):
                description = "".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                ).strip()
            else:
                description = str(content).strip()
        else:
            hf_id = MODEL_OPTIONS[model_name]

            # ── Offline env ───────────────────────────────────────────────────
            if offline_mode:
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                os.environ["HF_DATASETS_OFFLINE"] = "1"
                os.environ["HF_HUB_OFFLINE"] = "1"
            else:
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
                os.environ.pop("HF_DATASETS_OFFLINE", None)
                os.environ.pop("HF_HUB_OFFLINE", None)

            # ── Resolve source ────────────────────────────────────────────────
            if local_path and local_path.strip():
                source = local_path.strip()
            elif offline_mode:
                source = hf_id
            else:
                try:
                    from huggingface_hub import snapshot_download
                    source = snapshot_download(hf_id)
                except Exception as e:
                    print(f"[VisionDescribe] Download failed: {e}")
                    source = hf_id

            # ── Load if needed ────────────────────────────────────────────────
            target_device, input_device = _resolve_model_target(gpu_id)

            if (
                _INSTANCE["model"] is None
                or _INSTANCE["source"] != source
                or _INSTANCE.get("device") != target_device
                or _INSTANCE.get("backend") != backend
            ):
                # Clear any previous instance first
                if _INSTANCE["model"] is not None:
                    try:
                        _INSTANCE["model"].to("cpu")
                    except Exception:
                        pass
                    _INSTANCE["model"]     = None
                    _INSTANCE["processor"] = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                dtype = _get_torch_dtype_for_target(target_device)
                print(f"[VisionDescribe] Loading {model_name} on {target_device}...")

                _INSTANCE["processor"] = AutoProcessor.from_pretrained(
                    source, local_files_only=offline_mode
                )
                load_kwargs = _build_model_load_kwargs(target_device, offline_mode, dtype)
                print(f"[VisionDescribe] Loading with dtype={dtype} and device_map={load_kwargs['device_map']}")

                try:
                    _INSTANCE["model"] = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        source,
                        **load_kwargs,
                    )
                except Exception as e:
                    err = str(e).lower()
                    should_retry_fp16 = (
                        dtype == torch.bfloat16
                        and "cuda" in err
                        and ("invalid argument" in err or "acceleratorerror" in err)
                    )
                    if not should_retry_fp16:
                        raise

                    retry_dtype = torch.float16
                    retry_kwargs = _build_model_load_kwargs(target_device, offline_mode, retry_dtype)
                    print(
                        f"[VisionDescribe] BF16 load failed on {target_device}. "
                        f"Retrying with dtype={retry_dtype} and device_map={retry_kwargs['device_map']}."
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                    _INSTANCE["model"] = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        source,
                        **retry_kwargs,
                    )
                    dtype = retry_dtype
                if target_device not in ("cpu", "auto"):
                    print(f"[VisionDescribe] Moving model to {target_device} after CPU load...")
                    _INSTANCE["model"].to(target_device, dtype=dtype)
                _INSTANCE["model"].eval()
                _INSTANCE["source"] = source
                _INSTANCE["device"] = target_device
                _INSTANCE["backend"] = backend
                print("[VisionDescribe] Loaded.")

            processor = _INSTANCE["processor"]
            model     = _INSTANCE["model"]

            # ── Single inference ──────────────────────────────────────────────
            try:
                from qwen_vl_utils import process_vision_info
            except ImportError:
                raise ImportError("[VisionDescribe] Missing: qwen-vl-utils. Fix: pip install qwen-vl-utils then restart ComfyUI.")

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an image description tool for an AI video pipeline. "
                        "Describe exactly what you see in plain factual prose. "
                        "Be direct and accurate. Do not embellish or invent details."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text",  "text":  DESCRIBE_PROMPT},
                    ],
                },
            ]

            text_input = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(input_device)

            input_len = inputs["input_ids"].shape[1]

            tok = processor.tokenizer
            stop_ids = []
            if tok.eos_token_id is not None:
                stop_ids.append(tok.eos_token_id)
            for s in ["<|im_end|>", "<|endoftext|>"]:
                ids = tok.encode(s, add_special_tokens=False)
                if len(ids) == 1 and ids[0] not in stop_ids:
                    stop_ids.append(ids[0])

            pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=180,
                    temperature=0.3,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=pad_id,
                    eos_token_id=stop_ids,
                )

            new_tokens = out[0][input_len:]
            description = tok.decode(new_tokens, skip_special_tokens=True).strip()

            del out, inputs

        print(f"[VisionDescribe] Output: {len(description.split())} words.")

        # ── Unload immediately to free VRAM for the text node ─────────────────
        print("[VisionDescribe] Unloading — full hard VRAM free...")

        # Step 1: destroy every tensor in place so CUDA allocator releases pages
        if _INSTANCE["model"] is not None:
            try:
                for _name, module in list(_INSTANCE["model"].named_modules()):
                    for _pname, param in list(module.named_parameters(recurse=False)):
                        try:
                            param.data = torch.empty(0)
                        except Exception:
                            pass
                    for _bname, buf in list(module.named_buffers(recurse=False)):
                        try:
                            module._buffers[_bname] = None
                        except Exception:
                            pass
            except Exception as e:
                print(f"[VisionDescribe] Tensor destroy warning: {e}")

        # Step 2: delete Python references
        try:
            del _INSTANCE["model"]
        except Exception:
            pass
        try:
            del _INSTANCE["processor"]
        except Exception:
            pass

        _INSTANCE["model"]     = None
        _INSTANCE["processor"] = None
        _INSTANCE["source"]    = None
        _INSTANCE["backend"]   = None
        _INSTANCE["device"]    = None

        # Step 3: triple gc — catches circular refs from transformers internals
        gc.collect()
        gc.collect()
        gc.collect()

        # Step 4: full CUDA flush — same sequence as EasyPromptLD
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                # Reset the caching allocator — releases "reserved but not allocated" block
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"[VisionDescribe] CUDA flush warning: {e}")

        # Step 5: tell ComfyUI model manager to drop everything it holds too
        try:
            import comfy.model_management as mm
            mm.unload_all_models()
            mm.soft_empty_cache()
            print("[VisionDescribe] ComfyUI mm.unload_all_models + soft_empty_cache done.")
        except Exception as e:
            print(f"[VisionDescribe] ComfyUI mm call skipped: {e}")

        # Step 6: log final VRAM state
        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved  = torch.cuda.memory_reserved()  / 1024**3
                print(f"[VisionDescribe] VRAM after free: {allocated:.2f}GB allocated / {reserved:.2f}GB reserved")
            except Exception:
                pass
        else:
            print("[VisionDescribe] Model unloaded (no CUDA).")

        return (description,)


NODE_CLASS_MAPPINGS = {
    "LTX2VisionDescribe": LTX2VisionDescribe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTX2VisionDescribe": "LTX-2 Vision Describe By LoRa-Daddy",
}
