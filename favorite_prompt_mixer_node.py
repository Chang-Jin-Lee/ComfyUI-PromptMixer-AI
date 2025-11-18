# Save as: ComfyUI/custom_nodes/favorite_prompt_mixer_node.py
import json, random, re, requests, traceback, os, hashlib
from typing import List, Dict, Any, Tuple

# ---------- Comfy enums & helpers ----------
try:
    import comfy.samplers as comfy_samplers
    SAMPLER_ENUM   = comfy_samplers.KSampler.SAMPLERS
    SCHEDULER_ENUM = comfy_samplers.KSampler.SCHEDULERS
except Exception as e:
    SAMPLER_ENUM   = ("euler", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde")
    SCHEDULER_ENUM = ("normal", "karras", "exponential", "sgm_uniform")
    print("[FavoritePromptMixer] WARNING: comfy.samplers not found. Using fallback enum lists.", e)

# comfy core
try:
    import folder_paths
    import comfy.utils as cutils
    import comfy.sd as csd
    import nodes as comfy_nodes
except Exception as e:
    folder_paths = None
    cutils = None
    csd = None
    comfy_nodes = None
    print("[FavoritePromptMixer] WARNING: comfy internals not found; checkpoint/LoRA loading unavailable.", e)

# ========= (A) 프롬프트 블록 파서 =========
def _parse_user_prompts_block(block: str) -> List[str]:
    if not block or not block.strip():
        return []
    chunks = re.split(r"\n\s*\n+", block.strip(), flags=re.S)
    prompts: List[str] = []
    seen = set()
    for chunk in chunks:
        lines = []
        for raw in chunk.splitlines():
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            lines.append(s)
        if not lines:
            continue
        merged = " ".join(lines).strip()
        if merged and merged not in seen:
            prompts.append(merged)
            seen.add(merged)
    return prompts

# ========= (B) 헤더 파서 =========
_HEADER_PAIR_RE = re.compile(r"""
    (?P<key>[A-Za-z_][A-Za-z0-9_\- ]*)
    \s*[:=]\s*
    (?P<val>[^,;]+)
""", re.X)
_LORA_BLOCK_RE = re.compile(r"(?:\b(?:lora|loras|models)\b)\s*:\s*\{(?P<body>.*?)\}", re.I | re.S)

_INVISIBLE = dict.fromkeys(map(ord, [
    "\u200b","\u200c","\u200d","\ufeff","\u202a","\u202b","\u202c","\u202d","\u202e"
]), None)

def _strip_invisible(s: str) -> str:
    if not isinstance(s, str):
        return s
    return s.translate(_INVISIBLE)

def _parse_weight_pair(val: Any) -> Tuple[float, float]:
    """
    다양한 입력을 유연하게 해석:
      - 0.8                 -> (0.8, 0.8)
      - "0.7"               -> (0.7, 0.7)
      - "unet=0.6, te=0.5"  -> (0.6, 0.5)
      - {"unet":0.7,"te":0.4}-> (0.7, 0.4)
    기본값: (1.0, 1.0)
    """
    def _f(x, d):
        try: return float(x)
        except: return d

    if isinstance(val, (int, float)):
        f = float(val)
        return (f, f)

    if isinstance(val, dict):
        u = _f(val.get("unet", val.get("model", 1.0)), 1.0)
        t = _f(val.get("te", val.get("clip", 1.0)), 1.0)
        return (u, t)

    if isinstance(val, str):
        s = val.strip()
        if re.fullmatch(r"[+-]?(\d+(\.\d+)?|\.\d+)", s):
            f = float(s); return (f, f)
        # "unet=0.6, te=0.5" 같은 형태
        u = re.search(r"(?:unet|model)\s*=\s*([+-]?\d+(?:\.\d+)?)", s, re.I)
        t = re.search(r"(?:te|clip)\s*=\s*([+-]?\d+(?:\.\d+)?)", s, re.I)
        uu = float(u.group(1)) if u else 1.0
        tt = float(t.group(1)) if t else 1.0
        return (uu, tt)

    return (1.0, 1.0)

def _parse_lora_block(first_line: str) -> Dict[str, Tuple[float, float]]:
    """
    리턴: { "lora_name" : (unet_w, te_w), ... }
    """
    out: Dict[str, Tuple[float,float]] = {}
    if not first_line:
        return out
    m = _LORA_BLOCK_RE.search(_strip_invisible(first_line))
    if not m:
        return out
    body = m.group("body")
    # split by comma/semicolon but be tolerant with inner braces
    parts = re.split(r"(?<!\})\s*[,\n;]\s*(?![^{}]*\})", body.strip())
    for part in parts:
        if not part or ":" not in part:
            continue
        k, v = part.split(":", 1)
        name = k.strip().strip('"').strip("'")
        val_raw = v.strip()
        # dict-like?
        if val_raw.startswith("{") and val_raw.endswith("}"):
            try:
                obj = json.loads(val_raw.replace("'", '"'))
            except Exception:
                # fallback tolerant parse "unet:0.7, te:0.5"
                obj = {}
                for s in re.split(r"[,\n;]+", val_raw.strip("{} \t")):
                    if ":" in s:
                        kk, vv = s.split(":", 1)
                        obj[kk.strip()] = vv.strip()
            uw, tw = _parse_weight_pair(obj)
        else:
            uw, tw = _parse_weight_pair(val_raw)
        if name:
            out[name] = (uw, tw)
    return out

def _extract_header_and_body(text: str) -> Tuple[Tuple[str,int,float,str,str,str,Dict[str,Tuple[float,float]]], str]:
    """
    리턴: ((name, steps, cfg, sampler, scheduler, checkpoint, loras_dict), body_text)
    loras_dict: { lora_name : (unet_w, te_w) }
    """
    DEFAULTS = ("", 30, 7.0, "euler_ancestral", "karras", "", {})
    if not text:
        return DEFAULTS, ""
    lines = text.splitlines()

    first_idx = None
    for i, raw in enumerate(lines):
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        first_idx = i
        break
    if first_idx is None:
        return DEFAULTS, text

    first = _strip_invisible(lines[first_idx].strip())
    lowered = first.lower()
    if not any(k in lowered for k in ("name","steps","cfg","sampler","scheduler","checkpoint","lora","loras","models")):
        return DEFAULTS, text

    name_val, steps, cfg, sampler, scheduler, checkpoint = "", 30, 7.0, "euler_ancestral", "karras", ""
    loras: Dict[str,Tuple[float,float]] = {}
    try:
        for m in _HEADER_PAIR_RE.finditer(first):
            key = m.group("key").strip().lower()
            val = m.group("val").strip().strip('"').strip("'")
            if key == "name":
                name_val = val
            elif key == "steps":
                try: steps = int(float(val))
                except: pass
            elif key in ("cfg","cfg_scale","guidance","scale"):
                try: cfg = float(val)
                except: pass
            elif key in ("sampler","sampler_name"):
                sampler = val
            elif key in ("scheduler","schedule","sched"):
                scheduler = val
            elif key in ("checkpoint","ckpt","model","checkpoint_name"):
                checkpoint = val
    except Exception:
        traceback.print_exc()

    try:
        loras = _parse_lora_block(first)
    except Exception:
        traceback.print_exc()
        loras = {}

    body_lines = lines[:first_idx] + lines[first_idx+1:]
    body = "\n".join(body_lines)
    return (name_val, steps, cfg, sampler, scheduler, checkpoint, loras), body

# ========= (C) enum 키 노멀라이즈 =========
def _normalize_choice(enum_obj, name: str, default_key: str) -> str:
    if not name:
        return default_key
    n = name.strip().lower().replace(" ", "")
    keys = list(enum_obj.keys()) if hasattr(enum_obj, "keys") else list(enum_obj)
    for k in keys:
        if k.strip().lower().replace(" ", "") == n:
            return k
    return default_key

# ========= (D) 파일명 매칭 유틸 =========
def _resolve_from_list(target: str, avail: List[str]) -> str:
    if not target:
        return target
    t = _strip_invisible(target).strip().lower()
    # exact
    for a in avail:
        if a.lower() == t:
            return a
    # basename equal
    tbase = t.rsplit(".", 1)[0]
    for a in avail:
        if a.lower().rsplit(".", 1)[0] == tbase:
            return a
    # prefix
    for a in avail:
        if a.lower().startswith(tbase):
            return a
    return target

def _resolve_checkpoint_filename(name: str) -> str:
    if folder_paths is None or not name:
        return name
    return _resolve_from_list(name, folder_paths.get_filename_list("checkpoints"))

def _resolve_lora_filename(name: str) -> str:
    if folder_paths is None or not name:
        return name
    return _resolve_from_list(name, folder_paths.get_filename_list("loras"))

# ========= (E) Checkpoint 로드 & LoRA 적용 =========
def _load_checkpoint_clean(ckpt_name: str):
    if folder_paths is None or csd is None:
        raise RuntimeError("Comfy internals unavailable; cannot load checkpoint.")
    if not ckpt_name:
        raise RuntimeError("checkpoint is not specified in header.")

    ckpt_name = _resolve_checkpoint_filename(ckpt_name)
    ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
    print(f"[FavoritePromptMixer] Loading checkpoint: {ckpt_name} -> {ckpt_path}")

    # 1차: guess_config (모델/CLIP/VAE 모두 산출)
    try:
        out = csd.load_checkpoint_guess_config(
            ckpt_path,
            None, None,
            True,   # output_vae
            True,   # output_clip
            True    # embedding_directory (auto)
        )
        model, clip, vae = out[0], out[1], out[2]
        if clip is not None:
            return model, clip, vae, ckpt_name
        else:
            print("[FavoritePromptMixer] WARN: clip is None after guess_config. Falling back...")
    except Exception as e:
        print("[FavoritePromptMixer] WARN: guess_config failed:", e)

    # 2차: CheckpointLoaderSimple 폴백
    if comfy_nodes is None:
        raise RuntimeError("Cannot import Comfy nodes for fallback loader, and clip is None.")
    try:
        loader = comfy_nodes.CheckpointLoaderSimple()
        model, clip, vae = loader.load_checkpoint(ckpt_name)
        if clip is None:
            raise RuntimeError("Fallback loader also returned clip=None.")
        return model, clip, vae, ckpt_name
    except Exception as e:
        raise RuntimeError(f"Checkpoint fallback load failed: {e}")

def _apply_loras(model, clip, loras: Dict[str, Tuple[float,float]]):
    """
    loras: { name: (unet_w, te_w) }
    Comfy의 load_lora_for_models를 그대로 사용 (SDXL/SD1.x 모두 커버).
    """
    if not loras or csd is None or cutils is None or folder_paths is None:
        return model, clip

    for raw_name, (w_unet, w_te) in loras.items():
        lname = _resolve_lora_filename(raw_name)
        try:
            lpath = folder_paths.get_full_path("loras", lname)
        except Exception:
            print(f"[FavoritePromptMixer] LoRA not found in loras dir: '{raw_name}' (resolved '{lname}')")
            continue

        print(f"[FavoritePromptMixer] Applying LoRA: {raw_name} -> {lname} | unet={w_unet}, te={w_te}")
        try:
            lobj = cutils.load_torch_file(lpath, safe_load=True)
            model, clip = csd.load_lora_for_models(model, clip, lobj, float(w_unet), float(w_te))
        except Exception as e:
            # shape mismatch 등 문제 발생 시 나머지는 계속 적용
            print(f"[FavoritePromptMixer] Failed to apply LoRA '{raw_name}' -> '{lname}': {e}")
            traceback.print_exc()
    return model, clip

# ======================================================================================

class FavoritePromptMixerNode:
    """
    Favorite Prompt Mixer
    (+Header → name/steps/cfg/sampler/scheduler/checkpoint/LoRA
     → MODEL/CLIP/VAE & positive_text/negative_text)
    - positive/negative는 TEXT로만 출력합니다.
    - CLIP 인코딩은 외부의 'CLIP Text Encode' 노드에서 수행하세요.
    - 본 노드는 매 실행시 체크포인트를 재로드하여 이전 LoRA 잔존을 방지합니다.
    """

    @classmethod
    def INPUT_TYPES(cls):
        header_example = (
            "name : bluearchive, steps : 24, cfg : 7.0, sampler : euler_ancestral, scheduler : karras, "
            "checkpoint : unholyDesireMixSinister_v50.safetensors, "
            "lora : { ppw_v8_Illuv1_128.safetensors : 0.70, rouge-nikke-richy-v1_ixl.safetensors : {unet:0.8, te:0.5} }"
        )
        neg_default = "lowres, bad anatomy, bad hands, text, watermark, jpeg artifacts, out of frame, extra fingers, mutated hands, poorly drawn"
        return {
            "required": {
                "prompts_text": ("STRING", {
                    "multiline": True,
                    "default": header_example + "\n\n# --- 프롬프트 ---\nmasterpiece, best quality, 1girl, solo, full body",
                    "placeholder": header_example + "\n\n# --- 프롬프트 ---\n..."
                }),
                "always_prefix": ("STRING", {"default": "masterpiece, best quality"}),
                "always_suffix": ("STRING", {"default": ""}),
                "negative_text": ("STRING", {"default": neg_default, "multiline": True}),
                "backend": (["ollama", "openai_compat"],),
                "endpoint": ("STRING", {"default": "http://localhost:11434"}),
                "model": ("STRING", {"default": "llama3.1:8b-instruct-q4_K_M"}),
                "n_candidates": ("INT", {"default": 3, "min": 1, "max": 8}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
                "timeout_sec": ("INT", {"default": 20, "min": 5, "max": 120}),
                "theme": ("STRING", {"default": ""}),
                "style_lock": ("BOOLEAN", {"default": True}),
                "allow_palette": ("BOOLEAN", {"default": True}),
                "allow_style_notes": ("BOOLEAN", {"default": True}),
                "allow_camera": ("BOOLEAN", {"default": True}),
                "allow_lighting": ("BOOLEAN", {"default": True}),
                "select_mode": (["random", "index"],),
                "index": ("INT", {"default": 0, "min": 0, "max": 9999}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "lock_appearance": ("BOOLEAN", {"default": True}),
            }
        }

    # 출력:
    # name, steps, cfg, sampler, scheduler, model, clip, vae, positive_text, negative_text, checkpoint_name
    RETURN_TYPES = ("STRING","INT","FLOAT", SAMPLER_ENUM, SCHEDULER_ENUM, "MODEL","CLIP","VAE","STRING","STRING","STRING")
    RETURN_NAMES = ("name","steps","cfg","sampler_name","scheduler","model","clip","vae","positive_text","negative_text","checkpoint_name")
    FUNCTION = "build"
    CATEGORY = "Prompt/Favorites+AI"

    # ---------- Backends ----------
    def _ollama_chat(self, endpoint, model, messages, temperature, timeout):
        url = endpoint.rstrip("/") + "/api/chat"
        payload = {"model": model, "messages": messages, "stream": False, "options": {"temperature": float(temperature)}}
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return data.get("message", {}).get("content", "")

    def _openai_compat(self, endpoint, model, messages, temperature, timeout):
        url = endpoint.rstrip("/") + "/v1/chat/completions"
        payload = {"model": model, "messages": messages, "temperature": float(temperature), "stream": False}
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

    # ---------- Prompt templates ----------
    def _system_prompt(self):
        return (
            "You are a creative prompt designer for anime image generation.\n"
            "Given a strong base positive prompt, produce the most coherent, stylish additions:\n"
            "pose, outfit, background, optionally palette, style_notes, camera and lighting.\n"
            "Return VALID JSON only. SFW. Keep each field concise (<10 words)."
        )

    def _user_prompt(self, base: str, theme: str, style_lock: bool,
                     n_candidates: int, locked_text: str, lock_appearance: bool,
                     want_camera: bool, want_lighting: bool) -> str:
        lock_note = "strictly follow the base keywords and style" if style_lock else "loosely use the base as inspiration"
        theme_note = f"Theme: {theme}" if theme else "Theme: (none)"
        lock_clause = (
            f"""
Locked attributes from base (MUST stay EXACTLY as-is, no additions or changes):
{locked_text}

Rules:
- Do NOT add or change any tokens mentioning hair or eyes.
- Do NOT imply different hair/eye color through palette, lighting or style_notes.
"""
            if lock_appearance else
            "You may freely suggest variations, but keep coherence with the base prompt."
        )
        camera_clause = "- camera: optional composition notes\n" if want_camera else ""
        lighting_clause = "- lighting: optional lighting setup\n" if want_lighting else ""
        return f"""
Base (positive) prompt:
{base}

{theme_note}
Guideline: {lock_note}
{lock_clause}

TASK:
Generate {n_candidates} coherent candidates as JSON (key 'candidates'), each with:
- pose
- outfit
- background
- palette (optional)
- style_notes (optional)
{camera_clause}{lighting_clause}
Constraints:
- Concise (<10 words each)
- SFW only
- Output JSON only
"""

    # ---------- JSON parser ----------
    def _safe_json(self, text: str) -> Dict[str, Any]:
        cleaned = text.strip().strip("`").strip()
        m = re.search(r"\{.*\}", cleaned, re.S)
        if m:
            cleaned = m.group(0)
        return json.loads(cleaned)

    # ---------- Fallback ----------
    def _fallback_candidate(self) -> Dict[str, str]:
        poses = ["standing, hands at sides","kneeling, hands on lap","sitting on chair","walking forward","leaning on wall"]
        outfits = ["casual hoodie","school uniform","kimono with obi","futuristic bodysuit","elegant dress","armor suit"]
        bgs = ["in a cafe","on a beach","in a forest","futuristic city","rooftop at night"]
        return {
            "pose": random.choice(poses),
            "outfit": random.choice(outfits),
            "background": random.choice(bgs),
            "palette": "pastel pink and white",
            "style_notes": "soft rim light",
            "camera": "medium shot, eye-level",
            "lighting": "soft key, subtle rim"
        }

    # ---------- Helpers ----------
    _HAIR_EYE_KEYS = ("hair", "eyes")
    _HAIR_STYLE_HINTS = ("bangs","fringe","ponytail","twin tails","twintails","twin-tails","bun","hair bun","ahoge","bob","pixie","mohawk","long hair","short hair")

    def _extract_locked_tokens_from_base(self, base: str) -> List[str]:
        tokens = [t.strip() for t in re.split(r"[,\n/]+", base) if t.strip()]
        locked = []
        def hit(token: str) -> bool:
            tl = token.lower()
            if any(k in tl for k in self._HAIR_EYE_KEYS): return True
            if any(k in tl for k in self._HAIR_STYLE_HINTS): return True
            return False
        for tok in tokens:
            if hit(tok): locked.append(tok)
        seen, uniq = set(), []
        for x in locked:
            xl = x.lower()
            if xl not in seen:
                uniq.append(x); seen.add(xl)
        return uniq

    def _tokenize(self, s: str) -> List[str]:
        parts = re.split(r"[,\n/]+", s)
        tokens = []
        for p in parts:
            t = re.sub(r"\s+", " ", p.strip()).lower()
            if t: tokens.append(t)
        return tokens

    def _compose_prompt(self, base: str, cand: Dict[str, str],
                        allow_palette: bool, allow_style_notes: bool,
                        allow_camera: bool, allow_lighting: bool,
                        lock_appearance: bool, always_suffix: str) -> str:
        base_tokens = set(self._tokenize(base))
        pieces = [base.strip()]
        def is_locked_related(text: str) -> bool:
            tl = text.lower()
            if any(k in tl for k in self._HAIR_EYE_KEYS): return True
            if any(k in tl for k in self._HAIR_STYLE_HINTS): return True
            return False
        def _push_if_new(text: str):
            if not text: return
            for frag in [x.strip() for x in text.split(",") if x.strip()]:
                if lock_appearance and is_locked_related(frag): continue
                key = frag.lower()
                if key not in base_tokens:
                    pieces.append(frag); base_tokens.add(key)
        _push_if_new(cand.get("pose",""))
        _push_if_new(cand.get("outfit",""))
        _push_if_new(cand.get("background",""))
        if allow_palette:     _push_if_new(cand.get("palette",""))
        if allow_style_notes: _push_if_new(cand.get("style_notes",""))
        if allow_camera:      _push_if_new(cand.get("camera",""))
        if allow_lighting:    _push_if_new(cand.get("lighting",""))
        _push_if_new(always_suffix)
        return ", ".join([p for p in pieces if p])

    # ---------- build ----------
    def build(self, prompts_text, always_prefix, always_suffix, negative_text,
              backend, endpoint, model, n_candidates, temperature, timeout_sec,
              theme, style_lock, allow_palette, allow_style_notes, allow_camera, allow_lighting,
              select_mode, index, seed, lock_appearance):

        rnd = random.Random(seed if seed != 0 else None)

        # (1) 헤더 추출
        (name_val, steps_val, cfg_val, sampler_txt, scheduler_txt, ckpt_name, lora_dict), body_text = _extract_header_and_body(prompts_text)

        # (2) 본문에서 프롬프트 목록 파싱 + prefix
        user_prompts = _parse_user_prompts_block(body_text)
        prefixed = []
        for p in user_prompts or ["1girl, solo, full body, anime style"]:
            merged = (", ".join([x.strip() for x in (always_prefix + ", " + p.strip()).split(",") if x.strip()])
                      if always_prefix else p.strip())
            prefixed.append(merged)
        base = prefixed[max(0, min(index, len(prefixed)-1))] if (select_mode=="index") else rnd.choice(prefixed)

        # (3) 보조 디테일 생성
        locked_tokens = self._extract_locked_tokens_from_base(base) if lock_appearance else []
        locked_text = ", ".join(locked_tokens) if locked_tokens else "(none)"
        messages = [{"role":"system","content": self._system_prompt()},
                    {"role":"user","content": self._user_prompt(base, theme, style_lock, n_candidates, locked_text, lock_appearance,
                                                               want_camera=allow_camera, want_lighting=allow_lighting)}]
        candidates = []
        try:
            if backend == "ollama":
                text = self._ollama_chat(endpoint, model, messages, temperature, timeout_sec)
            else:
                text = self._openai_compat(endpoint, model, messages, temperature, timeout_sec)
            data = self._safe_json(text)
            for c in data.get("candidates", []):
                if isinstance(c, dict) and c.get("pose") and c.get("outfit") and c.get("background"):
                    candidates.append(c)
        except Exception as e:
            print("[FavoritePromptMixer] backend error:", e)
            traceback.print_exc()
        if not candidates:
            candidates.append(self._fallback_candidate())
        chosen = rnd.choice(candidates)

        final_text = self._compose_prompt(base, chosen,
                                          allow_palette, allow_style_notes,
                                          allow_camera, allow_lighting,
                                          lock_appearance, always_suffix)

        # (4) sampler/scheduler 정규화
        sampler_key   = _normalize_choice(SAMPLER_ENUM,   sampler_txt,   "euler_ancestral")
        scheduler_key = _normalize_choice(SCHEDULER_ENUM, scheduler_txt, "karras")

        # (5) 체크포인트 로드(매번 깨끗하게) + LoRA 적용
        model_obj, clip_obj, vae_obj, ckpt_name_resolved = _load_checkpoint_clean(ckpt_name)
        model_obj, clip_obj = _apply_loras(model_obj, clip_obj, lora_dict)

        # 디버그
        print("[FavoritePromptMixer] header -> name:", name_val,
              "| steps:", steps_val, "cfg:", cfg_val,
              "sampler:", sampler_key, "scheduler:", scheduler_key,
              "| checkpoint:", ckpt_name)
        print("[FavoritePromptMixer] loras (parsed):", lora_dict)
        print("[FavoritePromptMixer] final prompt   :", final_text)
        print("[FavoritePromptMixer] negative prompt:", (negative_text or "")[:120])

        # (6) 출력
        return (
            str(name_val or ""),
            int(steps_val),
            float(cfg_val),
            sampler_key,
            scheduler_key,
            model_obj,
            clip_obj,
            vae_obj,
            final_text,
            negative_text or "",
            ckpt_name_resolved,
        )

NODE_CLASS_MAPPINGS = {
    "FavoritePromptMixer": FavoritePromptMixerNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FavoritePromptMixer": "Favorite Prompt Mixer (+Header→ckpt/LoRA→MODEL/CLIP/VAE & texts)"
}
