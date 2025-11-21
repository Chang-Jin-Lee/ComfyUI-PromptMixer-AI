# Save as: ComfyUI/custom_nodes/favorite_prompt_mixer_node.py
import json, random, re, requests, traceback, os, glob
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

# ========= (B) 헤더 & LoRA 파서 =========
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
    def _f(x, d):
        try:
            return float(x)
        except Exception:
            return d
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
            f = float(s)
            return (f, f)
        u = re.search(r"(?:unet|model)\s*=\s*([+-]?\d+(?:\.\d+)?)", s, re.I)
        t = re.search(r"(?:te|clip)\s*=\s*([+-]?\d+(?:\.\d+)?)", s, re.I)
        uu = float(u.group(1)) if u else 1.0
        tt = float(t.group(1)) if t else 1.0
        return (uu, tt)
    return (1.0, 1.0)

def _parse_lora_block(first_line: str) -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
    if not first_line:
        return out
    m = _LORA_BLOCK_RE.search(_strip_invisible(first_line))
    if not m:
        return out
    body = m.group("body")
    parts = re.split(r"(?<!\})\s*[,\n;]\s*(?![^{}]*\})", body.strip())
    for part in parts:
        if not part or ":" not in part:
            continue
        k, v = part.split(":", 1)
        name = k.strip().strip('"').strip("'")
        val_raw = v.strip()
        if val_raw.startswith("{") and val_raw.endswith("}"):
            try:
                obj = json.loads(val_raw.replace("'", '"'))
            except Exception:
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

def _extract_header_and_body(text: str) -> Tuple[Tuple[str, int, float, str, str, str, Dict[str, Tuple[float, float]]], str]:
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
    if not any(k in lowered for k in ("name", "steps", "cfg", "sampler", "scheduler", "checkpoint", "lora", "loras", "models")):
        return DEFAULTS, text

    name_val, steps, cfg, sampler, scheduler, checkpoint = "", 30, 7.0, "euler_ancestral", "karras", ""
    loras: Dict[str, Tuple[float, float]] = {}
    try:
        for m in _HEADER_PAIR_RE.finditer(first):
            key = m.group("key").strip().lower()
            val = m.group("val").strip().strip('"').strip("'")
            if key == "name":
                name_val = val
            elif key == "steps":
                try:
                    steps = int(float(val))
                except Exception:
                    pass
            elif key in ("cfg", "cfg_scale", "guidance", "scale"):
                try:
                    cfg = float(val)
                except Exception:
                    pass
            elif key in ("sampler", "sampler_name"):
                sampler = val
            elif key in ("scheduler", "schedule", "sched"):
                scheduler = val
            elif key in ("checkpoint", "ckpt", "model", "checkpoint_name"):
                checkpoint = val
    except Exception:
        traceback.print_exc()

    try:
        loras = _parse_lora_block(first)
    except Exception:
        traceback.print_exc()
        loras = {}

    body_lines = lines[:first_idx] + lines[first_idx + 1:]
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
    for a in avail:
        if a.lower() == t:
            return a
    tbase = t.rsplit(".", 1)[0]
    for a in avail:
        if a.lower().rsplit(".", 1)[0] == tbase:
            return a
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

# ========= (E) Checkpoint 로드 =========
def _load_checkpoint_clean(ckpt_name: str):
    if folder_paths is None or csd is None:
        raise RuntimeError("Comfy internals unavailable.")
    if not ckpt_name:
        raise RuntimeError("checkpoint is not specified in header.")
    ckpt_name = _resolve_checkpoint_filename(ckpt_name)
    ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
    print(f"[FavoritePromptMixer] Loading checkpoint: {ckpt_name}")
    try:
        out = csd.load_checkpoint_guess_config(ckpt_path, None, None, True, True, True)
        model, clip, vae = out[0], out[1], out[2]
        if clip is not None:
            return model, clip, vae, ckpt_name
    except Exception as e:
        print("[FavoritePromptMixer] WARN: guess_config failed:", e)
    if comfy_nodes is None:
        raise RuntimeError("Comfy nodes import failed.")
    loader = comfy_nodes.CheckpointLoaderSimple()
    model, clip, vae = loader.load_checkpoint(ckpt_name)
    return model, clip, vae, ckpt_name

def _apply_loras(model, clip, loras: Dict[str, Tuple[float, float]]):
    if not loras or csd is None or cutils is None or folder_paths is None:
        return model, clip
    for raw_name, (w_unet, w_te) in loras.items():
        lname = _resolve_lora_filename(raw_name)
        try:
            lpath = folder_paths.get_full_path("loras", lname)
            lobj = cutils.load_torch_file(lpath, safe_load=True)
            model, clip = csd.load_lora_for_models(model, clip, lobj, float(w_unet), float(w_te))
            print(f"[FavoritePromptMixer] Applied LoRA: {lname} ({w_unet}, {w_te})")
        except Exception as e:
            print(f"[FavoritePromptMixer] Failed LoRA '{raw_name}': {e}")
    return model, clip

# ========= (F) 카테고리 파일 이름 고정 =========
CATEGORY_FILES = {
    "pose":       "pose.txt",
    "background": "background.txt",
    "outfit":     "outfit.txt",
    "state":      "state.txt",
}
CATEGORY_FILENAMES = set(CATEGORY_FILES.values())

# ======================================================================================

class FavoritePromptMixerNode:
    """
    Favorite Prompt Mixer
    - wildcard_dir 안에 pose/background/outfit/state.txt 를 두면,
      각 파일에서 한 줄씩 뽑아 '고정 조건'으로 사용.
    - LLM은 pose/background/outfit/state + 스타일/조명/팔레트만 보충.
    - 그 외의 모든 .txt 파일(예: character.txt, props.txt 등)은
      한 줄씩 뽑아서 LLM에 보내지 않고, 최종 프롬프트에 그대로 추가 태그로만 붙임.
    """

    @classmethod
    def INPUT_TYPES(cls):
        header_example = (
            "name : demo, steps : 24, cfg : 7.0, sampler : euler_ancestral, scheduler : karras, "
            "checkpoint : basic.safetensors, "
            "lora : { detail_slider.safetensors : 0.5 }"
        )
        neg_default = "lowres, bad anatomy, bad hands, text, watermark, jpeg artifacts"
        return {
            "required": {
                "prompts_text": ("STRING", {
                    "multiline": True,
                    "default": header_example + "\n\n# --- Base Prompts ---\n1girl, solo, masterpiece, best quality",
                }),
                "always_prefix": ("STRING", {"default": "masterpiece, best quality"}),
                "always_suffix": ("STRING", {"default": ""}),
                "negative_text": ("STRING", {"default": neg_default, "multiline": True}),
                "wildcard_dir": ("STRING", {"default": "", "placeholder": "C:/ComfyUI/wildcards/my_collection"}),

                "backend": (["ollama", "openai_compat"],),
                "endpoint": ("STRING", {"default": "http://localhost:11434"}),
                "model": ("STRING", {"default": "llama3.1:8b-instruct-q4_K_M"}),

                "n_candidates": ("INT", {"default": 3, "min": 1, "max": 8}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0}),
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

    RETURN_TYPES = ("STRING", "INT", "FLOAT", SAMPLER_ENUM, SCHEDULER_ENUM, "MODEL", "CLIP", "VAE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("name", "steps", "cfg", "sampler_name", "scheduler", "model", "clip", "vae", "positive_text", "negative_text", "checkpoint_name")
    FUNCTION = "build"
    CATEGORY = "Prompt/Favorites+AI"

    # ---------- Backends ----------
    def _ollama_chat(self, endpoint, model, messages, temperature, timeout):
        url = endpoint.rstrip("/") + "/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": float(temperature)},
        }
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "")

    def _openai_compat(self, endpoint, model, messages, temperature, timeout):
        url = endpoint.rstrip("/") + "/v1/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": float(temperature),
            "stream": False,
        }
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    # ---------- 카테고리별 TXT 한 줄씩 뽑기 ----------
    def _pick_category_lines(self, folder_path: str, rnd: random.Random) -> Dict[str, str]:
        """
        wildcard_dir 안에서 pose/background/outfit/state.txt 를 찾아,
        각 파일마다 빈 줄 / # 주석 제외 후 랜덤으로 한 줄을 뽑아 반환.
        비어 있거나 파일이 없으면 해당 카테고리는 생략.
        """
        result: Dict[str, str] = {}
        if not folder_path or not os.path.isdir(folder_path):
            print(f"[FavoritePromptMixer] INVALID wildcard_dir: {folder_path}")
            return result

        for cat, fname in CATEGORY_FILES.items():
            fpath = os.path.join(folder_path, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                candidates: List[str] = []
                with open(fpath, "r", encoding="utf-8") as f:
                    for raw in f:
                        line = raw.strip()
                        if not line or line.startswith("#"):
                            continue
                        if line.lower().startswith("text"):
                            line = line[4:].strip()
                        if line:
                            candidates.append(line)

                if not candidates:
                    print(f"[FavoritePromptMixer] {fname}: no usable lines")
                    continue

                chosen = rnd.choice(candidates)
                print(f"[FavoritePromptMixer] PICKED {cat} from {fname}: {chosen!r}")
                result[cat] = chosen
            except Exception as e:
                print(f"[FavoritePromptMixer] Error reading {fname}: {e}")
                traceback.print_exc()

        print(f"[FavoritePromptMixer] Category lines picked: {result}")
        return result

    # ---------- 기타 TXT 파일 한 줄씩 뽑기 (LLM에는 안 보내고, 프롬프트에만 추가) ----------
    def _pick_extra_txt_lines(self, folder_path: str, rnd: random.Random) -> List[str]:
        """
        wildcard_dir 안의 모든 *.txt 중에서
        CATEGORY_FILES 에 포함되지 않은 파일들을 '기타 txt'로 취급.
        각 파일에서 랜덤 한 줄 뽑아서 최종 프롬프트에 그대로 추가.
        """
        extras: List[str] = []
        if not folder_path or not os.path.isdir(folder_path):
            return extras

        all_txt = glob.glob(os.path.join(folder_path, "*.txt"))
        for fpath in all_txt:
            fname = os.path.basename(fpath)
            if fname in CATEGORY_FILENAMES:
                continue  # pose/background/outfit/state.txt 는 여기서 제외 (카테고리쪽에서 처리)
            try:
                candidates: List[str] = []
                with open(fpath, "r", encoding="utf-8") as f:
                    for raw in f:
                        line = raw.strip()
                        if not line or line.startswith("#"):
                            continue
                        if line.lower().startswith("text"):
                            line = line[4:].strip()
                        if line:
                            candidates.append(line)
                if not candidates:
                    print(f"[FavoritePromptMixer] extra {fname}: no usable lines")
                    continue
                chosen = rnd.choice(candidates)
                extras.append(chosen)
                print(f"[FavoritePromptMixer] EXTRA from {fname}: {chosen!r}")
            except Exception as e:
                print(f"[FavoritePromptMixer] Error reading extra {fname}: {e}")
                traceback.print_exc()

        if extras:
            print(f"[FavoritePromptMixer] Extra txt lines picked: {extras}")
        return extras

    # ---------- Prompt templates ----------
    def _system_prompt(self):
        return (
            "You are a prompt assistant for anime image generation.\n"
            "The base prompt ALREADY includes mandatory wildcard lines from external TXT files.\n"
            "You MUST NOT remove or contradict those fixed parts; only add small, coherent extra details.\n"
            "Return VALID JSON only. SFW."
        )

    def _user_prompt(self, base_with_wildcards: str, theme: str,
                     style_lock: bool, n_candidates: int,
                     cat_lines: Dict[str, str],
                     lock_appearance: bool) -> str:

        theme_note = f"Theme: {theme}" if theme else "Theme: (none)"
        style_note = (
            "Use the base as a strict style and content reference."
            if style_lock else
            "Use the base as inspiration; keep it consistent but slightly flexible."
        )

        cat_json = json.dumps({
            "pose":       cat_lines.get("pose", ""),
            "background": cat_lines.get("background", ""),
            "outfit":     cat_lines.get("outfit", ""),
            "state":      cat_lines.get("state", ""),
        }, ensure_ascii=False, indent=2)

        lock_clause = ""
        if lock_appearance:
            lock_clause = (
                "\nSome appearance tokens (hair/eyes) are considered locked and should not be overridden.\n"
                "Extra details should avoid contradicting these locked tokens."
            )

        return f"""
Base positive prompt (fixed, MUST NOT be removed):
{base_with_wildcards}

{theme_note}
Style guidance: {style_note}
{lock_clause}

Below is the BASE INFO per category. Treat these as fixed constraints:
{cat_json}

TASK:
Generate {n_candidates} coherent candidates as JSON (key "candidates").

Each candidate MUST:
- NOT delete or contradict the fixed base info.
- ONLY ADD extra details per category, if useful.

JSON shape:
{{
  "candidates": [
    {{
      "pose_extra": "...",
      "background_extra": "...",
      "outfit_extra": "...",
      "state_extra": "...",
      "palette": "...",
      "style_notes": "...",
      "camera": "...",
      "lighting": "..."
    }}
  ]
}}

Rules:
- Use *_extra fields ONLY for complementary details.
- Do NOT invent or modify character identity; character tags come from the base prompt or other TXT files and are fixed.
- If nothing extra is needed for a field, leave it as empty string.
- Keep everything SFW.
- Output VALID JSON only, no explanations.
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
        return {
            "pose_extra": "",
            "background_extra": "",
            "outfit_extra": "",
            "state_extra": "",
            "palette": "",
            "style_notes": "",
            "camera": "",
            "lighting": "",
        }

    # ---------- Helpers ----------
    _HAIR_EYE_KEYS = ("hair", "eyes")
    _HAIR_STYLE_HINTS = ("bangs", "fringe", "ponytail", "twin tails", "twintails", "bun", "ahoge", "bob", "pixie")

    def _extract_locked_tokens_from_base(self, base: str) -> List[str]:
        tokens = [t.strip() for t in re.split(r"[,\n/]+", base) if t.strip()]
        locked = []

        def hit(token: str) -> bool:
            tl = token.lower()
            if any(k in tl for k in self._HAIR_EYE_KEYS):
                return True
            if any(k in tl for k in self._HAIR_STYLE_HINTS):
                return True
            return False

        for tok in tokens:
            if hit(tok):
                locked.append(tok)
        return list(set(locked))

    def _tokenize(self, s: str) -> List[str]:
        parts = re.split(r"[,\n/]+", s)
        return [re.sub(r"\s+", " ", p.strip()).lower() for p in parts if p.strip()]

    def _compose_prompt(self, base_with_wildcards: str, cand: Dict[str, str],
                        allow_palette: bool, allow_style_notes: bool,
                        allow_camera: bool, allow_lighting: bool,
                        lock_appearance: bool, always_suffix: str) -> str:
        """
        base_with_wildcards 안에는 이미:
        - always_prefix + 유저 base 프롬프트
        - pose/background/outfit/state.txt 에서 뽑힌 한 줄들
        - 기타 txt에서 뽑힌 한 줄들
        이 전부 들어 있음 (고정).
        여기서는 LLM의 *_extra, palette, style_notes, camera, lighting 만 '추가'한다.
        """
        base_tokens = set(self._tokenize(base_with_wildcards))
        pieces = [base_with_wildcards.strip()]

        def is_locked_related(text: str) -> bool:
            tl = text.lower()
            if any(k in tl for k in self._HAIR_EYE_KEYS):
                return True
            if any(k in tl for k in self._HAIR_STYLE_HINTS):
                return True
            return False

        def _push_if_new(text: str):
            if not text:
                return
            for frag in [x.strip() for x in text.split(",") if x.strip()]:
                if lock_appearance and is_locked_related(frag):
                    continue
                key = frag.lower()
                if key not in base_tokens:
                    pieces.append(frag)
                    base_tokens.add(key)

        # 카테고리별 extra
        _push_if_new(cand.get("pose_extra", ""))
        _push_if_new(cand.get("background_extra", ""))
        _push_if_new(cand.get("outfit_extra", ""))
        _push_if_new(cand.get("state_extra", ""))

        # 스타일/팔레트/카메라/조명
        if allow_palette:
            _push_if_new(cand.get("palette", ""))
        if allow_style_notes:
            _push_if_new(cand.get("style_notes", ""))
        if allow_camera:
            _push_if_new(cand.get("camera", ""))
        if allow_lighting:
            _push_if_new(cand.get("lighting", ""))
        _push_if_new(always_suffix)

        return ", ".join([p for p in pieces if p])

    # ---------- build ----------
    def build(self, prompts_text, always_prefix, always_suffix, negative_text,
              wildcard_dir,
              backend, endpoint, model, n_candidates, temperature, timeout_sec,
              theme, style_lock, allow_palette, allow_style_notes, allow_camera, allow_lighting,
              select_mode, index, seed, lock_appearance):

        rnd = random.Random(seed if seed != 0 else None)

        # (1) 헤더 파싱
        (name_val, steps_val, cfg_val, sampler_txt, scheduler_txt, ckpt_name, lora_dict), body_text = _extract_header_and_body(prompts_text)

        # (2) Base prompt 선택
        user_prompts = _parse_user_prompts_block(body_text)
        prefixed: List[str] = []
        for p in user_prompts or ["1girl, solo"]:
            merged = (
                ", ".join([x.strip() for x in (always_prefix + ", " + p.strip()).split(",") if x.strip()])
                if always_prefix else p.strip()
            )
            prefixed.append(merged)
        base = prefixed[max(0, min(index, len(prefixed) - 1))] if (select_mode == "index") else rnd.choice(prefixed)

        # (3) 카테고리별 TXT 한 줄씩 (pose/background/outfit/state)
        cat_lines = self._pick_category_lines(wildcard_dir, rnd)

        # (4) 기타 txt 한 줄씩 (LLM과 무관, 그냥 태그 추가)
        extra_lines = self._pick_extra_txt_lines(wildcard_dir, rnd)

        # (5) base + 카테고리 + 기타 txt를 하나의 고정 프롬프트로 결합
        fixed_parts: List[str] = [base]
        for cat in ["pose", "background", "outfit", "state"]:
            if cat in cat_lines:
                fixed_parts.append(cat_lines[cat])
        for line in extra_lines:
            fixed_parts.append(line)

        base_with_wildcards = ", ".join(fixed_parts)

        # (6) LLM 요청 생성
        locked_tokens = self._extract_locked_tokens_from_base(base_with_wildcards) if lock_appearance else []
        if locked_tokens:
            print(f"[FavoritePromptMixer] Locked appearance tokens: {locked_tokens}")

        messages = [
            {"role": "system", "content": self._system_prompt()},
            {
                "role": "user",
                "content": self._user_prompt(
                    base_with_wildcards, theme, style_lock,
                    n_candidates, cat_lines, lock_appearance,
                ),
            },
        ]

        candidates: List[Dict[str, Any]] = []
        try:
            if backend == "ollama":
                text = self._ollama_chat(endpoint, model, messages, temperature, timeout_sec)
            else:
                text = self._openai_compat(endpoint, model, messages, temperature, timeout_sec)

            print(f"[FavoritePromptMixer] RAW LLM Response: {text}")
            data = self._safe_json(text)
            for c in data.get("candidates", []):
                if isinstance(c, dict):
                    candidates.append(c)
        except Exception as e:
            print("[FavoritePromptMixer] backend error:", e)
            traceback.print_exc()

        if not candidates:
            print("[FavoritePromptMixer] No valid candidates from backend, using fallback.")
            candidates.append(self._fallback_candidate())

        chosen = rnd.choice(candidates)

        # (7) 최종 프롬프트 조립
        final_text = self._compose_prompt(
            base_with_wildcards, chosen,
            allow_palette, allow_style_notes,
            allow_camera, allow_lighting,
            lock_appearance, always_suffix,
        )

        # (8) 체크포인트/LoRA 로드
        sampler_key = _normalize_choice(SAMPLER_ENUM, sampler_txt, "euler_ancestral")
        scheduler_key = _normalize_choice(SCHEDULER_ENUM, scheduler_txt, "karras")

        model_obj, clip_obj, vae_obj, ckpt_name_resolved = _load_checkpoint_clean(ckpt_name)
        model_obj, clip_obj = _apply_loras(model_obj, clip_obj, lora_dict)

        print(f"[FavoritePromptMixer] FINAL PROMPT: {final_text}")

        return (
            str(name_val or ""), int(steps_val), float(cfg_val),
            sampler_key, scheduler_key,
            model_obj, clip_obj, vae_obj,
            final_text, negative_text or "", ckpt_name_resolved,
        )

NODE_CLASS_MAPPINGS = {
    "FavoritePromptMixer": FavoritePromptMixerNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FavoritePromptMixer": "Favorite Prompt Mixer (File Input Added)"
}