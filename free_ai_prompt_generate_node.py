# Save as: ComfyUI/custom_nodes/free_ai_prompt_generate_node.py
import json, random, re, requests, traceback

class FreeAIPromptGenerateNode:
    """
    Generate a coherent (pose, outfit, background) with a local free LLM
    (Ollama or OpenAI-compatible servers). No predefined lists needed.

    The model produces multiple coherent candidates as JSON,
    and the node selects one and merges it with the base prompt.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "backend": (["ollama", "openai_compat"],),
                "endpoint": ("STRING", {"default": "http://localhost:11434"}),
                "model": ("STRING", {"default": "llama3.1:8b-instruct-q4_K_M"}),

                "base": ("STRING", {
                    "multiline": True,
                    "default": "masterpiece, best quality, 1girl, solo, full body, anime style"
                }),

                "theme": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "¿¹: spring picnic / cyberpunk city / japanese festival / beach sunset"
                }),

                "style_lock": ("BOOLEAN", {"default": True}),
                "allow_palette": ("BOOLEAN", {"default": True}),
                "allow_style_notes": ("BOOLEAN", {"default": True}),

                "n_candidates": ("INT", {"default": 3, "min": 1, "max": 8}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
                "timeout_sec": ("INT", {"default": 20, "min": 5, "max": 120}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "build"
    CATEGORY = "Prompt/Free-AI"

    # -------- Ollama backend
    def _ollama_chat(self, endpoint, model, messages, temperature, timeout):
        url_gen = endpoint.rstrip("/") + "/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": float(temperature)}
        }
        r = requests.post(url_gen, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return data.get("message", {}).get("content", "")

    # -------- OpenAI compatible backend
    def _openai_compat(self, endpoint, model, messages, temperature, timeout):
        url = endpoint.rstrip("/") + "/v1/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": float(temperature),
            "stream": False
        }
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

    # -------- Prompt templates
    def _system_prompt(self):
        return (
            "You are a creative prompt designer for anime image generation.\n"
            "Produce coherent, stylish combinations of pose, outfit, and background "
            "that match the given base keywords and optional theme.\n"
            "Only SFW ideas. Avoid explicit content. Return valid JSON only."
        )

    def _user_prompt(self, base, theme, style_lock, n_candidates):
        lock_note = (
            "strictly follow the base keywords and style" if style_lock
            else "loosely use the base as inspiration"
        )
        theme_note = f"Theme: {theme}" if theme else "Theme: (none)"
        return f"""
Base keywords: {base}
{theme_note}
Guideline: {lock_note}

TASK:
Generate {n_candidates} coherent candidates as JSON array (key 'candidates'), each with:
- pose: short phrase describing posture
- outfit: short phrase describing outfit
- background: short phrase describing setting
- palette: optional color harmony
- style_notes: optional lighting or tone hints

Constraints:
- Each field concise (<10 words)
- No extra text outside JSON
- SFW only

Example JSON:
{{
  "candidates": [
    {{
      "pose": "kneeling, hands on lap",
      "outfit": "spring dress with ribbon",
      "background": "cherry blossoms in park",
      "palette": "pastel pink and white",
      "style_notes": "soft rim light"
    }},
    {{
      "pose": "standing, arms crossed",
      "outfit": "casual hoodie and skirt",
      "background": "city street at dusk",
      "palette": "teal and neon purple",
      "style_notes": "bokeh lights"
    }}
  ]
}}
"""

    # -------- Parser
    def _safe_json(self, text):
        cleaned = text.strip().strip("`").strip()
        m = re.search(r"\{.*\}", cleaned, re.S)
        if m:
            cleaned = m.group(0)
        return json.loads(cleaned)

    # -------- Simple fallback if AI fails
    def _fallback_candidate(self):
        poses = [
            "standing, arms at sides", "kneeling, hands on lap",
            "sitting on a chair", "walking forward", "leaning against wall"
        ]
        outfits = [
            "casual hoodie", "school uniform", "kimono with obi",
            "futuristic bodysuit", "maid outfit", "armor suit"
        ]
        bgs = [
            "in a cafe", "on a beach", "in a forest", "in a futuristic city", "on a rooftop at night"
        ]
        return {
            "pose": random.choice(poses),
            "outfit": random.choice(outfits),
            "background": random.choice(bgs),
            "palette": "pastel pink and white",
            "style_notes": "soft rim light"
        }

    # -------- Combine to prompt
    def _compose_prompt(self, base, cand, allow_palette, allow_style_notes):
        parts = [base.strip()]
        for key in ["pose", "outfit", "background"]:
            if cand.get(key):
                parts.append(cand[key].strip())
        if allow_palette and cand.get("palette"):
            parts.append(cand["palette"].strip())
        if allow_style_notes and cand.get("style_notes"):
            parts.append(cand["style_notes"].strip())
        return ", ".join([p for p in parts if p])

    # -------- Main
    def build(self, backend, endpoint, model, base, theme, style_lock,
              allow_palette, allow_style_notes, n_candidates, temperature, timeout_sec):

        system = self._system_prompt()
        user = self._user_prompt(base, theme, style_lock, n_candidates)
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]

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
            print("[FreeAIPromptGenerate] backend error:", e)
            traceback.print_exc()

        if not candidates:
            candidates.append(self._fallback_candidate())

        chosen = random.choice(candidates)
        final_text = self._compose_prompt(base, chosen, allow_palette, allow_style_notes)
        print("[FreeAIPromptGenerate] chosen:", chosen)
        print("[FreeAIPromptGenerate] final prompt:", final_text)
        return (final_text,)


NODE_CLASS_MAPPINGS = {
    "FreeAIPromptGenerate": FreeAIPromptGenerateNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FreeAIPromptGenerate": "Free AI Prompt Generate (Pose/Outfit/Background)"
}


