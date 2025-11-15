# Save as: ComfyUI/custom_nodes/free_ai_prompt_node.py
import json
import random
import requests

class FreeAIPromptNode:
    """
    Generate (pose, outfit, background) via local free AI backends,
    then concat with base prompt.

    Backends:
      - "ollama": http://localhost:11434/api/generate (or /api/chat)
      - "openai_compat": http(s)://.../v1/chat/completions (LM Studio, vLLM, etc.)

    If backend call fails, falls back to local random lists.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "backend": (["ollama", "openai_compat"],),
                "endpoint": ("STRING", {"default": "http://localhost:11434"}),  # ollama 기본
                "model": ("STRING", {"default": "llama3.1:8b-instruct"}),
                "base": ("STRING", {
                    "multiline": True,
                    "default": "masterpiece, best quality, 1girl, solo, full body, anime style"
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default":
"""You are a prompt generator for anime image creation.
Output ONLY a single line in the exact format:
<pose>, <outfit>, <background>
Keep it short, no extra words."""
                }),
                "guidance_prompt": ("STRING", {
                    "multiline": True,
                    "default":
"""Generate a random combination for pose, outfit, and background.
Examples:
- kneeling pose, maid outfit, inside a cafe
- jumping, school uniform, in a park
- sitting on a chair, casual wear, in a classroom
Return ONLY one line: <pose>, <outfit>, <background>"""
                }),
                "joiner": ("STRING", {"default": ", "}),
                "timeout_sec": ("INT", {"default": 15, "min": 1, "max": 120}),
            },
            "optional": {
                # 로컬 백업용 리스트 (백엔드 실패 시 사용)
                "poses_fallback": ("STRING", {
                    "multiline": True,
                    "default": "standing with arms crossed\nkneeling on one knee\njumping\nsitting on a chair\nleaning against a wall\nholding umbrella"
                }),
                "outfits_fallback": ("STRING", {
                    "multiline": True,
                    "default": "school uniform\nmaid outfit\ncasual wear\nkimono\nfuturistic bodysuit\nlight armor"
                }),
                "backgrounds_fallback": ("STRING", {
                    "multiline": True,
                    "default": "in a classroom\nin a cafe\nin a park\non the beach\nin a futuristic city\nin a forest"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "build"
    CATEGORY = "Prompt/Free-AI"

    # ---- 유틸: 한 줄 파싱 & 트림
    def _parse_line(self, s: str):
        if not s:
            return None, None, None
        line = s.strip().splitlines()[0]
        # 코드블록/백틱 제거
        line = line.strip("` ").strip()
        # 접두사('pose: xxx, outfit: yyy' 형태 대응) 제거 시도
        for key in ["pose:", "outfit:", "background:"]:
            line = line.replace(key, "")
        # 콤마 구분으로 3부분으로 분할
        parts = [p.strip(" ,") for p in line.split(",")]
        if len(parts) >= 3:
            return parts[0], parts[1], ", ".join(parts[2:])  # 배경에 쉼표가 있을 수 있어 3번째 이후를 다시 합침
        return None, None, None

    def _pick(self, s: str) -> str:
        items = [x.strip() for x in s.splitlines() if x.strip()]
        return random.choice(items) if items else ""

    # ---- 백엔드: Ollama generate
    def _ollama_generate(self, endpoint, model, system_prompt, guidance_prompt, timeout_sec):
        # /api/generate (completion) 사용
        url = endpoint.rstrip("/") + "/api/generate"
        prompt = f"<<SYS>>\n{system_prompt}\n<</SYS>>\n{guidance_prompt}"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.9}
        }
        r = requests.post(url, json=payload, timeout=timeout_sec)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "")

    # ---- 백엔드: OpenAI 호환 (chat/completions)
    def _openai_compat(self, endpoint, model, system_prompt, guidance_prompt, timeout_sec):
        url = endpoint.rstrip("/") + "/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": guidance_prompt}
            ],
            "temperature": 0.9
        }
        r = requests.post(url, json=payload, timeout=timeout_sec)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

    def _fallback_triplet(self, poses, outfits, backgrounds):
        return self._pick(poses), self._pick(outfits), self._pick(backgrounds)

    # ---- 메인
    def build(self, backend, endpoint, model, base, system_prompt, guidance_prompt,
              joiner, timeout_sec, poses_fallback=None, outfits_fallback=None, backgrounds_fallback=None):

        pose = outfit = bg = None
        try:
            if backend == "ollama":
                text = self._ollama_generate(endpoint, model, system_prompt, guidance_prompt, timeout_sec)
            else:
                text = self._openai_compat(endpoint, model, system_prompt, guidance_prompt, timeout_sec)

            p, o, b = self._parse_line(text)
            if p and o and b:
                pose, outfit, bg = p, o, b

        except Exception:
            # 백엔드 실패 시 로컬 백업 사용
            pass

        if not (pose and outfit and bg):
            pose, outfit, bg = self._fallback_triplet(
                poses_fallback or "", outfits_fallback or "", backgrounds_fallback or ""
            )

        parts = [base.strip()]
        for x in (pose, outfit, bg):
            if x:
                parts.append(x)
        final_text = joiner.join(parts)
        return (final_text,)


NODE_CLASS_MAPPINGS = {
    "FreeAIPrompt": FreeAIPromptNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FreeAIPrompt": "Free AI Prompt (Ollama / OpenAI-Compat)"
}


