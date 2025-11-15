# load_text_batch_simple.py
# Drop this file into ComfyUI/custom_nodes/

import os
import glob
import json
import random
import hashlib
from typing import List, Tuple

COUNTER_DB_PATH = os.path.expanduser("~/.comfyui_text_batch_counters.json")

def _load_counter_db() -> dict:
    if os.path.exists(COUNTER_DB_PATH):
        try:
            with open(COUNTER_DB_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_counter_db(db: dict) -> None:
    try:
        with open(COUNTER_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(db, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _db_key(label: str, path: str, pattern: str) -> str:
    # label + path + pattern 조합으로 키 생성 (경로/패턴이 다르면 진행 인덱스를 분리)
    return f"{label}::{os.path.abspath(path)}::{pattern}"

def _get_sha256(file_path: str) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha256.update(block)
    return sha256.hexdigest()

class _TextBatchLoader:
    def __init__(self, directory_path: str, label: str, pattern: str):
        self.directory_path = directory_path
        self.label = label
        self.pattern = pattern
        self.text_paths: List[str] = []
        self._load_texts()
        self.text_paths.sort()

        self._db = _load_counter_db()
        self._key = _db_key(label, directory_path, pattern)
        if self._key not in self._db:
            self._db[self._key] = 0
            _save_counter_db(self._db)
        self.index = int(self._db.get(self._key, 0)) if self.text_paths else 0

    def _load_texts(self):
        # glob.escape로 경로 안전 처리, ** 사용 시 recursive=True 필요
        search_path = os.path.join(glob.escape(self.directory_path), self.pattern)
        for p in glob.glob(search_path, recursive=True):
            if os.path.isfile(p):
                self.text_paths.append(os.path.abspath(p))

    def _read_file(self, file_path: str, encoding: str, errors: str) -> str:
        with open(file_path, "r", encoding=encoding, errors=errors) as f:
            return f.read()

    def get_text_by_id(self, file_id: int, encoding: str, errors: str) -> Tuple[str, str]:
        if file_id < 0 or file_id >= len(self.text_paths):
            return None, None
        p = self.text_paths[file_id]
        try:
            return self._read_file(p, encoding, errors), os.path.basename(p)
        except Exception:
            return None, None

    def get_next_text(self, encoding: str, errors: str) -> Tuple[str, str]:
        if not self.text_paths:
            return None, None
        if self.index >= len(self.text_paths):
            self.index = 0
        p = self.text_paths[self.index]
        self.index += 1
        if self.index >= len(self.text_paths):
            self.index = 0
        # persist
        self._db[self._key] = self.index
        _save_counter_db(self._db)
        try:
            return self._read_file(p, encoding, errors), os.path.basename(p)
        except Exception:
            return None, None

    def get_current_filename(self) -> str:
        if not self.text_paths:
            return ""
        if self.index >= len(self.text_paths):
            self.index = 0
        return os.path.basename(self.text_paths[self.index])

class Load_Text_Batch_Simple:
    """
    Standalone text batch loader for ComfyUI (no WAS dependencies).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["single_text", "incremental_text", "random"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "index": ("INT", {"default": 0, "min": 0, "max": 150000, "step": 1}),
                "label": ("STRING", {"default": "Batch 001", "multiline": False}),
                "path": ("STRING", {"default": "", "multiline": False}),
                "pattern": ("STRING", {"default": "*.txt", "multiline": False}),
            },
            "optional": {
                "encoding": ("STRING", {"default": "utf-8", "multiline": False}),
                "errors": ("STRING", {"default": "ignore", "multiline": False}),  # 'strict'/'replace'/'ignore'
                "strip_trailing_newlines": (["false", "true"],),
                "filename_text_extension": (["true", "false"],),  # false면 확장자 제거
            },
        }

    RETURN_TYPES = ("STRING", "STRING")  # (text, filename_text)
    RETURN_NAMES = ("text", "filename_text")
    FUNCTION = "load_batch_texts"
    CATEGORY = "IO"

    def load_batch_texts(
        self,
        path,
        pattern="*.txt",
        index=0,
        mode="single_text",
        seed=0,
        label="Batch 001",
        encoding="utf-8",
        errors="ignore",
        strip_trailing_newlines="false",
        filename_text_extension="true",
    ):
        if not path or not os.path.exists(path):
            # Comfy 규약상 None 대신 빈 문자열 반환
            return ("", "")

        tl = _TextBatchLoader(path, label, pattern)

        if len(tl.text_paths) == 0:
            return ("", "")

        if mode == "single_text":
            text, filename = tl.get_text_by_id(index, encoding, errors)
            if text is None:
                return ("", "")
        elif mode == "incremental_text":
            text, filename = tl.get_next_text(encoding, errors)
            if text is None:
                return ("", "")
        else:  # random
            random.seed(seed)
            ridx = int(random.random() * len(tl.text_paths))
            text, filename = tl.get_text_by_id(ridx, encoding, errors)
            if text is None:
                return ("", "")

        if strip_trailing_newlines == "true":
            text = text.rstrip("\r\n")

        if filename_text_extension == "false":
            filename = os.path.splitext(filename)[0]

        return (text, filename)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # single_text 모드에서만 캐시 키(sha) 고정
        if kwargs.get("mode") != "single_text":
            return float("NaN")
        # path / label / pattern로 현재 파일을 찾아 sha 계산
        tl = _TextBatchLoader(kwargs.get("path", ""), kwargs.get("label", "Batch 001"), kwargs.get("pattern", "*.txt"))
        filename = tl.get_current_filename()
        if not filename:
            return float("NaN")
        file_path = os.path.join(kwargs.get("path", ""), filename)
        try:
            return _get_sha256(file_path)
        except Exception:
            return float("NaN")

# ComfyUI 노드 등록
NODE_CLASS_MAPPINGS = {
    "Load_Text_Batch_Simple": Load_Text_Batch_Simple,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Load_Text_Batch_Simple": "Load Text Batch (Simple)",
}


