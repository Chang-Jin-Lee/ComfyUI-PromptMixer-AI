## ComfyUI-PromptMixer-AI

텍스트 형식으로 체크포인트, 스텝, CFG, 샘플러/스케줄러, LoRA, 포지티브/네거티브 프롬프트를 한 번에 제어할 수 있는 ComfyUI 커스텀 노드 모음입니다. 레포를 `ComfyUI/custom_nodes/` 아래에 클론하면 자동 인식됩니다.

- 헤더 한 줄로 `checkpoint / steps / cfg / sampler / scheduler / lora` 등을 지정하고, MODEL/CLIP/VAE + positive/negative 텍스트까지 동시에 뽑아주는 워크플로우가 가능합니다.

### 포함 노드
- **FavoritePromptMixer**: 헤더+본문 텍스트를 파싱해 모델 로드(MODEL/CLIP/VAE)와 프롬프트(positive/negative)를 동시에 생성
- **FreeAIPromptGenerate**: 로컬 LLM(ollama/OpenAI 호환)으로 pose/outfit/background 후보 JSON을 생성 후 병합
- **FreeAIPrompt**: 로컬 LLM으로 단일 “<pose>, <outfit>, <background>” 라인을 받아 베이스 프롬프트와 합성
- **Load_Text_Batch_Simple**: 폴더에서 텍스트 파일을 순차/랜덤/인덱스로 불러오기

### 설치 (ComfyUI-Manager 권장)
1) ComfyUI 실행 후, 좌측의 Manager(노드 매니저)를 엽니다.  
2) 검색창에 `ComfyUI-PromptMixer-AI` 또는 `PromptMixer`를 입력하여 설치합니다.  
   - 만약 검색이 바로 뜨지 않으면 “Install via URL”에 이 레포 URL을 붙여넣어 설치할 수 있습니다.

수동 설치:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourname/ComfyUI-PromptMixer-AI.git
cd ComfyUI-PromptMixer-AI
pip install -r requirements.txt
```

### FavoritePromptMixer 사용 예시
`prompts_text` 입력에 아래와 같이 작성하면 됩니다.

```text
name: myrun, steps: 24, cfg: 7.0, sampler: euler_ancestral, scheduler: karras, checkpoint: your_model.safetensors, lora: { my_lora1.safetensors: 0.8, my_lora2.safetensors: 0.5 }

# --- 프롬프트 ---
masterpiece, best quality, 1girl, solo, full body
```

출력:
- `name/steps/cfg/sampler/scheduler`
- `MODEL/CLIP/VAE` (체크포인트 로드 및 LoRA 적용 결과)
- `positive_text / negative_text`

LLM 백엔드(옵션):
- `backend`: `ollama` 또는 `openai_compat`
- `endpoint`: ollama 기본은 `http://localhost:11434`
- `model`: 예) `llama3.1:8b-instruct-q4_K_M`

### 의존성
- `requests` (requirements.txt 포함)

### 호환성
- 최신 ComfyUI (기본 comfy 내장 모듈 `folder_paths`, `comfy.sd`, `comfy.samplers` 활용)

### 라이선스
- MIT (자유롭게 사용/수정/배포 가능)

---

### ComfyUI Manager에서 “검색으로” 잘 보이게 하려면 (레포 소유자용 팁)
- 레포 이름과 설명에 `ComfyUI`와 `custom nodes` 키워드를 포함하세요. (예: `ComfyUI-PromptMixer-AI`)
- GitHub Topics에 `comfyui`, `comfyui-node`, `comfyui-custom-nodes` 등을 추가하세요.
- README에 설치 방법(Manager 검색/URL 설치)을 명시하세요.


