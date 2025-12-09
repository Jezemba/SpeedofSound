# SpeedofSound: Visual Question Answering Benchmark Evaluation Framework

## Overview

This repository contains a comprehensive evaluation framework for benchmarking vision-language models (VLMs) on the **OpenSeeSimE-Structural** dataset, a fluid dynamics and scientific visualization benchmark. The framework supports multiple state-of-the-art models across different deployment methods (API-based and local inference) with standardized evaluation protocols to ensure reproducibility and fair comparison.

**Key Features:**
- Standardized prompting across all models for consistent evaluation
- Support for both image and video question answering
- Middle-frame-centered temporal sampling for video processing
- Checkpointing and resumability for long-running evaluations
- Multiple-choice format with exact-match evaluation
- Per-question-type accuracy metrics

---

## Table of Contents

1. [Dataset Information](#dataset-information)
2. [Standardized Prompts](#standardized-prompts)
3. [Model-Specific Parameters](#model-specific-parameters)
4. [Video Processing Settings](#video-processing-settings)
5. [Evaluation Metrics & Protocols](#evaluation-metrics--protocols)
6. [Installation & Setup](#installation--setup)
7. [Usage Examples](#usage-examples)
8. [Reproducibility Guidelines](#reproducibility-guidelines)
9. [API Endpoints & Authentication](#api-endpoints--authentication)
10. [Output Format](#output-format)
11. [Code Structure](#code-structure)

---

## Dataset Information

### Dataset Details

- **Name:** `JessicaE/OpenSeeSimE-Structural`
- **Source:** HuggingFace Datasets (private dataset requiring authentication)
- **Split:** `test` (standard evaluation split)
- **Access:** Requires HuggingFace API token with dataset access permissions

### Dataset Structure

Each example in the dataset contains the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `question` | str | The question text |
| `answer` | str | Ground truth answer |
| `answer_choices` | List[str] | Multiple-choice options (typically 2-4 choices) |
| `correct_choice_idx` | int | Index of the correct choice in `answer_choices` |
| `question_id` | int/str | Unique question identifier |
| `question_type` | str | Category/type of the question |
| `media_type` | str | Either `"image"` or `"video"` |
| `file_name` | str | Original file identifier |
| `source_file` | str | Source document reference |
| `image` | PIL.Image | PIL Image object (for image examples) |
| `video` | str/VideoDecoder | Path to video file or VideoDecoder object (for video examples) |

### Loading the Dataset

```python
from datasets import load_dataset

# Load the full dataset
dataset = load_dataset("JessicaE/OpenSeeSimE-Structural", split="test", token=True)

# Filter by media type (optional)
video_indices = [i for i, m in enumerate(dataset["media_type"]) if m == "video"]
video_dataset = dataset.select(video_indices)
```

**Note:** The `token=True` parameter automatically uses your HuggingFace token from the environment or cached credentials.

---

## Standardized Prompts

All models use **identical prompts** to ensure fair comparison and reproducibility. This standardization is critical for evaluation consistency.

### System Prompt

The following system prompt is used across all models to enforce structured output:

```python
SYSTEM_PROMPT = """You are a visual question answering assistant. You MUST follow this exact format:

FORMAT REQUIREMENTS:
Line 1: Copy the EXACT answer text from the provided options (word-for-word, including all symbols)
Line 2: One brief explanation sentence (10-15 words)

CRITICAL RULES:
1. The first line MUST be an EXACT COPY of one option - do not paraphrase or summarize
2. Copy ALL words, punctuation, and mathematical symbols exactly as shown in the option
3. Do NOT add phrases like 'The answer is' or explanatory text on line 1
4. Do NOT shorten or reword long options - copy them completely

EXAMPLE 1 (Simple):
Question: Is the sky blue?
Options: Yes, No
CORRECT:
Yes
The clear atmosphere scatters blue wavelengths effectively.

EXAMPLE 2 (Complex option with symbols):
Question: What is the range?
Options: Less than 10× min, More than 1000× min
CORRECT:
More than 1000× min
The values span from 7 billion to 1.6 trillion.

INCORRECT:
More than three orders of magnitude
(This paraphrases instead of copying the exact option)

Remember: Line 1 = EXACT COPY of option. Line 2 = explanation."""
```

### User Prompt Template

```python
USER_PROMPT_TEMPLATE = """{question}

Answer options:
{formatted_answer_choices}

Instructions:
1. First line: Provide ONLY your answer exactly as it appears in the options above (e.g., 'A', 'Yes', 'X axis', etc.). Do NOT add any other text on this line.
2. Second line onwards: Provide a brief summary (1-2 sentences) explaining your reasoning.

Answer:"""
```

**Example formatted prompt:**

```
What is the speed of sound in this medium?

Answer options:
- 343 m/s
- 1500 m/s
- 5000 m/s

Instructions:
1. First line: Provide ONLY your answer exactly as it appears in the options above (e.g., 'A', 'Yes', 'X axis', etc.). Do NOT add any other text on this line.
2. Second line onwards: Provide a brief summary (1-2 sentences) explaining your reasoning.

Answer:
```

### Video-Specific Prompt Modification

For video examples, the prompt is prefixed with frame context:

```python
f"These are {num_frames} frames from a video showing a sequence. {question}\n\n..."
```

---

## Model-Specific Parameters

This section documents all model-specific configurations for reproducibility.

### API-Based Models

| Model | Provider | Model ID | Frames (Video) | Frames (Image) | Max Tokens | Temperature | Top-p | Notes |
|-------|----------|----------|----------------|----------------|------------|-------------|-------|-------|
| **Claude Sonnet 4.5** | Anthropic | `claude-sonnet-4-5-20250929` | 8 | 1 | 512 | 0.0 | - | Deterministic evaluation |
| **Gemini 2.5 Flash** | Google | `gemini-2.5-flash` | 8 | 1 | - | 0.0 | - | Native video support |
| **GPT-5** | OpenAI | `gpt-5-2025-08-07` | 32 | Variable | 512 (video), 4096 (image) | - | - | Uses Responses API with reasoning |
| **Llama 3.2 90B** | Groq | `llama-3.2-90b-vision-preview` | 8 | 1 | 4096 | 0.0 | - | Groq API deployment |
| **Gemma 3 27B** | Google Cloud | `google/gemma-3-27b-it` | 32 | 1 | - | - | - | Cloud API or local |
| **Qwen3-VL 235B** | Alibaba | `Qwen3-VL-235B-A22B-Instruct` | Variable | Variable | 512 | 0.7 | 0.8 | Rate limited: 2s between requests |
| **InternVL 3.5 241B** | InternAI | `internvl3.5-241b-a28b` | Variable | Variable | 4096 | 0.0 | - | InternLM API endpoint |

### Local Inference Models (HuggingFace Transformers)

| Model | HF Model ID | Size | Frames | Max Tokens | Temperature | Do Sample | Dtype | Device Map |
|-------|-------------|------|--------|------------|-------------|-----------|-------|------------|
| **Gemma 3 4B** | `google/gemma-3-4b-it` | 4B | 32 | 512 | - | False | bfloat16 | auto |
| **Gemma 3 27B** | `google/gemma-3-27b-it` | 27B | 32 | - | - | False | bfloat16 | auto |
| **Qwen3-VL 2B** | `Qwen/Qwen3-VL-2B-Instruct` | 2B | 32 | - | 0.7 | True | bfloat16 | auto |
| **Qwen3-VL 8B** | `Qwen/Qwen3-VL-8B-Instruct` | 8B | 32 | 512 | - | False | bfloat16 | auto |
| **Qwen3-VL 8B (Large)** | `Qwen/Qwen3-VL-8B-Instruct` | 8B | 32 | - | 0.7 | True | bfloat16 | auto |
| **Llama 3.2 11B** | `meta-llama/Llama-3.2-11B-Vision-Instruct` | 11B | 32 | 4096 | - | False | bfloat16 | auto |
| **Llama 3.2 90B** | `meta-llama/Llama-3.2-90B-Vision-Instruct` | 90B | 8 | 4096 | 0.0 | - | bfloat16 | auto |
| **InternVL 3.5 1B** | `OpenGVLab/InternVL3_5-1B-Instruct` | 1B | 32 | 4096 | - | False | bfloat16 | auto |

### Model Initialization Examples

#### API-Based Models

**Claude (Anthropic):**
```python
import anthropic

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=512,
    temperature=0.0,
    system=system_prompt,
    messages=[{"role": "user", "content": content}]
)
```

**Gemini (Google):**
```python
import google.generativeai as genai

genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

generation_config = genai.types.GenerationConfig(
    temperature=0.0,
    max_output_tokens=512
)

response = model.generate_content(content, generation_config=generation_config)
```

**GPT-5 (OpenAI Responses API):**
```python
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

response = client.responses.create(
    model="gpt-5-2025-08-07",
    input=input_messages,
    reasoning={"effort": "minimal"},
    text={"verbosity": "medium"},
    max_output_tokens=512
)
```

#### Local Inference Models

**Gemma 3 (HuggingFace):**
```python
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch

model = Gemma3ForConditionalGeneration.from_pretrained(
    "google/gemma-3-4b-it",
    device_map="auto",
    torch_dtype=torch.bfloat16
).eval()

processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")

inputs = processor(text=messages, images=frames, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False
    )
```

---

## Video Processing Settings

### Frame Extraction Strategy

All models use a **middle-frame-centered symmetric sampling** strategy to ensure the most informative frame (typically at the video center) is always included.

#### Algorithm: Middle Frame Guarantee

```python
def extract_video_frames(video_path, num_frames=32):
    """
    Extract frames with guaranteed middle frame inclusion.

    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract

    Returns:
        List of PIL Image objects
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate middle frame
    middle_frame = total_frames // 2

    # Generate frame indices with middle frame guaranteed
    if num_frames % 2 == 1:
        # Odd number: sample symmetrically around middle
        half_segments = num_frames // 2
        indices_before = np.linspace(0, middle_frame - 1, half_segments, dtype=int)
        indices_after = np.linspace(middle_frame + 1, total_frames - 1, half_segments, dtype=int)
        frame_indices = np.concatenate([indices_before, [middle_frame], indices_after])
    else:
        # Even number: force middle frame at position num_frames//2
        half_segments = num_frames // 2
        if half_segments > 1:
            indices_before = np.linspace(0, middle_frame - 1, half_segments - 1, dtype=int)
        else:
            indices_before = np.array([], dtype=int)
        indices_after = np.linspace(middle_frame + 1, total_frames - 1, half_segments, dtype=int)
        frame_indices = np.concatenate([indices_before, [middle_frame], indices_after])

    # Ensure we have exactly num_frames
    frame_indices = frame_indices[:num_frames]

    # Extract frames
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)

    cap.release()
    return frames
```

#### Frame Count by Model Type

- **API Models (Claude, Gemini, Groq Llama):** 8 frames
  - Optimized for API token/cost efficiency
  - Still includes middle frame for critical information

- **Local Models (Gemma, Qwen, InternVL, Local Llama):** 32 frames
  - Full local compute allows higher frame counts
  - Better temporal resolution for video understanding

- **Video-Specific Fair Comparison Scripts:** 32 frames
  - `gpt_video.py`, `gemma_medium_video.py`, `qwen_medium_video.py`, `intern_medium_video.py`
  - Standardized at 32 frames for direct comparison

### Video Format & Encoding

#### Input Processing

```python
# OpenCV for video reading
cap = cv2.VideoCapture(video_path)

# Frame extraction
ret, frame = cap.read()  # Returns BGR format

# Color space conversion (BGR → RGB)
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Convert to PIL Image
pil_image = Image.fromarray(frame_rgb)  # uint8 format
```

#### API Transmission

```python
# For API-based models (Claude, GPT, etc.)
import base64
import io

def image_to_base64(image):
    """Convert PIL Image to base64 for API transmission"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str
```

- **Format:** PNG (lossless compression)
- **Encoding:** Base64
- **Data Type:** uint8 (0-255 range)
- **Color Space:** RGB

### Temporal Sampling Parameters

- **Default Time Bounds:** Full video (`start=-100000, end=100000`)
- **FPS Handling:** Preserves original FPS metadata from source video
- **Frame Selection:** Index-based (not time-based) for consistency

---

## Evaluation Metrics & Protocols

### Response Parsing

All model responses are parsed using a **strict two-line format**:

```python
def parse_model_response(response_text, answer_choices):
    """
    Parse the model's structured response.

    Args:
        response_text: Raw text from model
        answer_choices: List of valid answer choices

    Returns:
        tuple: (answer, explanation) - answer is None if invalid
    """
    lines = response_text.strip().split('\n')

    if len(lines) == 0:
        return None, ""

    # First line is the answer
    potential_answer = lines[0].strip()

    # Rest is explanation
    explanation = '\n'.join(lines[1:]).strip()

    # Validate answer is in the choices (case-insensitive)
    for choice in answer_choices:
        if potential_answer.lower() == choice.strip().lower():
            return choice.strip(), explanation

    # Invalid if not in choices
    return None, explanation
```

### Evaluation Function

```python
def evaluate_response(model_answer, ground_truth, answer_choices):
    """
    Evaluate model answer against ground truth.

    Args:
        model_answer: Parsed model answer
        ground_truth: Correct answer from dataset
        answer_choices: List of valid choices

    Returns:
        bool: True if correct, False otherwise
    """
    if model_answer is None:
        return False  # Failed to extract valid answer

    # Exact match comparison (case-insensitive)
    if model_answer.strip().lower() == ground_truth.strip().lower():
        return True

    return False
```

### Metrics Computed

#### Overall Metrics

1. **Overall Accuracy:**
   ```
   accuracy = (correct_answers / total_questions) × 100%
   ```

2. **Failed Answer Rate:**
   ```
   fail_rate = (answers_not_in_choices / total_questions) × 100%
   ```

3. **Success Rate:**
   ```
   success_rate = (valid_answers / total_questions) × 100%
   ```

#### Per-Question-Type Metrics

```python
# Calculated separately for each question_type
accuracy_by_type = {}
for q_type in unique_question_types:
    type_examples = [ex for ex in results if ex['question_type'] == q_type]
    type_correct = sum(1 for ex in type_examples if ex['correct'])
    accuracy_by_type[q_type] = (type_correct / len(type_examples)) × 100
```

### Evaluation Rules

- **Exact Match Only:** No fuzzy matching or partial credit
- **Case-Insensitive:** "Yes" matches "YES" or "yes"
- **Symbol Preservation:** Must match exactly including mathematical symbols (×, ±, %, etc.)
- **Multi-word Options:** Entire phrase must match word-for-word
- **No Paraphrasing:** Model must copy the exact option text

**Example valid matches:**
- Ground truth: `"More than 1000× min"` → Model answer: `"more than 1000× min"` ✅
- Ground truth: `"Yes"` → Model answer: `"YES"` ✅

**Example invalid matches:**
- Ground truth: `"More than 1000× min"` → Model answer: `"More than 1000 times min"` ❌
- Ground truth: `"X axis"` → Model answer: `"The X axis"` ❌

---

## Installation & Setup

### Dependencies

#### Core Dependencies (All Scripts)

```bash
pip install datasets transformers torch pillow opencv-python numpy pandas tqdm
```

#### API-Specific Dependencies

```bash
# Anthropic Claude
pip install anthropic

# Google Gemini
pip install google-generativeai

# OpenAI GPT
pip install openai

# Groq
pip install groq
```

#### Optional Dependencies

```bash
# For video processing optimization
pip install torchcodec

# For local model optimization
pip install accelerate bitsandbytes
```

### Environment Setup

#### 1. HuggingFace Authentication

```bash
# Method 1: Login via CLI
huggingface-cli login

# Method 2: Set token directly
export HUGGING_FACE_HUB_TOKEN="hf_..."
```

#### 2. API Key Configuration

Create a `.env` file or export environment variables:

```bash
# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Google Gemini
export GOOGLE_API_KEY="AIzaSy..."

# OpenAI
export OPENAI_API_KEY="sk-..."

# Groq
export GROQ_API_KEY="gsk_..."

# Alibaba DashScope
export DASHSCOPE_API_KEY="..."

# InternAI/InternLM
export INTERNLM_API_KEY="sk-..."

# Scaleway
export SCW_SECRET_KEY="..."
```

### Hardware Requirements

#### API-Based Models
- **CPU:** Any modern CPU
- **RAM:** 8GB+ recommended
- **GPU:** Not required
- **Storage:** ~10GB for dataset cache

#### Local Inference Models

| Model Size | GPU Memory | Recommended GPU |
|------------|------------|-----------------|
| 1B-4B | 8-16GB | RTX 3090, RTX 4090, A10 |
| 8B-12B | 24-40GB | A100 40GB, RTX 6000 Ada |
| 27B-90B | 80GB+ | A100 80GB, H100 |

**Note:** Using `device_map="auto"` enables automatic multi-GPU distribution for large models.

---

## Usage Examples

### Basic Usage

#### API-Based Models

**Claude (Anthropic):**
```bash
# All media types (default)
python claude.py

# Image only
python claude.py --media_type image

# Video only
python claude.py --media_type video

# Custom frame count
python claude.py --media_type video --num_video_frames 16
```

**Gemini (Google):**
```bash
python gemini.py --media_type all --num_video_frames 8
```

**GPT (OpenAI):**
```bash
# Standard GPT script
python gpt.py --media_type all

# Video-specific fair comparison (32 frames, 512 tokens)
python gpt_video.py --num_frames 32
```

**Llama 90B (Groq):**
```bash
python llama_large.py --media_type all
```

#### Local Inference Models

**Gemma 3:**
```bash
# Small model (4B)
python gemma_small.py --media_type all --num_frames 32

# Large model (27B)
python gemma_large.py --media_type all --num_frames 32

# Video-specific medium model
python gemma_medium_video.py --num_frames 32
```

**Qwen3-VL:**
```bash
# Small model (2B)
python qwen_small.py --mode process --model Qwen/Qwen3-VL-2B-Instruct

# Large model (8B)
python qwen_large.py --mode process --model Qwen/Qwen3-VL-8B-Instruct

# Video-specific medium model
python qwen_medium_video.py --num_frames 32
```

**Llama 3.2 Vision:**
```bash
# Medium model (11B)
python llama_medium.py --media_type all --num_frames 32

# Large model (90B)
python llama_large.py --media_type all --num_frames 8
```

**InternVL 3.5:**
```bash
# Small model (1B)
python internvl_small.py --media_type all --num_frames 32

# Medium model (1B)
python internvl_medium.py --media_type all --num_frames 32

# Large model (API-based 241B)
python internvl_large.py --media_type all

# Video-specific medium model
python intern_medium_video.py
```

### Advanced Usage

#### Parallel Processing

Some scripts support parallel processing for faster evaluation:

```bash
# GPT video with 4 workers
python gpt_video.py --num_workers 4

# Gemma large with 3 workers
python gemma_large.py --num_workers 3
```

#### Resuming Interrupted Runs

All scripts support automatic checkpointing:

```bash
# If interrupted, simply re-run the same command
python claude.py --media_type all

# The script will automatically:
# 1. Load existing checkpoint
# 2. Skip already processed examples
# 3. Continue from where it left off
```

#### Custom Output Paths

```bash
# Most scripts support custom output paths
python claude.py --output results_claude_custom.csv
```

---

## Reproducibility Guidelines

### Ensuring Reproducible Results

#### 1. Fixed Random Seeds

For local models, set random seeds before inference:

```python
import torch
import random
import numpy as np

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
```

#### 2. Deterministic Settings

- **Temperature:** Set to `0.0` for deterministic sampling (most models)
- **Do Sample:** Set to `False` to disable stochastic sampling
- **Top-p/Top-k:** Not used for evaluation (deterministic mode)

#### 3. Consistent Frame Extraction

- Use the provided `extract_video_frames()` function
- Ensure middle frame is always included
- Use integer frame indices (not time-based)

#### 4. Standardized Prompts

- Do not modify the system prompt
- Use identical user prompt format
- Maintain exact answer option formatting

#### 5. Version Control

Document all versions for reproducibility:

```python
# Example version documentation
versions = {
    "transformers": "4.45.0",
    "torch": "2.1.0",
    "datasets": "2.14.0",
    "anthropic": "0.40.0",
    "openai": "1.54.0",
    "model_id": "claude-sonnet-4-5-20250929",
    "dataset_id": "JessicaE/OpenSeeSimE-Structural",
    "dataset_commit": "main"  # or specific commit hash
}
```

### Reproducibility Checklist

- [ ] Environment variables set (API keys, HF token)
- [ ] Dependencies installed with specified versions
- [ ] Dataset accessible and authenticated
- [ ] Model parameters match specifications (temperature, max_tokens, etc.)
- [ ] Frame extraction uses middle-frame-centered algorithm
- [ ] System prompt unchanged from specification
- [ ] Evaluation uses exact-match comparison
- [ ] Random seeds set (for local models)

---

## API Endpoints & Authentication

### API Configuration

#### Anthropic (Claude)

```python
import anthropic

client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)
# Default endpoint: https://api.anthropic.com
```

**Environment Variable:** `ANTHROPIC_API_KEY`
**Format:** `sk-ant-...`

#### Google (Gemini)

```python
import google.generativeai as genai

genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
# Uses Google's generativeai library endpoint
```

**Environment Variable:** `GOOGLE_API_KEY`
**Format:** `AIzaSy...`

#### OpenAI (GPT)

```python
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)
# Default endpoint: https://api.openai.com/v1
```

**Environment Variable:** `OPENAI_API_KEY`
**Format:** `sk-...`

#### Groq (Llama)

```python
from groq import Groq

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)
# Default endpoint: https://api.groq.com/openai/v1
```

**Environment Variable:** `GROQ_API_KEY`
**Format:** `gsk_...`

#### Alibaba DashScope (Qwen API)

```python
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)
# Singapore endpoint (use dashscope.aliyuncs.com for Beijing)
```

**Environment Variable:** `DASHSCOPE_API_KEY`
**Endpoint:**
- Singapore: `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`
- Beijing: `https://dashscope.aliyuncs.com/compatible-mode/v1`

**Rate Limits:**
- 50 requests/minute
- 100,000 tokens/minute
- **Default delay:** 2 seconds between requests (implemented in code)

#### InternAI/InternLM (InternVL API)

```python
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("INTERNLM_API_KEY"),
    base_url="https://chat.intern-ai.org.cn/api/v1/"
)
```

**Environment Variable:** `INTERNLM_API_KEY`
**Format:** `sk-...`
**Endpoint:** `https://chat.intern-ai.org.cn/api/v1/`

### Rate Limiting

Some APIs require rate limiting to avoid quota errors:

```python
import time

# Alibaba DashScope example
DEFAULT_API_DELAY = 2.0  # seconds

for example in dataset:
    response = call_api(example)
    time.sleep(DEFAULT_API_DELAY)
```

**Models with built-in rate limiting:**
- Qwen3-VL 235B (Alibaba): 2s delay
- InternVL 241B (InternAI): As needed based on quotas

---

## Output Format

### CSV Output Structure

All scripts output results in CSV format with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `file_name` | str | Original file identifier from dataset |
| `source_file` | str | Source document reference |
| `question` | str | The question text |
| `question_type` | str | Category of the question |
| `question_id` | int/str | Unique question identifier |
| `answer` | str | Ground truth answer |
| `answer_choices` | str | JSON-encoded list of answer choices |
| `correct_choice_idx` | int | Index of correct choice |
| `model` | str | Model name/identifier |
| `model_answer` | str | Model's predicted answer (None if invalid) |
| `explanation` | str | Model's reasoning/explanation |
| `correct` | bool | Whether model answer is correct |
| `media_type` | str | "image" or "video" |
| `processing_time` | float | Time taken for this example (optional) |

### Example CSV Row

```csv
file_name,source_file,question,question_type,question_id,answer,answer_choices,correct_choice_idx,model,model_answer,explanation,correct,media_type,processing_time
video_001.mp4,experiment_A.pdf,"What is the flow regime?",classification,12,"Turbulent","[""Laminar"", ""Turbulent"", ""Transitional""]",1,claude-sonnet-4-5,Turbulent,The Reynolds number exceeds critical threshold.,True,video,2.34
```

### Checkpoint Format (Pickle)

For resumability, scripts save checkpoints in pickle format:

```python
checkpoint_data = {
    'processed_indices': set([0, 1, 2, ...]),  # Indices already processed
    'results': [
        {
            'file_name': '...',
            'question': '...',
            'model_answer': '...',
            'correct': True,
            ...
        },
        ...
    ]
}
```

**Checkpoint file naming:**
- `checkpoint_{model_name}_{media_type}.pkl`
- Example: `checkpoint_claude_all.pkl`

### Results Summary

At completion, scripts print a summary:

```
=== Results Summary ===
Total examples: 500
Overall accuracy: 78.4%
Failed to parse: 2 (0.4%)

Accuracy by question type:
- classification: 82.3% (120/146)
- comparison: 75.8% (91/120)
- measurement: 76.1% (89/117)
- reasoning: 79.5% (93/117)
```

---

## Code Structure

### Script Organization

```
SpeedofSound/
├── claude.py                    # Claude API (image + video)
├── gemini.py                    # Gemini API (image + video)
├── gpt.py                       # GPT API (image + video)
├── gpt_video.py                 # GPT video-specific (32 frames, 512 tokens)
├── llama_large.py               # Llama 90B via Groq (image + video)
├── llama_medium.py              # Llama 11B local (image + video)
├── gemma_small.py               # Gemma 4B local (image + video)
├── gemma_large.py               # Gemma 27B local/cloud (image + video)
├── gemma_medium_video.py        # Gemma video-specific (fair comparison)
├── qwen_small.py                # Qwen 2B local (image + video)
├── qwen_large.py                # Qwen 8B local (image + video)
├── qwen_medium_video.py         # Qwen video-specific (fair comparison)
├── internvl_small.py            # InternVL 1B local (image + video)
├── internvl_medium.py           # InternVL 1B local (image + video)
├── internvl_large.py            # InternVL 241B API (image + video)
├── intern_medium_video.py       # InternVL video-specific (fair comparison)
└── Subset Code/                 # Filtered subsets (e.g., question_id 12)
    ├── claude.py
    ├── gemini.py
    ├── gpt.py
    └── ...
```

### Script Categories

#### 1. Multi-Modal Scripts (Image + Video)

Support both image and video examples via `--media_type` flag:

```bash
python <model>.py --media_type all|image|video
```

**Examples:** `claude.py`, `gemini.py`, `gpt.py`, `llama_large.py`, `gemma_small.py`, etc.

#### 2. Video-Specific Scripts

Process only video examples with optimized settings:

```bash
python <model>_video.py
```

**Examples:** `gpt_video.py`, `gemma_medium_video.py`, `qwen_medium_video.py`, `intern_medium_video.py`

**Purpose:** Fair comparison with standardized settings (e.g., 32 frames, 512 max tokens)

#### 3. Subset Scripts

Located in `Subset Code/` directory. These scripts filter the dataset for specific subsets:

```python
# Example: Filter for question_id 12
dataset = load_dataset("JessicaE/OpenSeeSimE-Structural", split="test", token=True)
filtered = dataset.filter(lambda x: x['question_id'] == 12)
```

### Key Functions (Common Across Scripts)

#### 1. Dataset Loading
```python
def load_benchmark_dataset(dataset_name, split='test', media_type='all'):
    """Load and optionally filter dataset by media type"""
```

#### 2. Model Initialization
```python
def initialize_model(model_name):
    """Initialize model and processor/client"""
```

#### 3. Frame Extraction
```python
def extract_video_frames(video_path, num_frames=32):
    """Extract frames with middle-frame guarantee"""
```

#### 4. Prompt Building
```python
def build_system_prompt():
    """Build standardized system prompt"""

def prepare_messages(example, media_type, frames=None):
    """Format messages for specific model API"""
```

#### 5. Response Processing
```python
def parse_model_response(response_text, answer_choices):
    """Parse structured response into answer + explanation"""

def evaluate_response(model_answer, ground_truth, answer_choices):
    """Evaluate correctness of model answer"""
```

#### 6. Checkpointing
```python
def save_checkpoint(checkpoint_path, processed_indices, results):
    """Save progress for resumability"""

def load_checkpoint(checkpoint_path):
    """Load existing checkpoint if available"""
```

---

## Citation

If you use this evaluation framework, please cite:

```bibtex
@misc{speedofsound2024,
  title={SpeedofSound: Visual Question Answering Benchmark Evaluation Framework},
  author={[Your Name]},
  year={2024},
  howpublished={\url{https://github.com/Jezemba/SpeedofSound}}
}
```

---

## License

See [LICENSE](LICENSE) file for details.

---

## Troubleshooting

### Common Issues

#### 1. Dataset Access Error
```
Error: Repository not found
```
**Solution:** Ensure you're authenticated with HuggingFace and have access to `JessicaE/OpenSeeSimE-Structural`

#### 2. API Rate Limiting
```
Error: Rate limit exceeded
```
**Solution:** Increase delay between requests or use checkpointing to resume later

#### 3. Out of Memory (Local Models)
```
torch.cuda.OutOfMemoryError
```
**Solution:**
- Use smaller batch size
- Enable `device_map="auto"` for multi-GPU
- Use 8-bit quantization: `load_in_8bit=True`

#### 4. Video Decoding Error
```
ValueError: Could not read video
```
**Solution:** Ensure video file is accessible and in supported format (MP4, AVI, etc.)

#### 5. Invalid Model Response
```
Warning: Model answer not in choices
```
**Solution:** Check system prompt is properly formatted and model is following instructions

---

## Contact

For questions, issues, or contributions, please open an issue on GitHub.

---

**Last Updated:** 2025-12-09
