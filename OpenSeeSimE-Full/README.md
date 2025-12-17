# OpenSeeSimE-Full

**Shared utilities and infrastructure for evaluating vision-language models on the OpenSeeSimE-Structural benchmark.**

This repository provides standardized tools for prompt construction, response parsing, checkpoint management, and evaluation protocols to ensure reproducible and fair comparison of VLM performance on scientific visualization tasks.

---

## Overview

The OpenSeeSimE-Full repository includes:

- **Shared Utilities**: Consolidated functions for dataset loading, prompt construction, video processing, response parsing, and evaluation
- **Checkpoint Management**: Robust infrastructure for saving and resuming long-running evaluations
- **Setup Instructions**: Complete guide for dependencies, environment configuration, and dataset access

This repository is designed to serve as a foundation for researchers implementing their own model evaluation scripts while maintaining consistency with established benchmarking protocols.

---

## Table of Contents

1. [Features](#features)
2. [Installation & Setup](#installation--setup)
3. [Quick Start](#quick-start)
4. [Utilities Reference](#utilities-reference)
5. [Usage Examples](#usage-examples)
6. [Best Practices](#best-practices)
7. [Dataset Information](#dataset-information)
8. [Standardized Prompts](#standardized-prompts)
9. [Troubleshooting](#troubleshooting)
10. [Citation](#citation)

---

## Features

### Core Utilities

- **Dataset Loading**: Load and filter the OpenSeeSimE-Structural dataset by media type
- **Prompt Construction**: Build standardized system and user prompts for consistent evaluation
- **Video Processing**: Extract frames with middle-frame-centered symmetric sampling
- **Response Parsing**: Parse and validate model responses with exact-match checking
- **Evaluation**: Calculate accuracy metrics overall and per question type
- **Checkpoint Management**: Save and resume evaluation progress automatically

### Key Capabilities

- Standardized prompting across all models for fair comparison
- Support for both image and video question answering
- Middle-frame-centered temporal sampling for video processing
- Checkpointing and resumability for long-running evaluations
- Multiple-choice format with exact-match evaluation
- Per-question-type accuracy metrics

---

## Installation & Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/OpenSeeSimE-Full.git
cd OpenSeeSimE-Full
```

### Step 2: Install Dependencies

#### Core Dependencies (Required)

```bash
pip install -r requirements.txt
```

This includes:
- `datasets` - HuggingFace datasets library
- `transformers` - For model interfaces
- `torch` - PyTorch for tensor operations
- `pillow` - Image processing
- `opencv-python` - Video processing
- `numpy` - Numerical operations
- `pandas` - Data manipulation
- `tqdm` - Progress bars

#### API-Specific Dependencies (Optional)

Install based on which APIs you plan to use:

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

### Step 3: Environment Configuration

Create a `.env` file in the repository root or export environment variables:

```bash
# HuggingFace (Required for dataset access)
export HUGGING_FACE_HUB_TOKEN="hf_..."

# API Keys (Optional - based on which models you use)
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="AIzaSy..."
export OPENAI_API_KEY="sk-..."
export GROQ_API_KEY="gsk_..."
```

**Important**: The HuggingFace token is required and must have access to the `JessicaE/OpenSeeSimE-Structural` dataset.

#### HuggingFace Authentication

**Method 1: Login via CLI**
```bash
huggingface-cli login
```

**Method 2: Set token directly**
```bash
export HUGGING_FACE_HUB_TOKEN="hf_your_token_here"
```

**Method 3: Use `.env` file**
```bash
# Copy the example and fill in your tokens
cp .env.example .env
# Edit .env with your actual tokens
```

### Step 4: Verify Setup

Run a quick test to verify your setup:

```python
from utils import validate_environment, load_benchmark_dataset

# Check environment variables
env_status = validate_environment()
print("Environment status:", env_status)

# Try loading dataset
dataset = load_benchmark_dataset(media_type='image')
print(f"Successfully loaded {len(dataset)} examples")
```

---

## Quick Start

### Basic Example

```python
from utils import (
    load_benchmark_dataset,
    build_system_prompt,
    build_user_prompt,
    parse_model_response,
    evaluate_response,
    print_evaluation_summary
)

# Load dataset
dataset = load_benchmark_dataset(media_type='image')

# Get an example
example = dataset[0]

# Build prompts
system_prompt = build_system_prompt()
user_prompt = build_user_prompt(
    question=example['question'],
    answer_choices=example['answer_choices'],
    is_video=False
)

# ... call your model here with the prompts ...
# model_response = your_model.generate(system_prompt, user_prompt, example['image'])

# Parse response
model_answer, explanation = parse_model_response(
    model_response,
    example['answer_choices']
)

# Evaluate
is_correct = evaluate_response(
    model_answer,
    example['answer'],
    example['answer_choices']
)

print(f"Correct: {is_correct}")
```

### Complete Evaluation Loop

See `example_usage.py` for a complete working example that demonstrates:
- Loading the dataset
- Processing multiple examples
- Using checkpoints
- Calculating metrics
- Saving results

---

## Utilities Reference

### Dataset Loading

#### `load_benchmark_dataset(dataset_name, split, media_type)`

Load the OpenSeeSimE-Structural dataset with optional filtering.

```python
# Load all examples
dataset = load_benchmark_dataset()

# Load only images
dataset = load_benchmark_dataset(media_type='image')

# Load only videos
dataset = load_benchmark_dataset(media_type='video')
```

**Parameters:**
- `dataset_name` (str): HuggingFace dataset name (default: `"JessicaE/OpenSeeSimE-Structural"`)
- `split` (str): Dataset split (default: `"test"`)
- `media_type` (str): Filter by `'all'`, `'image'`, or `'video'` (default: `'all'`)

### Prompt Construction

#### `build_system_prompt()`

Returns the standardized system prompt used across all models.

```python
system_prompt = build_system_prompt()
```

#### `build_user_prompt(question, answer_choices, is_video, num_frames)`

Build formatted user prompt with question and answer choices.

```python
user_prompt = build_user_prompt(
    question="What is the flow direction?",
    answer_choices=["Left", "Right", "Up", "Down"],
    is_video=False
)
```

### Video Processing

#### `extract_video_frames(video_path, num_frames, middle_frame_guarantee)`

Extract frames from video with middle-frame-centered sampling.

```python
# Extract 8 frames with middle frame guaranteed
frames = extract_video_frames("video.mp4", num_frames=8)

# Extract 32 frames uniformly (no middle frame guarantee)
frames = extract_video_frames("video.mp4", num_frames=32, middle_frame_guarantee=False)
```

#### `image_to_base64(image, format)`

Convert PIL Image to base64 for API transmission.

```python
base64_str = image_to_base64(image, format="PNG")
```

### Response Parsing

#### `parse_model_response(response_text, answer_choices)`

Parse and validate model response.

```python
answer, explanation = parse_model_response(
    "Yes\nThe flow is turbulent.",
    ["Yes", "No"]
)
# Returns: ("Yes", "The flow is turbulent.")
```

**Returns:**
- `answer` (str or None): Validated answer, or None if invalid
- `explanation` (str): Model's explanation text

### Evaluation

#### `evaluate_response(model_answer, ground_truth, answer_choices)`

Evaluate model answer against ground truth.

```python
is_correct = evaluate_response(
    model_answer="Yes",
    ground_truth="Yes",
    answer_choices=["Yes", "No"]
)
# Returns: True
```

#### `calculate_accuracy_by_type(results)`

Calculate per-question-type accuracy metrics.

```python
accuracy_stats = calculate_accuracy_by_type(results)
# Returns: {'classification': {'accuracy': 0.85, 'correct': 85, 'total': 100}, ...}
```

#### `print_evaluation_summary(results, model_name)`

Print formatted evaluation summary and return statistics.

```python
summary = print_evaluation_summary(results, model_name="my-model")
```

### Checkpoint Management

#### `load_checkpoint(checkpoint_file)`

Load existing checkpoint to resume evaluation.

```python
processed_indices, results = load_checkpoint("checkpoint.pkl")
```

#### `save_checkpoint(checkpoint_file, processed_indices, results)`

Save current progress to checkpoint.

```python
save_checkpoint("checkpoint.pkl", processed_indices, results)
```

#### `save_results_to_csv(results, output_file)`

Save results to CSV file.

```python
save_results_to_csv(results, "my_results.csv")
```

#### `cleanup_checkpoint(checkpoint_file)`

Remove checkpoint file after successful completion.

```python
cleanup_checkpoint("checkpoint.pkl")
```

---

## Usage Examples

### Example 1: Evaluate on Image Examples

```python
from utils import load_benchmark_dataset, build_system_prompt, build_user_prompt

# Load image examples only
dataset = load_benchmark_dataset(media_type='image')

for i, example in enumerate(dataset):
    # Build prompts
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(
        question=example['question'],
        answer_choices=example['answer_choices']
    )

    # Get image
    image = example['image']

    # Call your model
    # response = your_model(system_prompt, user_prompt, image)

    if i >= 5:  # Process first 5 examples
        break
```

### Example 2: Evaluate on Video Examples

```python
from utils import (
    load_benchmark_dataset,
    extract_video_frames,
    build_user_prompt
)

# Load video examples only
dataset = load_benchmark_dataset(media_type='video')

for example in dataset:
    # Extract frames
    frames = extract_video_frames(
        example['video'],
        num_frames=8
    )

    # Build prompt with video context
    user_prompt = build_user_prompt(
        question=example['question'],
        answer_choices=example['answer_choices'],
        is_video=True,
        num_frames=len(frames)
    )

    # Call your model with frames
    # response = your_model(system_prompt, user_prompt, frames)
```

### Example 3: Complete Evaluation with Checkpointing

```python
from utils import (
    load_benchmark_dataset,
    load_checkpoint,
    save_checkpoint,
    parse_model_response,
    evaluate_response,
    save_results_to_csv,
    print_evaluation_summary,
    cleanup_checkpoint
)

# Setup
dataset = load_benchmark_dataset()
checkpoint_file = "my_checkpoint.pkl"
output_file = "my_results.csv"

# Load checkpoint
processed_indices, results = load_checkpoint(checkpoint_file)

# Process examples
for idx in range(len(dataset)):
    if idx in processed_indices:
        continue  # Skip already processed

    example = dataset[idx]

    # ... run your model ...
    # model_response = your_model.generate(...)

    # Parse and evaluate
    model_answer, explanation = parse_model_response(
        model_response,
        example['answer_choices']
    )

    is_correct = evaluate_response(
        model_answer,
        example['answer'],
        example['answer_choices']
    )

    # Store result
    result = {
        'question_id': example['question_id'],
        'question': example['question'],
        'question_type': example['question_type'],
        'answer': example['answer'],
        'model_answer': model_answer,
        'correct': is_correct,
        'explanation': explanation
    }
    results.append(result)
    processed_indices.add(idx)

    # Save checkpoint every 10 examples
    if len(results) % 10 == 0:
        save_checkpoint(checkpoint_file, processed_indices, results)
        save_results_to_csv(results, output_file)

# Final save
save_results_to_csv(results, output_file)
summary = print_evaluation_summary(results, "MyModel")
cleanup_checkpoint(checkpoint_file)
```

See `example_usage.py` for a complete working implementation.

---

## Best Practices

### 1. Use Standardized Prompts

Always use `build_system_prompt()` and `build_user_prompt()` to ensure consistency with benchmark protocols.

```python
# ✅ Good
system_prompt = build_system_prompt()
user_prompt = build_user_prompt(question, answer_choices)

# ❌ Bad - custom prompt breaks comparability
system_prompt = "You are a helpful assistant..."
```

### 2. Validate Responses

Always validate model responses using `parse_model_response()` before evaluation.

```python
# ✅ Good - validates answer is in choices
answer, explanation = parse_model_response(response, answer_choices)
if answer is None:
    print("Invalid response - not in answer choices")

# ❌ Bad - no validation
answer = response.split('\n')[0]
```

### 3. Use Checkpointing

For long evaluations, save checkpoints frequently.

```python
# Save every 10 examples
if len(results) % 10 == 0:
    save_checkpoint(checkpoint_file, processed_indices, results)
```

### 4. Frame Extraction Strategy

For video processing, use middle-frame-centered sampling for fair comparison.

```python
# ✅ Good - middle frame guaranteed
frames = extract_video_frames(video_path, num_frames=8, middle_frame_guarantee=True)

# ⚠️ Use only if you have a specific reason
frames = extract_video_frames(video_path, num_frames=8, middle_frame_guarantee=False)
```

### 5. Deterministic Settings

For evaluation, use deterministic model settings:
- Temperature: `0.0`
- Do sample: `False`
- No top-p/top-k sampling

---

## Dataset Information

### Dataset Details

- **Name**: `JessicaE/OpenSeeSimE-Structural`
- **Source**: HuggingFace Datasets (private dataset)
- **Split**: `test` (standard evaluation split)
- **Access**: Requires HuggingFace token with dataset permissions

### Dataset Structure

Each example contains:

| Field | Type | Description |
|-------|------|-------------|
| `question` | str | Question text |
| `answer` | str | Ground truth answer |
| `answer_choices` | List[str] | Multiple choice options |
| `correct_choice_idx` | int | Index of correct choice |
| `question_id` | int/str | Unique identifier |
| `question_type` | str | Question category |
| `media_type` | str | `"image"` or `"video"` |
| `image` | PIL.Image | Image (for image examples) |
| `video` | str | Video path (for video examples) |
| `file_name` | str | Original file identifier |
| `source_file` | str | Source document reference |

### Example Access

```python
dataset = load_benchmark_dataset()
example = dataset[0]

print(f"Question: {example['question']}")
print(f"Type: {example['question_type']}")
print(f"Choices: {example['answer_choices']}")
print(f"Answer: {example['answer']}")
```

---

## Standardized Prompts

### System Prompt

The system prompt enforces structured output with exact answer matching:

```python
system_prompt = build_system_prompt()
```

**Key requirements:**
- Line 1: Exact copy of answer from choices
- Line 2+: Brief explanation (10-15 words)
- No paraphrasing or summarizing
- Must include all symbols and punctuation

### User Prompt Format

```
{question}

Answer options:
- {choice_1}
- {choice_2}
- ...

Instructions:
1. First line: Provide ONLY your answer exactly as it appears in the options above.
2. Second line onwards: Provide a brief summary explaining your reasoning.

Answer:
```

### Video Prompt Modification

For video examples, the prompt includes frame context:

```
These are {num_frames} frames from a video showing a sequence. {question}
...
```

---

## Troubleshooting

### Common Issues

#### 1. Dataset Access Error

```
Error: Repository not found
```

**Solution**:
- Ensure you're authenticated with HuggingFace: `huggingface-cli login`
- Verify you have access to `JessicaE/OpenSeeSimE-Structural`
- Check your token has read permissions

#### 2. Video Decoding Error

```
ValueError: Could not read video
```

**Solution**:
- Ensure video file is accessible
- Check video format is supported (MP4, AVI, etc.)
- Verify OpenCV is properly installed: `pip install opencv-python`

#### 3. Import Errors

```
ModuleNotFoundError: No module named 'datasets'
```

**Solution**:
```bash
pip install -r requirements.txt
```

#### 4. Invalid Model Response

```
Warning: Model answer not in choices
```

**Solution**:
- Check system prompt is properly formatted
- Verify model is following instruction format
- Review model's raw output for debugging

#### 5. Out of Memory (Video Processing)

```
MemoryError: Unable to allocate array
```

**Solution**:
- Reduce `num_frames` parameter
- Process examples in smaller batches
- Close video capture objects: `cap.release()`

---

## Citation

If you use these utilities or the OpenSeeSimE-Structural benchmark, please cite:

```bibtex
@misc{openseeisme2024,
  title={OpenSeeSimE-Full: Shared Utilities for VLM Benchmark Evaluation},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yourusername/OpenSeeSimE-Full}}
}
```

---

## License

See [LICENSE](LICENSE) file for details.

---

## Contact

For questions, issues, or contributions:
- Open an issue on GitHub
- Email: your.email@example.com

---

**Last Updated**: 2024-12-17
