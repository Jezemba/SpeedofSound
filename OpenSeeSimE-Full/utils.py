#!/usr/bin/env python3
"""
OpenSeeSimE-Full Utilities
---------------------------
Consolidated utilities for prompt construction, response parsing,
checkpoint management, and evaluation for the OpenSeeSimE-Structural benchmark.

This module provides all the core functionality needed to:
- Load and filter the benchmark dataset
- Construct standardized prompts
- Extract and process video frames
- Parse and validate model responses
- Evaluate model performance
- Manage checkpoints for long-running evaluations
"""

import os
import io
import base64
import pickle
import numpy as np
import pandas as pd
from datasets import load_dataset
from PIL import Image
import cv2
from typing import List, Tuple, Dict, Any, Set, Optional


# =============================================================================
# DATASET LOADING
# =============================================================================

def load_benchmark_dataset(
    dataset_name: str = "JessicaE/OpenSeeSimE-Structural",
    split: str = 'test',
    media_type: str = 'all'
) -> Any:
    """
    Load the OpenSeeSimE-Structural benchmark dataset from HuggingFace.

    Args:
        dataset_name: Name of the dataset on HuggingFace
        split: Dataset split to load (default: 'test')
        media_type: Filter by media type - 'all', 'image', or 'video'

    Returns:
        Dataset object (or filtered dataset)

    Example:
        >>> dataset = load_benchmark_dataset(media_type='video')
        >>> print(f"Loaded {len(dataset)} video examples")
    """
    print(f"Loading dataset: {dataset_name} (split: {split})")
    dataset = load_dataset(dataset_name, split=split, token=True)
    print(f"Loaded {len(dataset)} examples")

    # Optional filtering by media type
    if media_type != "all":
        print(f"Filtering dataset for {media_type} examples only...")
        media_types = dataset["media_type"]
        filtered_indices = [i for i, mt in enumerate(media_types) if mt == media_type]
        dataset = dataset.select(filtered_indices)
        print(f"Remaining examples after filter: {len(dataset)}")

    return dataset


# =============================================================================
# PROMPT CONSTRUCTION
# =============================================================================

def build_system_prompt() -> str:
    """
    Build the standardized system prompt used across all models.

    This prompt enforces structured output with exact answer matching
    and brief explanations. It is critical for reproducible evaluation.

    Returns:
        System prompt string

    Example:
        >>> system_prompt = build_system_prompt()
        >>> print(system_prompt[:100])
    """
    return (
        "You are a visual question answering assistant. You MUST follow this exact format:\n\n"
        "FORMAT REQUIREMENTS:\n"
        "Line 1: Copy the EXACT answer text from the provided options (word-for-word, including all symbols)\n"
        "Line 2: One brief explanation sentence (10-15 words)\n\n"
        "CRITICAL RULES:\n"
        "1. The first line MUST be an EXACT COPY of one option - do not paraphrase or summarize\n"
        "2. Copy ALL words, punctuation, and mathematical symbols exactly as shown in the option\n"
        "3. Do NOT add phrases like 'The answer is' or explanatory text on line 1\n"
        "4. Do NOT shorten or reword long options - copy them completely\n\n"
        "EXAMPLE 1 (Simple):\n"
        "Question: Is the sky blue?\n"
        "Options: Yes, No\n"
        "CORRECT:\n"
        "Yes\n"
        "The clear atmosphere scatters blue wavelengths effectively.\n\n"
        "EXAMPLE 2 (Complex option with symbols):\n"
        "Question: What is the range?\n"
        "Options: Less than 10× min, More than 1000× min\n"
        "CORRECT:\n"
        "More than 1000× min\n"
        "The values span from 7 billion to 1.6 trillion.\n\n"
        "INCORRECT:\n"
        "More than three orders of magnitude\n"
        "(This paraphrases instead of copying the exact option)\n\n"
        "Remember: Line 1 = EXACT COPY of option. Line 2 = explanation."
    )


def build_user_prompt(
    question: str,
    answer_choices: List[str],
    is_video: bool = False,
    num_frames: int = 8
) -> str:
    """
    Build the standardized user prompt with question and answer choices.

    Args:
        question: The question text
        answer_choices: List of answer options
        is_video: Whether this is a video question (adds frame context)
        num_frames: Number of frames (for video questions)

    Returns:
        Formatted user prompt string

    Example:
        >>> prompt = build_user_prompt(
        ...     question="What is the flow direction?",
        ...     answer_choices=["Left to right", "Right to left"],
        ...     is_video=False
        ... )
    """
    # Add video context if needed
    if is_video:
        prompt = f"These are {num_frames} frames from a video showing a sequence. {question}\n\n"
    else:
        prompt = f"{question}\n\n"

    # Add answer choices
    if answer_choices and len(answer_choices) > 0:
        prompt += "Answer options:\n"
        for choice in answer_choices:
            prompt += f"- {choice}\n"
        prompt += "\n"

    # Add instruction for structured response
    prompt += "Instructions:\n"
    prompt += "1. First line: Provide ONLY your answer exactly as it appears in the options above (e.g., 'A', 'Yes', 'X axis', etc.). Do NOT add any other text on this line.\n"
    prompt += "2. Second line onwards: Provide a brief summary (1-2 sentences) explaining your reasoning.\n\n"
    prompt += "Answer:"

    return prompt


# =============================================================================
# VIDEO PROCESSING
# =============================================================================

def extract_video_frames(
    video_path: str,
    num_frames: int = 32,
    middle_frame_guarantee: bool = True
) -> List[Image.Image]:
    """
    Extract frames from video with optional middle-frame-centered sampling.

    This function implements the standardized frame extraction strategy used
    across all models. When middle_frame_guarantee=True, it ensures the middle
    frame of the video is always included, with other frames sampled symmetrically.

    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        middle_frame_guarantee: If True, guarantees middle frame inclusion

    Returns:
        List of PIL Image objects

    Raises:
        ValueError: If video cannot be read or has no frames

    Example:
        >>> frames = extract_video_frames("video.mp4", num_frames=8)
        >>> print(f"Extracted {len(frames)} frames")
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        raise ValueError(f"Could not read video: {video_path}")

    if middle_frame_guarantee:
        # Middle-frame-centered symmetric sampling
        middle_frame = total_frames // 2

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
    else:
        # Uniform sampling across entire video
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    # Extract frames
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)

    cap.release()
    return frames


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """
    Convert PIL Image to base64 string for API transmission.

    Args:
        image: PIL Image object
        format: Image format for encoding (default: PNG)

    Returns:
        Base64 encoded string

    Example:
        >>> img = Image.open("example.png")
        >>> base64_str = image_to_base64(img)
    """
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


# =============================================================================
# RESPONSE PARSING
# =============================================================================

def parse_model_response(
    response_text: str,
    answer_choices: List[str]
) -> Tuple[Optional[str], str]:
    """
    Parse the model's structured response into answer and explanation.

    Validates that the answer is one of the valid choices (case-insensitive).
    Returns None for the answer if the model's response is not a valid choice.

    Args:
        response_text: Raw text from model
        answer_choices: List of valid answer choices

    Returns:
        Tuple of (answer, explanation)
        - answer: The validated answer (or None if invalid)
        - explanation: The model's explanation text

    Example:
        >>> response = "Yes\\nThe sky appears blue due to scattering."
        >>> answer, explanation = parse_model_response(response, ["Yes", "No"])
        >>> print(f"Answer: {answer}")
        Answer: Yes
    """
    lines = response_text.strip().split('\n')

    if len(lines) == 0:
        return None, ""

    # First line is the answer
    potential_answer = lines[0].strip()

    # Rest is explanation
    explanation = '\n'.join(lines[1:]).strip()

    # Validate answer is in the choices
    if answer_choices and len(answer_choices) > 0:
        # Check if the potential answer matches any of the choices (case-insensitive)
        for choice in answer_choices:
            if potential_answer.lower() == choice.strip().lower():
                return choice.strip(), explanation

        # Model failed to provide a valid answer from the options
        return None, explanation
    else:
        # No choices provided, accept whatever the model said
        return potential_answer, explanation


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_response(
    model_answer: Optional[str],
    ground_truth: str,
    answer_choices: List[str]
) -> bool:
    """
    Evaluate model prediction against ground truth using exact matching.

    Args:
        model_answer: Model's extracted answer (or None if invalid)
        ground_truth: Correct answer from dataset
        answer_choices: List of valid answer choices

    Returns:
        True if correct, False otherwise

    Example:
        >>> is_correct = evaluate_response("Yes", "Yes", ["Yes", "No"])
        >>> print(is_correct)
        True
    """
    # If model failed to provide a valid answer, it's incorrect
    if model_answer is None:
        return False

    model_answer_clean = model_answer.strip()
    ground_truth_clean = ground_truth.strip()

    # Direct exact match (case-insensitive)
    if model_answer_clean.lower() == ground_truth_clean.lower():
        return True

    # Check if model answer exactly matches any of the choices
    # and that choice is the ground truth
    if answer_choices:
        for choice in answer_choices:
            if model_answer_clean.lower() == choice.strip().lower():
                if choice.strip().lower() == ground_truth_clean.lower():
                    return True

    return False


def calculate_accuracy_by_type(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Calculate per-question-type accuracy metrics.

    Args:
        results: List of result dictionaries with 'question_type' and 'correct' keys

    Returns:
        Dictionary mapping question types to accuracy statistics

    Example:
        >>> stats = calculate_accuracy_by_type(results)
        >>> print(stats['classification']['accuracy'])
        0.85
    """
    stats_by_type = {}

    for result in results:
        q_type = result.get('question_type')
        if q_type:
            if q_type not in stats_by_type:
                stats_by_type[q_type] = {'correct': 0, 'total': 0}

            stats_by_type[q_type]['total'] += 1
            if result.get('correct', False):
                stats_by_type[q_type]['correct'] += 1

    # Calculate accuracy percentages
    accuracy_by_type = {}
    for q_type, stats in stats_by_type.items():
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        accuracy_by_type[q_type] = {
            'accuracy': accuracy,
            'correct': stats['correct'],
            'total': stats['total']
        }

    return accuracy_by_type


def print_evaluation_summary(
    results: List[Dict[str, Any]],
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Print a formatted summary of evaluation results.

    Args:
        results: List of result dictionaries
        model_name: Name of the model being evaluated

    Returns:
        Dictionary with summary statistics

    Example:
        >>> summary = print_evaluation_summary(results, "claude-sonnet-4-5")
    """
    total = len(results)
    correct = sum(1 for r in results if r.get('correct', False))
    failed_answers = sum(1 for r in results if r.get('model_answer') in [None, 'None', ''])

    overall_accuracy = correct / total if total > 0 else 0
    fail_rate = failed_answers / total if total > 0 else 0

    # Calculate per-type accuracy
    accuracy_by_type = calculate_accuracy_by_type(results)

    # Print summary
    print("\n" + "="*80)
    print(f"EVALUATION RESULTS: {model_name}")
    print("="*80)
    print(f"\nOverall Accuracy: {overall_accuracy:.2%} ({correct}/{total})")
    print(f"Failed to provide valid answer: {failed_answers}/{total} ({fail_rate:.1%})")
    print(f"\nAccuracy by Question Type:")

    for q_type, stats in sorted(accuracy_by_type.items()):
        acc = stats['accuracy']
        corr = stats['correct']
        tot = stats['total']
        print(f"  {q_type}: {acc:.2%} ({corr}/{tot})")

    print("="*80)

    # Return summary dictionary
    return {
        'model': model_name,
        'overall_accuracy': overall_accuracy,
        'correct': correct,
        'total': total,
        'failed_answers': failed_answers,
        'fail_rate': fail_rate,
        'accuracy_by_type': accuracy_by_type
    }


# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================

def load_checkpoint(checkpoint_file: str) -> Tuple[Set[int], List[Dict[str, Any]]]:
    """
    Load checkpoint if it exists to resume interrupted evaluation.

    Args:
        checkpoint_file: Path to checkpoint pickle file

    Returns:
        Tuple of (processed_indices, results)
        - processed_indices: Set of already-processed example indices
        - results: List of result dictionaries

    Example:
        >>> processed, results = load_checkpoint("checkpoint.pkl")
        >>> print(f"Resuming from {len(processed)} processed examples")
    """
    if os.path.exists(checkpoint_file):
        print(f"\n📂 Loading checkpoint from {checkpoint_file}")
        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)

        processed_indices = checkpoint_data.get('processed_indices', set())
        results = checkpoint_data.get('results', [])
        print(f"   Resuming: {len(processed_indices)} examples already processed")
        return processed_indices, results
    else:
        return set(), []


def save_checkpoint(
    checkpoint_file: str,
    processed_indices: Set[int],
    results: List[Dict[str, Any]]
) -> None:
    """
    Save checkpoint for resumability.

    Args:
        checkpoint_file: Path to checkpoint pickle file
        processed_indices: Set of processed example indices
        results: List of result dictionaries

    Example:
        >>> save_checkpoint("checkpoint.pkl", processed_indices, results)
    """
    checkpoint_data = {
        'processed_indices': processed_indices,
        'results': results
    }
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)


def save_results_to_csv(
    results: List[Dict[str, Any]],
    output_file: str
) -> None:
    """
    Save results to CSV file.

    Args:
        results: List of result dictionaries
        output_file: Path to output CSV file

    Example:
        >>> save_results_to_csv(results, "results.csv")
    """
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to: {output_file}")


def cleanup_checkpoint(checkpoint_file: str) -> None:
    """
    Remove checkpoint file after successful completion.

    Args:
        checkpoint_file: Path to checkpoint pickle file

    Example:
        >>> cleanup_checkpoint("checkpoint.pkl")
    """
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"✅ Checkpoint file removed (processing complete)")


# =============================================================================
# HELPER UTILITIES
# =============================================================================

def format_answer_choices(answer_choices: List[str]) -> str:
    """
    Format answer choices as a bulleted list.

    Args:
        answer_choices: List of answer options

    Returns:
        Formatted string with bullet points

    Example:
        >>> formatted = format_answer_choices(["Yes", "No", "Maybe"])
        >>> print(formatted)
        - Yes
        - No
        - Maybe
    """
    return "\n".join([f"- {choice}" for choice in answer_choices])


def validate_environment() -> Dict[str, bool]:
    """
    Validate that required environment variables are set.

    Returns:
        Dictionary mapping variable names to their presence (True/False)

    Example:
        >>> env_status = validate_environment()
        >>> if not env_status['HUGGING_FACE_HUB_TOKEN']:
        ...     print("Warning: HuggingFace token not set")
    """
    required_vars = [
        'HUGGING_FACE_HUB_TOKEN',
        'ANTHROPIC_API_KEY',
        'GOOGLE_API_KEY',
        'OPENAI_API_KEY',
        'GROQ_API_KEY'
    ]

    status = {}
    for var in required_vars:
        status[var] = os.environ.get(var) is not None

    return status


# =============================================================================
# EXAMPLE DATASET ACCESS
# =============================================================================

def get_example_by_index(dataset: Any, index: int) -> Dict[str, Any]:
    """
    Safely get an example from the dataset by index.

    Args:
        dataset: HuggingFace dataset object
        index: Index of example to retrieve

    Returns:
        Dictionary with example data

    Example:
        >>> dataset = load_benchmark_dataset()
        >>> example = get_example_by_index(dataset, 0)
        >>> print(example['question'])
    """
    try:
        return dataset[index]
    except Exception as e:
        raise ValueError(f"Failed to load example at index {index}: {e}")


def filter_dataset_by_question_type(
    dataset: Any,
    question_type: str
) -> Any:
    """
    Filter dataset to only examples of a specific question type.

    Args:
        dataset: HuggingFace dataset object
        question_type: Type of questions to keep (e.g., 'classification')

    Returns:
        Filtered dataset

    Example:
        >>> dataset = load_benchmark_dataset()
        >>> filtered = filter_dataset_by_question_type(dataset, 'classification')
        >>> print(f"Filtered to {len(filtered)} classification questions")
    """
    print(f"Filtering dataset for question_type='{question_type}'...")
    filtered_indices = [
        i for i, qt in enumerate(dataset['question_type'])
        if qt == question_type
    ]
    filtered = dataset.select(filtered_indices)
    print(f"Remaining examples after filter: {len(filtered)}")
    return filtered


if __name__ == "__main__":
    # Quick test of utilities
    print("OpenSeeSimE-Full Utilities")
    print("=" * 80)
    print("\nAvailable functions:")
    print("  - load_benchmark_dataset()")
    print("  - build_system_prompt()")
    print("  - build_user_prompt()")
    print("  - extract_video_frames()")
    print("  - image_to_base64()")
    print("  - parse_model_response()")
    print("  - evaluate_response()")
    print("  - calculate_accuracy_by_type()")
    print("  - print_evaluation_summary()")
    print("  - load_checkpoint() / save_checkpoint()")
    print("  - save_results_to_csv()")
    print("\nFor usage examples, see example_usage.py")
