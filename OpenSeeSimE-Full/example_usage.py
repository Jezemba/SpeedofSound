#!/usr/bin/env python3
"""
Example Usage: OpenSeeSimE-Full Utilities
------------------------------------------
This script demonstrates how to use the shared utilities for evaluating
a vision-language model on the OpenSeeSimE-Structural benchmark.

This is a template that you can adapt for your specific model.
"""

import argparse
from tqdm import tqdm
from utils import (
    # Dataset
    load_benchmark_dataset,
    get_example_by_index,

    # Prompts
    build_system_prompt,
    build_user_prompt,

    # Video processing
    extract_video_frames,
    image_to_base64,

    # Response handling
    parse_model_response,
    evaluate_response,

    # Evaluation
    print_evaluation_summary,
    calculate_accuracy_by_type,

    # Checkpointing
    load_checkpoint,
    save_checkpoint,
    save_results_to_csv,
    cleanup_checkpoint,

    # Helpers
    validate_environment
)


def mock_model_inference(system_prompt, user_prompt, image=None, frames=None):
    """
    Mock model inference function.

    Replace this with your actual model API call or inference code.

    Args:
        system_prompt: The standardized system prompt
        user_prompt: The user prompt with question and choices
        image: PIL Image (for image examples)
        frames: List of PIL Images (for video examples)

    Returns:
        String response from the model
    """
    # This is a placeholder - replace with your actual model call
    # Examples:
    #
    # For Claude:
    # client = anthropic.Anthropic()
    # message = client.messages.create(
    #     model="claude-sonnet-4-5-20250929",
    #     max_tokens=512,
    #     temperature=0.0,
    #     system=system_prompt,
    #     messages=[{"role": "user", "content": content}]
    # )
    # return message.content[0].text
    #
    # For OpenAI:
    # client = OpenAI()
    # response = client.chat.completions.create(
    #     model="gpt-4o",
    #     messages=[
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": [
    #             {"type": "text", "text": user_prompt},
    #             {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_to_base64(image)}"}}
    #         ]}
    #     ]
    # )
    # return response.choices[0].message.content

    # Mock response for demonstration
    return "Option A\nThis is a placeholder explanation."


def run_evaluation(
    dataset_name="JessicaE/OpenSeeSimE-Structural",
    media_type="all",
    num_video_frames=8,
    checkpoint_file="checkpoint_example.pkl",
    output_file="results_example.csv",
    model_name="ExampleModel",
    max_examples=None
):
    """
    Run complete evaluation with checkpointing.

    Args:
        dataset_name: HuggingFace dataset name
        media_type: 'all', 'image', or 'video'
        num_video_frames: Number of frames to extract from videos
        checkpoint_file: Path to checkpoint file
        output_file: Path to output CSV
        model_name: Name of your model (for results)
        max_examples: Maximum number of examples to process (None = all)

    Returns:
        Dictionary with evaluation summary
    """
    print("="*80)
    print(f"OpenSeeSimE Evaluation: {model_name}")
    print("="*80)

    # Step 1: Validate environment
    print("\n[1/7] Validating environment...")
    env_status = validate_environment()
    if not env_status['HUGGING_FACE_HUB_TOKEN']:
        print("⚠️  Warning: HUGGING_FACE_HUB_TOKEN not set")
        print("   Set it with: export HUGGING_FACE_HUB_TOKEN='hf_...'")
        return None

    # Step 2: Load dataset
    print("\n[2/7] Loading dataset...")
    dataset = load_benchmark_dataset(
        dataset_name=dataset_name,
        media_type=media_type
    )

    # Limit examples if specified
    if max_examples and max_examples < len(dataset):
        print(f"   Limiting to first {max_examples} examples")
        dataset = dataset.select(range(max_examples))

    # Step 3: Load checkpoint
    print("\n[3/7] Loading checkpoint (if exists)...")
    processed_indices, results = load_checkpoint(checkpoint_file)

    # Step 4: Build system prompt
    print("\n[4/7] Building standardized prompts...")
    system_prompt = build_system_prompt()
    print(f"   System prompt: {len(system_prompt)} characters")

    # Step 5: Process examples
    print(f"\n[5/7] Processing examples...")
    print(f"   Total: {len(dataset)}")
    print(f"   Already processed: {len(processed_indices)}")
    print(f"   Remaining: {len(dataset) - len(processed_indices)}")

    for idx in tqdm(range(len(dataset)), desc="Evaluating"):
        # Skip if already processed
        if idx in processed_indices:
            continue

        try:
            # Get example
            example = get_example_by_index(dataset, idx)

            # Prepare based on media type
            media_type_ex = example['media_type']
            image = None
            frames = None

            if media_type_ex == 'image':
                # Single image
                image = example['image']
                user_prompt = build_user_prompt(
                    question=example['question'],
                    answer_choices=example['answer_choices'],
                    is_video=False
                )
            else:
                # Video - extract frames
                frames = extract_video_frames(
                    example['video'],
                    num_frames=num_video_frames
                )
                user_prompt = build_user_prompt(
                    question=example['question'],
                    answer_choices=example['answer_choices'],
                    is_video=True,
                    num_frames=len(frames)
                )

            # Run model inference
            # REPLACE mock_model_inference with your actual model call
            model_response = mock_model_inference(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image=image,
                frames=frames
            )

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

            # Store result
            result = {
                'file_name': example['file_name'],
                'source_file': example['source_file'],
                'question': example['question'],
                'question_type': example['question_type'],
                'question_id': example['question_id'],
                'answer': example['answer'],
                'answer_choices': str(example['answer_choices']),
                'correct_choice_idx': example['correct_choice_idx'],
                'model': model_name,
                'model_answer': model_answer if model_answer is not None else 'None',
                'explanation': explanation,
                'correct': is_correct,
                'media_type': example['media_type']
            }

            results.append(result)
            processed_indices.add(idx)

            # Save checkpoint periodically (every 10 examples)
            if len(results) % 10 == 0:
                save_checkpoint(checkpoint_file, processed_indices, results)
                save_results_to_csv(results, output_file)

        except Exception as e:
            print(f"\n❌ Error processing example {idx}: {e}")
            # Mark as processed to skip in future runs
            processed_indices.add(idx)
            save_checkpoint(checkpoint_file, processed_indices, results)
            continue

    # Step 6: Save final results
    print("\n[6/7] Saving final results...")
    save_results_to_csv(results, output_file)

    # Step 7: Calculate and display metrics
    print("\n[7/7] Calculating metrics...")
    summary = print_evaluation_summary(results, model_name=model_name)

    # Cleanup checkpoint
    cleanup_checkpoint(checkpoint_file)

    return summary


def example_single_inference():
    """
    Example: Process a single example (useful for debugging).
    """
    print("\n" + "="*80)
    print("Example: Single Inference")
    print("="*80)

    # Load dataset
    dataset = load_benchmark_dataset(media_type='image')

    # Get first example
    example = dataset[0]

    # Build prompts
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(
        question=example['question'],
        answer_choices=example['answer_choices']
    )

    print("\n--- Question ---")
    print(example['question'])

    print("\n--- Answer Choices ---")
    for i, choice in enumerate(example['answer_choices']):
        marker = "✓" if i == example['correct_choice_idx'] else " "
        print(f"  [{marker}] {choice}")

    print("\n--- System Prompt (first 200 chars) ---")
    print(system_prompt[:200] + "...")

    print("\n--- User Prompt ---")
    print(user_prompt)

    # Mock model response
    model_response = "Option A\nThis is an example explanation."

    print("\n--- Model Response ---")
    print(model_response)

    # Parse
    model_answer, explanation = parse_model_response(
        model_response,
        example['answer_choices']
    )

    print("\n--- Parsed ---")
    print(f"Answer: {model_answer}")
    print(f"Explanation: {explanation}")

    # Evaluate
    is_correct = evaluate_response(
        model_answer,
        example['answer'],
        example['answer_choices']
    )

    print("\n--- Evaluation ---")
    print(f"Ground truth: {example['answer']}")
    print(f"Model answer: {model_answer}")
    print(f"Correct: {is_correct}")


def example_video_processing():
    """
    Example: Process video frames.
    """
    print("\n" + "="*80)
    print("Example: Video Frame Extraction")
    print("="*80)

    # Load dataset
    dataset = load_benchmark_dataset(media_type='video')

    if len(dataset) == 0:
        print("No video examples found in dataset")
        return

    # Get first video example
    example = dataset[0]

    print(f"\nProcessing video: {example['file_name']}")

    # Extract frames
    print("Extracting 8 frames with middle frame guarantee...")
    frames = extract_video_frames(
        example['video'],
        num_frames=8,
        middle_frame_guarantee=True
    )

    print(f"Extracted {len(frames)} frames")
    print(f"Frame sizes: {[f.size for f in frames[:3]]} ...")

    # Build video prompt
    user_prompt = build_user_prompt(
        question=example['question'],
        answer_choices=example['answer_choices'],
        is_video=True,
        num_frames=len(frames)
    )

    print("\n--- Video Prompt (first 300 chars) ---")
    print(user_prompt[:300] + "...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Example usage of OpenSeeSimE-Full utilities"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "single", "video"],
        help="Run mode: 'full' (complete evaluation), 'single' (one example), 'video' (video demo)"
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="JessicaE/OpenSeeSimE-Structural",
        help="HuggingFace dataset name"
    )

    parser.add_argument(
        "--media_type",
        type=str,
        default="image",
        choices=["all", "image", "video"],
        help="Filter by media type"
    )

    parser.add_argument(
        "--num_video_frames",
        type=int,
        default=8,
        help="Number of frames to extract from videos"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="ExampleModel",
        help="Name of your model"
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default="results_example.csv",
        help="Output CSV file"
    )

    parser.add_argument(
        "--checkpoint_file",
        type=str,
        default="checkpoint_example.pkl",
        help="Checkpoint file path"
    )

    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum number of examples to process (for testing)"
    )

    args = parser.parse_args()

    if args.mode == "single":
        # Run single inference example
        example_single_inference()

    elif args.mode == "video":
        # Run video processing example
        example_video_processing()

    else:
        # Run full evaluation
        print("\n⚠️  NOTE: This example uses mock model inference!")
        print("   Replace mock_model_inference() with your actual model call.\n")

        summary = run_evaluation(
            dataset_name=args.dataset_name,
            media_type=args.media_type,
            num_video_frames=args.num_video_frames,
            checkpoint_file=args.checkpoint_file,
            output_file=args.output_file,
            model_name=args.model_name,
            max_examples=args.max_examples
        )

        if summary:
            print("\n✅ Evaluation complete!")
            print(f"   Overall accuracy: {summary['overall_accuracy']:.2%}")
            print(f"   Results saved to: {args.output_file}")
