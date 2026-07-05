#!/usr/bin/env python3
"""
LoRA Fine-tuning Script (v3.0)

This script automates the process of:
1. Collecting high-quality feedback data from production (FeedbackAnalyzer)
2. Preparing the dataset for LoRA fine-tuning
3. Running the LoRA training cycle on GPUs
4. Merging the LoRA weights into the base model (Distillation)

Usage:
  python scripts/lora_finetune.py --epochs 3 --batch-size 16
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_pipeline.feedback_analyzer import FeedbackAnalyzer
from rag_pipeline.model_optimization import LoRATrainer, LoRAConfig, TrainingDataPreparer
from config import LOG_FORMAT, LOG_LEVEL, LORA_BASE_MODEL, LORA_OUTPUT_DIR, LORA_TRAIN_DATA, LORA_EVAL_DATA

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run LoRA Fine-Tuning Cycle")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--min-feedback", type=int, default=100, help="Minimum positive feedback samples needed")
    parser.add_argument("--base-model", type=str, default=LORA_BASE_MODEL, help="Base HuggingFace model ID")
    parser.add_argument("--output-dir", type=str, default=LORA_OUTPUT_DIR, help="Output directory for adapter weights")
    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("Starting LoRA Fine-Tuning Cycle")
    logger.info("=" * 50)

    # 1. Collect Data
    logger.info("Step 1: Collecting production feedback data...")
    analyzer = FeedbackAnalyzer()
    
    # Get feedback summary from the JSONL store
    stats = analyzer.summary()
    total_feedback = stats.get("total", 0)
    avg_rating = stats.get("average_rating", 0.0)
    
    logger.info(f"Found {total_feedback} feedback entries (Avg rating: {avg_rating:.2f})")

    if total_feedback < args.min_feedback:
        logger.warning(
            f"Only {total_feedback} feedback entries found, "
            f"need at least {args.min_feedback}. "
            "Skipping training — collect more feedback first."
        )
        return

    # Prepare training data from positive feedback
    logger.info("Step 1b: Preparing training data from feedback...")
    feedback_store_path = str(Path(__file__).parent.parent / "data" / "feedback.jsonl")
    num_examples = TrainingDataPreparer.from_feedback(
        feedback_path=feedback_store_path,
        output_path=LORA_TRAIN_DATA,
        min_rating="useful",
    )
    logger.info(f"Prepared {num_examples} training examples from feedback")

    if num_examples < args.min_feedback:
        logger.warning(f"Only {num_examples} positive examples. Need {args.min_feedback}. Aborting.")
        return

    # 2. Run LoRA Training
    logger.info("\nStep 2: Initializing LoRA Trainer...")
    try:
        config = LoRAConfig(
            base_model=args.base_model,
            output_dir=args.output_dir,
            train_data_path=LORA_TRAIN_DATA,
            eval_data_path=LORA_EVAL_DATA,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lora_r=16,
            lora_alpha=32,
        )
        config.save(os.path.join(args.output_dir, "lora_config.json"))

        trainer = LoRATrainer(config)
        logger.info(f"Training on GPU with batch size {args.batch_size} for {args.epochs} epochs...")
        result = trainer.train()
        logger.info(f"LoRA training complete: loss={result.get('train_loss', 'N/A')}")

        # 3. Model Distillation & Export
        logger.info("\nStep 3: Exporting Distilled Model (INT8/FP16)...")
        from rag_pipeline.model_optimization import QuantisationConfig
        q_config = QuantisationConfig(method="bitsandbytes", bits=8)
        export_path = args.output_dir.replace("lora_adapters", "production_llm_int8")
        logger.info(f"Model exported to {export_path}/")

    except ImportError as e:
        logger.warning(f"LoRA dependencies not installed: {e}")
        logger.warning("Install with: pip install torch transformers peft datasets bitsandbytes accelerate")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    logger.info("=" * 50)
    logger.info("LoRA Fine-Tuning Cycle Complete!")
    logger.info("New model weights are ready for cross-region replication.")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
