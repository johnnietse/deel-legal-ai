# RAG Pipeline - Model Optimisation (Fine-tuning, Quantisation, Distillation)
"""
Model optimisation scaffolding inspired by ByteDance practices:
  - LoRA fine-tuning for domain-adapted legal models
  - INT8/FP16 quantisation for inference efficiency
  - Knowledge distillation (teacher → student) for cost reduction
  - GPU cluster auto-scaling configuration

ByteDance uses these techniques to:
  1. Reduce inference cost by 60% via distilled smaller models
  2. Improve domain accuracy by 15-25% via LoRA fine-tuning
  3. Achieve 2-4x inference speedup via INT8 quantisation

This module provides configuration, training scripts, and evaluation
hooks. Actual training requires GPU hardware and PyTorch/HuggingFace.

Requires: pip install torch transformers peft datasets bitsandbytes
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LoRA Fine-Tuning Configuration (ByteDance §7.2)
# ---------------------------------------------------------------------------

@dataclass
class LoRAConfig:
    """
    LoRA (Low-Rank Adaptation) configuration for domain fine-tuning.

    ByteDance fine-tunes their generation models on domain-specific data
    using LoRA to avoid catastrophic forgetting while adapting to the
    legal domain vocabulary and reasoning patterns.
    """
    # Base model
    base_model: str = "google/gemma-2-2b-it"  # Can use any HF model
    model_type: str = "causal_lm"

    # LoRA hyperparameters (ByteDance recommended)
    lora_r: int = 16                # Rank — higher = more parameters, better fit
    lora_alpha: int = 32            # Scaling factor (typically 2*r)
    lora_dropout: float = 0.05      # Regularisation
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj",       # MLP
    ])

    # Training hyperparameters
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    max_seq_length: int = 2048
    weight_decay: float = 0.01

    # Data
    train_data_path: str = ""       # JSONL with {instruction, input, output}
    eval_data_path: str = ""
    output_dir: str = ""

    # Hardware
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str):
        """Save config to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"LoRA config saved to {path}")

    @classmethod
    def load(cls, path: str) -> "LoRAConfig":
        """Load config from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class LoRAPresets:
    """Pre-configured LoRA settings for common use cases."""

    @staticmethod
    def legal_classification() -> LoRAConfig:
        """Optimised for worker classification / legal categorisation."""
        return LoRAConfig(
            lora_r=32,
            lora_alpha=64,
            num_epochs=5,
            learning_rate=1e-4,
            max_seq_length=2048,
        )

    @staticmethod
    def legal_generation() -> LoRAConfig:
        """Optimised for legal text generation / summarisation."""
        return LoRAConfig(
            lora_r=16,
            lora_alpha=32,
            num_epochs=3,
            learning_rate=2e-4,
            max_seq_length=4096,
        )

    @staticmethod
    def embedding_adaptation() -> LoRAConfig:
        """Optimised for embedding model domain adaptation."""
        return LoRAConfig(
            base_model="BAAI/bge-base-en-v1.5",
            model_type="encoder",
            lora_r=8,
            lora_alpha=16,
            num_epochs=2,
            learning_rate=5e-5,
            max_seq_length=512,
        )


# ---------------------------------------------------------------------------
# Fine-Tuning Data Preparation
# ---------------------------------------------------------------------------

class TrainingDataPreparer:
    """
    Prepare training data for LoRA fine-tuning from RAG pipeline data.

    Sources:
      1. Positive user feedback (from feedback_analyzer.py)
      2. Verified Q&A pairs (from verifier.py runs)
      3. Manually curated examples
    """

    @staticmethod
    def from_feedback(
        feedback_path: str,
        output_path: str,
        min_rating: str = "useful",
    ) -> int:
        """
        Convert positive feedback entries into training examples.

        Format: JSONL with {instruction, input, output}
        """
        entries = []
        try:
            with open(feedback_path, "r") as f:
                for line in f:
                    entry = json.loads(line.strip())
                    if entry.get("rating") == min_rating:
                        entries.append({
                            "instruction": "Answer the following legal question based on the provided context.",
                            "input": entry.get("query_text", ""),
                            "output": entry.get("answer_text", ""),
                        })
        except FileNotFoundError:
            logger.warning(f"Feedback file not found: {feedback_path}")
            return 0

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        logger.info(f"Prepared {len(entries)} training examples from feedback")
        return len(entries)

    @staticmethod
    def from_verified_qa(
        qa_pairs: List[Dict[str, str]],
        output_path: str,
    ) -> int:
        """
        Convert verified Q&A pairs into training data.

        Args:
            qa_pairs: List of {query, answer, sources} dicts
            output_path: Output JSONL path
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for pair in qa_pairs:
                entry = {
                    "instruction": "Answer the following legal question accurately with citations.",
                    "input": pair.get("query", ""),
                    "output": pair.get("answer", ""),
                }
                f.write(json.dumps(entry) + "\n")

        logger.info(f"Prepared {len(qa_pairs)} training examples from verified QA")
        return len(qa_pairs)


# ---------------------------------------------------------------------------
# LoRA Fine-Tuning Runner
# ---------------------------------------------------------------------------

class LoRATrainer:
    """
    LoRA fine-tuning runner using HuggingFace PEFT + Transformers.

    ByteDance §7.2: domain-specific fine-tuning with LoRA achieves
    comparable quality to full fine-tuning at 1/10th the cost.
    """

    def __init__(self, config: LoRAConfig):
        self.config = config

    def train(self) -> Dict[str, Any]:
        """
        Execute LoRA fine-tuning.

        Returns training metrics (loss, eval metrics).
        Requires GPU and the following packages:
          pip install torch transformers peft datasets bitsandbytes
        """
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                TrainingArguments,
                Trainer,
            )
            from peft import LoraConfig as PeftLoraConfig, get_peft_model
            from datasets import load_dataset
        except ImportError as e:
            logger.error(f"Missing dependency for LoRA training: {e}")
            return {"error": f"Missing dependency: {e}"}

        logger.info(f"Starting LoRA fine-tuning: {self.config.base_model}")

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=torch.float16 if self.config.fp16 else torch.bfloat16,
            device_map="auto",
        )

        # Apply LoRA
        peft_config = PeftLoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        # Load dataset
        dataset = load_dataset("json", data_files={
            "train": self.config.train_data_path,
            "eval": self.config.eval_data_path,
        })

        # Tokenize
        def tokenize(examples):
            texts = [
                f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
                for inst, inp, out in zip(
                    examples["instruction"],
                    examples["input"],
                    examples["output"],
                )
            ]
            return tokenizer(
                texts,
                truncation=True,
                max_length=self.config.max_seq_length,
                padding="max_length",
            )

        tokenized = dataset.map(tokenize, batched=True)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized.get("eval"),
        )

        # Train
        result = trainer.train()

        # Save
        model.save_pretrained(self.config.output_dir)
        tokenizer.save_pretrained(self.config.output_dir)

        metrics = {
            "train_loss": result.training_loss,
            "train_runtime": result.metrics.get("train_runtime", 0),
            "output_dir": self.config.output_dir,
        }
        logger.info(f"LoRA training complete: {metrics}")
        return metrics


# ---------------------------------------------------------------------------
# Quantisation Configuration (ByteDance §7.3)
# ---------------------------------------------------------------------------

@dataclass
class QuantisationConfig:
    """
    INT8/FP16 quantisation configuration.

    ByteDance §7.3: INT8 quantisation achieves 2-4x inference speedup
    with <1% quality degradation on RAG tasks.
    """
    method: str = "bitsandbytes"       # "bitsandbytes", "gptq", "awq"
    bits: int = 8                       # 4 or 8
    double_quant: bool = True           # Double quantisation for 4-bit
    quant_type: str = "nf4"            # "nf4" or "fp4" for 4-bit
    compute_dtype: str = "float16"

    def to_bnb_config(self):
        """Convert to BitsAndBytes configuration."""
        try:
            from transformers import BitsAndBytesConfig
            import torch

            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }

            if self.bits == 4:
                return BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=self.quant_type,
                    bnb_4bit_use_double_quant=self.double_quant,
                    bnb_4bit_compute_dtype=dtype_map.get(self.compute_dtype, torch.float16),
                )
            else:
                return BitsAndBytesConfig(load_in_8bit=True)
        except ImportError:
            logger.error("bitsandbytes not installed")
            return None


# ---------------------------------------------------------------------------
# Knowledge Distillation Configuration (ByteDance §7.4)
# ---------------------------------------------------------------------------

@dataclass
class DistillationConfig:
    """
    Knowledge distillation configuration (teacher → student).

    ByteDance §7.4: distil from large model (e.g., Gemini Pro) to
    smaller model (e.g., Gemma 2B) using RAG pipeline outputs as
    training signal. Reduces cost by 60% with 5-10% quality drop.
    """
    teacher_model: str = "gemini-2.0-flash"     # API-based teacher
    student_model: str = "google/gemma-2-2b-it"  # Local student
    temperature: float = 3.0                     # Distillation temperature
    alpha: float = 0.5                           # Weight: CE vs KD loss
    num_distillation_samples: int = 5000
    output_dir: str = ""

    def generate_training_plan(self) -> Dict[str, Any]:
        """Generate a training plan for distillation."""
        return {
            "step_1": "Run teacher model on N queries to generate responses",
            "step_2": "Pair each (query, context, teacher_response) as training data",
            "step_3": "Fine-tune student model using LoRA with teacher outputs as targets",
            "step_4": "Evaluate student vs teacher on held-out eval set",
            "step_5": "If quality gap < 10%, deploy student model for cost savings",
            "config": {
                "teacher": self.teacher_model,
                "student": self.student_model,
                "temperature": self.temperature,
                "samples": self.num_distillation_samples,
            },
        }


# ---------------------------------------------------------------------------
# GPU Auto-Scaling Configuration (ByteDance §8.2)
# ---------------------------------------------------------------------------

@dataclass
class AutoScalingConfig:
    """
    GPU cluster auto-scaling configuration.

    ByteDance §8.2: dynamic GPU allocation based on:
      - QPS (queries per second) → scale inference pods
      - Queue depth → scale batch processing pods
      - Model size → allocate appropriate GPU type
    """
    min_replicas: int = 1
    max_replicas: int = 8
    target_qps_per_replica: int = 10
    scale_up_threshold: float = 0.8    # CPU/GPU utilization
    scale_down_threshold: float = 0.3
    cooldown_seconds: int = 300         # 5 minutes between scaling events

    # GPU type mapping
    gpu_profiles: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "inference_small": {
            "gpu_type": "T4",
            "gpu_count": 1,
            "memory_gb": 16,
            "suitable_for": "INT8 quantised models < 7B params",
        },
        "inference_medium": {
            "gpu_type": "A10G",
            "gpu_count": 1,
            "memory_gb": 24,
            "suitable_for": "FP16 models 7-13B params",
        },
        "inference_large": {
            "gpu_type": "A100",
            "gpu_count": 1,
            "memory_gb": 80,
            "suitable_for": "FP16 models 13-70B params",
        },
        "training": {
            "gpu_type": "A100",
            "gpu_count": 4,
            "memory_gb": 320,
            "suitable_for": "LoRA fine-tuning on 7-70B models",
        },
    })

    def get_deployment_spec(self, model_size_b: float = 2.0) -> Dict[str, Any]:
        """
        Get recommended deployment spec for a given model size.

        Args:
            model_size_b: Model size in billions of parameters

        Returns:
            Deployment specification dict
        """
        if model_size_b <= 3:
            profile = "inference_small"
        elif model_size_b <= 13:
            profile = "inference_medium"
        else:
            profile = "inference_large"

        gpu_spec = self.gpu_profiles[profile]
        return {
            "profile": profile,
            "gpu": gpu_spec,
            "scaling": {
                "min_replicas": self.min_replicas,
                "max_replicas": self.max_replicas,
                "target_qps": self.target_qps_per_replica,
            },
            "estimated_latency_ms": {
                "p50": 200 if model_size_b <= 3 else 500,
                "p95": 500 if model_size_b <= 3 else 1500,
            },
        }


# ---------------------------------------------------------------------------
# Cross-Region Deployment Configuration (ByteDance §8.3)
# ---------------------------------------------------------------------------

@dataclass
class CrossRegionConfig:
    """
    Cross-region deployment configuration for high availability.

    ByteDance §8.3: multi-region deployment with:
      - Primary region for writes + reads
      - Read replicas in secondary regions
      - Automatic failover with health checks
    """
    primary_region: str = "us-east-1"
    replica_regions: List[str] = field(default_factory=lambda: ["eu-west-1", "ap-southeast-1"])

    # Vector DB replication
    vector_db_replication: Dict[str, Any] = field(default_factory=lambda: {
        "strategy": "async",              # "async" or "sync"
        "replication_lag_max_ms": 5000,   # Max acceptable lag
        "consistency_level": "eventual",   # "strong" or "eventual"
    })

    # Elasticsearch replication
    elasticsearch_replication: Dict[str, Any] = field(default_factory=lambda: {
        "cross_cluster_replication": True,
        "follower_indices": ["deel-legal-chunks"],
        "polling_interval_ms": 1000,
    })

    # Service routing
    routing: Dict[str, Any] = field(default_factory=lambda: {
        "strategy": "latency_based",    # "latency_based" or "geo_proximity"
        "failover_threshold_ms": 2000,
        "health_check_interval_s": 30,
    })

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
