#!/usr/bin/env python3
"""
Download models from HuggingFace for evaluation.

This script pre-downloads both the base and fine-tuned models
to avoid delays during evaluation.
"""

import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer


def download_model(model_name: str, cache_dir: str = None):
    """Download a model and tokenizer from HuggingFace."""
    print(f"\n{'=' * 60}")
    print(f"Downloading: {model_name}")
    print(f"{'=' * 60}")
    
    try:
        # Download tokenizer
        print("\n1. Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        print(f"   ✓ Tokenizer downloaded")
        print(f"   Vocab size: {len(tokenizer)}")
        
        # Download model (this will take a few minutes for ~1.2GB)
        print("\n2. Downloading model...")
        print("   (This may take several minutes for ~1.2GB...)")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            low_cpu_mem_usage=True,  # More efficient loading
        )
        print(f"   ✓ Model downloaded")
        print(f"   Parameters: {model.num_parameters():,}")
        
        # Verify model is loaded
        print("\n3. Verifying model...")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Device: {model.device}")
        print(f"   Dtype: {model.dtype}")
        
        # Clean up memory
        del model
        del tokenizer
        
        print(f"\n✅ {model_name} downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading {model_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download models from HuggingFace"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen3-Reranker-0.6B",
        help="Base model to download"
    )
    parser.add_argument(
        "--finetuned_model",
        type=str,
        default="codefactory4791/Qwen3-Reranker-HomeDepot",
        help="Fine-tuned model to download"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory (default: HuggingFace default cache)"
    )
    parser.add_argument(
        "--skip_base",
        action="store_true",
        help="Skip downloading base model (if already downloaded)"
    )
    parser.add_argument(
        "--skip_finetuned",
        action="store_true",
        help="Skip downloading fine-tuned model"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Model Download Script")
    print("=" * 60)
    print(f"Base model: {args.base_model}")
    print(f"Fine-tuned model: {args.finetuned_model}")
    if args.cache_dir:
        print(f"Cache directory: {args.cache_dir}")
    print("=" * 60)
    
    success_count = 0
    total_count = 0
    
    # Download base model
    if not args.skip_base:
        total_count += 1
        print(f"\n[1/2] Downloading BASE model...")
        if download_model(args.base_model, args.cache_dir):
            success_count += 1
    else:
        print(f"\nSkipping base model download")
    
    # Download fine-tuned model
    if not args.skip_finetuned:
        total_count += 1
        print(f"\n[2/2] Downloading FINE-TUNED model...")
        if download_model(args.finetuned_model, args.cache_dir):
            success_count += 1
    else:
        print(f"\nSkipping fine-tuned model download")
    
    # Summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"✅ Successfully downloaded: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("\n🎉 All models downloaded!")
        print("\nYou can now run evaluation:")
        print(f"  python scripts/evaluate_reranker.py \\")
        print(f"      --base_model {args.base_model} \\")
        print(f"      --finetuned_model {args.finetuned_model} \\")
        print(f"      --data_path data/home_depot.json \\")
        print(f"      --output_dir evaluation_results")
    else:
        print("\n⚠️  Some downloads failed. Check the errors above.")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
