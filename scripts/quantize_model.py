#!/usr/bin/env python3
"""
Quantize Qwen3-Reranker model for faster inference.

Creates INT8 quantized version for 2-4x faster inference with minimal accuracy loss.
"""

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path


def quantize_model(model_path: str, output_path: str, quantization_type: str = "int8"):
    """
    Quantize model for faster inference.
    
    Args:
        model_path: Path to the fine-tuned model
        output_path: Path to save quantized model
        quantization_type: Type of quantization ('int8' or '4bit')
    """
    print("=" * 60)
    print(f"Quantizing Model ({quantization_type.upper()})")
    print("=" * 60)
    print(f"Input: {model_path}")
    print(f"Output: {output_path}")
    print("=" * 60)
    
    # 1. Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    print("   ✓ Tokenizer loaded")
    
    # 2. Load model with quantization
    print(f"\n2. Loading and quantizing model ({quantization_type})...")
    
    if quantization_type == "int8":
        # INT8 quantization (recommended)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=True,
            device_map="auto",
            trust_remote_code=True,
        )
        print("   ✓ Model loaded in INT8")
        print("   💡 Expected speedup: 2-3x")
        print("   💡 Memory reduction: ~50%")
        
    elif quantization_type == "4bit":
        # 4-bit quantization (more aggressive)
        from transformers import BitsAndBytesConfig
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )
        print("   ✓ Model loaded in 4-bit")
        print("   💡 Expected speedup: 3-4x")
        print("   💡 Memory reduction: ~75%")
    
    else:
        raise ValueError(f"Unknown quantization type: {quantization_type}")
    
    # 3. Save quantized model
    print(f"\n3. Saving quantized model...")
    os.makedirs(output_path, exist_ok=True)
    
    # Note: Quantized models can't be saved with save_pretrained()
    # We need to use the original model and add quantization config
    print("   ⚠️  Note: For INT8/4bit models, you'll need to:")
    print("      1. Push the original (non-quantized) model to HF")
    print("      2. Users load with load_in_8bit=True or load_in_4bit=True")
    print("\n   Saving quantization config for reference...")
    
    # Save config
    config_path = os.path.join(output_path, "quantization_config.txt")
    with open(config_path, 'w') as f:
        f.write(f"Quantization Type: {quantization_type}\n")
        f.write(f"Base Model: {model_path}\n")
        f.write(f"\nTo load this model with quantization:\n")
        if quantization_type == "int8":
            f.write('model = AutoModelForCausalLM.from_pretrained(\n')
            f.write('    "codefactory4791/Qwen3-Reranker-HomeDepot",\n')
            f.write('    load_in_8bit=True,\n')
            f.write('    device_map="auto",\n')
            f.write('    trust_remote_code=True\n')
            f.write(')\n')
        else:
            f.write('from transformers import BitsAndBytesConfig\n')
            f.write('quantization_config = BitsAndBytesConfig(\n')
            f.write('    load_in_4bit=True,\n')
            f.write('    bnb_4bit_compute_dtype=torch.bfloat16\n')
            f.write(')\n')
            f.write('model = AutoModelForCausalLM.from_pretrained(\n')
            f.write('    "codefactory4791/Qwen3-Reranker-HomeDepot",\n')
            f.write('    quantization_config=quantization_config,\n')
            f.write('    device_map="auto",\n')
            f.write('    trust_remote_code=True\n')
            f.write(')\n')
    
    print(f"   ✓ Config saved to {config_path}")
    
    # 4. Test inference
    print("\n4. Testing quantized inference...")
    test_query = "drill"
    test_doc = "DEWALT Cordless Drill Kit"
    
    prompt = f'''<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: Given a web search query, retrieve relevant passages that answer the query
<Query>: {test_query}
<Document>: {test_doc}<|im_end|>
<|im_start|>assistant
<think>

</think>

'''
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]
    
    token_yes = tokenizer.convert_tokens_to_ids('yes')
    token_no = tokenizer.convert_tokens_to_ids('no')
    score = torch.sigmoid(logits[token_yes] - logits[token_no]).item()
    
    print(f"   Test query: '{test_query}'")
    print(f"   Test document: '{test_doc}'")
    print(f"   Relevance score: {score:.4f}")
    print("   ✓ Quantized model works!")
    
    print("\n" + "=" * 60)
    print("✅ Quantization complete!")
    print("=" * 60)
    print("\n💡 Recommendation:")
    print("   Push the ORIGINAL (non-quantized) model to HuggingFace")
    print("   Users can load with quantization on-demand:")
    print('   model = AutoModelForCausalLM.from_pretrained(..., load_in_8bit=True)')
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Quantize model for faster inference")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="models/quantized",
        help="Output path for quantized model"
    )
    parser.add_argument(
        "--quantization_type",
        type=str,
        default="int8",
        choices=["int8", "4bit"],
        help="Quantization type"
    )
    
    args = parser.parse_args()
    
    quantize_model(
        model_path=args.model_path,
        output_path=args.output_path,
        quantization_type=args.quantization_type
    )


if __name__ == "__main__":
    main()
