#!/usr/bin/env python3
"""
Simple KV Cache Test

This is a simplified test to measure KV cache performance without GPU compilation issues.
Tests the key scenarios for prefix caching effectiveness.
"""

import time
from typing import List, Tuple

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: VLLM not available.")

def test_kv_cache_scenarios():
    """Test KV cache performance with simple timing measurements."""
    
    if not VLLM_AVAILABLE:
        print("VLLM not available. Skipping benchmark.")
        return
    
    print("Initializing model with prefix caching...")
    
    # Use the simplest possible VLLM setup
    llm = LLM(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_model_len=256,  # Very small to avoid issues
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        enable_prefix_caching=True,
        trust_remote_code=True
    )
    
    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=20,
        stop=[".", "\n"]
    )
    
    print("Model loaded successfully!")
    
    # Test data
    base_prompt = "The quick brown fox jumps over the lazy dog. This is a test sentence for measuring"
    
    print("\n=== TEST 1: Cold vs Warm Cache ===")
    
    # Cold start - no cache
    print("Cold start (first time processing this prompt)...")
    start_time = time.time()
    outputs1 = llm.generate([base_prompt], sampling_params)
    cold_time = time.time() - start_time
    result1 = outputs1[0].outputs[0].text
    print(f"Cold start time: {cold_time:.3f}s")
    print(f"Generated: {result1[:50]}...")
    
    # Warm cache - same prompt again
    print("\nWarm cache (same prompt again)...")
    start_time = time.time()
    outputs2 = llm.generate([base_prompt], sampling_params)
    warm_time = time.time() - start_time
    result2 = outputs2[0].outputs[0].text
    print(f"Warm cache time: {warm_time:.3f}s")
    print(f"Generated: {result2[:50]}...")
    print(f"Speedup: {cold_time/warm_time:.2f}x")
    
    print("\n=== TEST 2: Prefix Extension ===")
    
    # Extend the prompt (should benefit from prefix cache)
    extended_prompt = base_prompt + " performance and efficiency"
    
    print("Extended prompt (should use cached prefix)...")
    start_time = time.time()
    outputs3 = llm.generate([extended_prompt], sampling_params)
    extended_time = time.time() - start_time
    result3 = outputs3[0].outputs[0].text
    print(f"Extended prompt time: {extended_time:.3f}s")
    print(f"Generated: {result3[:50]}...")
    print(f"Speedup vs cold start: {cold_time/extended_time:.2f}x")
    
    print("\n=== TEST 3: Multiple Similar Prompts ===")
    
    # Test multiple prompts with shared prefix
    prompts = [
        base_prompt + " in machine learning",
        base_prompt + " in web development", 
        base_prompt + " in data science"
    ]
    
    print("Multiple prompts with shared prefix...")
    start_time = time.time()
    outputs4 = llm.generate(prompts, sampling_params)
    batch_time = time.time() - start_time
    print(f"Batch processing time: {batch_time:.3f}s")
    print(f"Average per prompt: {batch_time/len(prompts):.3f}s")
    print(f"Speedup vs cold start: {cold_time/(batch_time/len(prompts)):.2f}x")
    
    for i, output in enumerate(outputs4):
        print(f"  Prompt {i+1}: {output.outputs[0].text[:30]}...")
    
    print("\n=== TEST 4: Simulated JSON Field Generation ===")
    
    # Simulate what prefilled-json does - iterative building
    json_base = 'Generate JSON: {"customer_id": "'
    
    print("Simulating iterative JSON generation...")
    
    # First field
    start_time = time.time()
    outputs5 = llm.generate([json_base], sampling_params)
    field1_time = time.time() - start_time
    field1_value = outputs5[0].outputs[0].text.split('"')[0]
    print(f"Field 1 time: {field1_time:.3f}s, Value: {field1_value}")
    
    # Second field (building on first)
    json_partial = json_base + field1_value + '", "issue_type": "'
    start_time = time.time()
    outputs6 = llm.generate([json_partial], sampling_params)
    field2_time = time.time() - start_time
    field2_value = outputs6[0].outputs[0].text.split('"')[0]
    print(f"Field 2 time: {field2_time:.3f}s, Value: {field2_value}")
    
    # Third field (building further)
    json_partial2 = json_partial + field2_value + '", "priority": "'
    start_time = time.time()
    outputs7 = llm.generate([json_partial2], sampling_params)
    field3_time = time.time() - start_time
    field3_value = outputs7[0].outputs[0].text.split('"')[0]
    print(f"Field 3 time: {field3_time:.3f}s, Value: {field3_value}")
    
    avg_field_time = (field1_time + field2_time + field3_time) / 3
    print(f"Average field generation time: {avg_field_time:.3f}s")
    
    # Compare to generating all at once
    full_json_prompt = 'Generate JSON: {"customer_id": "value1", "issue_type": "value2", "priority": "'
    start_time = time.time()
    outputs8 = llm.generate([full_json_prompt], sampling_params)
    full_time = time.time() - start_time
    print(f"Full JSON generation time: {full_time:.3f}s")
    print(f"Iterative vs Full ratio: {avg_field_time/full_time:.2f}x")
    
    print("\n=== SUMMARY ===")
    print(f"Cold start time: {cold_time:.3f}s")
    print(f"Warm cache speedup: {cold_time/warm_time:.2f}x")
    print(f"Prefix extension speedup: {cold_time/extended_time:.2f}x") 
    print(f"Batch processing speedup: {cold_time/(batch_time/len(prompts)):.2f}x")
    print(f"Iterative JSON overhead: {avg_field_time/full_time:.2f}x")
    
    print("\nKey Insights:")
    print("- Higher speedup values indicate better cache effectiveness")
    print("- Prefix caching should speed up repeated/similar prompts")
    print("- Iterative generation (like prefilled-json) may have overhead")

if __name__ == "__main__":
    test_kv_cache_scenarios()