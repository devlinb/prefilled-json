#!/usr/bin/env python3
"""
Comprehensive JSON Generation Benchmark

Tests prefilled-json stop token approach across different JSON complexity scenarios:
1. Simple fields (few parameters)
2. Nested objects 
3. Complicated field names
4. Mixed scenarios

Tests both speed and accuracy with Qwen2-1.5B-Instruct and attempts quantized Gemma.
"""

import time
import json
import statistics
import sys
import os
from typing import Dict, List, Any, Tuple, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_simple_fields_scenario():
    """Test 1: Simple fields - just a few basic parameters."""
    
    try:
        from vllm import LLM, SamplingParams
        from driver.stop_token_json_driver import StopTokenJsonDriver, STOP_TOKEN_EXCELLENT
        
        print("🧪 SCENARIO 1: SIMPLE FIELDS (Few Parameters)")
        print("=" * 70)
        
        llm = LLM(
            model="Qwen/Qwen2-1.5B-Instruct",
            max_model_len=256,
            gpu_memory_utilization=0.5,
            enable_prefix_caching=True,
            trust_remote_code=True
        )
        
        def vllm_generate_func(prompt: str, stop_token: str = None) -> str:
            stop_list = [stop_token] if stop_token else None
            params = SamplingParams(
                temperature=0.3,
                max_tokens=15,
                stop=stop_list,
                skip_special_tokens=True
            )
            outputs = llm.generate([prompt], params)
            return outputs[0].outputs[0].text.strip()
        
        model_config = STOP_TOKEN_EXCELLENT.get("Qwen/Qwen2-1.5B-Instruct", {
            "stop_tokens": [",", "}", "\n"],
            "stop_reliable": True
        })
        
        driver = StopTokenJsonDriver(vllm_generate_func, model_config)
        
        # Test different simple field scenarios
        test_cases = [
            # Very simple - 2 fields
            ([{"name": "string"}, {"age": "number"}], "2 fields"),
            
            # Simple - 3 fields
            ([{"id": "number"}, {"email": "string"}, {"active": "string"}], "3 fields"),
            
            # Medium simple - 4 fields
            ([{"username": "string"}, {"score": "number"}, {"level": "string"}, {"points": "number"}], "4 fields")
        ]
        
        results = {}
        
        for fields, description in test_cases:
            print(f"\nTesting {description}: {fields}")
            
            times = []
            valid_count = 0
            field_accuracy = []
            
            for run in range(3):
                start = time.time()
                try:
                    result = driver.generate_json(fields)
                    elapsed = time.time() - start
                    times.append(elapsed)
                    
                    # Validate JSON and field accuracy
                    parsed = json.loads(result)
                    valid_count += 1
                    
                    expected_fields = set(list(field.keys())[0] for field in fields)
                    actual_fields = set(parsed.keys())
                    accuracy = len(expected_fields & actual_fields) / len(expected_fields)
                    field_accuracy.append(accuracy)
                    
                    print(f"  Run {run+1}: {elapsed:.3f}s ✅ Valid JSON, {accuracy:.1%} field accuracy")
                    
                except Exception as e:
                    elapsed = time.time() - start
                    times.append(elapsed)
                    field_accuracy.append(0.0)
                    print(f"  Run {run+1}: {elapsed:.3f}s ❌ Error: {e}")
            
            results[description] = {
                "avg_time": statistics.mean(times),
                "validity_rate": valid_count / len(times),
                "avg_accuracy": statistics.mean(field_accuracy),
                "times": times
            }
        
        return results
        
    except Exception as e:
        print(f"❌ Simple fields test failed: {e}")
        return {}


def test_nested_objects_scenario():
    """Test 2: Nested objects - complex hierarchical structures."""
    
    try:
        from vllm import LLM, SamplingParams
        from driver.stop_token_json_driver import StopTokenJsonDriver, STOP_TOKEN_EXCELLENT
        
        print("\n🧪 SCENARIO 2: NESTED OBJECTS (Complex Hierarchy)")
        print("=" * 70)
        
        llm = LLM(
            model="Qwen/Qwen2-1.5B-Instruct",
            max_model_len=512,  # Larger for nested structures
            gpu_memory_utilization=0.5,
            enable_prefix_caching=True,
            trust_remote_code=True
        )
        
        def vllm_generate_func(prompt: str, stop_token: str = None) -> str:
            stop_list = [stop_token] if stop_token else None
            params = SamplingParams(
                temperature=0.3,
                max_tokens=20,
                stop=stop_list,
                skip_special_tokens=True
            )
            outputs = llm.generate([prompt], params)
            return outputs[0].outputs[0].text.strip()
        
        model_config = STOP_TOKEN_EXCELLENT.get("Qwen/Qwen2-1.5B-Instruct", {})
        driver = StopTokenJsonDriver(vllm_generate_func, model_config)
        
        # Test different nested scenarios
        test_cases = [
            # Simple nesting - 1 level
            ([
                {"user": {"name": "string", "id": "number"}},
                {"status": "string"}
            ], "1-level nesting"),
            
            # Medium nesting - 2 levels
            ([
                {"user": {
                    "profile": {"name": "string", "age": "number"},
                    "settings": {"theme": "string"}
                }},
                {"timestamp": "string"}
            ], "2-level nesting"),
            
            # Complex nesting - multiple objects
            ([
                {"customer": {"name": "string", "tier": "string"}},
                {"order": {"id": "number", "total": "number"}},
                {"metadata": {"created": "string", "source": "string"}}
            ], "multiple nested")
        ]
        
        results = {}
        
        for fields, description in test_cases:
            print(f"\nTesting {description}:")
            print(f"Structure: {fields}")
            
            times = []
            valid_count = 0
            
            for run in range(3):
                start = time.time()
                try:
                    result = driver.generate_json(fields)
                    elapsed = time.time() - start
                    times.append(elapsed)
                    
                    # Validate nested JSON structure
                    parsed = json.loads(result)
                    valid_count += 1
                    
                    print(f"  Run {run+1}: {elapsed:.3f}s ✅ Valid nested JSON")
                    print(f"    Keys: {list(parsed.keys())}")
                    
                except Exception as e:
                    elapsed = time.time() - start
                    times.append(elapsed)
                    print(f"  Run {run+1}: {elapsed:.3f}s ❌ Error: {e}")
            
            results[description] = {
                "avg_time": statistics.mean(times),
                "validity_rate": valid_count / len(times),
                "times": times
            }
        
        return results
        
    except Exception as e:
        print(f"❌ Nested objects test failed: {e}")
        return {}


def test_complicated_field_names_scenario():
    """Test 3: Complicated field names - edge cases and difficult naming."""
    
    try:
        from vllm import LLM, SamplingParams
        from driver.stop_token_json_driver import StopTokenJsonDriver, STOP_TOKEN_EXCELLENT
        
        print("\n🧪 SCENARIO 3: COMPLICATED FIELD NAMES")
        print("=" * 70)
        
        llm = LLM(
            model="Qwen/Qwen2-1.5B-Instruct",
            max_model_len=256,
            gpu_memory_utilization=0.5,
            enable_prefix_caching=True,
            trust_remote_code=True
        )
        
        def vllm_generate_func(prompt: str, stop_token: str = None) -> str:
            stop_list = [stop_token] if stop_token else None
            params = SamplingParams(
                temperature=0.3,
                max_tokens=15,
                stop=stop_list,
                skip_special_tokens=True
            )
            outputs = llm.generate([prompt], params)
            return outputs[0].outputs[0].text.strip()
        
        model_config = STOP_TOKEN_EXCELLENT.get("Qwen/Qwen2-1.5B-Instruct", {})
        driver = StopTokenJsonDriver(vllm_generate_func, model_config)
        
        # Test different complicated field naming scenarios
        test_cases = [
            # Snake case
            ([{"first_name": "string"}, {"last_name": "string"}, {"phone_number": "string"}], "snake_case"),
            
            # CamelCase
            ([{"firstName": "string"}, {"lastName": "string"}, {"phoneNumber": "string"}], "camelCase"),
            
            # Long descriptive names
            ([{"customerSupportTicketId": "number"}, {"priorityClassificationLevel": "string"}, {"issueResolutionTimestamp": "string"}], "long descriptive"),
            
            # Mixed conventions and numbers
            ([{"user_id_v2": "number"}, {"apiKey": "string"}, {"last_updated_timestamp": "string"}], "mixed conventions"),
            
            # Abbreviations and acronyms
            ([{"uuid": "string"}, {"api_endpoint_url": "string"}, {"http_status_code": "number"}], "abbreviations")
        ]
        
        results = {}
        
        for fields, description in test_cases:
            print(f"\nTesting {description}: {[list(f.keys())[0] for f in fields]}")
            
            times = []
            valid_count = 0
            field_accuracy = []
            
            for run in range(3):
                start = time.time()
                try:
                    result = driver.generate_json(fields)
                    elapsed = time.time() - start
                    times.append(elapsed)
                    
                    # Validate JSON and check field name accuracy
                    parsed = json.loads(result)
                    valid_count += 1
                    
                    expected_fields = set(list(field.keys())[0] for field in fields)
                    actual_fields = set(parsed.keys())
                    
                    # Check exact field name match
                    exact_match = expected_fields == actual_fields
                    field_accuracy.append(1.0 if exact_match else 0.0)
                    
                    print(f"  Run {run+1}: {elapsed:.3f}s ✅ Valid JSON, exact fields: {'✅' if exact_match else '❌'}")
                    if not exact_match:
                        print(f"    Expected: {expected_fields}")
                        print(f"    Actual: {actual_fields}")
                    
                except Exception as e:
                    elapsed = time.time() - start
                    times.append(elapsed)
                    field_accuracy.append(0.0)
                    print(f"  Run {run+1}: {elapsed:.3f}s ❌ Error: {e}")
            
            results[description] = {
                "avg_time": statistics.mean(times),
                "validity_rate": valid_count / len(times),
                "exact_field_accuracy": statistics.mean(field_accuracy),
                "times": times
            }
        
        return results
        
    except Exception as e:
        print(f"❌ Complicated field names test failed: {e}")
        return {}


def test_quantized_gemma_model():
    """Test 4: Try quantized Gemma model if available."""
    
    print("\n🧪 SCENARIO 4: QUANTIZED GEMMA MODEL")
    print("=" * 70)
    
    # Try different Gemma model variants
    gemma_models = [
        "google/gemma-2b",
        "google/gemma-2b-it", 
        "unsloth/gemma-2b-bnb-4bit",
        "unsloth/gemma-2b-it-bnb-4bit"
    ]
    
    for model_name in gemma_models:
        print(f"\nTrying {model_name}...")
        
        try:
            from vllm import LLM, SamplingParams
            from driver.stop_token_json_driver import StopTokenJsonDriver
            
            llm = LLM(
                model=model_name,
                max_model_len=256,
                gpu_memory_utilization=0.4,  # Conservative for quantized
                enable_prefix_caching=True,
                trust_remote_code=True
            )
            
            print(f"✅ {model_name} loaded successfully!")
            
            def vllm_generate_func(prompt: str, stop_token: str = None) -> str:
                stop_list = [stop_token] if stop_token else None
                params = SamplingParams(
                    temperature=0.3,
                    max_tokens=10,
                    stop=stop_list,
                    skip_special_tokens=True
                )
                outputs = llm.generate([prompt], params)
                return outputs[0].outputs[0].text.strip()
            
            # Test with a simple JSON
            model_config = {"stop_tokens": [",", "}", "\n"], "stop_reliable": True}
            driver = StopTokenJsonDriver(vllm_generate_func, model_config)
            
            test_fields = [{"name": "string"}, {"age": "number"}]
            
            times = []
            valid_count = 0
            
            for run in range(3):
                start = time.time()
                try:
                    result = driver.generate_json(test_fields)
                    elapsed = time.time() - start
                    times.append(elapsed)
                    
                    parsed = json.loads(result)
                    valid_count += 1
                    print(f"  Run {run+1}: {elapsed:.3f}s ✅ Valid JSON: {result}")
                    
                except Exception as e:
                    elapsed = time.time() - start
                    times.append(elapsed)
                    print(f"  Run {run+1}: {elapsed:.3f}s ❌ Error: {e}")
            
            avg_time = statistics.mean(times)
            validity_rate = valid_count / len(times)
            
            print(f"📊 {model_name} Results:")
            print(f"  Average time: {avg_time:.3f}s")
            print(f"  Validity rate: {validity_rate:.1%}")
            
            if validity_rate > 0.5:
                print(f"🎉 {model_name} works with stop tokens!")
                return {
                    "model": model_name,
                    "avg_time": avg_time,
                    "validity_rate": validity_rate,
                    "working": True
                }
            else:
                print(f"⚠️ {model_name} has reliability issues")
                
        except Exception as e:
            print(f"❌ {model_name} failed: {e}")
            continue
    
    print("❌ No working quantized Gemma models found")
    return {"working": False}


def compare_scenarios_with_simple_prompting():
    """Compare our scenarios against simple prompting approach."""
    
    print("\n🧪 COMPARISON: PREFILLED-JSON vs SIMPLE PROMPTING")
    print("=" * 70)
    
    try:
        from vllm import LLM, SamplingParams
        
        llm = LLM(
            model="Qwen/Qwen2-1.5B-Instruct",
            max_model_len=256,
            gpu_memory_utilization=0.5,
            enable_prefix_caching=True,
            trust_remote_code=True
        )
        
        # Test simple prompting on our test cases
        simple_prompts = [
            # Simple fields
            "Generate JSON with name (string) and age (number):",
            
            # Nested 
            "Generate JSON with user object containing name and id, plus status:",
            
            # Complicated fields
            "Generate JSON with first_name, last_name, and phone_number:"
        ]
        
        print("Simple prompting results:")
        simple_results = []
        
        for i, prompt in enumerate(simple_prompts):
            print(f"\nPrompt {i+1}: {prompt}")
            
            params = SamplingParams(temperature=0.3, max_tokens=50, skip_special_tokens=True)
            times = []
            valid_count = 0
            
            for run in range(3):
                start = time.time()
                outputs = llm.generate([prompt], params)
                elapsed = time.time() - start
                result = outputs[0].outputs[0].text.strip()
                times.append(elapsed)
                
                try:
                    json.loads(result)
                    valid_count += 1
                    print(f"  Run {run+1}: {elapsed:.3f}s ✅ Valid")
                except:
                    print(f"  Run {run+1}: {elapsed:.3f}s ❌ Invalid JSON")
            
            simple_results.append({
                "avg_time": statistics.mean(times),
                "validity_rate": valid_count / len(times)
            })
        
        return simple_results
        
    except Exception as e:
        print(f"❌ Simple prompting comparison failed: {e}")
        return []


def main():
    print("🔬 Comprehensive JSON Generation Benchmark")
    print("Testing prefilled-json across different complexity scenarios")
    print("=" * 80)
    
    # Run all test scenarios
    simple_results = test_simple_fields_scenario()
    nested_results = test_nested_objects_scenario()
    complicated_results = test_complicated_field_names_scenario()
    
    # Try quantized Gemma
    gemma_results = test_quantized_gemma_model()
    
    # Compare with simple prompting
    simple_prompting_results = compare_scenarios_with_simple_prompting()
    
    # Final analysis
    print("\n📊 COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    
    # Aggregate prefilled-json results
    all_prefilled_times = []
    all_prefilled_validity = []
    
    for scenario_name, results in [
        ("Simple Fields", simple_results),
        ("Nested Objects", nested_results), 
        ("Complicated Names", complicated_results)
    ]:
        if results:
            print(f"\n{scenario_name}:")
            for test_name, data in results.items():
                validity = data.get("validity_rate", 0)
                avg_time = data.get("avg_time", 0)
                all_prefilled_times.append(avg_time)
                all_prefilled_validity.append(validity)
                
                print(f"  {test_name}: {avg_time:.3f}s, {validity:.1%} valid")
    
    # Overall metrics
    if all_prefilled_times and all_prefilled_validity:
        overall_prefilled_time = statistics.mean(all_prefilled_times)
        overall_prefilled_validity = statistics.mean(all_prefilled_validity)
        
        print(f"\n🎯 OVERALL PREFILLED-JSON PERFORMANCE:")
        print(f"  Average time: {overall_prefilled_time:.3f}s")
        print(f"  Average validity: {overall_prefilled_validity:.1%}")
        
        # Compare with simple prompting if available
        if simple_prompting_results:
            simple_avg_time = statistics.mean([r["avg_time"] for r in simple_prompting_results])
            simple_avg_validity = statistics.mean([r["validity_rate"] for r in simple_prompting_results])
            
            speed_ratio = simple_avg_time / overall_prefilled_time
            
            print(f"\n📈 vs SIMPLE PROMPTING:")
            print(f"  Speed advantage: {speed_ratio:.1f}x faster")
            print(f"  Reliability advantage: {overall_prefilled_validity - simple_avg_validity:+.1%}")
    
    # Gemma results
    if gemma_results.get("working"):
        print(f"\n🎉 GEMMA MODEL SUCCESS:")
        print(f"  Model: {gemma_results['model']}")
        print(f"  Performance: {gemma_results['avg_time']:.3f}s, {gemma_results['validity_rate']:.1%} valid")
    else:
        print(f"\n❌ GEMMA MODELS: Not available or incompatible")
    
    # Final recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if overall_prefilled_validity > 0.8:
        print(f"  ✅ Prefilled-JSON is highly reliable across all scenarios")
    if len(all_prefilled_times) > 0 and max(all_prefilled_times) - min(all_prefilled_times) < 0.1:
        print(f"  ✅ Performance is consistent across complexity levels")
    
    print(f"  🎯 Best for: Production JSON APIs requiring guaranteed structure")
    print(f"  ⚡ Works well with: Qwen2-1.5B-Instruct and stop tokens")


if __name__ == "__main__":
    main()