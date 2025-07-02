#!/usr/bin/env python3
"""
Focused Complexity Benchmark

Tests prefilled-json across different JSON complexity scenarios with conservative settings
to avoid CUDA compilation issues while demonstrating the library's capabilities.
"""

import time
import json
import statistics
import sys
import os
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_single_test(driver, fields, description, max_runs=2):
    """Run a single test scenario with error handling."""
    
    print(f"\nTesting {description}:")
    print(f"Fields: {fields}")
    
    times = []
    valid_count = 0
    field_accuracy = []
    
    for run in range(max_runs):
        start = time.time()
        try:
            result = driver.generate_json(fields)
            elapsed = time.time() - start
            times.append(elapsed)
            
            # Validate JSON
            parsed = json.loads(result)
            valid_count += 1
            
            # Check field accuracy
            expected_fields = set()
            for field in fields:
                if isinstance(field, dict):
                    for key, value in field.items():
                        if isinstance(value, dict):
                            # Nested object
                            expected_fields.add(key)
                        else:
                            # Simple field
                            expected_fields.add(key)
            
            actual_fields = set(parsed.keys())
            accuracy = len(expected_fields & actual_fields) / len(expected_fields) if expected_fields else 1.0
            field_accuracy.append(accuracy)
            
            print(f"  Run {run+1}: {elapsed:.3f}s ✅ Valid JSON, {accuracy:.1%} field accuracy")
            print(f"    Result: {result}")
            
        except Exception as e:
            elapsed = time.time() - start
            times.append(elapsed)
            field_accuracy.append(0.0)
            print(f"  Run {run+1}: {elapsed:.3f}s ❌ Error: {str(e)[:50]}...")
    
    return {
        "avg_time": statistics.mean(times) if times else 0,
        "validity_rate": valid_count / len(times) if times else 0,
        "avg_accuracy": statistics.mean(field_accuracy) if field_accuracy else 0,
        "times": times
    }


def test_complexity_scenarios():
    """Test different JSON complexity scenarios."""
    
    try:
        from vllm import LLM, SamplingParams
        from driver.stop_token_json_driver import StopTokenJsonDriver, STOP_TOKEN_EXCELLENT
        
        print("🔬 Focused Complexity Benchmark")
        print("Testing prefilled-json across different JSON complexity scenarios")
        print("=" * 80)
        
        # Use conservative settings to avoid CUDA issues
        llm = LLM(
            model="Qwen/Qwen2-1.5B-Instruct",
            max_model_len=128,  # Very small
            gpu_memory_utilization=0.4,  # Conservative
            enable_prefix_caching=True,
            trust_remote_code=True
        )
        
        def vllm_generate_func(prompt: str, stop_token: str = None) -> str:
            stop_list = [stop_token] if stop_token else None
            params = SamplingParams(
                temperature=0.3,
                max_tokens=10,  # Small to avoid issues
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
        
        # Test scenarios with increasing complexity
        test_scenarios = [
            # Scenario 1: Simple fields (few parameters)
            {
                "name": "Simple Fields",
                "tests": [
                    ([{"name": "string"}], "1 field"),
                    ([{"name": "string"}, {"age": "number"}], "2 fields"),
                    ([{"id": "number"}, {"email": "string"}, {"active": "string"}], "3 fields")
                ]
            },
            
            # Scenario 2: Complicated field names
            {
                "name": "Complicated Field Names", 
                "tests": [
                    ([{"first_name": "string"}, {"last_name": "string"}], "snake_case"),
                    ([{"firstName": "string"}, {"lastName": "string"}], "camelCase"),
                    ([{"user_id": "number"}, {"api_key": "string"}], "mixed_naming")
                ]
            },
            
            # Scenario 3: Nested objects (simplified)
            {
                "name": "Nested Objects",
                "tests": [
                    ([{"user": {"name": "string"}}], "1-level nested"),
                    ([{"profile": {"id": "number", "name": "string"}}], "1-level multi-field")
                ]
            }
        ]
        
        all_results = {}
        
        for scenario in test_scenarios:
            print(f"\n🧪 SCENARIO: {scenario['name'].upper()}")
            print("=" * 60)
            
            scenario_results = {}
            
            for fields, description in scenario["tests"]:
                try:
                    result = run_single_test(driver, fields, description)
                    scenario_results[description] = result
                except Exception as e:
                    print(f"❌ Test {description} failed completely: {e}")
                    scenario_results[description] = {"avg_time": 0, "validity_rate": 0, "avg_accuracy": 0}
            
            all_results[scenario["name"]] = scenario_results
        
        return all_results
        
    except Exception as e:
        print(f"❌ Complexity test failed: {e}")
        return {}


def test_simple_prompting_comparison():
    """Compare with simple prompting approach."""
    
    try:
        from vllm import LLM, SamplingParams
        
        print("\n🧪 SIMPLE PROMPTING COMPARISON")
        print("=" * 60)
        
        llm = LLM(
            model="Qwen/Qwen2-1.5B-Instruct",
            max_model_len=128,
            gpu_memory_utilization=0.4,
            enable_prefix_caching=True,
            trust_remote_code=True
        )
        
        # Test simple prompting on basic scenarios
        test_cases = [
            ("Generate JSON with name (string):", "simple"),
            ("Generate JSON with name and age:", "two_fields"),
            ("Generate JSON with user_id and email:", "naming_test")
        ]
        
        simple_results = {}
        
        for prompt, test_name in test_cases:
            print(f"\nTesting {test_name}: {prompt}")
            
            params = SamplingParams(temperature=0.3, max_tokens=30, skip_special_tokens=True)
            times = []
            valid_count = 0
            
            for run in range(2):  # Conservative number of runs
                try:
                    start = time.time()
                    outputs = llm.generate([prompt], params)
                    elapsed = time.time() - start
                    result = outputs[0].outputs[0].text.strip()
                    times.append(elapsed)
                    
                    # Try to parse as JSON
                    json.loads(result)
                    valid_count += 1
                    print(f"  Run {run+1}: {elapsed:.3f}s ✅ Valid JSON")
                    print(f"    Result: {result}")
                    
                except json.JSONDecodeError:
                    print(f"  Run {run+1}: {elapsed:.3f}s ❌ Invalid JSON")
                    print(f"    Result: {result[:50]}...")
                except Exception as e:
                    elapsed = time.time() - start if 'start' in locals() else 0
                    times.append(elapsed)
                    print(f"  Run {run+1}: {elapsed:.3f}s ❌ Error: {str(e)[:30]}...")
            
            simple_results[test_name] = {
                "avg_time": statistics.mean(times) if times else 0,
                "validity_rate": valid_count / len(times) if times else 0
            }
        
        return simple_results
        
    except Exception as e:
        print(f"❌ Simple prompting comparison failed: {e}")
        return {}


def analyze_results(prefilled_results, simple_results):
    """Analyze and compare results across scenarios."""
    
    print("\n📊 COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    
    # Aggregate prefilled-json results
    all_prefilled_times = []
    all_prefilled_validity = []
    all_prefilled_accuracy = []
    
    print("\nPrefilled-JSON Results by Scenario:")
    
    for scenario_name, scenario_data in prefilled_results.items():
        if scenario_data:
            print(f"\n{scenario_name}:")
            
            for test_name, data in scenario_data.items():
                validity = data.get("validity_rate", 0)
                avg_time = data.get("avg_time", 0)
                accuracy = data.get("avg_accuracy", 0)
                
                if avg_time > 0:  # Only include successful tests
                    all_prefilled_times.append(avg_time)
                    all_prefilled_validity.append(validity)
                    all_prefilled_accuracy.append(accuracy)
                
                print(f"  {test_name}: {avg_time:.3f}s, {validity:.1%} valid, {accuracy:.1%} accurate")
    
    # Simple prompting results
    if simple_results:
        print(f"\nSimple Prompting Results:")
        simple_times = []
        simple_validity = []
        
        for test_name, data in simple_results.items():
            validity = data.get("validity_rate", 0)
            avg_time = data.get("avg_time", 0)
            
            if avg_time > 0:
                simple_times.append(avg_time)
                simple_validity.append(validity)
            
            print(f"  {test_name}: {avg_time:.3f}s, {validity:.1%} valid")
    
    # Overall comparison
    if all_prefilled_times and simple_times:
        print(f"\n🎯 OVERALL COMPARISON:")
        
        prefilled_avg_time = statistics.mean(all_prefilled_times)
        prefilled_avg_validity = statistics.mean(all_prefilled_validity)
        prefilled_avg_accuracy = statistics.mean(all_prefilled_accuracy)
        
        simple_avg_time = statistics.mean(simple_times)
        simple_avg_validity = statistics.mean(simple_validity)
        
        speed_ratio = simple_avg_time / prefilled_avg_time if prefilled_avg_time > 0 else 1.0
        
        print(f"  Prefilled-JSON: {prefilled_avg_time:.3f}s avg, {prefilled_avg_validity:.1%} valid, {prefilled_avg_accuracy:.1%} accurate")
        print(f"  Simple Prompting: {simple_avg_time:.3f}s avg, {simple_avg_validity:.1%} valid")
        print(f"  Speed Advantage: {speed_ratio:.1f}x {'faster' if speed_ratio > 1 else 'slower'}")
        print(f"  Reliability Advantage: {prefilled_avg_validity - simple_avg_validity:+.1%}")
    
    # Complexity analysis
    if prefilled_results:
        print(f"\n🔧 COMPLEXITY ANALYSIS:")
        
        # Check if performance degrades with complexity
        if "Simple Fields" in prefilled_results and "Complicated Field Names" in prefilled_results:
            simple_avg = statistics.mean([d["avg_time"] for d in prefilled_results["Simple Fields"].values() if d["avg_time"] > 0])
            complex_avg = statistics.mean([d["avg_time"] for d in prefilled_results["Complicated Field Names"].values() if d["avg_time"] > 0])
            
            if simple_avg > 0 and complex_avg > 0:
                complexity_overhead = (complex_avg - simple_avg) / simple_avg * 100
                print(f"  Complexity overhead: {complexity_overhead:+.1f}% (complex vs simple fields)")
        
        # Check nested object performance
        if "Nested Objects" in prefilled_results:
            nested_results = [d for d in prefilled_results["Nested Objects"].values() if d["validity_rate"] > 0]
            if nested_results:
                nested_success_rate = len(nested_results) / len(prefilled_results["Nested Objects"])
                print(f"  Nested object success rate: {nested_success_rate:.1%}")
    
    # Final recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if all_prefilled_validity and statistics.mean(all_prefilled_validity) > 0.8:
        print(f"  ✅ Prefilled-JSON shows high reliability across complexity levels")
    
    if simple_times and all_prefilled_times:
        speed_advantage = statistics.mean(simple_times) / statistics.mean(all_prefilled_times)
        if speed_advantage > 1.2:
            print(f"  ⚡ Prefilled-JSON is significantly faster than simple prompting")
        elif speed_advantage > 1.0:
            print(f"  🏃 Prefilled-JSON has speed advantages")
    
    print(f"  🎯 Best use case: Structured JSON APIs requiring guaranteed field compliance")
    print(f"  🔧 Works well with: Qwen2-1.5B-Instruct + stop tokens + prefix caching")


def main():
    print("🔬 Focused Complexity Benchmark")
    print("Testing prefilled-json with conservative settings to avoid CUDA issues")
    print("=" * 80)
    
    # Run main complexity tests
    prefilled_results = test_complexity_scenarios()
    
    # Run simple prompting comparison
    simple_results = test_simple_prompting_comparison()
    
    # Analyze results
    if prefilled_results or simple_results:
        analyze_results(prefilled_results, simple_results)
    else:
        print("❌ All tests failed - check VLLM setup and GPU memory")
    
    print(f"\n🎉 Benchmark complete! Check results above for performance insights.")


if __name__ == "__main__":
    main()