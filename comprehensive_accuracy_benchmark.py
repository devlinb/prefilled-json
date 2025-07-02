#!/usr/bin/env python3
"""
Comprehensive Accuracy Benchmark

Compare multiple approaches for JSON generation:
1. Our prefilled-json with stop tokens (Qwen, Phi-3.5 full, Phi-3.5 GPTQ)
2. Simple prompting
3. VLLM's built-in JSON mode (guided generation)
4. VLLM's constrained generation

Test accuracy, speed, and reliability across different JSON complexity scenarios.
"""

import time
import json
import statistics
import sys
import os
from typing import Dict, List, Any, Tuple, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_prefilled_json_models():
    """Test our prefilled-json approach with different models."""
    
    print("🔧 Testing Prefilled-JSON with Multiple Models")
    print("=" * 70)
    
    models_to_test = [
        {
            "name": "Qwen2-1.5B-Instruct",
            "model": "Qwen/Qwen2-1.5B-Instruct",
            "config": {"stop_tokens": [",", "}", "\n"], "stop_reliable": True},
            "memory_util": 0.25,
            "disable_sliding": False
        },
        {
            "name": "Phi-3.5-GPTQ-4bit",
            "model": "thesven/Phi-3.5-mini-instruct-GPTQ-4bit", 
            "config": {"stop_tokens": [",", "}", "\n", "<|end|>"], "stop_reliable": True},
            "memory_util": 0.35,
            "disable_sliding": True
        },
        {
            "name": "Phi-3.5-Full",
            "model": "microsoft/Phi-3.5-mini-instruct",
            "config": {"stop_tokens": [",", "}", "\n", "<|end|>"], "stop_reliable": True},
            "memory_util": 0.4,
            "disable_sliding": True
        }
    ]
    
    # Test scenarios with varying complexity
    test_scenarios = [
        {
            "name": "simple_fields",
            "fields": [{"name": "string"}, {"age": "number"}],
            "description": "Simple 2-field JSON"
        },
        {
            "name": "multiple_fields", 
            "fields": [{"id": "number"}, {"name": "string"}, {"email": "string"}, {"active": "string"}],
            "description": "Multiple fields (4)"
        },
        {
            "name": "complex_naming",
            "fields": [{"first_name": "string"}, {"last_name": "string"}, {"phone_number": "string"}],
            "description": "Complex field names"
        },
        {
            "name": "nested_object",
            "fields": [{"user": {"name": "string", "id": "number"}}, {"timestamp": "string"}],
            "description": "Nested object structure"
        }
    ]
    
    results = {}
    
    for model_info in models_to_test:
        print(f"\n🚀 Testing {model_info['name']}")
        
        try:
            from vllm import LLM, SamplingParams
            from driver.stop_token_json_driver import StopTokenJsonDriver
            
            # Load model with appropriate settings
            llm_kwargs = {
                "model": model_info["model"],
                "max_model_len": 512,
                "gpu_memory_utilization": model_info["memory_util"],
                "enable_prefix_caching": True,
                "trust_remote_code": True,
                "dtype": "float16"
            }
            
            if model_info["disable_sliding"]:
                llm_kwargs["disable_sliding_window"] = True
            
            llm = LLM(**llm_kwargs)
            
            def generate_func(prompt: str, stop_token: str = None) -> str:
                stop_list = [stop_token] if stop_token else None
                params = SamplingParams(
                    temperature=0.5,
                    max_tokens=20,
                    stop=stop_list,
                    skip_special_tokens=True
                )
                outputs = llm.generate([prompt], params)
                return outputs[0].outputs[0].text.strip()
            
            driver = StopTokenJsonDriver(generate_func, model_info["config"])
            
            model_results = {}
            
            for scenario in test_scenarios:
                print(f"  Testing {scenario['description']}...")
                
                times = []
                valid_count = 0
                field_accuracy = []
                
                for run in range(3):
                    start = time.time()
                    try:
                        result = driver.generate_json(scenario["fields"])
                        elapsed = time.time() - start
                        times.append(elapsed)
                        
                        # Validate JSON
                        parsed = json.loads(result)
                        valid_count += 1
                        
                        # Calculate field accuracy
                        expected_fields = set()
                        for field in scenario["fields"]:
                            if isinstance(field, dict):
                                expected_fields.update(field.keys())
                        
                        actual_fields = set(parsed.keys())
                        accuracy = len(expected_fields & actual_fields) / len(expected_fields)
                        field_accuracy.append(accuracy)
                        
                    except Exception as e:
                        elapsed = time.time() - start
                        times.append(elapsed)
                        field_accuracy.append(0.0)
                
                model_results[scenario["name"]] = {
                    "avg_time": statistics.mean(times),
                    "validity_rate": valid_count / 3,
                    "avg_accuracy": statistics.mean(field_accuracy),
                    "description": scenario["description"]
                }
                
                print(f"    {scenario['description']}: {valid_count}/3 valid, {statistics.mean(field_accuracy):.1%} accurate")
            
            results[model_info["name"]] = model_results
            print(f"  ✅ {model_info['name']} completed")
            
        except Exception as e:
            print(f"  ❌ {model_info['name']} failed: {e}")
            results[model_info["name"]] = {}
    
    return results


def test_simple_prompting():
    """Test simple prompting approach."""
    
    print(f"\n📝 Testing Simple Prompting")
    print("=" * 70)
    
    try:
        from vllm import LLM, SamplingParams
        
        # Use Qwen for simple prompting comparison
        llm = LLM(
            model="Qwen/Qwen2-1.5B-Instruct",
            max_model_len=256,
            gpu_memory_utilization=0.2,
            enable_prefix_caching=True,
            trust_remote_code=True
        )
        
        test_prompts = [
            ("Generate JSON with name and age fields:", "simple_fields"),
            ("Generate JSON with id, name, email, and active fields:", "multiple_fields"),
            ("Generate JSON with first_name, last_name, and phone_number:", "complex_naming"),
            ("Generate JSON with user object containing name and id, plus timestamp:", "nested_object")
        ]
        
        results = {}
        
        for prompt, scenario_name in test_prompts:
            print(f"  Testing {scenario_name}...")
            
            params = SamplingParams(
                temperature=0.5,
                max_tokens=80,
                skip_special_tokens=True
            )
            
            times = []
            valid_count = 0
            over_generation_count = 0
            
            for run in range(3):
                start = time.time()
                outputs = llm.generate([prompt], params)
                elapsed = time.time() - start
                result = outputs[0].outputs[0].text.strip()
                times.append(elapsed)
                
                try:
                    # Look for JSON in the response
                    json_start = result.find('{')
                    if json_start != -1:
                        json_part = result[json_start:]
                        json_end = json_part.rfind('}') + 1
                        if json_end > 0:
                            json_str = json_part[:json_end]
                            parsed = json.loads(json_str)
                            valid_count += 1
                            
                            # Check for over-generation (extra text after JSON)
                            remaining = result[json_start + json_end:].strip()
                            if remaining:
                                over_generation_count += 1
                        
                except json.JSONDecodeError:
                    pass
            
            results[scenario_name] = {
                "avg_time": statistics.mean(times),
                "validity_rate": valid_count / 3,
                "over_generation_rate": over_generation_count / 3,
                "description": f"Simple prompting - {scenario_name}"
            }
            
            print(f"    {scenario_name}: {valid_count}/3 valid, {over_generation_count}/3 over-generated")
        
        return results
        
    except Exception as e:
        print(f"  ❌ Simple prompting failed: {e}")
        return {}


def test_vllm_json_mode():
    """Test VLLM's built-in JSON mode (guided generation)."""
    
    print(f"\n🎯 Testing VLLM JSON Mode (Guided Generation)")
    print("=" * 70)
    
    try:
        from vllm import LLM, SamplingParams
        
        # Test with a suitable model for guided generation
        llm = LLM(
            model="Qwen/Qwen2-1.5B-Instruct",
            max_model_len=256,
            gpu_memory_utilization=0.2,
            enable_prefix_caching=True,
            trust_remote_code=True
        )
        
        # Define JSON schemas for guided generation
        schemas = {
            "simple_fields": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "number"}
                },
                "required": ["name", "age"]
            },
            "multiple_fields": {
                "type": "object", 
                "properties": {
                    "id": {"type": "number"},
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                    "active": {"type": "string"}
                },
                "required": ["id", "name", "email", "active"]
            },
            "complex_naming": {
                "type": "object",
                "properties": {
                    "first_name": {"type": "string"},
                    "last_name": {"type": "string"},
                    "phone_number": {"type": "string"}
                },
                "required": ["first_name", "last_name", "phone_number"]
            },
            "nested_object": {
                "type": "object",
                "properties": {
                    "user": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "id": {"type": "number"}
                        },
                        "required": ["name", "id"]
                    },
                    "timestamp": {"type": "string"}
                },
                "required": ["user", "timestamp"]
            }
        }
        
        results = {}
        
        for scenario_name, schema in schemas.items():
            print(f"  Testing {scenario_name} with guided generation...")
            
            try:
                params = SamplingParams(
                    temperature=0.5,
                    max_tokens=50,
                    guided_json=schema,  # Use guided JSON generation
                    skip_special_tokens=True
                )
                
                prompt = f"Generate a JSON object that matches the schema:"
                
                times = []
                valid_count = 0
                schema_compliance = []
                
                for run in range(3):
                    start = time.time()
                    outputs = llm.generate([prompt], params)
                    elapsed = time.time() - start
                    result = outputs[0].outputs[0].text.strip()
                    times.append(elapsed)
                    
                    try:
                        parsed = json.loads(result)
                        valid_count += 1
                        
                        # Check schema compliance
                        required_fields = schema.get("required", [])
                        actual_fields = set(parsed.keys())
                        compliance = len(set(required_fields) & actual_fields) / len(required_fields)
                        schema_compliance.append(compliance)
                        
                    except json.JSONDecodeError:
                        schema_compliance.append(0.0)
                
                results[scenario_name] = {
                    "avg_time": statistics.mean(times),
                    "validity_rate": valid_count / 3,
                    "schema_compliance": statistics.mean(schema_compliance),
                    "description": f"VLLM JSON mode - {scenario_name}"
                }
                
                print(f"    {scenario_name}: {valid_count}/3 valid, {statistics.mean(schema_compliance):.1%} compliant")
                
            except Exception as e:
                print(f"    {scenario_name}: ❌ Error: {e}")
                results[scenario_name] = {
                    "avg_time": 0,
                    "validity_rate": 0,
                    "schema_compliance": 0,
                    "description": f"VLLM JSON mode - {scenario_name} (failed)"
                }
        
        return results
        
    except Exception as e:
        print(f"  ❌ VLLM JSON mode failed: {e}")
        return {}


def test_constrained_generation():
    """Test VLLM's constrained generation with regex."""
    
    print(f"\n🔒 Testing Constrained Generation (Regex)")
    print("=" * 70)
    
    try:
        from vllm import LLM, SamplingParams
        import re
        
        llm = LLM(
            model="Qwen/Qwen2-1.5B-Instruct",
            max_model_len=256,
            gpu_memory_utilization=0.2,
            enable_prefix_caching=True,
            trust_remote_code=True
        )
        
        # Define regex patterns for JSON structures
        patterns = {
            "simple_fields": r'\{"name":\s*"[^"]*",\s*"age":\s*\d+\}',
            "multiple_fields": r'\{"id":\s*\d+,\s*"name":\s*"[^"]*",\s*"email":\s*"[^"]*",\s*"active":\s*"[^"]*"\}',
        }
        
        results = {}
        
        for scenario_name, pattern in patterns.items():
            print(f"  Testing {scenario_name} with regex constraint...")
            
            try:
                params = SamplingParams(
                    temperature=0.5,
                    max_tokens=50,
                    guided_regex=pattern,  # Use regex constraint
                    skip_special_tokens=True
                )
                
                prompt = f"Generate JSON:"
                
                times = []
                valid_count = 0
                pattern_match = []
                
                for run in range(3):
                    start = time.time()
                    outputs = llm.generate([prompt], params)
                    elapsed = time.time() - start
                    result = outputs[0].outputs[0].text.strip()
                    times.append(elapsed)
                    
                    try:
                        parsed = json.loads(result)
                        valid_count += 1
                        
                        # Check pattern compliance
                        matches = bool(re.match(pattern, result))
                        pattern_match.append(matches)
                        
                    except json.JSONDecodeError:
                        pattern_match.append(False)
                
                results[scenario_name] = {
                    "avg_time": statistics.mean(times),
                    "validity_rate": valid_count / 3,
                    "pattern_compliance": sum(pattern_match) / len(pattern_match),
                    "description": f"Constrained generation - {scenario_name}"
                }
                
                print(f"    {scenario_name}: {valid_count}/3 valid, {sum(pattern_match)}/3 pattern match")
                
            except Exception as e:
                print(f"    {scenario_name}: ❌ Error: {e}")
                results[scenario_name] = {
                    "avg_time": 0,
                    "validity_rate": 0, 
                    "pattern_compliance": 0,
                    "description": f"Constrained generation - {scenario_name} (failed)"
                }
        
        return results
        
    except Exception as e:
        print(f"  ❌ Constrained generation failed: {e}")
        return {}


def analyze_results(prefilled_results, simple_results, vllm_json_results, constrained_results):
    """Analyze and compare all results."""
    
    print(f"\n📊 COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    
    # Aggregate results by approach
    approaches = {
        "Prefilled-JSON (Qwen)": prefilled_results.get("Qwen2-1.5B-Instruct", {}),
        "Prefilled-JSON (Phi-3.5 GPTQ)": prefilled_results.get("Phi-3.5-GPTQ-4bit", {}),
        "Prefilled-JSON (Phi-3.5 Full)": prefilled_results.get("Phi-3.5-Full", {}),
        "Simple Prompting": simple_results,
        "VLLM JSON Mode": vllm_json_results,
        "Constrained Generation": constrained_results
    }
    
    print(f"\n📋 DETAILED RESULTS BY APPROACH:")
    
    for approach_name, approach_results in approaches.items():
        if not approach_results:
            print(f"\n❌ {approach_name}: No results")
            continue
            
        print(f"\n✅ {approach_name}:")
        
        # Calculate overall metrics
        all_validity = []
        all_times = []
        all_accuracy = []
        
        for scenario_name, data in approach_results.items():
            validity = data.get("validity_rate", 0)
            avg_time = data.get("avg_time", 0)
            accuracy = data.get("avg_accuracy", data.get("schema_compliance", data.get("pattern_compliance", 0)))
            
            all_validity.append(validity)
            all_times.append(avg_time)
            all_accuracy.append(accuracy)
            
            print(f"  {scenario_name}: {validity:.1%} valid, {avg_time:.3f}s, {accuracy:.1%} accurate")
        
        if all_validity:
            overall_validity = statistics.mean(all_validity)
            overall_time = statistics.mean(all_times)
            overall_accuracy = statistics.mean(all_accuracy)
            
            print(f"  📊 Overall: {overall_validity:.1%} valid, {overall_time:.3f}s avg, {overall_accuracy:.1%} accurate")
    
    # Find best approach
    print(f"\n🏆 PERFORMANCE RANKING:")
    
    rankings = []
    for approach_name, approach_results in approaches.items():
        if not approach_results:
            continue
            
        all_validity = [data.get("validity_rate", 0) for data in approach_results.values()]
        all_times = [data.get("avg_time", 0) for data in approach_results.values() if data.get("avg_time", 0) > 0]
        
        if all_validity:
            overall_validity = statistics.mean(all_validity)
            overall_time = statistics.mean(all_times) if all_times else float('inf')
            
            # Score: high validity, low time
            score = overall_validity / (overall_time + 0.001)  # Avoid division by zero
            rankings.append((approach_name, overall_validity, overall_time, score))
    
    rankings.sort(key=lambda x: x[3], reverse=True)
    
    for i, (approach, validity, time, score) in enumerate(rankings):
        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else "  "
        print(f"  {medal} {approach}: {validity:.1%} valid, {time:.3f}s (score: {score:.2f})")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if rankings:
        best_approach = rankings[0][0]
        best_validity = rankings[0][1]
        
        if best_validity >= 0.9:
            print(f"  🎉 {best_approach} is EXCELLENT for production use")
        elif best_validity >= 0.8:
            print(f"  ✅ {best_approach} is GOOD for production use")
        else:
            print(f"  ⚠️  All approaches need improvement for production")
        
        # Check if our prefilled-json approach is competitive
        prefilled_approaches = [r for r in rankings if "Prefilled-JSON" in r[0]]
        if prefilled_approaches:
            best_prefilled = prefilled_approaches[0]
            if best_prefilled == rankings[0]:
                print(f"  🏆 Our Prefilled-JSON approach is the BEST overall!")
            elif best_prefilled[1] >= 0.9:
                print(f"  🎯 Our Prefilled-JSON approach is highly competitive")
        
        # Over-generation analysis
        if "Simple Prompting" in [r[0] for r in rankings]:
            simple_over_gen = statistics.mean([
                data.get("over_generation_rate", 0) 
                for data in simple_results.values()
            ])
            if simple_over_gen > 0.3:
                print(f"  ⚠️  Simple prompting has {simple_over_gen:.1%} over-generation issues")


def main():
    print("🔬 Comprehensive JSON Generation Accuracy Benchmark")
    print("Comparing prefilled-json vs alternatives on L4 GPU")
    print("=" * 80)
    
    # Test all approaches
    print("🔥 Running comprehensive benchmark suite...")
    
    # Test 1: Our prefilled-json approach with multiple models
    prefilled_results = test_prefilled_json_models()
    
    # Test 2: Simple prompting baseline
    simple_results = test_simple_prompting()
    
    # Test 3: VLLM's JSON mode (guided generation)
    vllm_json_results = test_vllm_json_mode()
    
    # Test 4: Constrained generation with regex
    constrained_results = test_constrained_generation()
    
    # Analyze and compare all results
    analyze_results(prefilled_results, simple_results, vllm_json_results, constrained_results)
    
    print(f"\n🎯 CONCLUSION:")
    print(f"This benchmark provides comprehensive comparison across:")
    print(f"  - 3 prefilled-json models (Qwen, Phi-3.5 GPTQ, Phi-3.5 Full)")
    print(f"  - Simple prompting baseline")
    print(f"  - VLLM's built-in JSON mode")
    print(f"  - VLLM's constrained generation")
    print(f"  - 4 complexity scenarios each")
    print(f"Enabling informed decisions for production JSON generation!")


if __name__ == "__main__":
    main()