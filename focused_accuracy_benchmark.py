#!/usr/bin/env python3
"""
Focused Accuracy Benchmark

Efficient comparison of key approaches:
1. Our best prefilled-json model (Phi-3.5 GPTQ 4-bit)
2. Simple prompting 
3. VLLM's JSON mode
4. VLLM's constrained generation

Focus on accuracy, speed, and reliability metrics.
"""

import time
import json
import statistics
import sys
import os
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_prefilled_json_approach():
    """Test our best prefilled-json model (Phi-3.5 GPTQ 4-bit)."""
    
    print("🔧 Testing Prefilled-JSON (Phi-3.5 GPTQ 4-bit)")
    print("=" * 60)
    
    try:
        from vllm import LLM, SamplingParams
        from driver.stop_token_json_driver import StopTokenJsonDriver
        
        print("Loading Phi-3.5 GPTQ 4-bit model...")
        llm = LLM(
            model="thesven/Phi-3.5-mini-instruct-GPTQ-4bit",
            max_model_len=512,
            gpu_memory_utilization=0.4,
            enable_prefix_caching=True,
            disable_sliding_window=True,
            trust_remote_code=True,
            dtype="float16"
        )
        
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
        
        config = {"stop_tokens": [",", "}", "\n", "<|end|>"], "stop_reliable": True}
        driver = StopTokenJsonDriver(generate_func, config)
        
        # Test scenarios
        test_scenarios = [
            ([{"name": "string"}, {"age": "number"}], "simple_fields"),
            ([{"id": "number"}, {"name": "string"}, {"email": "string"}, {"active": "string"}], "multiple_fields"),
            ([{"first_name": "string"}, {"last_name": "string"}, {"phone_number": "string"}], "complex_naming"),
            ([{"user": {"name": "string", "id": "number"}}, {"timestamp": "string"}], "nested_object")
        ]
        
        results = {}
        
        for fields, scenario_name in test_scenarios:
            print(f"  Testing {scenario_name}...")
            
            times = []
            valid_count = 0
            field_accuracy = []
            
            for run in range(3):
                start = time.time()
                try:
                    result = driver.generate_json(fields)
                    elapsed = time.time() - start
                    times.append(elapsed)
                    
                    # Validate JSON
                    parsed = json.loads(result)
                    valid_count += 1
                    
                    # Calculate field accuracy
                    expected_fields = set()
                    for field in fields:
                        if isinstance(field, dict):
                            expected_fields.update(field.keys())
                    
                    actual_fields = set(parsed.keys())
                    accuracy = len(expected_fields & actual_fields) / len(expected_fields)
                    field_accuracy.append(accuracy)
                    
                    print(f"    Run {run+1}: {elapsed:.3f}s ✅ {result}")
                    
                except Exception as e:
                    elapsed = time.time() - start
                    times.append(elapsed)
                    field_accuracy.append(0.0)
                    print(f"    Run {run+1}: {elapsed:.3f}s ❌ {str(e)[:40]}...")
            
            results[scenario_name] = {
                "avg_time": statistics.mean(times),
                "validity_rate": valid_count / 3,
                "avg_accuracy": statistics.mean(field_accuracy)
            }
            
            print(f"    Summary: {valid_count}/3 valid, {statistics.mean(field_accuracy):.1%} accurate")
        
        return results, llm
        
    except Exception as e:
        print(f"❌ Prefilled-JSON test failed: {e}")
        return {}, None


def test_simple_prompting_with_llm(llm):
    """Test simple prompting with the same model."""
    
    print(f"\n📝 Testing Simple Prompting (Same Model)")
    print("=" * 60)
    
    try:
        from vllm import SamplingParams
        
        test_prompts = [
            ("Generate JSON with name and age fields:", "simple_fields"),
            ("Generate JSON with id, name, email, and active fields:", "multiple_fields"), 
            ("Generate JSON with first_name, last_name, and phone_number:", "complex_naming"),
            ("Generate JSON with user object containing name and id, plus timestamp:", "nested_object")
        ]
        
        params = SamplingParams(
            temperature=0.5,
            max_tokens=80,
            skip_special_tokens=True
        )
        
        results = {}
        
        for prompt, scenario_name in test_prompts:
            print(f"  Testing {scenario_name}...")
            
            times = []
            valid_count = 0
            over_generation_count = 0
            
            for run in range(3):
                start = time.time()
                outputs = llm.generate([prompt], params)
                elapsed = time.time() - start
                result = outputs[0].outputs[0].text.strip()
                times.append(elapsed)
                
                print(f"    Run {run+1}: {elapsed:.3f}s → {result[:50]}...")
                
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
                            
                            # Check for over-generation
                            remaining = result[json_start + json_end:].strip()
                            if remaining:
                                over_generation_count += 1
                                print(f"      ⚠️  Over-generation: {len(remaining)} extra chars")
                        
                except json.JSONDecodeError:
                    print(f"      ❌ Invalid JSON")
            
            results[scenario_name] = {
                "avg_time": statistics.mean(times),
                "validity_rate": valid_count / 3,
                "over_generation_rate": over_generation_count / 3
            }
            
            print(f"    Summary: {valid_count}/3 valid, {over_generation_count}/3 over-generated")
        
        return results
        
    except Exception as e:
        print(f"❌ Simple prompting test failed: {e}")
        return {}


def test_vllm_json_mode_with_llm(llm):
    """Test VLLM's JSON mode with the same model."""
    
    print(f"\n🎯 Testing VLLM JSON Mode (Guided Generation)")
    print("=" * 60)
    
    try:
        from vllm import SamplingParams
        from vllm.sampling_params import GuidedDecodingParams
        
        # Define JSON schemas
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
            print(f"  Testing {scenario_name}...")
            
            try:
                guided_decoding_params = GuidedDecodingParams(json=schema)
                params = SamplingParams(
                    temperature=0.5,
                    max_tokens=50,
                    guided_decoding=guided_decoding_params,
                    skip_special_tokens=True
                )
                
                prompt = "Generate a JSON object:"
                
                times = []
                valid_count = 0
                schema_compliance = []
                
                for run in range(3):
                    start = time.time()
                    outputs = llm.generate([prompt], params)
                    elapsed = time.time() - start
                    result = outputs[0].outputs[0].text.strip()
                    times.append(elapsed)
                    
                    print(f"    Run {run+1}: {elapsed:.3f}s → {result}")
                    
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
                    "schema_compliance": statistics.mean(schema_compliance)
                }
                
                print(f"    Summary: {valid_count}/3 valid, {statistics.mean(schema_compliance):.1%} compliant")
                
            except Exception as e:
                print(f"    ❌ {scenario_name}: {e}")
                results[scenario_name] = {
                    "avg_time": 0,
                    "validity_rate": 0,
                    "schema_compliance": 0
                }
        
        return results
        
    except Exception as e:
        print(f"❌ VLLM JSON mode test failed: {e}")
        return {}


def test_constrained_generation_with_llm(llm):
    """Test constrained generation with regex."""
    
    print(f"\n🔒 Testing Constrained Generation (Regex)")
    print("=" * 60)
    
    try:
        from vllm import SamplingParams
        from vllm.sampling_params import GuidedDecodingParams
        import re
        
        # Simple regex patterns for JSON
        patterns = {
            "simple_fields": r'\{"name":\s*"[^"]*",\s*"age":\s*\d+\}',
            "multiple_fields": r'\{"id":\s*\d+,\s*"name":\s*"[^"]*",\s*"email":\s*"[^"]*",\s*"active":\s*"[^"]*"\}'
        }
        
        results = {}
        
        for scenario_name, pattern in patterns.items():
            print(f"  Testing {scenario_name}...")
            
            try:
                guided_decoding_params = GuidedDecodingParams(regex=pattern)
                params = SamplingParams(
                    temperature=0.5,
                    max_tokens=50,
                    guided_decoding=guided_decoding_params,
                    skip_special_tokens=True
                )
                
                prompt = "Generate JSON:"
                
                times = []
                valid_count = 0
                pattern_match = []
                
                for run in range(3):
                    start = time.time()
                    outputs = llm.generate([prompt], params)
                    elapsed = time.time() - start
                    result = outputs[0].outputs[0].text.strip()
                    times.append(elapsed)
                    
                    print(f"    Run {run+1}: {elapsed:.3f}s → {result}")
                    
                    try:
                        parsed = json.loads(result)
                        valid_count += 1
                        
                        # Check pattern match
                        matches = bool(re.match(pattern, result))
                        pattern_match.append(matches)
                        
                    except json.JSONDecodeError:
                        pattern_match.append(False)
                
                results[scenario_name] = {
                    "avg_time": statistics.mean(times),
                    "validity_rate": valid_count / 3,
                    "pattern_compliance": sum(pattern_match) / len(pattern_match)
                }
                
                print(f"    Summary: {valid_count}/3 valid, {sum(pattern_match)}/3 pattern match")
                
            except Exception as e:
                print(f"    ❌ {scenario_name}: {e}")
                results[scenario_name] = {
                    "avg_time": 0,
                    "validity_rate": 0,
                    "pattern_compliance": 0
                }
        
        return results
        
    except Exception as e:
        print(f"❌ Constrained generation test failed: {e}")
        return {}


def analyze_and_compare_results(prefilled_results, simple_results, json_mode_results, constrained_results):
    """Analyze and compare all approaches."""
    
    print(f"\n📊 COMPREHENSIVE COMPARISON")
    print("=" * 80)
    
    approaches = {
        "Prefilled-JSON (Stop Tokens)": prefilled_results,
        "Simple Prompting": simple_results,
        "VLLM JSON Mode": json_mode_results,
        "Constrained Generation": constrained_results
    }
    
    # Calculate overall metrics for each approach
    approach_summaries = {}
    
    for approach_name, results in approaches.items():
        if not results:
            continue
            
        all_validity = []
        all_times = []
        all_accuracy = []
        
        for scenario_data in results.values():
            validity = scenario_data.get("validity_rate", 0)
            avg_time = scenario_data.get("avg_time", 0)
            accuracy = scenario_data.get("avg_accuracy", scenario_data.get("schema_compliance", scenario_data.get("pattern_compliance", 0)))
            
            all_validity.append(validity)
            if avg_time > 0:
                all_times.append(avg_time)
            all_accuracy.append(accuracy)
        
        if all_validity:
            approach_summaries[approach_name] = {
                "overall_validity": statistics.mean(all_validity),
                "overall_time": statistics.mean(all_times) if all_times else 0,
                "overall_accuracy": statistics.mean(all_accuracy),
                "scenario_count": len(all_validity)
            }
    
    # Display results
    print(f"\n📋 OVERALL PERFORMANCE BY APPROACH:")
    
    for approach_name, summary in approach_summaries.items():
        validity = summary["overall_validity"]
        avg_time = summary["overall_time"]
        accuracy = summary["overall_accuracy"]
        count = summary["scenario_count"]
        
        print(f"\n✅ {approach_name}:")
        print(f"   Validity Rate: {validity:.1%}")
        print(f"   Average Time: {avg_time:.3f}s")
        print(f"   Accuracy/Compliance: {accuracy:.1%}")
        print(f"   Scenarios Tested: {count}")
    
    # Ranking
    print(f"\n🏆 RANKING BY OVERALL PERFORMANCE:")
    
    rankings = []
    for approach_name, summary in approach_summaries.items():
        validity = summary["overall_validity"]
        time = summary["overall_time"]
        accuracy = summary["overall_accuracy"]
        
        # Combined score: validity + accuracy - time_penalty
        time_penalty = min(time * 0.1, 0.2)  # Cap time penalty
        score = (validity + accuracy) / 2 - time_penalty
        
        rankings.append((approach_name, validity, time, accuracy, score))
    
    rankings.sort(key=lambda x: x[4], reverse=True)
    
    for i, (approach, validity, time, accuracy, score) in enumerate(rankings):
        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else "  "
        print(f"  {medal} {approach}")
        print(f"      Score: {score:.3f} (Validity: {validity:.1%}, Time: {time:.3f}s, Accuracy: {accuracy:.1%})")
    
    # Key insights
    print(f"\n💡 KEY INSIGHTS:")
    
    if rankings:
        best_approach, best_validity, best_time, best_accuracy, best_score = rankings[0]
        
        print(f"  🎯 Best Overall: {best_approach}")
        
        if best_validity >= 0.95:
            print(f"  🎉 Excellent reliability: {best_validity:.1%} JSON validity")
        elif best_validity >= 0.8:
            print(f"  ✅ Good reliability: {best_validity:.1%} JSON validity")
        else:
            print(f"  ⚠️  Reliability concerns: {best_validity:.1%} JSON validity")
        
        # Check for over-generation in simple prompting
        if "Simple Prompting" in simple_results:
            over_gen_rates = [data.get("over_generation_rate", 0) for data in simple_results.values()]
            avg_over_gen = statistics.mean(over_gen_rates)
            if avg_over_gen > 0.3:
                print(f"  ⚠️  Simple prompting has {avg_over_gen:.1%} over-generation rate")
        
        # Compare prefilled-json performance
        prefilled_rank = next((i for i, (name, _, _, _, _) in enumerate(rankings) if "Prefilled-JSON" in name), None)
        if prefilled_rank == 0:
            print(f"  🏆 Our Prefilled-JSON approach is the BEST!")
        elif prefilled_rank is not None and prefilled_rank <= 1:
            print(f"  🎯 Our Prefilled-JSON approach is highly competitive (rank {prefilled_rank + 1})")
    
    # Final recommendation
    print(f"\n🎯 RECOMMENDATION:")
    if rankings and rankings[0][1] >= 0.9:
        print(f"  ✅ Use {rankings[0][0]} for production JSON generation")
        print(f"  📈 Reliability: {rankings[0][1]:.1%}, Speed: {rankings[0][2]:.3f}s")
    else:
        print(f"  ⚠️  All approaches need improvement for high-reliability production use")
        print(f"  💡 Consider hybrid approaches or additional validation")


def main():
    print("🔬 Focused JSON Generation Accuracy Benchmark")
    print("Comparing approaches using same model for fairness")
    print("=" * 70)
    
    # Test 1: Our prefilled-json approach
    prefilled_results, llm = test_prefilled_json_approach()
    
    if not llm:
        print("❌ Could not load model, aborting benchmark")
        return
    
    # Test 2-4: Other approaches using the same model
    simple_results = test_simple_prompting_with_llm(llm)
    json_mode_results = test_vllm_json_mode_with_llm(llm) 
    constrained_results = test_constrained_generation_with_llm(llm)
    
    # Analyze all results
    analyze_and_compare_results(prefilled_results, simple_results, json_mode_results, constrained_results)
    
    print(f"\n🎉 Benchmark Complete!")
    print(f"📊 Results show relative performance of all JSON generation approaches")


if __name__ == "__main__":
    main()