#!/usr/bin/env python3
"""
Prefilled-JSON vs Baseline Benchmark

This benchmark compares two approaches to JSON generation:
1. Baseline (standard generation with schema prompt)
2. Prefilled-JSON library (iterative field generation)
"""

import json
import os
import sys
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Add parent directory to path to find vllm_plugin
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError as e:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None

try:
    from vllm_plugin.json_prefilled_plugin import generate_with_json_prefilled
    PREFILLED_AVAILABLE = True
except ImportError as e:
    PREFILLED_AVAILABLE = False
    generate_with_json_prefilled = None

@dataclass
class BenchmarkResult:
    """Results from a generation method."""
    method: str
    success: bool
    valid_json: bool
    correct_fields: int
    total_fields: int
    execution_time: float
    generated_text: str
    error: Optional[str] = None

# Conversation context
CONVERSATION_CONTEXT = """Customer: I'm CS-789456. CloudSync Pro keeps crashing when syncing files. This is my third call about this! Very frustrated.

Agent: I see your previous 2 tickets. This is high priority - escalating to senior team. They resolve similar issues in 6 hours.

Customer: Perfect, call me on my phone please.

Extract the information and generate a JSON object with these exact fields:
{
  "customer_id": "string - customer ID from conversation",
  "issue_category": "string - type of technical issue", 
  "priority_level": "string - urgency level",
  "product_name": "string - affected product name",
  "issue_description": "string - brief description of the problem",
  "customer_sentiment": "string - customer's emotional state",
  "estimated_resolution_time": "number - hours until resolution",
  "requires_escalation": "boolean - whether escalation is needed",
  "contact_method": "string - preferred contact method",
  "previous_tickets": "number - count of prior tickets for this issue"
}

JSON:"""

# Expected values for validation
EXPECTED_VALUES = {
    "customer_id": "CS-789456",
    "issue_category": "crash",
    "priority_level": "high", 
    "product_name": "CloudSync Pro",
    "issue_description": "crashing when syncing",
    "customer_sentiment": "frustrated",
    "estimated_resolution_time": 6,
    "requires_escalation": True,
    "contact_method": "phone", 
    "previous_tickets": 2
}


def validate_json_output(output: str, expected: Dict[str, Any]) -> tuple[bool, int, int]:
    """Validate generated JSON against expected values."""
    print(f"Validating output: {output[:100]}...")
    
    # Try to extract JSON from the output
    json_text = output.strip()
    
    # Look for JSON-like content in the output
    start_idx = json_text.find('{')
    end_idx = json_text.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_text = json_text[start_idx:end_idx+1]
    
    try:
        parsed = json.loads(json_text)
        print(f"Successfully parsed JSON with {len(parsed)} fields")
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return False, 0, len(expected)
    
    correct_fields = 0
    total_fields = len(expected)
    
    for key, expected_value in expected.items():
        if key in parsed:
            actual_value = parsed[key]
            
            if isinstance(expected_value, bool):
                if isinstance(actual_value, bool) and actual_value == expected_value:
                    correct_fields += 1
                elif isinstance(actual_value, str) and actual_value.lower() in ['true', 'false']:
                    if (actual_value.lower() == 'true') == expected_value:
                        correct_fields += 1
            elif isinstance(expected_value, (int, float)):
                if isinstance(actual_value, (int, float)) and abs(actual_value - expected_value) <= 1:
                    correct_fields += 1
                elif isinstance(actual_value, str):
                    try:
                        num_val = float(actual_value)
                        if abs(num_val - expected_value) <= 1:
                            correct_fields += 1
                    except ValueError:
                        pass
            elif isinstance(expected_value, str):
                if isinstance(actual_value, str):
                    expected_lower = expected_value.lower()
                    actual_lower = actual_value.lower()
                    
                    if (expected_lower == actual_lower or 
                        expected_lower in actual_lower or 
                        actual_lower in expected_lower or
                        any(word in actual_lower for word in expected_lower.split()) or
                        any(word in expected_lower for word in actual_lower.split())):
                        correct_fields += 1
    
    print(f"Scored {correct_fields}/{total_fields} fields correctly")
    return True, correct_fields, total_fields

class PrefilledJsonBenchmark:
    """Benchmark for comparing baseline vs prefilled-JSON generation approaches."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        self.model_name = model_name
        self.llm = None
        
    def initialize_model(self):
        """Initialize the VLLM model."""
        print(f"Initializing {self.model_name}...")
        self.llm = LLM(
            model=self.model_name,
            max_model_len=512,
            gpu_memory_utilization=0.6,
            enable_prefix_caching=True,
            trust_remote_code=True
        )
        print("Model initialized successfully.")
    
    def run_baseline_generation(self) -> BenchmarkResult:
        """Run baseline JSON generation without assistance."""
        print("\n=== BASELINE: Standard Generation ===")
        start_time = time.time()
        
        try:
            sampling_params = SamplingParams(
                temperature=0.1,
                max_tokens=200,
                stop=["\n\n", "---"]
            )
            
            outputs = self.llm.generate([CONVERSATION_CONTEXT], sampling_params)
            generated_text = outputs[0].outputs[0].text.strip()
            
            execution_time = time.time() - start_time
            is_valid, correct_fields, total_fields = validate_json_output(generated_text, EXPECTED_VALUES)
            
            return BenchmarkResult(
                method="baseline",
                success=True,
                valid_json=is_valid,
                correct_fields=correct_fields,
                total_fields=total_fields,
                execution_time=execution_time,
                generated_text=generated_text
            )
            
        except Exception as e:
            return BenchmarkResult(
                method="baseline",
                success=False,
                valid_json=False,
                correct_fields=0,
                total_fields=len(EXPECTED_VALUES),
                execution_time=time.time() - start_time,
                generated_text="",
                error=str(e)
            )
    
    def run_prefilled_generation(self) -> BenchmarkResult:
        """Run prefilled-json library generation."""
        print("\n=== PREFILLED: Iterative Field Generation ===")
        start_time = time.time()
        
        try:
            json_fields = [
                {"customer_id": "string"},
                {"issue_category": "string"},
                {"priority_level": "string"}, 
                {"product_name": "string"},
                {"issue_description": "string"},
                {"customer_sentiment": "string"},
                {"estimated_resolution_time": "number"},
                {"requires_escalation": "string"},
                {"contact_method": "string"},
                {"previous_tickets": "number"}
            ]
            
            outputs = generate_with_json_prefilled(
                engine=self.llm,
                prompts=[CONVERSATION_CONTEXT],
                json_prefilled_fields=json_fields
            )
            
            generated_text = outputs[0].strip()
            execution_time = time.time() - start_time
            
            is_valid, correct_fields, total_fields = validate_json_output(generated_text, EXPECTED_VALUES)
            
            return BenchmarkResult(
                method="prefilled",
                success=True,
                valid_json=is_valid,
                correct_fields=correct_fields,
                total_fields=total_fields,
                execution_time=execution_time,
                generated_text=generated_text
            )
            
        except Exception as e:
            return BenchmarkResult(
                method="prefilled",
                success=False,
                valid_json=False,
                correct_fields=0,
                total_fields=len(EXPECTED_VALUES),
                execution_time=time.time() - start_time,
                generated_text="",
                error=str(e)
            )
    
    
    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all benchmark methods and return results."""
        if not VLLM_AVAILABLE:
            print("VLLM not available. Cannot run benchmarks.")
            return []
        
        if not PREFILLED_AVAILABLE:
            print("Prefilled-JSON plugin not available. Cannot run benchmarks.")
            return []
        
        print("VLLM and Prefilled-JSON are available! Starting benchmark...")
        self.initialize_model()
        results = []
        
        # Run baseline vs prefilled comparison
        results.append(self.run_baseline_generation())
        results.append(self.run_prefilled_generation())
        
        return results
    
    def print_results(self, results: List[BenchmarkResult]):
        """Print comprehensive benchmark results."""
        print("\n" + "="*80)
        print("PREFILLED-JSON BENCHMARK RESULTS")
        print("="*80)
        
        # Results table
        print(f"\n{'Method':<15} {'Valid JSON':<10} {'Accuracy':<10} {'Time (s)':<10} {'Success':<8}")
        print("-" * 65)
        
        for result in results:
            accuracy = f"{result.correct_fields}/{result.total_fields} ({result.correct_fields/result.total_fields*100:.1f}%)"
            print(f"{result.method:<15} {str(result.valid_json):<10} {accuracy:<10} {result.execution_time:<10.3f} {str(result.success):<8}")
        
        # Detailed outputs
        print("\nDETAILED OUTPUTS:")
        print("="*50)
        
        for result in results:
            print(f"\n{result.method.upper()}:")
            print("-" * 40)
            if result.error:
                print(f"ERROR: {result.error}")
            else:
                print(result.generated_text[:300] + ("..." if len(result.generated_text) > 300 else ""))
        
        # Analysis
        print("\nANALYSIS:")
        print("="*30)
        
        successful_results = [r for r in results if r.success and r.valid_json]
        if successful_results:
            best_accuracy = max(successful_results, key=lambda x: x.correct_fields)
            fastest = min(successful_results, key=lambda x: x.execution_time)
            
            print(f"🎯 Best accuracy: {best_accuracy.method} ({best_accuracy.correct_fields}/{best_accuracy.total_fields})")
            print(f"⚡ Fastest: {fastest.method} ({fastest.execution_time:.3f}s)")
            
            # Compare against baseline
            baseline = next((r for r in results if r.method == "baseline"), None)
            if baseline and baseline.success:
                print(f"\nComparison vs Baseline:")
                for result in results:
                    if result.method != "baseline" and result.success:
                        acc_diff = result.correct_fields - baseline.correct_fields
                        time_ratio = result.execution_time / baseline.execution_time
                        print(f"  {result.method}: {acc_diff:+d} fields, {time_ratio:.2f}x time")
        
        print(f"\nKEY INSIGHTS:")
        print("- Prefilled-JSON uses iterative field generation with stop tokens")
        print("- Baseline relies on model to generate complete JSON")
        print("- Compare accuracy vs speed tradeoffs")
        print("- Prefilled approach should be more reliable for small models")

def main():
    """Run the prefilled-JSON benchmark."""
    benchmark = PrefilledJsonBenchmark()
    results = benchmark.run_all_benchmarks()
    benchmark.print_results(results)

if __name__ == "__main__":
    main()