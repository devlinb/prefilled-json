#!/usr/bin/env python3
"""
Benchmark for testing JSON generation accuracy with small LLMs.

This benchmark simulates a customer support scenario where an LLM needs to 
extract structured information from a conversation and format it as JSON.
"""

import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

try:
    from vllm import LLM, SamplingParams
    from prefilled_json.vllm_integration import generate_with_json_prefilled
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: VLLM not available. Install with 'pip install vllm' to run benchmarks.")

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    method: str
    success: bool
    valid_json: bool
    correct_fields: int
    total_fields: int
    execution_time: float
    generated_text: str
    error: Optional[str] = None

# Define the target JSON schema for the customer support ticket
TARGET_SCHEMA = {
    "customer_id": "string",
    "issue_category": "string", 
    "priority_level": "string",
    "product_name": "string",
    "issue_description": "string",
    "customer_sentiment": "string",
    "estimated_resolution_time": "number",
    "requires_escalation": "boolean",
    "contact_method": "string",
    "previous_tickets": "number"
}

# The conversation context that provides information to fill the JSON
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

# Expected correct values for validation
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
    """
    Validate generated JSON against expected values.
    
    Returns:
        (is_valid_json, correct_fields, total_fields)
    """
    print(f"Validating output: {output[:200]}...")
    
    # Try to extract JSON from the output
    json_text = output.strip()
    
    # Look for JSON-like content in the output
    start_idx = json_text.find('{')
    end_idx = json_text.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_text = json_text[start_idx:end_idx+1]
    
    try:
        parsed = json.loads(json_text)
        print(f"Successfully parsed JSON: {parsed}")
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return False, 0, len(expected)
    
    correct_fields = 0
    total_fields = len(expected)
    
    for key, expected_value in expected.items():
        if key in parsed:
            actual_value = parsed[key]
            print(f"Checking {key}: expected='{expected_value}', actual='{actual_value}'")
            
            if isinstance(expected_value, bool):
                # Accept boolean values or string representations
                if isinstance(actual_value, bool) and actual_value == expected_value:
                    correct_fields += 1
                    print(f"  ✓ Boolean match")
                elif isinstance(actual_value, str) and actual_value.lower() in ['true', 'false']:
                    if (actual_value.lower() == 'true') == expected_value:
                        correct_fields += 1
                        print(f"  ✓ Boolean string match")
            elif isinstance(expected_value, (int, float)):
                if isinstance(actual_value, (int, float)) and abs(actual_value - expected_value) <= 1:
                    correct_fields += 1
                    print(f"  ✓ Number match")
                elif isinstance(actual_value, str):
                    try:
                        num_val = float(actual_value)
                        if abs(num_val - expected_value) <= 1:
                            correct_fields += 1
                            print(f"  ✓ Number string match")
                    except ValueError:
                        pass
            elif isinstance(expected_value, str):
                if isinstance(actual_value, str):
                    # More flexible string matching for content extraction
                    expected_lower = expected_value.lower()
                    actual_lower = actual_value.lower()
                    
                    # Direct match or substring match
                    if (expected_lower == actual_lower or 
                        expected_lower in actual_lower or 
                        actual_lower in expected_lower or
                        # Handle variations like "crash" vs "crashing"
                        any(word in actual_lower for word in expected_lower.split()) or
                        any(word in expected_lower for word in actual_lower.split())):
                        correct_fields += 1
                        print(f"  ✓ String match")
        else:
            print(f"Missing field: {key}")
    
    print(f"Total score: {correct_fields}/{total_fields}")
    return True, correct_fields, total_fields

def run_baseline_generation(llm: Any, prompt: str) -> BenchmarkResult:
    """Run baseline JSON generation without prefilled assistance."""
    start_time = time.time()
    
    try:
        sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=512,
            stop=["\n\n", "---"]
        )
        
        outputs = llm.generate([prompt], sampling_params)
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

def run_prefilled_generation(llm: Any, prompt: str) -> BenchmarkResult:
    """Run JSON generation with prefilled assistance."""
    start_time = time.time()
    
    try:
        # Convert schema to field specifications for prefilled JSON
        json_fields = [
            {"customer_id": "string"},
            {"issue_category": "string"},
            {"priority_level": "string"}, 
            {"product_name": "string"},
            {"issue_description": "string"},
            {"customer_sentiment": "string"},
            {"estimated_resolution_time": "number"},
            {"requires_escalation": "string"},  # Note: using string for boolean compatibility
            {"contact_method": "string"},
            {"previous_tickets": "number"}
        ]
        
        outputs = generate_with_json_prefilled(
            engine=llm,
            prompts=[prompt],
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

def print_results(results: List[BenchmarkResult]):
    """Print benchmark results in a formatted table."""
    print("\n" + "="*80)
    print("CUSTOMER SUPPORT JSON EXTRACTION BENCHMARK RESULTS")
    print("="*80)
    
    for result in results:
        print(f"\nMethod: {result.method.upper()}")
        print(f"Success: {result.success}")
        print(f"Valid JSON: {result.valid_json}")
        print(f"Correct Fields: {result.correct_fields}/{result.total_fields} ({result.correct_fields/result.total_fields*100:.1f}%)")
        print(f"Execution Time: {result.execution_time:.2f}s")
        
        if result.error:
            print(f"Error: {result.error}")
        
        print(f"Generated Output:")
        print("-" * 40)
        print(result.generated_text)
        print("-" * 40)

def main():
    """Run the customer support JSON extraction benchmark."""
    if not VLLM_AVAILABLE:
        print("VLLM is required to run benchmarks. Install with: pip install vllm")
        return
    
    # Initialize small model for testing
    print("Initializing model...")
    
    # Try TinyLlama first as it's better for instruction following
    try:
        llm = LLM(
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            max_model_len=1024,
            gpu_memory_utilization=0.7,
            enforce_eager=True,
            trust_remote_code=True
        )
        print("Successfully loaded TinyLlama-1.1B-Chat-v1.0")
    except Exception as e:
        print(f"Failed to load TinyLlama: {e}")
        print("Trying Phi-3.5-mini...")
        try:
            llm = LLM(
                model="microsoft/Phi-3.5-mini-instruct",
                max_model_len=1024,
                gpu_memory_utilization=0.7,
                enforce_eager=True,
                trust_remote_code=True
            )
            print("Successfully loaded Phi-3.5-mini-instruct")
        except Exception as e2:
            print(f"Failed to load Phi-3.5: {e2}")
            print("Trying GPT-2...")
            try:
                llm = LLM(
                    model="gpt2",
                    max_model_len=512,
                    gpu_memory_utilization=0.7,
                    enforce_eager=True
                )
                print("Successfully loaded GPT-2")
            except Exception as e3:
                print(f"All models failed: {e3}")
                return
    
    print("Running benchmarks...")
    
    results = []
    
    # Run baseline generation
    print("Running baseline generation...")
    baseline_result = run_baseline_generation(llm, CONVERSATION_CONTEXT)
    results.append(baseline_result)
    
    # Run prefilled generation
    print("Running prefilled generation...")
    prefilled_result = run_prefilled_generation(llm, CONVERSATION_CONTEXT)
    results.append(prefilled_result)
    
    # Print results
    print_results(results)
    
    # Summary comparison
    if len(results) == 2:
        baseline, prefilled = results
        print(f"\nSUMMARY COMPARISON:")
        print(f"Baseline accuracy: {baseline.correct_fields}/{baseline.total_fields} ({baseline.correct_fields/baseline.total_fields*100:.1f}%)")
        print(f"Prefilled accuracy: {prefilled.correct_fields}/{prefilled.total_fields} ({prefilled.correct_fields/prefilled.total_fields*100:.1f}%)")
        
        if prefilled.correct_fields > baseline.correct_fields:
            improvement = prefilled.correct_fields - baseline.correct_fields
            print(f"✅ Prefilled JSON improved accuracy by {improvement} fields!")
        elif prefilled.correct_fields == baseline.correct_fields:
            print("📊 Both methods achieved the same accuracy")
        else:
            print("❌ Baseline performed better than prefilled")

if __name__ == "__main__":
    main()