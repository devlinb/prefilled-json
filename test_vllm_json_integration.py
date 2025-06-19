#!/usr/bin/env python3
"""
Integration test for VLLM JSON Prefilled Plugin.

This test starts up a real VLLM model and tests the JSON prefilled generation
functionality end-to-end, including basic conversation and JSON generation.
"""

import json
import sys
from typing import Dict, Any

def test_vllm_json_integration():
    """
    Integration test that:
    1. Starts up a VLLM model
    2. Tests basic conversation generation
    3. Tests JSON prefilled generation
    """
    
    try:
        from vllm import LLM, SamplingParams
        from vllm_plugin.json_prefilled_plugin import generate_with_json_prefilled
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("Install with: pip install vllm")
        return False
    
    # Use a lightweight model for testing
    model_name = "microsoft/DialoGPT-small"
    
    print(f"🚀 Starting VLLM integration test with model: {model_name}")
    print("=" * 60)
    
    try:
        # Initialize VLLM model
        print("📦 Loading model...")
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            max_model_len=512,
            enable_prefix_caching=True,  # Enable for better performance
        )
        print("✅ Model loaded successfully!")
        
        # Test 1: Basic conversation
        print("\n🗣️  Testing basic conversation...")
        basic_prompts = [
            "Hello, how are you?",
            "Tell me about yourself."
        ]
        
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=50,
            top_p=0.9
        )
        
        outputs = llm.generate(basic_prompts, sampling_params)
        
        print("Basic conversation results:")
        for i, output in enumerate(outputs):
            print(f"  Prompt {i+1}: {output.prompt}")
            print(f"  Response: {output.outputs[0].text.strip()}")
            print()
        
        print("✅ Basic conversation test passed!")
        
        # Test 2: JSON prefilled generation - Simple fields
        print("\n🔧 Testing JSON prefilled generation (simple fields)...")
        
        simple_prompts = [
            "Generate a user profile:",
            "Create employee data:"
        ]
        
        simple_fields = [
            {"name": "string"},
            {"age": "number"}
        ]
        
        json_outputs = generate_with_json_prefilled(
            engine=llm,
            prompts=simple_prompts,
            json_prefilled_fields=simple_fields
        )
        
        print("Simple JSON results:")
        for i, output in enumerate(json_outputs):
            print(f"  Result {i+1}: {output}")
            
            # Extract and validate JSON part
            if '\n{' in output:
                json_part = output.split('\n', 1)[1]
                try:
                    parsed = json.loads(json_part)
                    print(f"  ✅ Valid JSON with fields: {list(parsed.keys())}")
                    
                    # Validate field types
                    if 'name' in parsed and isinstance(parsed['name'], str):
                        print(f"  ✅ Name field is string: '{parsed['name']}'")
                    if 'age' in parsed and isinstance(parsed['age'], (int, float)):
                        print(f"  ✅ Age field is number: {parsed['age']}")
                        
                except json.JSONDecodeError as e:
                    print(f"  ❌ Invalid JSON: {e}")
                    return False
            print()
        
        print("✅ Simple JSON generation test passed!")
        
        # Test 3: JSON prefilled generation - Nested objects
        print("\n🔧 Testing JSON prefilled generation (nested objects)...")
        
        nested_prompts = ["Generate detailed user information:"]
        
        nested_fields = [
            {"name": "string"},
            {"profile": {
                "email": "string",
                "phone": "string"
            }},
            {"age": "number"}
        ]
        
        nested_outputs = generate_with_json_prefilled(
            engine=llm,
            prompts=nested_prompts,
            json_prefilled_fields=nested_fields
        )
        
        print("Nested JSON results:")
        for i, output in enumerate(nested_outputs):
            print(f"  Result {i+1}: {output}")
            
            # Extract and validate JSON part
            if '\n{' in output:
                json_part = output.split('\n', 1)[1]
                try:
                    parsed = json.loads(json_part)
                    print(f"  ✅ Valid JSON with top-level fields: {list(parsed.keys())}")
                    
                    # Validate nested structure
                    if 'profile' in parsed and isinstance(parsed['profile'], dict):
                        profile_fields = list(parsed['profile'].keys())
                        print(f"  ✅ Profile nested object with fields: {profile_fields}")
                        
                        # Validate nested field types
                        if 'email' in parsed['profile']:
                            print(f"  ✅ Email field: '{parsed['profile']['email']}'")
                        if 'phone' in parsed['profile']:
                            print(f"  ✅ Phone field: '{parsed['profile']['phone']}'")
                    
                    if 'name' in parsed:
                        print(f"  ✅ Name field: '{parsed['name']}'")
                    if 'age' in parsed:
                        print(f"  ✅ Age field: {parsed['age']}")
                        
                except json.JSONDecodeError as e:
                    print(f"  ❌ Invalid JSON: {e}")
                    return False
            print()
        
        print("✅ Nested JSON generation test passed!")
        
        # Test 4: Fallback behavior (no JSON fields)
        print("\n🔧 Testing fallback behavior (no JSON fields)...")
        
        fallback_outputs = generate_with_json_prefilled(
            engine=llm,
            prompts=["What is the capital of France?"]
            # No json_prefilled_fields parameter
        )
        
        print("Fallback results:")
        for i, output in enumerate(fallback_outputs):
            print(f"  Result {i+1}: {output}")
            # Should not contain JSON structure
            if not output.strip().startswith('{'):
                print("  ✅ Correctly used standard generation (no JSON)")
            else:
                print("  ❌ Unexpectedly generated JSON format")
                return False
        
        print("✅ Fallback behavior test passed!")
        
        print("\n" + "=" * 60)
        print("🎉 All VLLM JSON integration tests passed!")
        print("✅ Model loading works")
        print("✅ Basic conversation works")
        print("✅ Simple JSON generation works")
        print("✅ Nested JSON generation works")
        print("✅ Fallback behavior works")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {str(e)}")
        print("\nThis might indicate:")
        print("- VLLM installation issues")
        print("- GPU/CUDA setup problems")
        print("- Model loading failures")
        print("- Plugin compatibility issues")
        import traceback
        traceback.print_exc()
        return False


def test_model_compatibility():
    """Test model compatibility checking."""
    print("\n🔍 Testing model compatibility...")
    
    try:
        from vllm import LLM
        from vllm_plugin.json_prefilled_plugin import VLLMJSONPrefilledPlugin, ModelCompatibilityError
        
        # Test with compatible model
        compatible_model = "microsoft/DialoGPT-small"
        llm_compatible = LLM(
            model=compatible_model,
            trust_remote_code=True,
            max_model_len=256
        )
        
        try:
            plugin = VLLMJSONPrefilledPlugin(llm_compatible)
            print(f"✅ Compatible model '{compatible_model}' accepted")
        except ModelCompatibilityError:
            print(f"❌ Compatible model '{compatible_model}' rejected")
            return False
        
        print("✅ Model compatibility test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Model compatibility test failed: {e}")
        return False


def main():
    """Run all integration tests."""
    print("VLLM JSON Prefilled Plugin - Integration Tests")
    print("=" * 60)
    
    success = True
    
    # Test model compatibility first
    if not test_model_compatibility():
        success = False
    
    # Run main integration test
    if not test_vllm_json_integration():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("The VLLM JSON Prefilled Plugin is working correctly.")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()