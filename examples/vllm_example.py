#!/usr/bin/env python3
"""
VLLM integration examples for prefilled-json package.

Run this with:
    pip install prefilled-json[vllm]
    python examples/vllm_example.py
"""

import sys


def vllm_basic_example():
    """Basic VLLM integration example."""
    print("=== VLLM Basic Example ===")
    
    try:
        from vllm import LLM
        from prefilled_json.vllm_integration import generate_with_json_prefilled
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("Install with: pip install prefilled-json[vllm]")
        return False
    
    try:
        # Load a compatible model (use a small one for demo)
        print("Loading model...")
        llm = LLM(
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            max_model_len=512,
            enforce_eager=True  # For compatibility
        )
        print("✅ Model loaded!")
        
        # Generate JSON with simple fields
        outputs = generate_with_json_prefilled(
            engine=llm,
            prompts=["Generate user profile:"],
            json_prefilled_fields=[
                {"name": "string"},
                {"age": "number"},
                {"city": "string"}
            ]
        )
        
        print(f"Result: {outputs[0]}")
        return True
        
    except Exception as e:
        print(f"❌ VLLM example failed: {e}")
        return False


def vllm_advanced_example():
    """Advanced VLLM example with nested JSON."""
    print("\n=== VLLM Advanced Example ===")
    
    try:
        from vllm import LLM
        from prefilled_json.vllm_integration import generate_with_json_prefilled
    except ImportError:
        print("❌ VLLM not available")
        return False
    
    try:
        # Use a model that's good for JSON generation
        llm = LLM(
            model="microsoft/DialoGPT-small",
            max_model_len=256,
            enforce_eager=True
        )
        
        # Generate more complex JSON
        outputs = generate_with_json_prefilled(
            engine=llm,
            prompts=[
                "Create employee record:",
                "Generate product data:"
            ],
            json_prefilled_fields=[
                {"employee": {
                    "name": "string",
                    "id": "number"
                }},
                {"department": "string"},
                {"active": "string"}
            ]
        )
        
        print("Results:")
        for i, output in enumerate(outputs):
            print(f"  {i+1}: {output}")
            
        return True
        
    except Exception as e:
        print(f"❌ Advanced example failed: {e}")
        return False


def compatibility_test_example():
    """Test model compatibility."""
    print("\n=== Model Compatibility Test ===")
    
    try:
        from vllm import LLM
        from prefilled_json.vllm_integration import VLLMJSONPrefilledPlugin
    except ImportError:
        print("❌ VLLM not available")
        return False
    
    def test_model_compatibility(model_name):
        """Test if a model is compatible."""
        try:
            print(f"Testing {model_name}...")
            llm = LLM(model=model_name, max_model_len=128, enforce_eager=True)
            plugin = VLLMJSONPrefilledPlugin(llm)
            print(f"✅ {model_name} is compatible!")
            return True
        except Exception as e:
            print(f"❌ {model_name}: {e}")
            return False
    
    # Test various models
    test_models = [
        "microsoft/DialoGPT-small",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    ]
    
    compatible_count = 0
    for model in test_models:
        if test_model_compatibility(model):
            compatible_count += 1
    
    print(f"\n📊 Results: {compatible_count}/{len(test_models)} models compatible")
    return compatible_count > 0


def alternative_import_example():
    """Show alternative import patterns."""
    print("\n=== Alternative Import Patterns ===")
    
    # Method 1: Direct import
    try:
        from prefilled_json.vllm_integration import generate_with_json_prefilled
        print("✅ Method 1: Direct import works")
    except ImportError:
        print("❌ Method 1: Direct import failed")
    
    # Method 2: Top-level import (when VLLM available)
    try:
        from prefilled_json import generate_with_json_prefilled
        print("✅ Method 2: Top-level import works")
    except ImportError:
        print("❌ Method 2: Top-level import failed")
    
    # Method 3: Check availability
    try:
        import prefilled_json
        if hasattr(prefilled_json, 'generate_with_json_prefilled'):
            print("✅ Method 3: VLLM functions available at package level")
        else:
            print("ℹ️  Method 3: VLLM functions not available at package level")
    except ImportError:
        print("❌ Method 3: Package import failed")


if __name__ == "__main__":
    print("🚀 VLLM Integration Examples")
    print("=" * 50)
    
    # Show import patterns first
    alternative_import_example()
    
    # Try basic example
    if vllm_basic_example():
        # Only run advanced examples if basic works
        vllm_advanced_example()
        compatibility_test_example()
    
    print("\n✅ VLLM examples completed!")
    print("\nNote: Examples use small models for demonstration.")
    print("For production, use larger models like Qwen/Qwen2.5-1.5B-Instruct")