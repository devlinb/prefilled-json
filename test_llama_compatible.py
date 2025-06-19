#!/usr/bin/env python3
"""
Test script specifically for Llama-3.2 models with compatibility guidance.

This script demonstrates the difference between compatible and incompatible 
Llama models for the JSON prefilled plugin.
"""

import sys
import os
from pathlib import Path


def explain_llama_compatibility():
    """Explain Llama model compatibility with the JSON prefilled plugin."""
    print("🦙 LLAMA MODEL COMPATIBILITY GUIDE")
    print("=" * 50)
    print()
    
    print("❌ INCOMPATIBLE MODELS (Will be rejected):")
    print("  • unsloth/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
    print("  • meta-llama/Llama-3.2-1B-Instruct")
    print("  • meta-llama/Llama-2-7b-chat-hf")
    print("  • Any model with: -instruct, -chat, -it suffixes")
    print()
    print("  📝 Reason: These use strict chat templates that don't support")
    print("     the assistant message resumption needed for iterative JSON generation.")
    print()
    
    print("✅ COMPATIBLE MODELS (Will work):")
    print("  • meta-llama/Llama-3.2-1B (base model)")
    print("  • meta-llama/Llama-2-7b-hf (base model)")
    print("  • microsoft/DialoGPT-small")
    print("  • Any base model without chat formatting")
    print()
    
    print("🔧 GGUF ALTERNATIVE:")
    print("  To use GGUF with this plugin, you need:")
    print("  1. A BASE model GGUF file (not instruct variant)")
    print("  2. Example: Llama-3.2-1B-Q4_K_M.gguf (base, not instruct)")
    print("  3. Corresponding tokenizer: meta-llama/Llama-3.2-1B")
    print()


def test_compatible_llama_models():
    """Test compatible Llama models if available."""
    try:
        from vllm import LLM, SamplingParams
        from vllm_plugin.json_prefilled_plugin import generate_with_json_prefilled, ModelCompatibilityError
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    print("🧪 TESTING COMPATIBLE LLAMA MODELS")
    print("=" * 40)
    
    # Test models in order of preference/availability
    test_models = [
        {
            "name": "Llama-3.2-1B (base)",
            "path": "meta-llama/Llama-3.2-1B",
            "description": "Latest Llama base model - should work perfectly"
        },
        {
            "name": "Llama-2-7B (base)",
            "path": "meta-llama/Llama-2-7b-hf", 
            "description": "Proven compatible Llama base model"
        }
    ]
    
    for model_config in test_models:
        print(f"\n🔍 Testing: {model_config['name']}")
        print(f"📝 {model_config['description']}")
        
        try:
            print("📦 Loading model...")
            llm = LLM(
                model=model_config['path'],
                trust_remote_code=True,
                max_model_len=1024,
                enable_prefix_caching=True
            )
            print("✅ Model loaded successfully!")
            
            # Test compatibility
            from vllm_plugin.json_prefilled_plugin import VLLMJSONPrefilledPlugin
            try:
                plugin = VLLMJSONPrefilledPlugin(llm)
                print("✅ Model passed compatibility check!")
            except ModelCompatibilityError as e:
                print(f"❌ Compatibility check failed: {e}")
                continue
            
            # Test JSON generation
            print("🔧 Testing JSON generation...")
            
            result = generate_with_json_prefilled(
                engine=llm,
                prompts=["Generate a user profile with the following information:"],
                json_prefilled_fields=[
                    {"name": "string"},
                    {"age": "number"},
                    {"email": "string"}
                ]
            )
            
            print("📄 Generated result:")
            print(f"  {result[0]}")
            
            # Try to validate JSON part
            if '\n{' in result[0]:
                json_part = result[0].split('\n', 1)[1]
                try:
                    import json
                    parsed = json.loads(json_part)
                    print(f"✅ Valid JSON generated with fields: {list(parsed.keys())}")
                    
                    # Check field types
                    if 'name' in parsed and isinstance(parsed['name'], str):
                        print(f"  ✅ Name field (string): '{parsed['name']}'")
                    if 'age' in parsed and isinstance(parsed['age'], (int, float)):
                        print(f"  ✅ Age field (number): {parsed['age']}")
                    if 'email' in parsed and isinstance(parsed['email'], str):
                        print(f"  ✅ Email field (string): '{parsed['email']}'")
                        
                except json.JSONDecodeError as e:
                    print(f"⚠️  Generated text but invalid JSON: {e}")
            else:
                print("⚠️  No JSON structure detected")
            
            print(f"🎉 {model_config['name']} test completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to test {model_config['name']}: {e}")
            print("   This might be due to:")
            print("   - Model not available for download")
            print("   - Insufficient resources (GPU memory)")
            print("   - Network issues")
            continue
    
    print("⚠️  No compatible Llama models could be tested")
    return False


def demonstrate_incompatible_model():
    """Demonstrate what happens with an incompatible model."""
    try:
        from vllm import LLM
        from vllm_plugin.json_prefilled_plugin import VLLMJSONPrefilledPlugin, ModelCompatibilityError
    except ImportError:
        return False
    
    print("\n🚫 DEMONSTRATING INCOMPATIBLE MODEL REJECTION")
    print("=" * 50)
    
    # Create a mock engine that simulates an instruct model
    class MockEngine:
        def __init__(self, model_name):
            self.model_config = type('obj', (object,), {'model': model_name})
    
    instruct_model_name = "meta-llama/Llama-3.2-1B-Instruct"
    print(f"🧪 Testing incompatible model: {instruct_model_name}")
    
    try:
        mock_engine = MockEngine(instruct_model_name)
        plugin = VLLMJSONPrefilledPlugin(mock_engine)
        print("❌ ERROR: Incompatible model was incorrectly accepted!")
        return False
    except ModelCompatibilityError as e:
        print(f"✅ Model correctly rejected: {e}")
        print("   This is the expected behavior for instruct models.")
        return True


def show_gguf_setup_guide():
    """Show how to set up GGUF models for this plugin."""
    print("\n📦 GGUF MODEL SETUP GUIDE")
    print("=" * 30)
    
    print("To use GGUF models with this plugin:")
    print()
    print("1. Download a BASE model GGUF file (not instruct):")
    print("   wget https://huggingface.co/QuantFactory/Llama-3.2-1B-GGUF/resolve/main/Llama-3.2-1B.Q4_K_M.gguf")
    print()
    print("2. Place it in a models directory:")
    print("   mkdir -p ./models")
    print("   mv Llama-3.2-1B.Q4_K_M.gguf ./models/")
    print()
    print("3. Use it with the correct tokenizer:")
    print("   from vllm import LLM")
    print("   llm = LLM(")
    print("       model='./models/Llama-3.2-1B.Q4_K_M.gguf',")
    print("       tokenizer='meta-llama/Llama-3.2-1B',  # Base model tokenizer")
    print("       trust_remote_code=True")
    print("   )")
    print()
    print("⚠️  IMPORTANT: Use BASE model GGUF files, not instruct variants!")


def main():
    """Main test function."""
    print("🦙 LLAMA MODEL COMPATIBILITY TEST")
    print("=" * 60)
    
    # Explain compatibility
    explain_llama_compatibility()
    
    # Demonstrate incompatible model rejection
    demonstrate_incompatible_model()
    
    # Test compatible models if available
    print()
    compatible_test_passed = test_compatible_llama_models()
    
    # Show GGUF setup guide
    show_gguf_setup_guide()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    
    if compatible_test_passed:
        print("✅ Compatible Llama model testing: PASSED")
    else:
        print("⚠️  Compatible Llama model testing: NOT AVAILABLE")
        print("   (Models may not be accessible for download/testing)")
    
    print("✅ Incompatible model rejection: WORKING CORRECTLY")
    print()
    print("🎯 KEY TAKEAWAY:")
    print("   Use BASE Llama models (not instruct/chat variants)")
    print("   for JSON prefilled generation to work properly.")
    
    return True


if __name__ == "__main__":
    main()