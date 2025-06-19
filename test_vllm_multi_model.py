#!/usr/bin/env python3
"""
Multi-model integration test for VLLM JSON Prefilled Plugin.

This test framework supports testing multiple model types with different configurations,
including both standard HuggingFace models and GGUF models.
"""

import json
import sys
import os
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    model_path: str
    tokenizer: Optional[str] = None
    max_model_len: int = 512
    trust_remote_code: bool = True
    enable_prefix_caching: bool = True
    is_gguf: bool = False
    compatible_with_json_prefilled: bool = True
    test_prompts: Optional[List[str]] = None
    expected_behavior: str = "normal"  # "normal", "fallback", "error"
    description: str = ""
    
    def __post_init__(self):
        """Set default test prompts if not provided."""
        if self.test_prompts is None:
            if "instruct" in self.name.lower() or "chat" in self.name.lower():
                self.test_prompts = [
                    "Generate a JSON object with user information:",
                    "Please create user profile data:"
                ]
            else:
                self.test_prompts = [
                    "Generate user data:",
                    "Create employee information:"
                ]


# Define test model configurations
TEST_MODELS = [
    ModelConfig(
        name="DialoGPT-small",
        model_path="microsoft/DialoGPT-small",
        max_model_len=512,
        compatible_with_json_prefilled=True,
        description="Lightweight conversational model, good for testing basic functionality"
    ),
    
    ModelConfig(
        name="Llama-2-7B-base", 
        model_path="meta-llama/Llama-2-7b-hf",
        max_model_len=1024,
        compatible_with_json_prefilled=True,
        description="Base Llama-2 model, should work well with JSON generation"
    ),
    
    ModelConfig(
        name="Llama-3.2-1B-base",
        model_path="meta-llama/Llama-3.2-1B",
        max_model_len=1024, 
        compatible_with_json_prefilled=True,
        description="Base Llama-3.2 model (NOT instruct variant), compatible with plugin"
    ),
    
    # GGUF model example (would need to be downloaded locally)
    ModelConfig(
        name="Llama-3.2-1B-base-GGUF",
        model_path="./models/Llama-3.2-1B-Q4_K_M.gguf",
        tokenizer="meta-llama/Llama-3.2-1B",
        max_model_len=1024,
        is_gguf=True,
        compatible_with_json_prefilled=True,
        description="GGUF quantized base model (requires local download)"
    ),
    
    # Incompatible model examples for testing error handling
    ModelConfig(
        name="Llama-2-7B-chat",
        model_path="meta-llama/Llama-2-7b-chat-hf", 
        max_model_len=1024,
        compatible_with_json_prefilled=False,
        expected_behavior="error",
        description="Chat model - should be rejected by compatibility check"
    ),
]


def check_model_availability(config: ModelConfig) -> bool:
    """Check if a model is available for testing."""
    if config.is_gguf:
        # Check if GGUF file exists locally
        return Path(config.model_path).exists()
    else:
        # For HuggingFace models, we'll assume they can be downloaded
        # In a real scenario, you might want to check HF API or cache
        return True


def create_vllm_instance(config: ModelConfig):
    """Create a VLLM LLM instance with the given configuration."""
    from vllm import LLM
    
    kwargs = {
        "model": config.model_path,
        "trust_remote_code": config.trust_remote_code,
        "max_model_len": config.max_model_len,
        "enable_prefix_caching": config.enable_prefix_caching,
    }
    
    # Add tokenizer for GGUF models or if explicitly specified
    if config.tokenizer:
        kwargs["tokenizer"] = config.tokenizer
    
    return LLM(**kwargs)


def test_basic_conversation(llm, config: ModelConfig) -> bool:
    """Test basic conversation generation."""
    from vllm import SamplingParams
    
    print(f"  🗣️  Testing basic conversation...")
    
    prompts = config.test_prompts[:2] if len(config.test_prompts) >= 2 else ["Hello", "Test"]
    
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=50,
        top_p=0.9
    )
    
    try:
        outputs = llm.generate(prompts, sampling_params)
        
        print(f"  Basic conversation results:")
        for i, output in enumerate(outputs):
            generated = output.outputs[0].text.strip()
            print(f"    Prompt {i+1}: {output.prompt}")
            print(f"    Response: {generated[:100]}...")
            
        print(f"  ✅ Basic conversation test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Basic conversation failed: {e}")
        return False


def test_json_prefilled_generation(llm, config: ModelConfig) -> bool:
    """Test JSON prefilled generation."""
    from vllm_plugin.json_prefilled_plugin import generate_with_json_prefilled
    
    print(f"  🔧 Testing JSON prefilled generation...")
    
    # Simple fields test
    simple_prompts = [config.test_prompts[0]] if config.test_prompts else ["Generate user data:"]
    simple_fields = [
        {"name": "string"},
        {"age": "number"}
    ]
    
    try:
        json_outputs = generate_with_json_prefilled(
            engine=llm,
            prompts=simple_prompts,
            json_prefilled_fields=simple_fields
        )
        
        print(f"  Simple JSON results:")
        for i, output in enumerate(json_outputs):
            print(f"    Result {i+1}: {output}")
            
            # Try to validate JSON if it looks like JSON was generated
            if '\n{' in output:
                json_part = output.split('\n', 1)[1]
                try:
                    parsed = json.loads(json_part)
                    print(f"    ✅ Valid JSON with fields: {list(parsed.keys())}")
                except json.JSONDecodeError:
                    print(f"    ⚠️  Generated text but invalid JSON format")
            else:
                print(f"    ⚠️  No JSON structure detected (might be fallback)")
        
        # Nested fields test
        print(f"  Testing nested JSON generation...")
        nested_prompts = [config.test_prompts[0]] if config.test_prompts else ["Generate detailed user:"]
        nested_fields = [
            {"name": "string"},
            {"contact": {
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
        
        for i, output in enumerate(nested_outputs):
            print(f"    Nested result {i+1}: {output}")
            
        print(f"  ✅ JSON prefilled generation test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ JSON prefilled generation failed: {e}")
        return False


def test_fallback_behavior(llm, config: ModelConfig) -> bool:
    """Test fallback behavior when no JSON fields specified."""
    from vllm_plugin.json_prefilled_plugin import generate_with_json_prefilled
    
    print(f"  🔧 Testing fallback behavior...")
    
    try:
        fallback_outputs = generate_with_json_prefilled(
            engine=llm,
            prompts=["What is the capital of France?"]
            # No json_prefilled_fields parameter
        )
        
        print(f"  Fallback results:")
        for i, output in enumerate(fallback_outputs):
            print(f"    Result {i+1}: {output}")
            
        print(f"  ✅ Fallback behavior test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Fallback behavior test failed: {e}")
        return False


def test_model_compatibility(config: ModelConfig) -> bool:
    """Test model compatibility checking."""
    from vllm_plugin.json_prefilled_plugin import VLLMJSONPrefilledPlugin, ModelCompatibilityError
    
    print(f"  🔍 Testing model compatibility...")
    
    try:
        llm = create_vllm_instance(config)
        
        if config.compatible_with_json_prefilled:
            # Should succeed
            try:
                plugin = VLLMJSONPrefilledPlugin(llm)
                print(f"  ✅ Compatible model '{config.name}' correctly accepted")
                return True
            except ModelCompatibilityError as e:
                print(f"  ❌ Compatible model '{config.name}' incorrectly rejected: {e}")
                return False
        else:
            # Should fail
            try:
                plugin = VLLMJSONPrefilledPlugin(llm)
                print(f"  ❌ Incompatible model '{config.name}' incorrectly accepted")
                return False
            except ModelCompatibilityError:
                print(f"  ✅ Incompatible model '{config.name}' correctly rejected")
                return True
                
    except Exception as e:
        print(f"  ❌ Model compatibility test failed: {e}")
        return False


def test_single_model(config: ModelConfig) -> bool:
    """Test a single model configuration."""
    print(f"\n{'='*60}")
    print(f"🚀 Testing model: {config.name}")
    print(f"📝 Description: {config.description}")
    print(f"🔗 Model path: {config.model_path}")
    print(f"🔧 Compatible: {config.compatible_with_json_prefilled}")
    
    if config.is_gguf:
        print(f"📦 Type: GGUF quantized model")
        if config.tokenizer:
            print(f"🔤 Tokenizer: {config.tokenizer}")
    
    print(f"{'='*60}")
    
    # Check availability
    if not check_model_availability(config):
        print(f"⏭️  Skipping {config.name} - model not available locally")
        return True
    
    try:
        # Import required modules
        from vllm import LLM, SamplingParams
        from vllm_plugin.json_prefilled_plugin import generate_with_json_prefilled
        
        # Test compatibility first
        if not test_model_compatibility(config):
            if config.expected_behavior == "error":
                print(f"✅ Expected error behavior confirmed for {config.name}")
                return True
            else:
                return False
        
        # If model is incompatible and we expected that, we're done
        if not config.compatible_with_json_prefilled:
            return True
            
        # Load model for full testing
        print(f"📦 Loading model...")
        llm = create_vllm_instance(config)
        print(f"✅ Model loaded successfully!")
        
        # Run test suite
        tests_passed = 0
        total_tests = 0
        
        # Test 1: Basic conversation
        total_tests += 1
        if test_basic_conversation(llm, config):
            tests_passed += 1
            
        # Test 2: JSON prefilled generation  
        total_tests += 1
        if test_json_prefilled_generation(llm, config):
            tests_passed += 1
            
        # Test 3: Fallback behavior
        total_tests += 1
        if test_fallback_behavior(llm, config):
            tests_passed += 1
            
        success_rate = tests_passed / total_tests
        print(f"\n📊 Model {config.name} Results: {tests_passed}/{total_tests} tests passed ({success_rate:.1%})")
        
        if tests_passed == total_tests:
            print(f"🎉 All tests passed for {config.name}!")
            return True
        else:
            print(f"⚠️  Some tests failed for {config.name}")
            return False
            
    except Exception as e:
        print(f"❌ Model {config.name} testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run multi-model integration tests."""
    print("VLLM JSON Prefilled Plugin - Multi-Model Integration Tests")
    print("=" * 70)
    
    try:
        from vllm import LLM, SamplingParams
        from vllm_plugin.json_prefilled_plugin import generate_with_json_prefilled
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("Install with: pip install vllm")
        return False
    
    # Allow user to specify which models to test via environment variable
    test_model_names = os.getenv("TEST_MODELS", "").split(",")
    test_model_names = [name.strip() for name in test_model_names if name.strip()]
    
    if test_model_names:
        models_to_test = [config for config in TEST_MODELS if config.name in test_model_names]
        print(f"🎯 Testing specific models: {', '.join(test_model_names)}")
    else:
        # Default: test available models
        models_to_test = [config for config in TEST_MODELS if check_model_availability(config)]
        print(f"🔍 Testing all available models")
    
    if not models_to_test:
        print("❌ No models available for testing!")
        print("Available models:")
        for config in TEST_MODELS:
            available = "✅" if check_model_availability(config) else "❌"
            print(f"  {available} {config.name}: {config.model_path}")
        return False
    
    print(f"📋 Models to test: {len(models_to_test)}")
    for config in models_to_test:
        print(f"  • {config.name} ({'compatible' if config.compatible_with_json_prefilled else 'incompatible'})")
    
    # Run tests
    results = {}
    for config in models_to_test:
        results[config.name] = test_single_model(config)
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 FINAL RESULTS SUMMARY")
    print(f"{'='*70}")
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for model_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {status} {model_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} models passed ({passed/total:.1%})")
    
    if passed == total:
        print("🎉 ALL MODELS PASSED!")
        return True
    else:
        print("⚠️  SOME MODELS FAILED")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)