#!/usr/bin/env python3
"""
Integration test script for prefilled-json library.

This script tests both the core driver and the VLLM plugin (if available).
"""

import sys
from typing import Optional


def test_core_driver():
    """Test the core JsonFieldDriver functionality."""
    print("=== Testing Core Driver ===")
    
    from driver.json_driver import JsonFieldDriver
    
    # Mock generate function for testing
    call_count = 0
    def mock_generate(prompt: str, stop_token: Optional[str] = None) -> str:
        nonlocal call_count
        call_count += 1
        
        if call_count == 1:  # name field
            return '"Alice"'
        elif call_count == 2:  # age field
            return '30'
        else:
            return '""'
    
    driver = JsonFieldDriver(generate=mock_generate)
    
    # Test basic fields
    fields = [{"name": "string"}, {"age": "number"}]
    result = driver.generate_json(fields)
    
    expected = '{"name": "Alice", "age": 30}'
    assert result == expected, f"Expected {expected}, got {result}"
    
    print("✓ Basic field generation works")
    
    # Test nested objects
    call_count = 0
    def mock_generate_nested(prompt: str, stop_token: Optional[str] = None) -> str:
        nonlocal call_count
        call_count += 1
        
        if call_count == 1:  # name
            return '"Bob"'
        elif call_count == 2:  # address.street
            return '"123 Main St"'
        elif call_count == 3:  # address.city
            return '"Seattle"'
        elif call_count == 4:  # age
            return '25'
        else:
            return '""'
    
    driver_nested = JsonFieldDriver(generate=mock_generate_nested)
    
    nested_fields = [
        {"name": "string"},
        {"address": {
            "street": "string",
            "city": "string"
        }},
        {"age": "number"}
    ]
    
    result = driver_nested.generate_json(nested_fields)
    expected = '{"name": "Bob", "address": {"street": "123 Main St", "city": "Seattle"}, "age": 25}'
    assert result == expected, f"Expected {expected}, got {result}"
    
    print("✓ Nested object generation works")
    print("Core driver tests passed!\n")


def test_vllm_plugin():
    """Test the VLLM plugin if available."""
    print("=== Testing VLLM Plugin ===")
    
    try:
        from vllm_plugin.json_prefilled_plugin import (
            VLLMJSONPrefilledPlugin, 
            ModelCompatibilityError,
            generate_with_json_prefilled
        )
        print("✓ VLLM plugin imports successfully")
    except ImportError as e:
        print(f"✗ VLLM plugin import failed: {e}")
        print("Install with: pip install -e '.[vllm]'")
        return False
    
    # Test model compatibility checking
    class MockEngine:
        def __init__(self, model_name):
            self.model_config = type('obj', (object,), {'model': model_name})
    
    # Test compatible model
    try:
        engine = MockEngine("meta-llama/Llama-2-7b-hf")
        plugin = VLLMJSONPrefilledPlugin(engine)
        print("✓ Compatible model accepted")
    except ModelCompatibilityError:
        print("✗ Compatible model rejected")
        return False
    
    # Test incompatible model
    try:
        engine_chat = MockEngine("meta-llama/Llama-2-7b-chat-hf")
        plugin_chat = VLLMJSONPrefilledPlugin(engine_chat)
        print("✗ Incompatible model accepted (should fail)")
        return False
    except ModelCompatibilityError:
        print("✓ Incompatible model correctly rejected")
    
    # Test session management
    plugin = VLLMJSONPrefilledPlugin(MockEngine("meta-llama/Llama-2-7b-hf"))
    
    session_id = plugin.create_session("test prompt", [{"name": "string"}])
    assert session_id in plugin.active_sessions
    print("✓ Session creation works")
    
    plugin.cleanup_session(session_id)
    assert session_id not in plugin.active_sessions
    print("✓ Session cleanup works")
    
    print("VLLM plugin tests passed!\n")
    return True


def run_example_without_vllm():
    """Run an example that doesn't require actual VLLM installation."""
    print("=== Example Without VLLM ===")
    
    from driver.json_driver import JsonFieldDriver
    
    # Simulate a simple LLM that generates predictable responses
    responses = [
        '"John Doe"',           # name
        '28',                   # age  
        '"Software Engineer"',  # job title
        '"TechCorp"',          # company
        '"john@example.com"'    # email
    ]
    response_iter = iter(responses)
    
    def simple_llm_generate(prompt: str, stop_token: Optional[str] = None) -> str:
        """Simple mock LLM that returns predictable responses."""
        try:
            return next(response_iter)
        except StopIteration:
            return '""'
    
    # Create driver
    driver = JsonFieldDriver(generate=simple_llm_generate)
    
    # Define employee profile fields
    fields = [
        {"name": "string"},
        {"age": "number"},
        {"job": {
            "title": "string", 
            "company": "string"
        }},
        {"email": "string"}
    ]
    
    # Generate JSON
    result = driver.generate_json(fields)
    
    print("Generated JSON:")
    print(result)
    
    # Verify it's valid JSON
    import json
    try:
        parsed = json.loads(result)
        print("✓ Generated valid JSON")
        print(f"✓ Contains {len(parsed)} top-level fields")
        print(f"✓ Job field is nested object with {len(parsed['job'])} fields")
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON generated: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("Prefilled JSON Integration Tests")
    print("=" * 40)
    
    success = True
    
    try:
        test_core_driver()
    except Exception as e:
        print(f"✗ Core driver tests failed: {e}")
        success = False
    
    try:
        vllm_available = test_vllm_plugin()
    except Exception as e:
        print(f"✗ VLLM plugin tests failed: {e}")
        vllm_available = False
        # Don't mark as failure since VLLM might not be installed
    
    try:
        run_example_without_vllm()
    except Exception as e:
        print(f"✗ Example failed: {e}")
        success = False
    
    print("=" * 40)
    if success:
        print("✓ All available tests passed!")
        if vllm_available:
            print("✓ VLLM plugin is available and working")
        else:
            print("ⓘ VLLM plugin not available (install with: pip install -e '.[vllm]')")
    else:
        print("✗ Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()