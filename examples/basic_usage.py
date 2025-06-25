#!/usr/bin/env python3
"""
Basic usage examples for prefilled-json package.
"""

from prefilled_json import JsonFieldDriver, StreamingJsonFieldDriver


def mock_llm_generate(prompt: str, stop_token: str = None) -> str:
    """Mock LLM function for demonstration."""
    print(f"LLM called with: {prompt!r}, stop_token: {stop_token}")
    
    # Mock responses based on the last field being requested
    # Look for the field that ends with ': ' (the one being generated)
    if prompt.endswith('"name": '):
        return '"Alice"'
    elif prompt.endswith('"age": '):
        return '25'
    elif prompt.endswith('"city": '):
        return '"Seattle"'
    elif prompt.endswith('"street": '):
        return '"123 Main St"'
    elif prompt.endswith('"zip": '):
        return '98101'
    else:
        return '"default"'


def basic_usage_example():
    """Demonstrate basic usage with traditional stop token approach."""
    print("=== Basic Usage (Traditional Approach) ===")
    
    # Create driver with your LLM function
    driver = JsonFieldDriver(generate=mock_llm_generate)
    
    # Define fields to generate
    fields = [
        {"name": "string"},
        {"age": "number"},
        {"city": "string"}
    ]
    
    # Generate JSON
    result = driver.generate_json(fields)
    print(f"Result: {result}")
    print()


def streaming_usage_example():
    """Demonstrate streaming approach with pattern matching."""
    print("=== Streaming Usage (Pattern Matching Approach) ===")
    
    # Mock function that over-generates (like modern models)
    def over_generating_llm(prompt: str, stop_token: str = None) -> str:
        print(f"LLM called with: {prompt!r}, stop_token: {stop_token}")
        
        if prompt.endswith('"name": '):
            return '"Alice", "age": 30, "city": "Seattle", "email": "alice@example.com"'
        elif prompt.endswith('"age": '):
            return '25, "city": "Seattle", "active": true, "score": 100'
        elif prompt.endswith('"city": '):
            return '"Seattle", "state": "WA", "country": "USA"'
        else:
            return '"default", "extra": "data"'
    
    # Create streaming driver
    driver = StreamingJsonFieldDriver(generate=over_generating_llm)
    
    # Define same fields
    fields = [
        {"name": "string"},
        {"age": "number"},
        {"city": "string"}
    ]
    
    # Generate JSON - driver extracts only what's needed
    result = driver.generate_json(fields)
    print(f"Result: {result}")
    print()


def nested_json_example():
    """Demonstrate nested JSON generation."""
    print("=== Nested JSON Example ===")
    
    def nested_mock_llm(prompt: str, stop_token: str = None) -> str:
        print(f"LLM called with: {prompt!r}")
        
        # Handle nested structure responses
        if prompt.endswith('"name": '):
            return '"John Doe"'
        elif prompt.endswith('"street": '):
            return '"123 Main St"'
        elif prompt.endswith('"zip": '):
            return '98101'
        elif prompt.endswith('"age": '):
            return '30'
        else:
            return '"default"'
    
    driver = JsonFieldDriver(generate=nested_mock_llm)
    
    # Define nested structure
    fields = [
        {"name": "string"},
        {"address": {
            "street": "string",
            "zip": "number"
        }},
        {"age": "number"}
    ]
    
    result = driver.generate_json(fields)
    print(f"Result: {result}")
    print()


if __name__ == "__main__":
    basic_usage_example()
    streaming_usage_example()
    nested_json_example()
    
    print("✅ Basic usage examples completed!")
    print("\nFor VLLM integration examples, see vllm_example.py")