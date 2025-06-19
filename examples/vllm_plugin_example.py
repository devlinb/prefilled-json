"""
Example usage of the VLLM JSON Prefilled Plugin

This example demonstrates how to use the VLLM plugin to generate JSON
with small parameter models through iterative field completion.
"""

from vllm import LLM
from vllm_plugin import generate_with_json_prefilled


def main():
    """Main example function."""
    
    # Initialize VLLM with a compatible base model
    # Note: Use base models, not chat/instruct variants
    llm = LLM(
        model="meta-llama/Llama-2-7b-hf",  # Base model, not -chat
        enable_prefix_caching=True,        # Important for performance
        max_model_len=2048,
        gpu_memory_utilization=0.8
    )
    
    print("=== Basic JSON Generation ===")
    
    # Basic user profile generation
    basic_fields = [
        {"name": "string"},
        {"age": "number"},
        {"city": "string"}
    ]
    
    outputs = generate_with_json_prefilled(
        engine=llm,
        prompts=["Generate a user profile:"],
        json_prefilled_fields=basic_fields
    )
    
    print("Input:", "Generate a user profile:")
    print("Output:", outputs[0])
    print()
    
    print("=== Nested Object Generation ===")
    
    # Nested object example
    nested_fields = [
        {"name": "string"},
        {"address": {
            "street": "string",
            "city": "string", 
            "zip": "number"
        }},
        {"age": "number"}
    ]
    
    outputs = generate_with_json_prefilled(
        engine=llm,
        prompts=["Generate detailed user information:"],
        json_prefilled_fields=nested_fields
    )
    
    print("Input:", "Generate detailed user information:")
    print("Output:", outputs[0])
    print()
    
    print("=== Multiple Prompts ===")
    
    # Generate multiple JSON objects with different prompts
    prompts = [
        "Create a product listing:",
        "Generate employee record:",
        "Make a book entry:"
    ]
    
    product_fields = [
        {"name": "string"},
        {"price": "number"},
        {"category": "string"}
    ]
    
    outputs = generate_with_json_prefilled(
        engine=llm,
        prompts=prompts,
        json_prefilled_fields=product_fields
    )
    
    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        print(f"Prompt {i+1}:", prompt)
        print("Output:", output)
        print()
    
    print("=== Complex Nested Structure ===")
    
    # Deeply nested example
    complex_fields = [
        {"user": {
            "profile": {
                "name": "string",
                "contact": {
                    "email": "string",
                    "phone": "string"
                }
            },
            "settings": {
                "theme": "string",
                "notifications": "number"
            }
        }},
        {"timestamp": "number"}
    ]
    
    outputs = generate_with_json_prefilled(
        engine=llm,
        prompts=["Generate complex user data structure:"],
        json_prefilled_fields=complex_fields
    )
    
    print("Input:", "Generate complex user data structure:")
    print("Output:", outputs[0])
    print()
    
    print("=== Fallback to Standard Generation ===")
    
    # When no JSON fields are specified, it works like normal VLLM
    outputs = generate_with_json_prefilled(
        engine=llm,
        prompts=["Write a short story about a robot:"]
        # No json_prefilled_fields parameter
    )
    
    print("Input:", "Write a short story about a robot:")
    print("Output:", outputs[0])


def demonstrate_error_handling():
    """Demonstrate error handling scenarios."""
    
    print("=== Error Handling Examples ===")
    
    try:
        # This will fail - chat model not supported
        llm_chat = LLM(model="meta-llama/Llama-2-7b-chat-hf")
        
        outputs = generate_with_json_prefilled(
            engine=llm_chat,
            prompts=["Generate data:"],
            json_prefilled_fields=[{"name": "string"}]
        )
        
    except Exception as e:
        print(f"Expected error with chat model: {e}")
    
    # Demonstrate graceful fallback
    print("\nWith graceful fallback, the function still returns results:")
    
    llm_chat = LLM(model="meta-llama/Llama-2-7b-chat-hf")
    
    # This will fallback to standard generation
    outputs = generate_with_json_prefilled(
        engine=llm_chat,
        prompts=["Generate some data:"],
        json_prefilled_fields=[{"name": "string"}]
    )
    
    print("Fallback output:", outputs[0])


if __name__ == "__main__":
    # Run main examples
    main()
    
    # Demonstrate error handling
    demonstrate_error_handling()