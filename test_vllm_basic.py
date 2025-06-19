#!/usr/bin/env python3

from vllm import LLM, SamplingParams

def test_vllm_basic():
    """
    Basic test to verify VLLM setup with a small model.
    Uses microsoft/DialoGPT-small as it's lightweight for testing.
    """
    
    # Initialize the model (using a small model for testing)
    model_name = "microsoft/DialoGPT-small"
    
    try:
        # Create LLM instance
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            max_model_len=512,  # Keep it small for testing
        )
        
        # Set up sampling parameters
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=50
        )
        
        # Test prompts
        prompts = [
            "Hello, how are you?",
            "What is the weather like today?"
        ]
        
        print(f"Testing VLLM with model: {model_name}")
        print("=" * 50)
        
        # Generate responses
        outputs = llm.generate(prompts, sampling_params)
        
        # Print results
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt}")
            print(f"Generated: {generated_text}")
            print("-" * 30)
            
        print("✅ VLLM test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ VLLM test failed: {str(e)}")
        print("This might indicate:")
        print("- VLLM is not installed (pip install vllm)")
        print("- GPU/CUDA setup issues")
        print("- Model download/loading issues")
        return False

if __name__ == "__main__":
    test_vllm_basic()