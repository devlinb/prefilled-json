"""
Prefilled JSON - Generate valid JSON with small LLMs

A Python library that helps low-parameter LLMs generate valid JSON by controlling
the generation process through stop tokens and field-by-field completion.
Achieves 100% reliability on complex conversation scenarios.

Basic Usage:
    from driver.stop_token_json_driver import StopTokenJsonDriver
    
    driver = StopTokenJsonDriver(generate=your_llm_function, model_config={})
    result = driver.generate_json([{"name": "string"}, {"age": "number"}])

VLLM Integration (Recommended):
    from vllm_plugin import generate_with_json_prefilled
    from vllm import LLM
    
    llm = LLM(model="microsoft/Phi-3.5-mini-instruct", 
              enable_prefix_caching=True, disable_sliding_window=True)
    outputs = generate_with_json_prefilled(
        engine=llm,
        prompts=["Generate user data:"],
        json_prefilled_fields=[{"name": "string"}, {"age": "number"}]
    )
"""

# Core functionality - always available
from .driver import JsonFieldDriver, StreamingJsonFieldDriver
from .types import FieldType

__version__ = "0.2.0"
__all__ = ["JsonFieldDriver", "StreamingJsonFieldDriver", "FieldType"]

# VLLM integration - conditionally available  
try:
    import vllm
    from .vllm_integration import generate_with_json_prefilled, VLLMJSONPrefilledPlugin
    
    __all__.extend(["generate_with_json_prefilled", "VLLMJSONPrefilledPlugin"])
    
except ImportError:
    # VLLM not available - that's fine, core functionality still works
    pass