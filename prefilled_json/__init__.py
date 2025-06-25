"""
Prefilled JSON - Generate valid JSON with small LLMs

A Python library that helps low-parameter LLMs generate valid JSON by controlling
the generation process through iterative field-by-field completion.

Basic Usage:
    from prefilled_json import JsonFieldDriver
    
    driver = JsonFieldDriver(generate=your_llm_function)
    result = driver.generate_json([{"name": "string"}, {"age": "number"}])

VLLM Integration:
    from prefilled_json.vllm_integration import generate_with_json_prefilled
    from vllm import LLM
    
    llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")
    outputs = generate_with_json_prefilled(
        engine=llm,
        prompts=["Generate user data:"],
        json_prefilled_fields=[{"name": "string"}, {"age": "number"}]
    )
"""

# Core functionality - always available
from .driver import JsonFieldDriver, StreamingJsonFieldDriver
from .types import FieldType

__version__ = "0.1.0"
__all__ = ["JsonFieldDriver", "StreamingJsonFieldDriver", "FieldType"]

# VLLM integration - conditionally available  
try:
    import vllm
    from .vllm_integration import generate_with_json_prefilled, VLLMJSONPrefilledPlugin
    
    __all__.extend(["generate_with_json_prefilled", "VLLMJSONPrefilledPlugin"])
    
except ImportError:
    # VLLM not available - that's fine, core functionality still works
    pass