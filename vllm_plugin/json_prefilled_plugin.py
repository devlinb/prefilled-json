"""
VLLM Plugin for JSON Prefilled Generation

This plugin enables VLLM to generate JSON through iterative field-by-field completion,
specifically designed for small parameter models that struggle with complete JSON generation.
"""

import asyncio
import uuid
from typing import List, Dict, Any, Optional, Union, Callable
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from driver.json_driver import JsonFieldDriver, FieldType

logger = logging.getLogger(__name__)


@dataclass
class GenerationSession:
    """Tracks state for an ongoing JSON generation session."""
    session_id: str
    original_prompt: str
    fields: List[Dict[str, FieldType]]
    partial_json: str = ""
    current_field_index: int = 0
    completed: bool = False
    result: Optional[str] = None


class ModelCompatibilityError(Exception):
    """Raised when model is incompatible with JSON prefilled generation."""
    pass


class VLLMJSONPrefilledPlugin:
    """
    VLLM Plugin that enables JSON prefilled generation through iterative field completion.
    
    This plugin intercepts generation requests that include json_prefilled_fields parameter
    and orchestrates the iterative generation process using the JsonFieldDriver.
    """
    
    def __init__(self, vllm_engine):
        """
        Initialize the plugin with a VLLM engine.
        
        Args:
            vllm_engine: The VLLM LLM engine instance
        """
        self.engine = vllm_engine
        self.active_sessions: Dict[str, GenerationSession] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Validate model compatibility
        self._check_model_compatibility()
        
    def _check_model_compatibility(self):
        """
        Check if the current model supports assistant message resumption.
        
        Raises:
            ModelCompatibilityError: If model is incompatible
        """
        if hasattr(self.engine, 'model_config'):
            model_name = getattr(self.engine.model_config, 'model', '')
        else:
            # Fallback - try to get model name from engine
            model_name = getattr(self.engine, 'model_name', '')
            
        if not model_name:
            logger.warning("Could not determine model name for compatibility check")
            return
            
        # Models with strict chat templates that don't support resumption
        incompatible_patterns = [
            '-chat', '-instruct', '-it', 'chatglm', 'vicuna'
        ]
        
        model_lower = model_name.lower()
        for pattern in incompatible_patterns:
            if pattern in model_lower:
                raise ModelCompatibilityError(
                    f"Model '{model_name}' uses strict chat template and is incompatible "
                    f"with JSON prefilled generation. Use base models without chat formatting."
                )
                
        logger.info(f"Model '{model_name}' appears compatible with JSON prefilled generation")
    
    def create_session(self, prompt: str, fields: List[Dict[str, FieldType]]) -> str:
        """
        Create a new JSON generation session.
        
        Args:
            prompt: The original user prompt
            fields: Field specifications for JSON generation
            
        Returns:
            Session ID string
        """
        session_id = str(uuid.uuid4())
        session = GenerationSession(
            session_id=session_id,
            original_prompt=prompt,
            fields=fields,
            partial_json="",
            current_field_index=0
        )
        self.active_sessions[session_id] = session
        return session_id
    
    def cleanup_session(self, session_id: str):
        """Remove completed session from active sessions."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
    
    def generate_json_iteratively(self, prompt: str, fields: List[Dict[str, FieldType]]) -> str:
        """
        Generate JSON using iterative field-by-field completion.
        
        Args:
            prompt: Original user prompt
            fields: List of field specifications
            
        Returns:
            Complete JSON string
        """
        def vllm_generate_func(prompt_text: str, stop_token: Optional[str] = None) -> str:
            """Generate function that interfaces with VLLM engine."""
            # Prepare generation parameters
            stop_sequences = [stop_token] if stop_token else None
            
            # Use VLLM's generate method
            outputs = self.engine.generate([prompt_text], 
                                         sampling_params=self._get_sampling_params(stop_sequences))
            
            if not outputs or not outputs[0].outputs:
                raise RuntimeError("VLLM generation failed - no outputs returned")
                
            # Return the generated text
            return outputs[0].outputs[0].text
        
        # Create JsonFieldDriver with VLLM generate function
        driver = JsonFieldDriver(generate=vllm_generate_func)
        
        # Generate the JSON
        json_result = driver.generate_json(fields)
        
        # Return the complete response (original prompt + JSON)
        return f"{prompt}\n{json_result}"
    
    def _get_sampling_params(self, stop_sequences: Optional[List[str]] = None):
        """
        Get sampling parameters for VLLM generation.
        
        Args:
            stop_sequences: Optional stop sequences for generation
            
        Returns:
            SamplingParams object
        """
        try:
            from vllm import SamplingParams
            return SamplingParams(
                temperature=0.1,  # Low temperature for more deterministic JSON
                max_tokens=100,   # Reasonable limit for individual fields
                stop=stop_sequences,
                skip_special_tokens=True
            )
        except ImportError:
            # Fallback if SamplingParams not available
            logger.warning("Could not import SamplingParams, using default parameters")
            return None


def generate_with_json_prefilled(
    engine,
    prompts: Union[str, List[str]], 
    json_prefilled_fields: Optional[List[Dict[str, FieldType]]] = None,
    **kwargs
) -> List[str]:
    """
    Generate text with optional JSON prefilled mode.
    
    This function extends VLLM's generate method to support JSON prefilled generation
    when json_prefilled_fields parameter is provided.
    
    Args:
        engine: VLLM LLM engine
        prompts: Input prompts (string or list of strings)
        json_prefilled_fields: Optional field specifications for JSON generation
        **kwargs: Additional arguments passed to VLLM generate
        
    Returns:
        List of generated responses
        
    Example:
        >>> outputs = generate_with_json_prefilled(
        ...     engine=llm,
        ...     prompts=["Generate user data:"],
        ...     json_prefilled_fields=[{"name": "string"}, {"age": "number"}]
        ... )
        >>> print(outputs[0])
        Generate user data:
        {"name": "Alice", "age": 30}
    """
    # Convert single prompt to list
    if isinstance(prompts, str):
        prompts = [prompts]
    
    # If no JSON prefilled fields, use standard VLLM generation
    if not json_prefilled_fields:
        outputs = engine.generate(prompts, **kwargs)
        return [output.outputs[0].text for output in outputs]
    
    # Generate with JSON prefilled for each prompt
    results = []
    for prompt in prompts:
        try:
            # Create plugin instance (may fail with incompatible models)
            plugin = VLLMJSONPrefilledPlugin(engine)
            result = plugin.generate_json_iteratively(prompt, json_prefilled_fields)
            results.append(result)
        except Exception as e:
            logger.error(f"JSON prefilled generation failed for prompt '{prompt}': {e}")
            # Fallback to standard generation
            outputs = engine.generate([prompt], **kwargs)
            results.append(outputs[0].outputs[0].text)
    
    return results