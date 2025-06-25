"""
VLLM integration for JSON prefilled generation.

This module provides seamless integration with VLLM using the streaming approach
for reliable JSON generation with modern instruction-tuned models.
"""

import asyncio
import uuid
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from .types import FieldType
from .driver import StreamingJsonFieldDriver

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
    and orchestrates the iterative generation process using the StreamingJsonFieldDriver.
    """
    
    def __init__(self, vllm_engine):
        """
        Initialize the plugin with a VLLM engine.
        
        Args:
            vllm_engine: The VLLM LLM engine instance
            
        Raises:
            ModelCompatibilityError: If the model is incompatible with JSON prefilled generation
        """
        self.engine = vllm_engine
        self.active_sessions: Dict[str, GenerationSession] = {}
        
        # Test model compatibility
        self._test_model_compatibility()
    
    def _test_model_compatibility(self):
        """
        Test if the current model is compatible with JSON prefilled generation.
        
        Tests technical capabilities rather than relying on naming patterns.
        """
        try:
            model_name = getattr(self.engine.model_config, 'model', 'unknown_model')
            tokenizer = self._get_tokenizer()
            
            if tokenizer is None:
                # If we can't access the tokenizer, assume compatible (base model)
                logger.info(f"Model '{model_name}' tokenizer not accessible - assuming compatible (base model)")
                return
            
            # Test message resumption capability
            is_compatible = self._test_message_resumption(tokenizer, model_name)
            
            if is_compatible:
                logger.info(f"Model '{model_name}' supports message resumption - compatible with JSON prefilled generation")
            else:
                raise ModelCompatibilityError(
                    f"Model '{model_name}' does not support assistant message resumption. "
                    f"This model uses a rigid chat template that enforces strict role alternation. "
                    f"Compatible models include: Qwen series, Phi-3 series, Gemma series, and most base models."
                )
                
        except Exception as e:
            if isinstance(e, ModelCompatibilityError):
                raise
            # For other errors, log but assume compatible
            logger.debug(f"Compatibility test error (assuming compatible): {e}")
    
    def _get_tokenizer(self):
        """Get the tokenizer from the VLLM engine."""
        try:
            # Try different ways to access the tokenizer
            if hasattr(self.engine, 'llm_engine') and hasattr(self.engine.llm_engine, 'tokenizer'):
                return self.engine.llm_engine.tokenizer
            elif hasattr(self.engine, 'tokenizer'):
                return self.engine.tokenizer
            elif hasattr(self.engine, 'llm_engine') and hasattr(self.engine.llm_engine, 'model_executor'):
                model_executor = self.engine.llm_engine.model_executor
                if hasattr(model_executor, 'tokenizer'):
                    return model_executor.tokenizer
            
            # Try to access through model config
            if hasattr(self.engine, 'model_config'):
                model_config = self.engine.model_config
                if hasattr(model_config, 'tokenizer'):
                    return model_config.tokenizer
                    
            return None
        except Exception as e:
            logger.debug(f"Error accessing tokenizer: {e}")
            return None
    
    def _test_message_resumption(self, tokenizer, model_name: str) -> bool:
        """
        Test if the model/tokenizer supports message resumption by checking
        technical capabilities rather than name patterns.
        """
        try:
            # Test 1: Check if tokenizer has apply_chat_template method
            if not hasattr(tokenizer, 'apply_chat_template'):
                logger.debug(f"Model '{model_name}' tokenizer lacks apply_chat_template, assuming compatible (base model)")
                return True
            
            # Test 2: Try to apply a chat template with message continuation
            test_messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"}
            ]
            
            # Test with continue_final_message=True to see if model supports resumption
            try:
                result_with_continuation = tokenizer.apply_chat_template(
                    test_messages, continue_final_message=True, tokenize=False
                )
                result_without_continuation = tokenizer.apply_chat_template(
                    test_messages, continue_final_message=False, tokenize=False
                )
                
                # If the results are different, the model supports resumption
                supports_continuation = result_with_continuation != result_without_continuation
                
                if supports_continuation:
                    logger.debug(f"Model '{model_name}' supports continue_final_message parameter")
                    
                    # Test 3: Test flexible template acceptance
                    if self._test_flexible_template(tokenizer, model_name):
                        return True
                    
                    # Test 4: Analyze chat template for rigidity
                    return self._analyze_chat_template_flexibility(tokenizer, model_name)
                else:
                    logger.debug(f"Model '{model_name}' does not support continue_final_message parameter")
                    return False
                    
            except Exception as e:
                logger.debug(f"Model '{model_name}' failed continue_final_message test: {e}")
                return False
                
        except Exception as e:
            logger.debug(f"Message resumption test failed for '{model_name}': {e}")
            return True  # Assume compatible on error
    
    def _test_flexible_template(self, tokenizer, model_name: str) -> bool:
        """Test if the model accepts flexible custom templates."""
        try:
            # Create a flexible template that allows assistant message continuation
            flexible_template = """{% for message in messages %}{{ message.role }}: {{ message.content }}
{% endfor %}"""
            
            test_messages = [
                {"role": "user", "content": "Test"},
                {"role": "assistant", "content": "Response"},
                {"role": "assistant", "content": "Continuation"}  # Two assistant messages in a row
            ]
            
            # Try to apply the flexible template
            original_template = getattr(tokenizer, 'chat_template', None)
            tokenizer.chat_template = flexible_template
            
            result = tokenizer.apply_chat_template(
                test_messages,
                add_generation_prompt=False,
                tokenize=False
            )
            
            # Restore original template
            if original_template:
                tokenizer.chat_template = original_template
            
            # If we got here without exception, flexible templates work
            logger.debug(f"Model '{model_name}' accepts flexible custom templates")
            return True
            
        except Exception as e:
            logger.debug(f"Model '{model_name}' rejects flexible templates: {e}")
            return False
    
    def _analyze_chat_template_flexibility(self, tokenizer, model_name: str) -> bool:
        """
        Analyze the chat template content to determine if it has strict role validation.
        """
        try:
            template = getattr(tokenizer, 'chat_template', '')
            if not template:
                logger.debug(f"Model '{model_name}' has no chat template - assuming compatible")
                return True
            
            # Look for patterns that indicate rigid role enforcement
            rigid_patterns = [
                'raise_exception',  # Explicit error raising
                'loop.index0 % 2',  # Role alternation checking
                'must alternate',   # Error messages about alternation
                'user/assistant/user',  # Specific role pattern requirements
            ]
            
            template_lower = template.lower()
            rigid_indicators = sum(1 for pattern in rigid_patterns if pattern in template_lower)
            
            if rigid_indicators >= 2:
                logger.debug(f"Model '{model_name}' has rigid chat template with {rigid_indicators} strict patterns")
                return False
            else:
                logger.debug(f"Model '{model_name}' has flexible chat template")
                return True
                
        except Exception as e:
            logger.debug(f"Error analyzing chat template for '{model_name}': {e}")
            return True  # Assume compatible on error
    
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
        Generate JSON using iterative field-by-field completion with streaming approach.
        
        Args:
            prompt: Original user prompt
            fields: List of field specifications
            
        Returns:
            Complete JSON string
        """
        def vllm_generate_func(prompt_text: str, stop_token: Optional[str] = None) -> str:
            """Generate function that interfaces with VLLM engine using streaming approach."""
            # Note: stop_token parameter is ignored - streaming approach uses pattern matching
            outputs = self.engine.generate([prompt_text], 
                                         sampling_params=self._get_sampling_params())
            
            if not outputs or not outputs[0].outputs:
                raise RuntimeError("VLLM generation failed - no outputs returned")
                
            # Return the generated text for pattern extraction
            return outputs[0].outputs[0].text
        
        # Create StreamingJsonFieldDriver with VLLM generate function
        driver = StreamingJsonFieldDriver(generate=vllm_generate_func)
        
        # Generate the JSON
        json_result = driver.generate_json(fields)
        
        # Return the complete response (original prompt + JSON)
        return f"{prompt}\n{json_result}"
    
    def _get_sampling_params(self):
        """
        Get sampling parameters optimized for streaming approach with pattern extraction.
        
        Returns:
            SamplingParams object configured for pattern extraction
        """
        try:
            from vllm import SamplingParams
            return SamplingParams(
                temperature=0.1,  # Low temperature for more deterministic JSON
                max_tokens=50,    # Allow more tokens for over-generation
                stop=None,        # No stop sequences - we extract manually
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