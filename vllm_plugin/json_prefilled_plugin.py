"""
VLLM Plugin for JSON Prefilled Generation

This plugin enables VLLM to generate JSON through iterative field-by-field completion,
specifically designed for small parameter models that struggle with complete JSON generation.

Uses streaming approach with pattern matching to extract field values, eliminating 
reliance on stop tokens which modern instruction-tuned models often ignore.
"""

import asyncio
import uuid
import re
from typing import List, Dict, Any, Optional, Union, Callable
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from driver.json_driver import FieldType

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
        """
        self.engine = vllm_engine
        self.active_sessions: Dict[str, GenerationSession] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Validate model compatibility
        self._check_model_compatibility()
        
    def _check_model_compatibility(self):
        """
        Check if the current model supports assistant message resumption by testing
        actual technical capabilities rather than name patterns.
        
        Raises:
            ModelCompatibilityError: If model is incompatible
        """
        model_name = self._get_model_name()
        
        # Test 1: Check if we can access the tokenizer
        tokenizer = self._get_tokenizer()
        if not tokenizer:
            logger.warning(f"Could not access tokenizer for model '{model_name}', assuming compatible")
            return
            
        # Test 2: Check message resumption capability
        resumption_supported = self._test_message_resumption(tokenizer, model_name)
        
        if not resumption_supported:
            raise ModelCompatibilityError(
                f"Model '{model_name}' does not support assistant message resumption. "
                f"This is typically caused by strict chat templates that enforce turn-taking. "
                f"Try using a base model variant or a model with flexible chat templates."
            )
                
        logger.info(f"Model '{model_name}' supports message resumption - compatible with JSON prefilled generation")
    
    def _get_model_name(self) -> str:
        """Get the model name from the VLLM engine."""
        if hasattr(self.engine, 'model_config'):
            model_name = getattr(self.engine.model_config, 'model', '')
        else:
            # Fallback - try to get model name from engine
            model_name = getattr(self.engine, 'model_name', '')
            
        if not model_name:
            model_name = "unknown_model"
            
        return model_name
    
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
        
        Args:
            tokenizer: The model's tokenizer
            model_name: Name of the model for logging
            
        Returns:
            bool: True if resumption is supported, False otherwise
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
                    test_messages,
                    continue_final_message=True,
                    add_generation_prompt=False,
                    tokenize=False  # Return string, not tokens
                )
                
                # Test without continuation for comparison
                result_without_continuation = tokenizer.apply_chat_template(
                    test_messages,
                    continue_final_message=False,
                    add_generation_prompt=False,
                    tokenize=False
                )
                
                # If the results are different, the model supports continuation
                supports_continuation = result_with_continuation != result_without_continuation
                
                if supports_continuation:
                    logger.debug(f"Model '{model_name}' supports continue_final_message parameter")
                    return True
                else:
                    logger.debug(f"Model '{model_name}' ignores continue_final_message parameter")
                    
            except Exception as e:
                logger.debug(f"Model '{model_name}' failed continue_final_message test: {e}")
            
            # Test 3: Check if we can create a custom flexible template
            flexible_template_works = self._test_flexible_template(tokenizer, model_name)
            if flexible_template_works:
                return True
            
            # Test 4: Check if model has a rigid chat template by examining template content
            template_is_flexible = self._analyze_chat_template_flexibility(tokenizer, model_name)
            if template_is_flexible:
                return True
                
            # If all tests fail, model likely has rigid chat template
            logger.debug(f"Model '{model_name}' appears to have rigid chat template - incompatible")
            return False
            
        except Exception as e:
            logger.warning(f"Error testing message resumption for '{model_name}': {e}")
            # On error, assume compatible to avoid false negatives
            return True
    
    def _test_flexible_template(self, tokenizer, model_name: str) -> bool:
        """Test if we can apply a flexible custom template."""
        try:
            # Create a simple template that doesn't enforce role alternation
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
        
        Args:
            tokenizer: The model's tokenizer
            model_name: Name of the model for logging
            
        Returns:
            bool: True if template appears flexible, False if rigid
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


class StreamingJsonFieldDriver:
    """
    JSON Field Driver that uses streaming approach with pattern matching.
    
    This driver eliminates reliance on stop tokens by allowing models to generate
    more content than needed, then extracting precise field values using regex patterns.
    """
    
    def __init__(self, generate: Callable[[str, Optional[str]], str]):
        """
        Initialize the streaming driver.
        
        Args:
            generate: Function that takes a prompt and optional stop token (ignored),
                     and returns the generated value as a string.
        """
        self.generate = generate

    def generate_json(self, fields: List[Dict[str, FieldType]]) -> str:
        """
        Generate JSON by iteratively prompting an LLM to fill in each field value.
        Uses pattern matching to extract clean field values from over-generated content.

        Args:
            fields: A list of dictionaries, each with one key (field name)
                   and its type: 'string', 'number', or a nested object specification.
                   
        Returns:
            A valid JSON string.
        """
        return self._generate_object(fields)

    def _generate_object(self, fields: List[Dict[str, FieldType]]) -> str:
        """
        Generate a JSON object from field specifications using streaming approach.
        
        Args:
            fields: List of field specifications
            
        Returns:
            JSON object string
        """
        json_parts = ["{"]

        for i, field_spec in enumerate(fields):
            assert len(field_spec) == 1, "Each field specification must have exactly one field"
            field_name, field_type = next(iter(field_spec.items()))

            # Add field name
            current_json = "".join(json_parts)
            prompt = current_json + f'"{field_name}": '
            
            # Determine if this is the last field
            is_last_field = i == len(fields) - 1

            if isinstance(field_type, dict):
                # Handle nested object
                nested_fields = [{key: value} for key, value in field_type.items()]
                nested_json = self._generate_object(nested_fields)
                json_parts.append(f'"{field_name}": {nested_json}')
            else:
                # Handle primitive field (string/number) with streaming extraction
                # Generate with no stop token - we'll extract manually
                raw_output = self.generate(prompt, None)
                
                # Extract field value using pattern matching
                value = self._extract_field_value(raw_output, field_type, field_name)
                json_parts.append(f'"{field_name}": {value}')

            # Add comma if not the last field
            if not is_last_field:
                json_parts.append(", ")

        json_parts.append("}")
        return "".join(json_parts)
    
    def _extract_field_value(self, raw_output: str, field_type: str, field_name: str) -> str:
        """
        Extract the field value from raw model output using pattern matching.
        
        Args:
            raw_output: Raw model output that may contain over-generation
            field_type: "string" or "number"
            field_name: Name of the field (for error reporting)
            
        Returns:
            Properly formatted field value
            
        Raises:
            ValueError: If field value cannot be extracted or validated
        """
        if field_type == "string":
            return self._extract_string_value(raw_output)
        elif field_type == "number":
            return self._extract_number_value(raw_output, field_name)
        else:
            raise ValueError(f"Unsupported field type: {field_type}")
    
    def _extract_string_value(self, raw_output: str) -> str:
        """Extract a string value from raw output."""
        # Pattern 1: Quoted string followed by delimiter or end
        match = re.match(r'^"([^"]*)"(?:[,}\s]|$)', raw_output.strip())
        if match:
            return f'"{match.group(1)}"'
        
        # Pattern 2: Unquoted text before delimiter
        match = re.match(r'^([^,}]+)(?:[,}]|$)', raw_output.strip())
        if match:
            text = match.group(1).strip().strip('"')
            return f'"{text}"'
        
        # Pattern 3: First word/phrase as fallback
        words = raw_output.strip().split()
        if words:
            first_word = words[0].rstrip(',}').strip('"')
            return f'"{first_word}"'
        
        # Ultimate fallback
        return '"default"'
    
    def _extract_number_value(self, raw_output: str, field_name: str) -> str:
        """Extract a numeric value from raw output."""
        # Pattern 1: Number followed by delimiter or end
        match = re.match(r'^(\d+(?:\.\d+)?)(?:[,}\s]|$)', raw_output.strip())
        if match:
            number_str = match.group(1)
            # Validate it's a proper number
            try:
                float(number_str)
                return number_str
            except ValueError:
                pass
        
        # Pattern 2: Number anywhere in the beginning
        match = re.search(r'(\d+(?:\.\d+)?)', raw_output[:30])  # Check first 30 chars
        if match:
            number_str = match.group(1)
            try:
                float(number_str)
                return number_str
            except ValueError:
                pass
        
        # If no valid number found, raise an error
        raise ValueError(f"Generated value for field '{field_name}' is not a valid number: {raw_output.strip()[:50]}")


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