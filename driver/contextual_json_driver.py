"""
Contextual JSON Driver

Implements JSON generation that maintains conversation context across field generations.
This approach preserves the full context while building JSON incrementally.
"""

import re
import json
from typing import Optional, Dict, List, Any, Callable, Union, Literal

# Import types from json_driver module
FieldType = Union[Literal["string", "number"], Dict[str, Any]]


class ContextualJsonDriver:
    """
    JSON generation driver that maintains conversation context across field generations.
    
    This driver addresses the context fragmentation issue in the original StopTokenJsonDriver
    by maintaining the full conversation context throughout the iterative JSON building process.
    """
    
    def __init__(self, llm_engine, tokenizer, base_context: str = ""):
        """
        Initialize the contextual driver.
        
        Args:
            llm_engine: VLLM LLM engine instance
            tokenizer: Tokenizer for the model
            base_context: Base conversation context (e.g., chat history, extraction prompt)
        """
        self.llm_engine = llm_engine
        self.tokenizer = tokenizer
        self.base_context = base_context
        
        # Session state for maintaining context
        self.session_state = None
        self.accumulated_tokens = []
        
    def generate_json(self, fields: List[Dict[str, FieldType]], 
                     extraction_instruction: str = "Return ONLY a JSON object with the relevant data. No explanation, no extra text:") -> str:
        """
        Generate JSON using contextual session-based generation.
        
        Args:
            fields: List of field specifications, each containing exactly one field
            extraction_instruction: Instruction for JSON extraction task
            
        Returns:
            Complete JSON string
        """
        if not fields:
            return "{}"
        
        # Initialize session with full context
        full_context = f"{self.base_context}\n\n{extraction_instruction}\n{{"
        
        # Start contextual generation session
        self._initialize_session(full_context)
        
        try:
            # Build JSON incrementally while maintaining context
            json_content = self._build_json_contextually(fields)
            return "{" + json_content + "}"
            
        finally:
            # Clean up session state
            self._cleanup_session()
    
    def _initialize_session(self, initial_context: str):
        """Initialize a generation session with the full context."""
        from vllm import SamplingParams
        
        # Tokenize the initial context to establish session state
        self.session_tokens = self.tokenizer.encode(initial_context)
        self.accumulated_output = ""
        
        # Create sampling params for contextual generation
        self.sampling_params = SamplingParams(
            temperature=0.3,
            max_tokens=15,  # Short generations per field
            skip_special_tokens=True
        )
    
    def _build_json_contextually(self, fields: List[Dict[str, FieldType]]) -> str:
        """Build JSON content while maintaining full conversation context."""
        json_parts = []
        
        for i, field_spec in enumerate(fields):
            # Validate field specification
            assert len(field_spec) == 1, "Each field specification must have exactly one field"
            
            field_name, field_type = next(iter(field_spec.items()))
            is_last_field = (i == len(fields) - 1)
            
            # Handle nested objects
            if isinstance(field_type, dict):
                nested_json = self._generate_nested_contextually(field_type)
                field_content = f'"{field_name}": {nested_json}'
            else:
                # Generate field value maintaining full context
                field_content = self._generate_field_contextually(field_name, field_type, is_last_field)
            
            json_parts.append(field_content)
            
            # Add comma if not the last field
            if not is_last_field:
                json_parts.append(", ")
        
        return "".join(json_parts)
    
    def _generate_field_contextually(self, field_name: str, field_type: str, is_last: bool) -> str:
        """Generate a single field while maintaining conversation context."""
        
        # Build the current prompt including all context and JSON so far
        current_json_part = f'"{field_name}": '
        
        # Create full prompt with accumulated context
        full_prompt = self._build_contextual_prompt(current_json_part)
        
        # Set appropriate stop tokens
        stop_tokens = None if is_last else [",", "}", "\n"]
        
        # Update sampling params with current stop tokens
        current_params = self.sampling_params.__class__(
            temperature=self.sampling_params.temperature,
            max_tokens=self.sampling_params.max_tokens,
            stop=stop_tokens,
            skip_special_tokens=self.sampling_params.skip_special_tokens
        )
        
        # Generate field value with full context
        outputs = self.llm_engine.generate([full_prompt], current_params)
        raw_result = outputs[0].outputs[0].text.strip()
        
        # Extract and clean the field value
        field_value = self._extract_field_value(raw_result, field_type)
        
        # Update accumulated output for next iteration
        self.accumulated_output += current_json_part + field_value
        
        return f'"{field_name}": {field_value}'
    
    def _build_contextual_prompt(self, next_json_part: str) -> str:
        """Build full contextual prompt including conversation and accumulated JSON."""
        # Reconstruct the full prompt with context
        base_tokens = self.session_tokens
        accumulated_json = "{" + self.accumulated_output + next_json_part
        
        # Decode base context and append current JSON state
        base_context = self.tokenizer.decode(base_tokens)
        full_prompt = base_context + self.accumulated_output + next_json_part
        
        return full_prompt
    
    def _extract_field_value(self, raw_result: str, field_type: str) -> str:
        """Extract clean field value from generation result."""
        # Clean up whitespace and common trailing punctuation
        cleaned = raw_result.strip().rstrip(",}")
        
        # Handle common patterns from over-generation
        if '\n' in cleaned and ('>>>' in cleaned or 'print(' in cleaned or 'def ' in cleaned):
            cleaned = cleaned.split('\n')[0].strip()
        
        # Remove trailing quotes if they appear to be closing the JSON
        cleaned = cleaned.rstrip('",}')
        
        if field_type == "string":
            return self._sanitize_string_value(cleaned)
        
        elif field_type == "number":
            # For numbers, extract the first valid number
            match = re.search(r'-?\d+(?:\.\d+)?', cleaned)
            if match:
                return match.group(0)
            else:
                return "0"
        
        return cleaned
    
    def _sanitize_string_value(self, raw_value: str) -> str:
        """Sanitize string value to ensure valid JSON."""
        # If already quoted, extract the content for processing
        if raw_value.startswith('"') and raw_value.endswith('"'):
            content = raw_value[1:-1]
        elif '"' in raw_value:
            # Find first quoted string
            match = re.search(r'"([^"]*)"', raw_value)
            content = match.group(1) if match else raw_value
        else:
            # Use the raw value, but take only the first reasonable part
            words = raw_value.split()
            if len(words) > 5:  # Likely over-generated, take first few words
                content = " ".join(words[:3])
            else:
                content = raw_value
        
        # Remove or escape invalid JSON control characters
        content = self._escape_json_string(content)
        
        # Return properly quoted string
        return f'"{content}"'
    
    def _escape_json_string(self, text: str) -> str:
        """Escape a string for valid JSON according to RFC 7159."""
        # Replace invalid control characters
        replacements = {
            '\n': '\\n',      # Newline
            '\r': '\\r',      # Carriage return  
            '\t': '\\t',      # Tab
            '\b': '\\b',      # Backspace
            '\f': '\\f',      # Form feed
            '"': '\\"',       # Quote
            '\\': '\\\\',     # Backslash
        }
        
        # Apply standard JSON escape sequences
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        
        # Remove any remaining control characters (ASCII 0-31 except allowed ones)
        cleaned_chars = []
        for char in text:
            if ord(char) >= 32 or char in '\n\r\t\b\f':
                cleaned_chars.append(char)
            else:
                continue
        
        text = ''.join(cleaned_chars)
        
        # Limit length to prevent over-generation issues
        if len(text) > 100:  # Reasonable limit for most fields
            if ' ' in text[:100]:
                last_space = text[:100].rfind(' ')
                text = text[:last_space]
            else:
                text = text[:100]
        
        return text
    
    def _generate_nested_contextually(self, nested_fields: Dict[str, FieldType]) -> str:
        """Generate a nested JSON object while maintaining context."""
        if not nested_fields:
            return "{}"
        
        # Convert nested dict to list format for recursive generation
        field_list = [{name: field_type} for name, field_type in nested_fields.items()]
        
        # Create a sub-driver for the nested object with current context
        nested_context = self._build_contextual_prompt("")
        nested_driver = ContextualJsonDriver(self.llm_engine, self.tokenizer, nested_context)
        
        # Generate nested object
        return nested_driver.generate_json(field_list, "")
    
    def _cleanup_session(self):
        """Clean up session state."""
        self.session_state = None
        self.accumulated_tokens = []
        self.accumulated_output = ""


def create_contextual_generate_func(llm_engine, tokenizer, base_context: str):
    """
    Create a contextual generate function for use with existing driver interfaces.
    
    Args:
        llm_engine: VLLM LLM engine instance
        tokenizer: Tokenizer for the model
        base_context: Base conversation context
        
    Returns:
        Generate function that maintains context
    """
    driver = ContextualJsonDriver(llm_engine, tokenizer, base_context)
    
    def contextual_generate(fields: List[Dict[str, FieldType]]) -> str:
        """Generate JSON maintaining conversation context."""
        return driver.generate_json(fields)
    
    return contextual_generate