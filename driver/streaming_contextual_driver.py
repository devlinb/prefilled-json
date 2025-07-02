"""
Streaming Contextual JSON Driver

Uses a streaming approach that maintains full conversation context by treating 
JSON generation as a single continuous conversation rather than multiple separate calls.
"""

import json
import re
from typing import Optional, Dict, List, Any, Union, Literal

FieldType = Union[Literal["string", "number"], Dict[str, Any]]


class StreamingContextualDriver:
    """
    JSON generation driver that uses streaming to maintain conversation context.
    
    Instead of making multiple separate LLM calls for each field, this driver
    generates the complete JSON in a single streaming session, using the LLM's
    natural generation capabilities while guiding the structure.
    """
    
    def __init__(self, llm_engine, tokenizer):
        """
        Initialize the streaming contextual driver.
        
        Args:
            llm_engine: VLLM LLM engine instance
            tokenizer: Tokenizer for the model
        """
        self.llm_engine = llm_engine
        self.tokenizer = tokenizer
    
    def generate_json(self, fields: List[Dict[str, FieldType]], 
                     base_context: str = "",
                     extraction_instruction: str = "Return ONLY a JSON object with the relevant data. No explanation, no extra text:") -> str:
        """
        Generate JSON using streaming contextual generation.
        
        Args:
            fields: List of field specifications
            base_context: Full conversation context
            extraction_instruction: Instruction for JSON extraction
            
        Returns:
            Complete JSON string
        """
        # Create field guidance for the LLM
        field_guidance = self._create_field_guidance(fields)
        
        # Build complete prompt with context and structure guidance
        full_prompt = f"""{base_context}

{extraction_instruction}

Please extract the information into a JSON object with the following structure:
{field_guidance}

JSON:"""
        
        # Generate with streaming approach
        return self._generate_with_streaming(full_prompt, fields)
    
    def _create_field_guidance(self, fields: List[Dict[str, FieldType]]) -> str:
        """Create human-readable field guidance for the LLM."""
        field_descriptions = []
        
        for field_spec in fields:
            field_name, field_type = next(iter(field_spec.items()))
            
            if isinstance(field_type, dict):
                # Nested object
                nested_fields = []
                for sub_name, sub_type in field_type.items():
                    nested_fields.append(f'  "{sub_name}": ({sub_type})')
                nested_desc = "{\n" + ",\n".join(nested_fields) + "\n}"
                field_descriptions.append(f'"{field_name}": {nested_desc}')
            else:
                field_descriptions.append(f'"{field_name}": ({field_type})')
        
        return "{\n" + ",\n".join(field_descriptions) + "\n}"
    
    def _generate_with_streaming(self, prompt: str, fields: List[Dict[str, FieldType]]) -> str:
        """Generate JSON using a streaming approach that maintains context."""
        from vllm import SamplingParams
        
        # Configure for JSON generation
        params = SamplingParams(
            temperature=0.1,  # Lower temperature for more structured output
            max_tokens=300,   # Enough tokens for complete JSON
            skip_special_tokens=True,
            # Let the model generate naturally, then post-process
        )
        
        # Generate the complete JSON in one go
        outputs = self.llm_engine.generate([prompt], params)
        raw_result = outputs[0].outputs[0].text.strip()
        
        # Post-process to ensure valid JSON structure
        return self._post_process_json(raw_result, fields)
    
    def _post_process_json(self, raw_result: str, fields: List[Dict[str, FieldType]]) -> str:
        """Post-process the generated text to ensure valid JSON."""
        
        # Find JSON content in the response
        json_start = raw_result.find('{')
        if json_start == -1:
            # No JSON found, create minimal structure
            return self._create_fallback_json(fields)
        
        # Extract potential JSON
        json_text = raw_result[json_start:]
        
        # Try to find the end of JSON
        brace_count = 0
        json_end = len(json_text)
        
        for i, char in enumerate(json_text):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_end = i + 1
                    break
        
        json_candidate = json_text[:json_end]
        
        # Try to parse and fix common issues
        try:
            # First attempt: direct parsing
            parsed = json.loads(json_candidate)
            return json.dumps(parsed, separators=(',', ':'))
            
        except json.JSONDecodeError:
            # Second attempt: fix common issues
            fixed_json = self._fix_json_issues(json_candidate)
            try:
                parsed = json.loads(fixed_json)
                return json.dumps(parsed, separators=(',', ':'))
            except json.JSONDecodeError:
                # Final fallback: create structured response
                return self._create_structured_fallback(raw_result, fields)
    
    def _fix_json_issues(self, json_text: str) -> str:
        """Fix common JSON syntax issues."""
        
        # Remove trailing commas
        json_text = re.sub(r',(\s*[}\]])', r'\1', json_text)
        
        # Fix unquoted keys (basic cases)
        json_text = re.sub(r'(\w+):', r'"\1":', json_text)
        
        # Fix single quotes to double quotes
        json_text = json_text.replace("'", '"')
        
        # Remove control characters that break JSON
        json_text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_text)
        
        # Ensure proper closing
        open_braces = json_text.count('{')
        close_braces = json_text.count('}')
        if open_braces > close_braces:
            json_text += '}' * (open_braces - close_braces)
        
        return json_text
    
    def _create_fallback_json(self, fields: List[Dict[str, FieldType]]) -> str:
        """Create a minimal valid JSON structure when generation fails."""
        result = {}
        
        for field_spec in fields:
            field_name, field_type = next(iter(field_spec.items()))
            
            if isinstance(field_type, dict):
                # Create nested object with empty strings/zeros
                nested = {}
                for sub_name, sub_type in field_type.items():
                    if sub_type == "string":
                        nested[sub_name] = ""
                    elif sub_type == "number":
                        nested[sub_name] = 0
                result[field_name] = nested
            else:
                if field_type == "string":
                    result[field_name] = ""
                elif field_type == "number":
                    result[field_name] = 0
        
        return json.dumps(result, separators=(',', ':'))
    
    def _create_structured_fallback(self, raw_text: str, fields: List[Dict[str, FieldType]]) -> str:
        """Create structured JSON by extracting information from raw text."""
        result = {}
        
        # Extract information for each field from the raw text
        for field_spec in fields:
            field_name, field_type = next(iter(field_spec.items()))
            
            if isinstance(field_type, dict):
                # Handle nested objects
                nested = {}
                for sub_name, sub_type in field_type.items():
                    value = self._extract_field_from_text(raw_text, sub_name, sub_type)
                    nested[sub_name] = value
                result[field_name] = nested
            else:
                value = self._extract_field_from_text(raw_text, field_name, field_type)
                result[field_name] = value
        
        return json.dumps(result, separators=(',', ':'))
    
    def _extract_field_from_text(self, text: str, field_name: str, field_type: str) -> Any:
        """Extract a specific field value from raw text."""
        
        # Look for patterns like "field_name": "value" or "field_name": value
        pattern = rf'"{field_name}":\s*"?([^",}}\n]+)"?'
        match = re.search(pattern, text, re.IGNORECASE)
        
        if match:
            value = match.group(1).strip()
            
            if field_type == "number":
                # Extract number
                num_match = re.search(r'-?\d+(?:\.\d+)?', value)
                return float(num_match.group(0)) if num_match else 0
            else:
                # String value
                return value.strip('"')
        
        # Fallback values
        if field_type == "number":
            return 0
        else:
            return ""


def create_streaming_generate_func(llm_engine, tokenizer):
    """
    Create a streaming generate function for use with existing interfaces.
    
    Args:
        llm_engine: VLLM LLM engine instance
        tokenizer: Tokenizer for the model
        
    Returns:
        Generate function that uses streaming contextual approach
    """
    driver = StreamingContextualDriver(llm_engine, tokenizer)
    
    def streaming_generate(fields: List[Dict[str, FieldType]], base_context: str = "") -> str:
        """Generate JSON using streaming contextual approach."""
        return driver.generate_json(fields, base_context)
    
    return streaming_generate