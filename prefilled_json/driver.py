"""Core JSON field drivers for LLM-based JSON generation."""

import re
from typing import List, Dict, Optional
from .types import FieldType, GenerateFunc


class JsonFieldDriver:
    """
    Traditional JSON field driver that uses stop tokens.
    
    This driver is suitable for custom LLM implementations that respect stop tokens.
    For modern instruction-tuned models, use StreamingJsonFieldDriver instead.
    """
    
    def __init__(self, generate: GenerateFunc):
        """
        Initialize the driver with a generate function.
        
        Args:
            generate: A function that takes a prompt and optional stop token,
                     and returns the generated value as a string.
        """
        self.generate = generate

    def generate_json(self, fields: List[Dict[str, FieldType]]) -> str:
        """
        Generate JSON by iteratively prompting an LLM to fill in each field value.

        Args:
            fields: A list of dictionaries, each with one key (field name)
                   and its type: 'string', 'number', or a nested object specification.
                   
        Returns:
            A valid JSON string.
        """
        return self._generate_object(fields)

    def _generate_object(self, fields: List[Dict[str, FieldType]]) -> str:
        """
        Generate a JSON object from field specifications.
        
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
                # Handle primitive field (string/number)
                stop = "," if not is_last_field else None
                value = self.generate(prompt, stop)

                # Strip trailing commas or close-brace if generated prematurely
                value = value.strip().rstrip(',}').strip()

                if field_type == "string":
                    if not (value.startswith('"') and value.endswith('"')):
                        value = '"' + value.strip('"') + '"'
                elif field_type == "number":
                    try:
                        float(value)  # just to validate
                    except ValueError:
                        raise ValueError(f"Generated value for field '{field_name}' is not a valid number: {value}")
                else:
                    raise ValueError(f"Unsupported field type: {field_type}")

                json_parts.append(f'"{field_name}": {value}')

            # Add comma if not the last field
            if not is_last_field:
                json_parts.append(", ")

        json_parts.append("}")
        return "".join(json_parts)


class StreamingJsonFieldDriver:
    """
    Modern JSON field driver that uses pattern matching instead of stop tokens.
    
    This driver works reliably with modern instruction-tuned models by allowing
    them to over-generate content and then extracting precise field values using
    regex patterns. This is the recommended approach for most use cases.
    """
    
    def __init__(self, generate: GenerateFunc):
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