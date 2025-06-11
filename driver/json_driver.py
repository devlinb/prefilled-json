from typing import List, Literal, Callable, Optional, Dict, Union, Any

FieldType = Union[Literal["string", "number"], Dict[str, Any]]
GenerateFunc = Callable[[str, Optional[str]], str]

class JsonFieldDriver:
    def __init__(self, generate: GenerateFunc):
        """
        :param generate: A function that takes a prompt and optional stop token,
        and returns the generated value as a string.
        """
        self.generate = generate

    def generate_json(self, fields: List[Dict[str, FieldType]]) -> str:
        """
        Generate JSON by iteratively prompting an LLM to fill in each field value.

        :param fields: A list of dictionaries, each with one key (field name)
        and its type: 'string', 'number', or a nested object specification.
        :return: A valid JSON string.
        """
        return self._generate_object(fields)

    def _generate_object(self, fields: List[Dict[str, FieldType]]) -> str:
        """
        Generate a JSON object from field specifications.
        
        :param fields: List of field specifications
        :return: JSON object string
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
