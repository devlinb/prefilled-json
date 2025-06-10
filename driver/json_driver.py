from typing import List, Literal, Callable, Optional, Dict, Union

FieldType = Literal["string", "number"]
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
        and its type: 'string' or 'number'.
        :return: A valid JSON string.
        """
        json_parts = ["{"]

        for i, field_spec in enumerate(fields):
            assert len(field_spec) == 1, "Each field specification must have exactly one field"
            field_name, field_type = next(iter(field_spec.items()))

            # Add field name
            prompt = "".join(json_parts) + f'"{field_name}": '
            stop = "," if i < len(fields) - 1 else None  # Let model stop naturally on final field

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
            if i < len(fields) - 1:
                json_parts.append(", ")

        json_parts.append("}")
        return "".join(json_parts)
