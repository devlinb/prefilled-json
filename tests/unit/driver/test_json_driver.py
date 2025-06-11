import pytest
from unittest.mock import Mock, call
from typing import Optional

from driver.json_driver import JsonFieldDriver


class TestJsonFieldDriver:
    
    def test_init(self):
        """Test that JsonFieldDriver initializes correctly with a generate function."""
        mock_generate = Mock()
        driver = JsonFieldDriver(mock_generate)
        assert driver.generate == mock_generate

    def test_generate_json_single_string_field(self):
        """Test generating JSON with a single string field."""
        def mock_generate(prompt: str, stop: Optional[str]) -> str:
            # For single field, no stop token is passed (None)
            assert stop is None
            assert prompt == '{"name": '
            return '"Alice"}'  # LLM might add closing brace
        
        driver = JsonFieldDriver(mock_generate)
        fields = [{"name": "string"}]
        result = driver.generate_json(fields)
        
        assert result == '{"name": "Alice"}'

    def test_generate_json_single_number_field(self):
        """Test generating JSON with a single number field."""
        def mock_generate(prompt: str, stop: Optional[str]) -> str:
            assert stop is None  # No stop token for single field
            assert prompt == '{"age": '
            return '25}'  # LLM might add closing brace
        
        driver = JsonFieldDriver(mock_generate)
        fields = [{"age": "number"}]
        result = driver.generate_json(fields)
        
        assert result == '{"age": 25}'

    def test_generate_json_multiple_fields_incremental_prompts(self):
        """Test that prompts build incrementally as JSON is constructed."""
        mock_generate = Mock()
        
        # Mock responses for each field
        mock_generate.side_effect = [
            '"Alice",',      # First field with comma
            '30,',           # Second field with comma  
            '"Seattle"}'     # Last field with closing brace
        ]
        
        driver = JsonFieldDriver(mock_generate)
        fields = [
            {"name": "string"},
            {"age": "number"},
            {"city": "string"}
        ]
        result = driver.generate_json(fields)
        
        # Verify the incremental prompts sent to LLM
        expected_calls = [
            call('{"name": ', ","),                              # First field
            call('{"name": "Alice", "age": ', ","),              # Second field  
            call('{"name": "Alice", "age": 30, "city": ', None)  # Last field
        ]
        
        assert mock_generate.call_args_list == expected_calls
        assert result == '{"name": "Alice", "age": 30, "city": "Seattle"}'

    def test_generate_json_strips_trailing_punctuation(self):
        """Test that trailing commas and braces are properly stripped."""
        def mock_generate(prompt: str, stop: Optional[str]) -> str:
            return '"Alice",}'  # LLM adds both comma and brace
        
        driver = JsonFieldDriver(mock_generate)
        fields = [{"name": "string"}]
        result = driver.generate_json(fields)
        
        assert result == '{"name": "Alice"}'

    def test_generate_json_adds_quotes_to_unquoted_string(self):
        """Test that unquoted string values get properly quoted."""
        def mock_generate(prompt: str, stop: Optional[str]) -> str:
            return 'Alice'  # LLM returns unquoted string
        
        driver = JsonFieldDriver(mock_generate)
        fields = [{"name": "string"}]
        result = driver.generate_json(fields)
        
        assert result == '{"name": "Alice"}'

    def test_generate_json_preserves_quoted_strings(self):
        """Test that already quoted strings are preserved."""
        def mock_generate(prompt: str, stop: Optional[str]) -> str:
            return '"Alice"'  # LLM returns properly quoted string
        
        driver = JsonFieldDriver(mock_generate)
        fields = [{"name": "string"}]
        result = driver.generate_json(fields)
        
        assert result == '{"name": "Alice"}'

    def test_generate_json_number_validation(self):
        """Test that number fields are validated."""
        def mock_generate(prompt: str, stop: Optional[str]) -> str:
            return '42.5'  # Valid number
        
        driver = JsonFieldDriver(mock_generate)
        fields = [{"score": "number"}]
        result = driver.generate_json(fields)
        
        assert result == '{"score": 42.5}'

    def test_generate_json_invalid_number_raises_error(self):
        """Test that invalid number values raise ValueError."""
        def mock_generate(prompt: str, stop: Optional[str]) -> str:
            return 'not_a_number'
        
        driver = JsonFieldDriver(mock_generate)
        fields = [{"age": "number"}]
        
        with pytest.raises(ValueError, match="Generated value for field 'age' is not a valid number"):
            driver.generate_json(fields)

    def test_generate_json_unsupported_field_type_raises_error(self):
        """Test that unsupported field types raise ValueError."""
        def mock_generate(prompt: str, stop: Optional[str]) -> str:
            return 'value'
        
        driver = JsonFieldDriver(mock_generate)
        fields = [{"field": "boolean"}]  # Unsupported type
        
        with pytest.raises(ValueError, match="Unsupported field type: boolean"):
            driver.generate_json(fields)

    def test_generate_json_field_spec_validation(self):
        """Test that field specifications with multiple keys raise AssertionError."""
        def mock_generate(prompt: str, stop: Optional[str]) -> str:
            return '""'
        
        driver = JsonFieldDriver(mock_generate)
        fields = [{"name": "string", "age": "number"}]  # Invalid: multiple keys
        
        with pytest.raises(AssertionError, match="Each field specification must have exactly one field"):
            driver.generate_json(fields)

    def test_generate_json_stop_token_behavior(self):
        """Test that stop tokens are used correctly for controlling generation."""
        mock_generate = Mock()
        mock_generate.side_effect = ['"Alice",', '30']
        
        driver = JsonFieldDriver(mock_generate)
        fields = [
            {"name": "string"},
            {"age": "number"}
        ]
        
        result = driver.generate_json(fields)
        
        # Verify stop token usage - args are positional, kwargs are keyword
        calls = mock_generate.call_args_list
        assert calls[0].args[1] == ","     # Stop token for first field
        assert calls[1].args[1] is None    # No stop token for last field
        
        assert result == '{"name": "Alice", "age": 30}'

    def test_generate_json_empty_fields_list(self):
        """Test behavior with empty fields list."""
        def mock_generate(prompt: str, stop: Optional[str]) -> str:
            return ""
        
        driver = JsonFieldDriver(mock_generate)
        fields = []
        result = driver.generate_json(fields)
        
        assert result == "{}"

    def test_generate_json_whitespace_handling(self):
        """Test that whitespace is properly handled in generated values."""
        def mock_generate(prompt: str, stop: Optional[str]) -> str:
            return '  "Alice"  ,  '  # LLM adds extra whitespace
        
        driver = JsonFieldDriver(mock_generate)
        fields = [{"name": "string"}]
        result = driver.generate_json(fields)
        
        assert result == '{"name": "Alice"}'

    def test_generate_json_final_field_with_comma(self):
        """Test that small LLMs adding comma to final field is handled correctly."""
        call_count = 0
        def mock_generate(prompt: str, stop: Optional[str]) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # First call for "name" field
                return '"Alice",'
            elif call_count == 2:  # Second call for "age" field
                return '30,'  # Final field - small LLM doesn't know it's final, adds comma
            return '""'
        
        driver = JsonFieldDriver(mock_generate)
        fields = [
            {"name": "string"},
            {"age": "number"}
        ]
        result = driver.generate_json(fields)
        
        assert result == '{"name": "Alice", "age": 30}'

    def test_generate_json_single_field_with_comma(self):
        """Test single field where LLM adds comma even though it's the only field."""
        def mock_generate(prompt: str, stop: Optional[str]) -> str:
            return '"Alice",'  # Small LLM adds comma even on single field
        
        driver = JsonFieldDriver(mock_generate)
        fields = [{"name": "string"}]
        result = driver.generate_json(fields)
        
        assert result == '{"name": "Alice"}'

    def test_generate_json_final_number_field_with_comma(self):
        """Test final number field where small LLM adds comma."""
        def mock_generate(prompt: str, stop: Optional[str]) -> str:
            return '42,'  # Small LLM adds comma to final number field
        
        driver = JsonFieldDriver(mock_generate)
        fields = [{"score": "number"}]
        result = driver.generate_json(fields)
        
        assert result == '{"score": 42}'

    def test_generate_json_mixed_trailing_punctuation(self):
        """Test various combinations of trailing punctuation from small LLMs."""
        responses = [
            '"Alice",',    # First field with comma
            '30,}',        # Second field with comma and brace (confused LLM)
            '"Seattle",,'  # Final field with double comma (very confused LLM)
        ]
        response_iter = iter(responses)
        
        def mock_generate(prompt: str, stop: Optional[str]) -> str:
            return next(response_iter)
        
        driver = JsonFieldDriver(mock_generate)
        fields = [
            {"name": "string"},
            {"age": "number"},
            {"city": "string"}
        ]
        result = driver.generate_json(fields)
        
        assert result == '{"name": "Alice", "age": 30, "city": "Seattle"}'

    def test_generate_json_simple_nested_object(self):
        """Test generating JSON with a simple nested object."""
        call_count = 0
        def mock_generate(prompt: str, stop: Optional[str]) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # name field
                return '"Alice",'
            elif call_count == 2:  # address.street field
                return '"123 Main St",'
            elif call_count == 3:  # address.city field
                return '"Seattle"'
            elif call_count == 4:  # age field
                return '30'
            return '""'
        
        driver = JsonFieldDriver(mock_generate)
        fields = [
            {"name": "string"},
            {"address": {
                "street": "string",
                "city": "string"
            }},
            {"age": "number"}
        ]
        result = driver.generate_json(fields)
        
        expected = '{"name": "Alice", "address": {"street": "123 Main St", "city": "Seattle"}, "age": 30}'
        assert result == expected

    def test_generate_json_nested_object_with_numbers(self):
        """Test nested object containing number fields."""
        call_count = 0
        def mock_generate(prompt: str, stop: Optional[str]) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # name field
                return '"Bob",'
            elif call_count == 2:  # coordinates.lat field
                return '47.6062,'
            elif call_count == 3:  # coordinates.lng field
                return '-122.3321'
            return '""'
        
        driver = JsonFieldDriver(mock_generate)
        fields = [
            {"name": "string"},
            {"coordinates": {
                "lat": "number",
                "lng": "number"
            }}
        ]
        result = driver.generate_json(fields)
        
        expected = '{"name": "Bob", "coordinates": {"lat": 47.6062, "lng": -122.3321}}'
        assert result == expected

    def test_generate_json_deeply_nested_object(self):
        """Test deeply nested objects (3 levels)."""
        call_count = 0
        def mock_generate(prompt: str, stop: Optional[str]) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # name
                return '"Charlie",'
            elif call_count == 2:  # profile.contact.email
                return '"charlie@example.com",'
            elif call_count == 3:  # profile.contact.phone
                return '"555-1234",'
            elif call_count == 4:  # profile.age
                return '28'
            return '""'
        
        driver = JsonFieldDriver(mock_generate)
        fields = [
            {"name": "string"},
            {"profile": {
                "contact": {
                    "email": "string",
                    "phone": "string"
                },
                "age": "number"
            }}
        ]
        result = driver.generate_json(fields)
        
        expected = '{"name": "Charlie", "profile": {"contact": {"email": "charlie@example.com", "phone": "555-1234"}, "age": 28}}'
        assert result == expected

    def test_generate_json_multiple_nested_objects(self):
        """Test multiple nested objects at the same level."""
        call_count = 0
        def mock_generate(prompt: str, stop: Optional[str]) -> str:
            nonlocal call_count
            call_count += 1
            responses = [
                '"Diana",',           # name
                '"123 Oak St",',      # address.street
                '"Portland",',        # address.city
                '"Engineer",',        # job.title
                '"TechCorp"'          # job.company
            ]
            return responses[call_count - 1] if call_count <= len(responses) else '""'
        
        driver = JsonFieldDriver(mock_generate)
        fields = [
            {"name": "string"},
            {"address": {
                "street": "string",
                "city": "string"
            }},
            {"job": {
                "title": "string",
                "company": "string"
            }}
        ]
        result = driver.generate_json(fields)
        
        expected = '{"name": "Diana", "address": {"street": "123 Oak St", "city": "Portland"}, "job": {"title": "Engineer", "company": "TechCorp"}}'
        assert result == expected

    def test_generate_json_nested_object_with_trailing_punctuation(self):
        """Test that nested objects handle LLM trailing punctuation correctly."""
        call_count = 0
        def mock_generate(prompt: str, stop: Optional[str]) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # name
                return '"Eve",'
            elif call_count == 2:  # settings.theme
                return '"dark",}'  # LLM adds both comma and brace
            elif call_count == 3:  # settings.lang
                return '"en",'  # LLM adds comma to final nested field
            return '""'
        
        driver = JsonFieldDriver(mock_generate)
        fields = [
            {"name": "string"},
            {"settings": {
                "theme": "string",
                "lang": "string"
            }}
        ]
        result = driver.generate_json(fields)
        
        expected = '{"name": "Eve", "settings": {"theme": "dark", "lang": "en"}}'
        assert result == expected

    def test_generate_json_single_nested_object(self):
        """Test JSON with only a nested object."""
        call_count = 0
        def mock_generate(prompt: str, stop: Optional[str]) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # user.name
                return '"Frank",'
            elif call_count == 2:  # user.id
                return '12345'
            return '""'
        
        driver = JsonFieldDriver(mock_generate)
        fields = [
            {"user": {
                "name": "string",
                "id": "number"
            }}
        ]
        result = driver.generate_json(fields)
        
        expected = '{"user": {"name": "Frank", "id": 12345}}'
        assert result == expected

    def test_generate_json_empty_nested_object(self):
        """Test nested object with no fields."""
        def mock_generate(prompt: str, stop: Optional[str]) -> str:
            return '"Grace"'
        
        driver = JsonFieldDriver(mock_generate)
        fields = [
            {"name": "string"},
            {"metadata": {}}
        ]
        result = driver.generate_json(fields)
        
        expected = '{"name": "Grace", "metadata": {}}'
        assert result == expected