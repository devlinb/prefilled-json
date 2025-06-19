import pytest
from unittest.mock import Mock
from vllm_plugin.json_prefilled_plugin import StreamingJsonFieldDriver


class TestStreamingJsonFieldDriver:
    """Test the StreamingJsonFieldDriver with pattern matching approach."""
    
    def setup_method(self):
        """Setup mock generate function for each test."""
        self.mock_generate = Mock()
        self.driver = StreamingJsonFieldDriver(generate=self.mock_generate)
    
    def test_extract_string_value_quoted(self):
        """Test extracting quoted string values."""
        # Standard quoted string
        result = self.driver._extract_string_value('"Alice", "age": 30')
        assert result == '"Alice"'
        
        # Quoted string at end
        result = self.driver._extract_string_value('"Bob"}')
        assert result == '"Bob"'
        
        # Simple quoted string
        result = self.driver._extract_string_value('"Carol"')
        assert result == '"Carol"'
    
    def test_extract_string_value_unquoted(self):
        """Test extracting unquoted string values."""
        # Unquoted text before comma
        result = self.driver._extract_string_value('Alice, age: 30')
        assert result == '"Alice"'
        
        # Unquoted text before brace
        result = self.driver._extract_string_value('Bob}')
        assert result == '"Bob"'
        
        # Single word
        result = self.driver._extract_string_value('Carol')
        assert result == '"Carol"'
    
    def test_extract_string_value_fallback(self):
        """Test fallback cases for string extraction."""
        # Multiple words - should take first
        result = self.driver._extract_string_value('John Doe is a person')
        assert result == '"John Doe is a person"'
        
        # Empty input
        result = self.driver._extract_string_value('')
        assert result == '"default"'
        
        # Just whitespace
        result = self.driver._extract_string_value('   ')
        assert result == '"default"'
    
    def test_extract_number_value_valid(self):
        """Test extracting valid number values."""
        # Integer with comma
        result = self.driver._extract_number_value('25, "city": "NYC"', 'age')
        assert result == '25'
        
        # Integer with brace
        result = self.driver._extract_number_value('30}', 'age')
        assert result == '30'
        
        # Float value
        result = self.driver._extract_number_value('42.5, more', 'price')
        assert result == '42.5'
        
        # Just number
        result = self.driver._extract_number_value('100', 'score')
        assert result == '100'
        
        # Number in text
        result = self.driver._extract_number_value('age 25 years', 'age')
        assert result == '25'
    
    def test_extract_number_value_invalid(self):
        """Test number extraction with invalid inputs."""
        # No number found
        with pytest.raises(ValueError, match="is not a valid number"):
            self.driver._extract_number_value('no numbers here', 'age')
        
        # Text only
        with pytest.raises(ValueError, match="is not a valid number"):
            self.driver._extract_number_value('abc def', 'count')
    
    def test_generate_simple_json(self):
        """Test generating simple JSON with string and number fields."""
        # Setup mock responses for field generation
        self.mock_generate.side_effect = [
            '"Alice", "age": 30, "city": "Seattle"',  # name field response
            '25, "city": "Seattle", "active": true'   # age field response
        ]
        
        fields = [
            {"name": "string"},
            {"age": "number"}
        ]
        
        result = self.driver.generate_json(fields)
        
        # Verify the complete JSON structure
        assert result == '{"name": "Alice", "age": 25}'
        
        # Verify generate was called with correct prompts
        assert self.mock_generate.call_count == 2
        calls = self.mock_generate.call_args_list
        
        # First call for name field
        assert calls[0][0][0] == '{"name": '
        assert calls[0][0][1] is None  # No stop token
        
        # Second call for age field
        assert calls[1][0][0] == '{"name": "Alice", "age": '
        assert calls[1][0][1] is None  # No stop token
    
    def test_generate_nested_json(self):
        """Test generating nested JSON structures."""
        # Setup mock responses
        self.mock_generate.side_effect = [
            '"John", "id": 123, "dept": "Engineering"',  # employee.name
            '456, "dept": "Engineering"',                # employee.id  
            '"Engineering", "location": "Seattle"'       # department
        ]
        
        fields = [
            {"employee": {
                "name": "string",
                "id": "number"
            }},
            {"department": "string"}
        ]
        
        result = self.driver.generate_json(fields)
        
        expected = '{"employee": {"name": "John", "id": 456}, "department": "Engineering"}'
        assert result == expected
    
    def test_generate_single_field(self):
        """Test generating JSON with single field."""
        self.mock_generate.return_value = '"test_value", "extra": "data"'
        
        fields = [{"test_field": "string"}]
        result = self.driver.generate_json(fields)
        
        assert result == '{"test_field": "test_value"}'
    
    def test_extract_field_value_string_type(self):
        """Test _extract_field_value with string type."""
        result = self.driver._extract_field_value('"Alice"', "string", "name")
        assert result == '"Alice"'
    
    def test_extract_field_value_number_type(self):
        """Test _extract_field_value with number type."""
        result = self.driver._extract_field_value('25', "number", "age")
        assert result == '25'
    
    def test_extract_field_value_unsupported_type(self):
        """Test _extract_field_value with unsupported type."""
        with pytest.raises(ValueError, match="Unsupported field type"):
            self.driver._extract_field_value('value', "boolean", "flag")
    
    def test_multiple_fields_with_over_generation(self):
        """Test handling multiple fields when model over-generates."""
        # Model generates complete JSON structures but we extract just what we need
        self.mock_generate.side_effect = [
            '"Alice Johnson", "age": 30, "city": "Seattle", "email": "alice@example.com"',
            '28, "city": "Portland", "phone": "555-1234"',
            '"Portland", "state": "Oregon", "country": "USA"'
        ]
        
        fields = [
            {"name": "string"},
            {"age": "number"}, 
            {"city": "string"}
        ]
        
        result = self.driver.generate_json(fields)
        
        expected = '{"name": "Alice Johnson", "age": 28, "city": "Portland"}'
        assert result == expected
        
        # Verify all generate calls were made without stop tokens
        for call in self.mock_generate.call_args_list:
            assert call[0][1] is None  # No stop token parameter