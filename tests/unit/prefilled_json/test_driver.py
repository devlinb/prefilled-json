import pytest
from unittest.mock import Mock
from prefilled_json.driver import JsonFieldDriver, StreamingJsonFieldDriver


class TestJsonFieldDriver:
    """Test the traditional JsonFieldDriver with stop tokens."""
    
    def setup_method(self):
        """Setup mock generate function for each test."""
        self.mock_generate = Mock()
        self.driver = JsonFieldDriver(generate=self.mock_generate)
    
    def test_init(self):
        """Test that JsonFieldDriver initializes correctly with a generate function."""
        mock_generate = Mock()
        driver = JsonFieldDriver(mock_generate)
        assert driver.generate == mock_generate
    
    def test_generate_json_single_string_field(self):
        """Test generating JSON with a single string field."""
        self.mock_generate.return_value = '"Alice"'
        
        fields = [{"name": "string"}]
        result = self.driver.generate_json(fields)
        
        assert result == '{"name": "Alice"}'
        self.mock_generate.assert_called_once_with('{"name": ', None)
    
    def test_generate_json_single_number_field(self):
        """Test generating JSON with a single number field."""
        self.mock_generate.return_value = '30'
        
        fields = [{"age": "number"}]
        result = self.driver.generate_json(fields)
        
        assert result == '{"age": 30}'
        self.mock_generate.assert_called_once_with('{"age": ', None)
    
    def test_generate_json_multiple_fields_incremental_prompts(self):
        """Test that multiple fields generate incremental prompts with stop tokens."""
        self.mock_generate.side_effect = ['"Alice"', '30', '"Seattle"']
        
        fields = [{"name": "string"}, {"age": "number"}, {"city": "string"}]
        result = self.driver.generate_json(fields)
        
        expected_calls = [
            (('{"name": ', ','), {}),
            (('{"name": "Alice", "age": ', ','), {}),
            (('{"name": "Alice", "age": 30, "city": ', None), {})
        ]
        assert self.mock_generate.call_args_list == expected_calls
        assert result == '{"name": "Alice", "age": 30, "city": "Seattle"}'


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
        
        # Both calls should use None for stop token (streaming approach)
        assert calls[0][0][0] == '{"name": '
        assert calls[0][0][1] is None
        
        assert calls[1][0][0] == '{"name": "Alice", "age": '
        assert calls[1][0][1] is None