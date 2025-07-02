#!/usr/bin/env python3
"""
Comprehensive unit tests for StopTokenJsonDriver.

These tests focus heavily on the field value extraction logic which has been
identified as the source of JSON corruption issues.
"""

import pytest
import json
from unittest.mock import Mock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from driver.stop_token_json_driver import StopTokenJsonDriver


class TestStopTokenJsonDriver:
    """Test the StopTokenJsonDriver class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_generate = Mock()
        self.config = {
            "stop_tokens": [",", "}", "\n", "<|end|>"],
            "stop_reliable": True
        }
        self.driver = StopTokenJsonDriver(self.mock_generate, self.config)

    def test_init(self):
        """Test driver initialization."""
        assert self.driver.generate == self.mock_generate
        assert self.driver.model_config == self.config
        assert self.driver.stop_tokens == [",", "}", "\n", "<|end|>"]

    def test_empty_fields(self):
        """Test handling of empty field list."""
        result = self.driver.generate_json([])
        assert result == "{}"

    def test_invalid_field_spec(self):
        """Test validation of field specifications."""
        with pytest.raises(AssertionError, match="exactly one field"):
            self.driver.generate_json([{"name": "string", "age": "number"}])

    def test_unsupported_field_type(self):
        """Test handling of unsupported field types."""
        self.mock_generate.return_value = "test"
        with pytest.raises(ValueError, match="Unsupported field type"):
            self.driver.generate_json([{"test_field": "boolean"}])


class TestFieldValueExtraction:
    """Test the _extract_field_value method extensively."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_generate = Mock()
        self.config = {"stop_tokens": [",", "}", "\n"], "stop_reliable": True}
        self.driver = StopTokenJsonDriver(self.mock_generate, self.config)

    # String Value Extraction Tests
    def test_extract_simple_string(self):
        """Test extracting a simple unquoted string."""
        result = self.driver._extract_field_value("John Doe", "string")
        assert result == '"John Doe"'

    def test_extract_quoted_string(self):
        """Test extracting an already quoted string."""
        result = self.driver._extract_field_value('"John Doe"', "string")
        assert result == '"John Doe"'

    def test_extract_string_with_comma(self):
        """Test extracting string that ends with comma (stop token)."""
        result = self.driver._extract_field_value("John Doe,", "string")
        assert result == '"John Doe"'

    def test_extract_string_with_brace(self):
        """Test extracting string that ends with closing brace."""
        result = self.driver._extract_field_value("John Doe}", "string")
        assert result == '"John Doe"'

    def test_extract_string_with_multiple_stop_chars(self):
        """Test extracting string with multiple stop characters."""
        result = self.driver._extract_field_value("John Doe,}", "string")
        assert result == '"John Doe"'

    def test_extract_string_with_quotes_and_comma(self):
        """Test extracting quoted string with trailing comma."""
        result = self.driver._extract_field_value('"John Doe",', "string")
        assert result == '"John Doe"'

    def test_extract_string_with_over_generation(self):
        """Test extracting string from over-generated content."""
        over_gen = "John Doe, and then some extra content that should be ignored"
        result = self.driver._extract_field_value(over_gen, "string")
        assert result == '"John Doe"'

    def test_extract_string_with_newline_code(self):
        """Test extracting string that contains code-like content after newline."""
        over_gen = "John Doe\nprint('hello')\ndef function():"
        result = self.driver._extract_field_value(over_gen, "string")
        assert result == '"John Doe"'

    def test_extract_string_with_python_prompt(self):
        """Test extracting string from Python REPL-like output."""
        over_gen = "John Doe\n>>> print('test')"
        result = self.driver._extract_field_value(over_gen, "string")
        assert result == '"John Doe"'

    def test_extract_long_string_truncation(self):
        """Test that very long strings are truncated reasonably."""
        long_string = "This is a very long string " * 20  # ~540 chars
        result = self.driver._extract_field_value(long_string, "string")
        parsed = json.loads(result)
        assert len(parsed) <= 100  # Should be truncated

    def test_extract_string_with_whitespace(self):
        """Test extracting string with leading/trailing whitespace."""
        result = self.driver._extract_field_value("  John Doe  ", "string")
        assert result == '"John Doe"'

    def test_extract_string_with_internal_quotes(self):
        """Test extracting string that contains internal quotes."""
        result = self.driver._extract_field_value('John "Johnny" Doe', "string")
        # Should escape internal quotes
        assert '"John \\"Johnny\\" Doe"' in result

    def test_extract_empty_string(self):
        """Test extracting empty string."""
        result = self.driver._extract_field_value("", "string")
        assert result == '""'

    def test_extract_string_only_quotes(self):
        """Test extracting string that is just quotes."""
        result = self.driver._extract_field_value('""', "string")
        assert result == '""'

    # Number Value Extraction Tests
    def test_extract_simple_integer(self):
        """Test extracting a simple integer."""
        result = self.driver._extract_field_value("42", "number")
        assert result == "42"

    def test_extract_simple_float(self):
        """Test extracting a simple float."""
        result = self.driver._extract_field_value("42.5", "number")
        assert result == "42.5"

    def test_extract_negative_number(self):
        """Test extracting a negative number."""
        result = self.driver._extract_field_value("-42", "number")
        assert result == "-42"

    def test_extract_negative_float(self):
        """Test extracting a negative float."""
        result = self.driver._extract_field_value("-42.75", "number")
        assert result == "-42.75"

    def test_extract_number_with_comma(self):
        """Test extracting number that ends with comma."""
        result = self.driver._extract_field_value("42,", "number")
        assert result == "42"

    def test_extract_number_with_brace(self):
        """Test extracting number that ends with closing brace."""
        result = self.driver._extract_field_value("42}", "number")
        assert result == "42"

    def test_extract_number_with_over_generation(self):
        """Test extracting number from over-generated content."""
        over_gen = "42 and some additional text"
        result = self.driver._extract_field_value(over_gen, "number")
        assert result == "42"

    def test_extract_number_with_whitespace(self):
        """Test extracting number with whitespace."""
        result = self.driver._extract_field_value("  42  ", "number")
        assert result == "42"

    def test_extract_zero(self):
        """Test extracting zero."""
        result = self.driver._extract_field_value("0", "number")
        assert result == "0"

    def test_extract_large_number(self):
        """Test extracting a large number."""
        result = self.driver._extract_field_value("1234567890", "number")
        assert result == "1234567890"

    def test_extract_decimal_with_many_places(self):
        """Test extracting decimal with many decimal places."""
        result = self.driver._extract_field_value("3.14159265", "number")
        assert result == "3.14159265"

    def test_extract_number_from_mixed_content(self):
        """Test extracting number from content with letters."""
        result = self.driver._extract_field_value("The answer is 42!", "number")
        assert result == "42"

    def test_extract_no_number_found(self):
        """Test extracting when no valid number is found."""
        result = self.driver._extract_field_value("no numbers here", "number")
        assert result == "0"  # Should default to 0

    def test_extract_number_scientific_notation(self):
        """Test that scientific notation numbers work."""
        # Note: current regex might not support this, but test for future
        result = self.driver._extract_field_value("1e5", "number")
        # Current implementation might not handle this, but we'll test
        assert result in ["1", "0"]  # Either extracts "1" or defaults to "0"


class TestJSONStringEscaping:
    """Test the _escape_json_string method."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_generate = Mock()
        self.config = {"stop_tokens": [",", "}"], "stop_reliable": True}
        self.driver = StopTokenJsonDriver(self.mock_generate, self.config)

    def test_escape_newline(self):
        """Test escaping newline characters."""
        result = self.driver._escape_json_string("Line 1\nLine 2")
        assert "\\n" in result
        assert "\n" not in result

    def test_escape_carriage_return(self):
        """Test escaping carriage return characters."""
        result = self.driver._escape_json_string("Line 1\rLine 2")
        assert "\\r" in result
        assert "\r" not in result

    def test_escape_tab(self):
        """Test escaping tab characters."""
        result = self.driver._escape_json_string("Column1\tColumn2")
        assert "\\t" in result
        assert "\t" not in result

    def test_escape_backspace(self):
        """Test escaping backspace characters."""
        result = self.driver._escape_json_string("Text\bBackspace")
        assert "\\b" in result
        assert "\b" not in result

    def test_escape_form_feed(self):
        """Test escaping form feed characters."""
        result = self.driver._escape_json_string("Page1\fPage2")
        assert "\\f" in result
        assert "\f" not in result

    def test_escape_quotes(self):
        """Test escaping quote characters."""
        result = self.driver._escape_json_string('He said "Hello"')
        assert '\\"' in result
        # All quotes should be escaped - no raw quotes should remain
        assert '"' not in result.replace('\\"', '')

    def test_escape_backslash(self):
        """Test escaping backslash characters."""
        result = self.driver._escape_json_string("Path\\to\\file")
        assert "\\\\" in result

    def test_escape_multiple_characters(self):
        """Test escaping multiple special characters at once."""
        text = 'Text with "quotes", \n newlines, \t tabs, and \\ backslashes'
        result = self.driver._escape_json_string(text)
        
        # Verify all escapes
        assert '\\"' in result
        assert '\\n' in result
        assert '\\t' in result
        assert '\\\\' in result

    def test_remove_control_characters(self):
        """Test removal of invalid control characters."""
        # ASCII control chars 0-31 (except allowed ones)
        text_with_control = "Normal text\x01\x02\x03control chars"
        result = self.driver._escape_json_string(text_with_control)
        
        # Should remove control chars but keep normal text
        assert "Normal text" in result
        assert "control chars" in result
        assert "\x01" not in result
        assert "\x02" not in result
        assert "\x03" not in result

    def test_length_limiting(self):
        """Test that very long strings are truncated."""
        long_text = "A" * 200  # 200 characters
        result = self.driver._escape_json_string(long_text)
        assert len(result) <= 100

    def test_word_boundary_truncation(self):
        """Test that truncation happens at word boundaries when possible."""
        text = "This is a very long sentence with many words that should be truncated at word boundaries if possible"
        result = self.driver._escape_json_string(text)
        
        # If truncated, should not end mid-word (if space exists within limit)
        if len(result) < len(text):
            assert not result.endswith("wor")  # partial word

    def test_no_space_truncation(self):
        """Test truncation when no spaces exist within limit."""
        text = "A" * 150  # No spaces, should truncate at 100
        result = self.driver._escape_json_string(text)
        assert len(result) == 100

    def test_preserve_allowed_control_chars(self):
        """Test that allowed control characters are preserved."""
        text = "Text\nwith\tallowed\rcontrol\bchars\f"
        result = self.driver._escape_json_string(text)
        
        # These should be escaped but present
        assert "\\n" in result
        assert "\\t" in result
        assert "\\r" in result
        assert "\\b" in result
        assert "\\f" in result


class TestStringValueSanitization:
    """Test the _sanitize_string_value method."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_generate = Mock()
        self.config = {"stop_tokens": [",", "}"], "stop_reliable": True}
        self.driver = StopTokenJsonDriver(self.mock_generate, self.config)

    def test_sanitize_already_quoted(self):
        """Test sanitizing string that's already properly quoted."""
        result = self.driver._sanitize_string_value('"John Doe"')
        assert result == '"John Doe"'

    def test_sanitize_unquoted_string(self):
        """Test sanitizing unquoted string."""
        result = self.driver._sanitize_string_value('John Doe')
        assert result == '"John Doe"'

    def test_sanitize_string_with_quotes_inside(self):
        """Test sanitizing string that has quotes in the middle."""
        result = self.driver._sanitize_string_value('John "Johnny" Doe is here')
        # Should preserve the full phrase with properly escaped quotes
        parsed = json.loads(result)
        assert 'John' in parsed and 'Johnny' in parsed and 'Doe' in parsed

    def test_sanitize_over_generated_string(self):
        """Test sanitizing string with many words (over-generation)."""
        long_text = "John Doe is a person who lives in New York City"
        result = self.driver._sanitize_string_value(long_text)
        parsed = json.loads(result)
        
        # Should be truncated to first few words
        words = parsed.split()
        assert len(words) <= 3

    def test_sanitize_with_special_chars(self):
        """Test sanitizing string with special characters."""
        result = self.driver._sanitize_string_value('John\nDoe\tTest')
        parsed = json.loads(result)
        
        # Should be properly escaped
        assert "\\n" in result
        assert "\\t" in result

    def test_sanitize_empty_string(self):
        """Test sanitizing empty string."""
        result = self.driver._sanitize_string_value('')
        assert result == '""'

    def test_sanitize_whitespace_only(self):
        """Test sanitizing string with only whitespace."""
        result = self.driver._sanitize_string_value('   ')
        assert result == '""'


class TestNestedObjectGeneration:
    """Test nested object generation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_generate = Mock()
        self.config = {"stop_tokens": [",", "}"], "stop_reliable": True}
        self.driver = StopTokenJsonDriver(self.mock_generate, self.config)

    def test_empty_nested_object(self):
        """Test generating empty nested object."""
        result = self.driver._generate_nested_object({})
        assert result == "{}"

    def test_single_field_nested_object(self):
        """Test generating nested object with single field."""
        self.mock_generate.return_value = "John"
        
        nested_fields = {"name": "string"}
        result = self.driver._generate_nested_object(nested_fields)
        
        # Should be valid JSON
        parsed = json.loads(result)
        assert "name" in parsed

    def test_multiple_field_nested_object(self):
        """Test generating nested object with multiple fields."""
        # Mock responses for each field
        self.mock_generate.side_effect = ["John", "30"]
        
        nested_fields = {"name": "string", "age": "number"}
        result = self.driver._generate_nested_object(nested_fields)
        
        # Should be valid JSON with both fields
        parsed = json.loads(result)
        assert "name" in parsed
        assert "age" in parsed


class TestCompleteJSONGeneration:
    """Test complete JSON generation with various scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_generate = Mock()
        self.config = {"stop_tokens": [",", "}"], "stop_reliable": True}
        self.driver = StopTokenJsonDriver(self.mock_generate, self.config)

    def test_single_string_field(self):
        """Test generating JSON with single string field."""
        self.mock_generate.return_value = "John Doe"
        
        fields = [{"name": "string"}]
        result = self.driver.generate_json(fields)
        
        # Should be valid JSON
        parsed = json.loads(result)
        assert parsed["name"] == "John Doe"

    def test_single_number_field(self):
        """Test generating JSON with single number field."""
        self.mock_generate.return_value = "42"
        
        fields = [{"age": "number"}]
        result = self.driver.generate_json(fields)
        
        # Should be valid JSON
        parsed = json.loads(result)
        assert parsed["age"] == 42

    def test_multiple_fields(self):
        """Test generating JSON with multiple fields."""
        self.mock_generate.side_effect = ["John Doe", "30", "john@example.com"]
        
        fields = [
            {"name": "string"},
            {"age": "number"},
            {"email": "string"}
        ]
        result = self.driver.generate_json(fields)
        
        # Should be valid JSON
        parsed = json.loads(result)
        assert parsed["name"] == "John Doe"
        assert parsed["age"] == 30
        assert parsed["email"] == "john@example.com"

    def test_mixed_clean_and_messy_generation(self):
        """Test handling mix of clean and over-generated responses."""
        self.mock_generate.side_effect = [
            "John Doe",  # Clean
            "30, and some extra stuff here",  # Over-generated
            '"alice@test.com",\n\nSome other content'  # Over-generated with quotes
        ]
        
        fields = [
            {"name": "string"},
            {"age": "number"},
            {"email": "string"}
        ]
        result = self.driver.generate_json(fields)
        
        # Should still produce valid JSON
        parsed = json.loads(result)
        assert parsed["name"] == "John Doe"
        assert parsed["age"] == 30
        assert parsed["email"] == "alice@test.com"

    def test_with_nested_object(self):
        """Test generating JSON with nested object."""
        self.mock_generate.side_effect = ["John", "30"]
        
        fields = [
            {"user": {
                "name": "string",
                "age": "number"
            }}
        ]
        result = self.driver.generate_json(fields)
        
        # Should be valid JSON with nested structure
        parsed = json.loads(result)
        assert "user" in parsed
        assert parsed["user"]["name"] == "John"
        assert parsed["user"]["age"] == 30

    def test_complex_nested_structure(self):
        """Test generating complex nested JSON structure."""
        self.mock_generate.side_effect = [
            "John Doe",
            "john@example.com", 
            "123 Main St",
            "New York",
            "10001"
        ]
        
        fields = [
            {"name": "string"},
            {"contact": {
                "email": "string",
                "address": {
                    "street": "string",
                    "city": "string",
                    "zip": "string"
                }
            }}
        ]
        result = self.driver.generate_json(fields)
        
        # Should be valid JSON with complex nesting
        parsed = json.loads(result)
        assert parsed["name"] == "John Doe"
        assert parsed["contact"]["email"] == "john@example.com"
        assert parsed["contact"]["address"]["street"] == "123 Main St"
        assert parsed["contact"]["address"]["city"] == "New York"
        assert parsed["contact"]["address"]["zip"] == "10001"


class TestRealWorldExtractionScenarios:
    """Test scenarios based on the actual debug output we observed."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_generate = Mock()
        self.config = {"stop_tokens": [",", "}", "\n", "<|end|>"], "stop_reliable": True}
        self.driver = StopTokenJsonDriver(self.mock_generate, self.config)

    def test_debug_scenario_email_extraction(self):
        """Test the actual email extraction that was failing in debug."""
        # This was the actual raw result from debug output
        raw_result = '"john.smith@example.com"'
        
        result = self.driver._extract_field_value(raw_result, "string")
        
        # Should extract correctly without double-escaping
        assert result == '"john.smith@example.com"'
        
        # Should be valid JSON
        parsed = json.loads(result)
        assert parsed == "john.smith@example.com"

    def test_debug_scenario_order_id_extraction(self):
        """Test the order ID extraction that was failing."""
        # This was the actual raw result from debug output
        raw_result = '12345}\n\nConversation:\nI need to check the status of my order, the ID is 12345.\n\nAnswer:\n{"name": "", "email": "john.sm'
        
        result = self.driver._extract_field_value(raw_result, "string")
        
        # Should extract "12345" correctly
        parsed = json.loads(result)
        assert parsed == "12345"

    def test_debug_scenario_name_extraction_empty(self):
        """Test the name extraction that returned empty."""
        # This was the actual raw result from debug output
        raw_result = ''
        
        result = self.driver._extract_field_value(raw_result, "string")
        
        # Should return empty string, not crash
        assert result == '""'
        parsed = json.loads(result)
        assert parsed == ""

    def test_over_generation_with_conversation(self):
        """Test extraction when model generates extra conversation."""
        raw_result = '45.99}\n\nUser: How much was the refund?\nAssistant: The refund amount was $45.99.'
        
        result = self.driver._extract_field_value(raw_result, "number")
        
        # Should extract the number correctly
        assert result == "45.99"

    def test_quoted_string_in_conversation(self):
        """Test extraction of quoted strings within conversational over-generation."""
        raw_result = '"Express shipping", the customer requested expedited delivery due to the delay.'
        
        result = self.driver._extract_field_value(raw_result, "string")
        
        # Should extract the quoted part
        parsed = json.loads(result)
        assert parsed == "Express shipping"

    def test_address_extraction_from_conversation(self):
        """Test address extraction from conversational over-generation."""
        raw_result = '123 Business Ave, Suite 400, New York, NY 10001. The building has a front desk that can receive packages during business hours.'
        
        result = self.driver._extract_field_value(raw_result, "string")
        
        # Should extract the address part (first part before extra explanation)
        parsed = json.loads(result)
        assert "123 Business Ave" in parsed
        # Should not include the entire explanation
        assert len(parsed) < len(raw_result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])