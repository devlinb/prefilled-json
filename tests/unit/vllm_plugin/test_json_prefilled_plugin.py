import pytest
from unittest.mock import Mock, MagicMock, patch, call
from typing import List, Dict, Any, Optional

from vllm_plugin.json_prefilled_plugin import (
    VLLMJSONPrefilledPlugin, 
    GenerationSession, 
    ModelCompatibilityError,
    generate_with_json_prefilled
)


class MockVLLMOutput:
    """Mock VLLM output structure."""
    def __init__(self, text: str):
        self.text = text


class MockVLLMGenerationOutput:
    """Mock VLLM generation output structure."""
    def __init__(self, outputs: List[MockVLLMOutput]):
        self.outputs = outputs


class TestVLLMJSONPrefilledPlugin:
    
    def setup_method(self):
        """Setup mock VLLM engine for each test."""
        self.mock_engine = Mock()
        self.mock_engine.model_config = Mock()
        self.mock_engine.model_config.model = "meta-llama/Llama-2-7b-hf"
        
    def test_init_compatible_model(self):
        """Test plugin initialization with compatible model."""
        plugin = VLLMJSONPrefilledPlugin(self.mock_engine)
        assert plugin.engine == self.mock_engine
        assert isinstance(plugin.active_sessions, dict)
        assert len(plugin.active_sessions) == 0
    
    def test_init_incompatible_model_raises_error(self):
        """Test that incompatible models raise ModelCompatibilityError."""
        self.mock_engine.model_config.model = "meta-llama/Llama-2-7b-chat-hf"
        
        with pytest.raises(ModelCompatibilityError, match="uses strict chat template"):
            VLLMJSONPrefilledPlugin(self.mock_engine)
    
    def test_model_compatibility_check_patterns(self):
        """Test various model name patterns for compatibility."""
        test_cases = [
            ("meta-llama/Llama-2-7b-hf", True),  # Compatible
            ("microsoft/DialoGPT-medium", True),  # Compatible
            ("meta-llama/Llama-2-7b-chat-hf", False),  # Chat model
            ("microsoft/DialoGPT-medium-instruct", False),  # Instruct model
            ("THUDM/chatglm-6b", False),  # ChatGLM
            ("lmsys/vicuna-7b-v1.5", False),  # Vicuna
        ]
        
        for model_name, should_be_compatible in test_cases:
            self.mock_engine.model_config.model = model_name
            
            if should_be_compatible:
                # Should not raise exception
                plugin = VLLMJSONPrefilledPlugin(self.mock_engine)
                assert plugin is not None
            else:
                # Should raise ModelCompatibilityError
                with pytest.raises(ModelCompatibilityError):
                    VLLMJSONPrefilledPlugin(self.mock_engine)
    
    def test_create_session(self):
        """Test session creation."""
        plugin = VLLMJSONPrefilledPlugin(self.mock_engine)
        
        prompt = "Generate user data:"
        fields = [{"name": "string"}, {"age": "number"}]
        
        session_id = plugin.create_session(prompt, fields)
        
        assert session_id in plugin.active_sessions
        session = plugin.active_sessions[session_id]
        assert session.original_prompt == prompt
        assert session.fields == fields
        assert session.current_field_index == 0
        assert not session.completed
    
    def test_cleanup_session(self):
        """Test session cleanup."""
        plugin = VLLMJSONPrefilledPlugin(self.mock_engine)
        
        # Create session
        session_id = plugin.create_session("test", [{"name": "string"}])
        assert session_id in plugin.active_sessions
        
        # Cleanup session
        plugin.cleanup_session(session_id)
        assert session_id not in plugin.active_sessions
    
    @patch('vllm.SamplingParams')
    def test_get_sampling_params(self, mock_sampling_params):
        """Test sampling parameters generation."""
        plugin = VLLMJSONPrefilledPlugin(self.mock_engine)
        
        stop_sequences = [",", "}"]
        plugin._get_sampling_params(stop_sequences)
        
        mock_sampling_params.assert_called_once_with(
            temperature=0.1,
            max_tokens=100,
            stop=stop_sequences,
            skip_special_tokens=True
        )
    
    def test_generate_json_iteratively_simple_fields(self):
        """Test iterative JSON generation with simple fields."""
        # Mock VLLM engine responses
        mock_outputs = [
            [MockVLLMGenerationOutput([MockVLLMOutput('"Alice"')])],
            [MockVLLMGenerationOutput([MockVLLMOutput('30')])]
        ]
        self.mock_engine.generate.side_effect = mock_outputs
        
        plugin = VLLMJSONPrefilledPlugin(self.mock_engine)
        
        prompt = "Generate user data:"
        fields = [{"name": "string"}, {"age": "number"}]
        
        result = plugin.generate_json_iteratively(prompt, fields)
        
        expected = 'Generate user data:\n{"name": "Alice", "age": 30}'
        assert result == expected
        
        # Verify VLLM was called with correct prompts
        assert self.mock_engine.generate.call_count == 2
        call_args = [call.args[0][0] for call in self.mock_engine.generate.call_args_list]
        assert '{"name": ' in call_args[0]
        assert '{"name": "Alice", "age": ' in call_args[1]
    
    def test_generate_json_iteratively_with_nested_objects(self):
        """Test iterative JSON generation with nested objects."""
        # Mock responses for nested structure
        mock_outputs = [
            [MockVLLMGenerationOutput([MockVLLMOutput('"Alice"')])],  # name
            [MockVLLMGenerationOutput([MockVLLMOutput('"123 Main St"')])],  # address.street
            [MockVLLMGenerationOutput([MockVLLMOutput('"Seattle"')])],  # address.city
            [MockVLLMGenerationOutput([MockVLLMOutput('30')])]  # age
        ]
        self.mock_engine.generate.side_effect = mock_outputs
        
        plugin = VLLMJSONPrefilledPlugin(self.mock_engine)
        
        prompt = "Generate user data:"
        fields = [
            {"name": "string"},
            {"address": {
                "street": "string",
                "city": "string"
            }},
            {"age": "number"}
        ]
        
        result = plugin.generate_json_iteratively(prompt, fields)
        
        expected = 'Generate user data:\n{"name": "Alice", "address": {"street": "123 Main St", "city": "Seattle"}, "age": 30}'
        assert result == expected
    
    def test_generate_json_iteratively_engine_failure(self):
        """Test handling of VLLM engine failures."""
        # Mock engine to return no outputs
        self.mock_engine.generate.return_value = []
        
        plugin = VLLMJSONPrefilledPlugin(self.mock_engine)
        
        with pytest.raises(RuntimeError, match="VLLM generation failed"):
            plugin.generate_json_iteratively("test", [{"name": "string"}])
    
    def test_generate_json_iteratively_empty_output(self):
        """Test handling of empty outputs from VLLM."""
        # Mock engine to return output with no generated text
        mock_output = MockVLLMGenerationOutput([])
        self.mock_engine.generate.return_value = [mock_output]
        
        plugin = VLLMJSONPrefilledPlugin(self.mock_engine)
        
        with pytest.raises(RuntimeError, match="VLLM generation failed"):
            plugin.generate_json_iteratively("test", [{"name": "string"}])


class TestGenerateWithJSONPrefilled:
    
    def setup_method(self):
        """Setup mock VLLM engine for each test."""
        self.mock_engine = Mock()
        self.mock_engine.model_config = Mock()
        self.mock_engine.model_config.model = "meta-llama/Llama-2-7b-hf"
    
    def test_generate_with_json_prefilled_single_prompt(self):
        """Test JSON prefilled generation with single prompt."""
        # Mock VLLM responses
        mock_outputs = [
            [MockVLLMGenerationOutput([MockVLLMOutput('"Alice"')])],
            [MockVLLMGenerationOutput([MockVLLMOutput('25')])]
        ]
        self.mock_engine.generate.side_effect = mock_outputs
        
        result = generate_with_json_prefilled(
            engine=self.mock_engine,
            prompts="Generate user:",
            json_prefilled_fields=[{"name": "string"}, {"age": "number"}]
        )
        
        assert len(result) == 1
        assert 'Generate user:\n{"name": "Alice", "age": 25}' == result[0]
    
    def test_generate_with_json_prefilled_multiple_prompts(self):
        """Test JSON prefilled generation with multiple prompts."""
        # Mock VLLM responses for two separate JSON generations
        mock_outputs = [
            # First prompt responses
            [MockVLLMGenerationOutput([MockVLLMOutput('"Alice"')])],
            [MockVLLMGenerationOutput([MockVLLMOutput('25')])],
            # Second prompt responses
            [MockVLLMGenerationOutput([MockVLLMOutput('"Bob"')])],
            [MockVLLMGenerationOutput([MockVLLMOutput('30')])]
        ]
        self.mock_engine.generate.side_effect = mock_outputs
        
        prompts = ["Generate user 1:", "Generate user 2:"]
        fields = [{"name": "string"}, {"age": "number"}]
        
        results = generate_with_json_prefilled(
            engine=self.mock_engine,
            prompts=prompts,
            json_prefilled_fields=fields
        )
        
        assert len(results) == 2
        assert 'Generate user 1:\n{"name": "Alice", "age": 25}' == results[0]
        assert 'Generate user 2:\n{"name": "Bob", "age": 30}' == results[1]
    
    def test_generate_without_json_prefilled_fields(self):
        """Test fallback to standard VLLM generation when no JSON fields specified."""
        mock_output = MockVLLMGenerationOutput([MockVLLMOutput("Standard response")])
        self.mock_engine.generate.return_value = [mock_output]
        
        result = generate_with_json_prefilled(
            engine=self.mock_engine,
            prompts=["Regular prompt"]
        )
        
        assert len(result) == 1
        assert result[0] == "Standard response"
        
        # Verify standard generate was called
        self.mock_engine.generate.assert_called_once_with(["Regular prompt"])
    
    def test_generate_with_json_prefilled_exception_fallback(self):
        """Test fallback to standard generation when JSON prefilled fails."""
        # Create a mock engine that will cause an exception during plugin creation
        mock_engine_failing = Mock()
        mock_engine_failing.model_config = Mock()
        mock_engine_failing.model_config.model = "meta-llama/Llama-2-7b-hf"  # Compatible model
        
        # Mock standard generation response
        mock_output = MockVLLMGenerationOutput([MockVLLMOutput("Fallback response")])
        mock_engine_failing.generate.return_value = [mock_output]
        
        # Mock the plugin creation to raise an exception
        with patch('vllm_plugin.json_prefilled_plugin.VLLMJSONPrefilledPlugin') as mock_plugin_class:
            mock_plugin_class.side_effect = RuntimeError("Plugin creation failed")
            
            result = generate_with_json_prefilled(
                engine=mock_engine_failing,
                prompts=["Test prompt"],
                json_prefilled_fields=[{"name": "string"}]
            )
        
        assert len(result) == 1
        assert result[0] == "Fallback response"
    
    def test_generate_with_json_prefilled_empty_fields(self):
        """Test generation with empty field list."""
        # For empty fields, the normal generate path should be used
        mock_output = MockVLLMGenerationOutput([MockVLLMOutput("Regular response")])
        self.mock_engine.generate.return_value = [mock_output]
        
        result = generate_with_json_prefilled(
            engine=self.mock_engine,
            prompts=["Generate empty JSON:"],
            json_prefilled_fields=[]
        )
        
        assert len(result) == 1
        assert result[0] == "Regular response"
    
    def test_prompt_type_conversion(self):
        """Test that single string prompt is converted to list."""
        mock_outputs = [
            [MockVLLMGenerationOutput([MockVLLMOutput('"test"')])]
        ]
        self.mock_engine.generate.side_effect = mock_outputs
        
        # Pass string instead of list
        result = generate_with_json_prefilled(
            engine=self.mock_engine,
            prompts="Single prompt:",  # String, not list
            json_prefilled_fields=[{"field": "string"}]
        )
        
        assert len(result) == 1
        assert 'Single prompt:\n{"field": "test"}' == result[0]


class TestGenerationSession:
    
    def test_generation_session_creation(self):
        """Test GenerationSession dataclass creation."""
        session = GenerationSession(
            session_id="test-123",
            original_prompt="Generate data:",
            fields=[{"name": "string"}]
        )
        
        assert session.session_id == "test-123"
        assert session.original_prompt == "Generate data:"
        assert session.fields == [{"name": "string"}]
        assert session.partial_json == ""
        assert session.current_field_index == 0
        assert not session.completed
        assert session.result is None
    
    def test_generation_session_with_custom_values(self):
        """Test GenerationSession with custom initial values."""
        session = GenerationSession(
            session_id="test-456",
            original_prompt="Test:",
            fields=[{"name": "string"}, {"age": "number"}],
            partial_json='{"name": "Alice"',
            current_field_index=1,
            completed=True,
            result='{"name": "Alice", "age": 30}'
        )
        
        assert session.session_id == "test-456"
        assert session.partial_json == '{"name": "Alice"'
        assert session.current_field_index == 1
        assert session.completed
        assert session.result == '{"name": "Alice", "age": 30}'