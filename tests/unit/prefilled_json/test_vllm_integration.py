import pytest
from unittest.mock import Mock, patch
from prefilled_json.vllm_integration import (
    VLLMJSONPrefilledPlugin,
    generate_with_json_prefilled,
    ModelCompatibilityError
)


class MockVLLMOutput:
    """Mock VLLM output structure."""
    def __init__(self, text: str):
        self.text = text


class MockVLLMGenerationOutput:
    """Mock VLLM generation output structure."""
    def __init__(self, outputs):
        self.outputs = outputs


class TestVLLMJSONPrefilledPlugin:
    
    def setup_method(self):
        """Setup mock VLLM engine for each test."""
        self.mock_engine = Mock()
        self.mock_engine.model_config = Mock()
        self.mock_engine.model_config.model = "Qwen/Qwen2.5-1.5B-Instruct"
        
        # Mock successful compatibility test
        with patch.object(VLLMJSONPrefilledPlugin, '_test_model_compatibility'):
            self.plugin = VLLMJSONPrefilledPlugin(self.mock_engine)
    
    def test_init_compatible_model(self):
        """Test plugin initialization with compatible model."""
        assert self.plugin.engine == self.mock_engine
        assert isinstance(self.plugin.active_sessions, dict)
        assert len(self.plugin.active_sessions) == 0
    
    @patch('vllm.SamplingParams')
    def test_get_sampling_params(self, mock_sampling_params):
        """Test sampling parameters generation for streaming approach."""
        self.plugin._get_sampling_params()
        
        mock_sampling_params.assert_called_once_with(
            temperature=0.1,
            max_tokens=50,
            stop=None,  # No stop sequences for streaming approach
            skip_special_tokens=True
        )
    
    def test_generate_json_iteratively_simple_fields(self):
        """Test iterative JSON generation with simple fields."""
        # Mock VLLM engine responses with streaming output
        mock_outputs = [
            [MockVLLMGenerationOutput([MockVLLMOutput('"Alice", "age": 30')])],  # name response
            [MockVLLMGenerationOutput([MockVLLMOutput('25, "city": "Seattle"')])]  # age response
        ]
        self.mock_engine.generate.side_effect = mock_outputs
        
        prompt = "Generate user data:"
        fields = [{"name": "string"}, {"age": "number"}]
        
        result = self.plugin.generate_json_iteratively(prompt, fields)
        
        expected = 'Generate user data:\n{"name": "Alice", "age": 25}'
        assert result == expected
        
        # Verify VLLM was called with correct prompts (no stop tokens)
        assert self.mock_engine.generate.call_count == 2


class TestGenerateWithJSONPrefilled:
    
    def setup_method(self):
        """Setup mock VLLM engine."""
        self.mock_engine = Mock()
        self.mock_engine.model_config = Mock()
        self.mock_engine.model_config.model = "Qwen/Qwen2.5-1.5B-Instruct"
    
    @patch('prefilled_json.vllm_integration.VLLMJSONPrefilledPlugin')
    def test_generate_with_json_prefilled_single_prompt(self, mock_plugin_class):
        """Test JSON prefilled generation with single prompt."""
        # Setup mock plugin
        mock_plugin = Mock()
        mock_plugin.generate_json_iteratively.return_value = "Generated JSON response"
        mock_plugin_class.return_value = mock_plugin
        
        # Test single prompt
        result = generate_with_json_prefilled(
            engine=self.mock_engine,
            prompts="Test prompt",
            json_prefilled_fields=[{"name": "string"}]
        )
        
        assert result == ["Generated JSON response"]
        mock_plugin_class.assert_called_once_with(self.mock_engine)
        mock_plugin.generate_json_iteratively.assert_called_once_with(
            "Test prompt", [{"name": "string"}]
        )
    
    def test_generate_without_json_prefilled_fields(self):
        """Test standard generation when no JSON fields provided."""
        # Mock standard VLLM response
        mock_output = MockVLLMGenerationOutput([MockVLLMOutput("Standard response")])
        self.mock_engine.generate.return_value = [mock_output]
        
        result = generate_with_json_prefilled(
            engine=self.mock_engine,
            prompts=["Test prompt"],
            json_prefilled_fields=None
        )
        
        assert result == ["Standard response"]
        self.mock_engine.generate.assert_called_once()