# Distribution Guide

## Package Structure

This package uses a **single package with optional dependencies** approach:

### Core Package
```bash
pip install prefilled-json
```

Provides:
- `JsonFieldDriver` - Core JSON generation with custom LLMs - relies on stop tokens, not 100% reliable
- `StreamingJsonFieldDriver` - Streams output and backtracks to the last good result
- `FieldType` - Type definitions

### VLLM Integration (Optional)
```bash
pip install prefilled-json[vllm]
```

Additional provides:
- `generate_with_json_prefilled` - High-level VLLM API
- `VLLMJSONPrefilledPlugin` - VLLM plugin class
- Automatic model compatibility detection

## Usage Patterns

### Core Only (No VLLM dependency)
```python
from prefilled_json import JsonFieldDriver, StreamingJsonFieldDriver

def my_llm_generate(prompt, stop_token=None):
    # Your LLM implementation
    return response

# Traditional approach (for LLMs that respect stop tokens)
driver = JsonFieldDriver(generate=my_llm_generate)
result = driver.generate_json([{"name": "string"}, {"age": "number"}])

# Streaming approach (for modern instruction-tuned models)
streaming_driver = StreamingJsonFieldDriver(generate=my_llm_generate)
result = streaming_driver.generate_json([{"name": "string"}, {"age": "number"}])
```

### VLLM Integration (Method 1: Direct Import)
```python
from prefilled_json.vllm_integration import generate_with_json_prefilled
from vllm import LLM

llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")
outputs = generate_with_json_prefilled(
    engine=llm,
    prompts=["Generate user data:"],
    json_prefilled_fields=[{"name": "string"}, {"age": "number"}]
)
```

### VLLM Integration (Method 2: Top-level Import)
```python
from prefilled_json import generate_with_json_prefilled, VLLMJSONPrefilledPlugin
from vllm import LLM

# When VLLM is available, functions are also available at top level
llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")
outputs = generate_with_json_prefilled(
    engine=llm,
    prompts=["Generate user data:"],
    json_prefilled_fields=[{"name": "string"}, {"age": "number"}]
)
```

## Publishing to PyPI

### 1. Build the Package
```bash
pip install build twine
python -m build
```

### 2. Test Upload (TestPyPI)
```bash
python -m twine upload --repository testpypi dist/*
```

### 3. Production Upload
```bash
python -m twine upload dist/*
```

### 4. Installation Testing
```bash
# Test core functionality
pip install prefilled-json
python -c "from prefilled_json import JsonFieldDriver; print('Core works!')"

# Test VLLM integration
pip install prefilled-json[vllm]
python -c "from prefilled_json.vllm import generate_with_json_prefilled; print('VLLM works!')"
```

## Version Management

Update version in:
- `pyproject.toml` (line 7)
- `__init__.py` (line 25)

## Dependencies

**Core**: No dependencies (pure Python)
**VLLM**: Optional dependency on `vllm>=0.2.0`
**Dev**: Test and linting tools

This structure provides:
- ✅ **Separation of concerns**: Core and VLLM are separate
- ✅ **Optional dependencies**: Users only install what they need
- ✅ **Clean imports**: Intuitive package structure
- ✅ **Backward compatibility**: Existing code continues to work