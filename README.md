# Prefilled JSON

A Python library that helps low-parameter LLMs generate valid JSON by controlling the generation process through iterative field-by-field completion.

Small/low-parameter LLMs struggle to generate valid JSON, this library helps them out  by prefilling JSON field names and using pattern matching to extract clean field values.

What this does:

1. **Controls the generation process**: The library fills in JSON field names and structure
2. **Letting the LLM focus on values**: The LLM only generates field values
3. **Using pattern extraction**: Uses regex patterns to extract precise field values from model output
4. **Ensuring valid structure**: The library maintains proper JSON syntax throughout

## How It Works

The library uses a **streaming approach with pattern matching** for modern LLMs. Stop tokens are not 100% reliable across models, using streaming output is the only way I was able to make this 100% reliable. Accordingly, models may over-generate content, which this library then backtracks through and  extracts precise field values using regex patterns. This approach works reliably with state-of-the-art models like Qwen, Phi-3.5, and Gemma. It should work fine even with highly quantized low-param models.

## Architecture

### Core Components

- **StreamingJsonFieldDriver**: Pattern-matching based JSON generation that works with modern models
- **JsonFieldDriver**: Traditional stop-token based driver (for custom implementations)
- **VLLM Plugin**: Seamless integration with VLLM using the streaming approach

## VLLM Integration

The library includes a VLLM plugin with intelligent model compatibility detection that runs a bunch of checks to see if the loaded model is compatible.

### Model Compatibility

The plugin automatically detects compatible models by testing:
- Assistant message resumption capabilities
- Chat template flexibility  
- `continue_final_message` parameter support
- Custom template acceptance

#### Sample Models

**Chat:**
```python
# Qwen models (excellent JSON generation)
"Qwen/Qwen2.5-0.5B-Instruct"     # 0.5B - Ultra lightweight
"Qwen/Qwen2.5-1.5B-Instruct"     # 1.5B - Best balance
"Qwen/Qwen2.5-3B-Instruct"       # 3B - Production ready
"Qwen/Qwen2.5-7B-Instruct"       # 7B - Maximum performance
"Qwen/Qwen2.5-Coder-1.5B-Instruct" # 1.5B - Code/JSON specialized

# Microsoft Phi models (excellent chat flexibility)
"microsoft/phi-2"                 # 2.7B - Versatile base/chat
"microsoft/Phi-3-mini-4k-instruct" # 3.8B - Strong reasoning
"microsoft/Phi-3.5-mini-instruct" # 3.8B - Latest with 128K context

# Google Gemma models (production tested)
"google/gemma-2b-it"             # 2B - Efficient chat
"google/gemma-7b-it"             # 7B - High performance chat
```

**Base Models**
```python
"meta-llama/Llama-3.2-1B"        # 1B - Latest Llama base
"meta-llama/Llama-3.2-3B"        # 3B - Balanced base model
"TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T" # 1.1B - Ultra efficient
"microsoft/DialoGPT-medium"      # 345M - Proven compatibility
```

#### Incompatible Models
Models with rigid chat templates that enforce strict role alternation:
- `meta-llama/Llama-2-7b-chat-hf` (rigid template)
- `meta-llama/Llama-3.1-8B-Instruct` (strict turn-taking)
- Most models with very strict chat formatting

### Quick VLLM Usage

```python
from vllm import LLM
from vllm_plugin import generate_with_json_prefilled

# Initialize with a compatible model (auto-detected)
llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct", enable_prefix_caching=True)

# Generate JSON with simple API
outputs = generate_with_json_prefilled(
    engine=llm,
    prompts=["Generate user data:"],
    json_prefilled_fields=[{"name": "string"}, {"age": "number"}]
)

print(outputs[0])
# Output: Generate user data:
# {"name": "Alice", "age": 30}
```

### Testing Model Compatibility

```python
from vllm import LLM
from vllm_plugin.json_prefilled_plugin import VLLMJSONPrefilledPlugin

def test_model(model_name):
    try:
        llm = LLM(model=model_name, trust_remote_code=True)
        plugin = VLLMJSONPrefilledPlugin(llm)
        print(f"✅ {model_name} is compatible!")
        return True
    except Exception as e:
        print(f"❌ {model_name}: {e}")
        return False

# Test any model
test_model("your-model-here")
```

See `examples/vllm_plugin_example.py` for more detailed usage examples and `TESTING.md` for comprehensive testing instructions.

The library uses pattern matching to extract clean field values from model output, automatically handling over-generation and ensuring valid JSON structure.

## What it doesn't do

Because we focus on reliable JSON generation, some advanced features are not supported:

1. Fancy JSON schema restrictions on field values
2. Types other than string and number (object nesting is supported)
3. Optional fields

## Usage

### VLLM Integration (Recommended)

```python
from vllm import LLM
from vllm_plugin import generate_with_json_prefilled

# Initialize VLLM with any compatible model
llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)

# Generate JSON with simple API
outputs = generate_with_json_prefilled(
    engine=llm,
    prompts=["Create user profile:"],
    json_prefilled_fields=[
        {"name": "string"},
        {"age": "number"},
        {"city": "string"}
    ]
)

print(outputs[0])
# Output: Create user profile:
# {"name": "Alice", "age": 30, "city": "Seattle"}

# Nested object usage
nested_outputs = generate_with_json_prefilled(
    engine=llm,
    prompts=["Generate complete user data:"],
    json_prefilled_fields=[
        {"name": "string"},
        {"address": {
            "street": "string",
            "city": "string",
            "zip": "number"
        }},
        {"age": "number"}
    ]
)

print(nested_outputs[0])
# Output: Generate complete user data:
# {"name": "Alice", "address": {"street": "123 Main St", "city": "Seattle", "zip": 98101}, "age": 30}
```

### Custom Driver Usage (Advanced)

For custom LLM implementations, you can use the core driver directly:

```python
from driver.json_driver import JsonFieldDriver

# Define your generation function
def my_generate_func(prompt: str, stop_token: str = None) -> str:
    # Your LLM call here - stop_token parameter available but not required
    return your_llm_response

# Create driver and generate JSON
driver = JsonFieldDriver(generate=my_generate_func)
result = driver.generate_json([{"name": "string"}, {"age": "number"}])
```

## How the Streaming Approach Works

The library uses a sophisticated pattern-matching approach that works reliably with modern instruction-tuned models:

1. **Step 1**: Sends `'{"name": '` to LLM (no stop tokens)
   - LLM generates: `'"Alice", "age": 30, "city": "Seattle", "email": "alice@example.com"'`
   - Library extracts: `'"Alice"'` using regex pattern matching

2. **Step 2**: Sends `'{"name": "Alice", "age": '` to LLM
   - LLM generates: `'25, "city": "Seattle", "active": true'`
   - Library extracts: `'25'` as the numeric value

3. **Step 3**: Sends `'{"name": "Alice", "age": 25, "city": '` to LLM
   - LLM generates: `'"Seattle"}, "country": "USA"'`
   - Library extracts: `'"Seattle"'` using pattern matching

4. **Final result**: `'{"name": "Alice", "age": 25, "city": "Seattle"}'`

This approach **works with modern instruction-tuned models** because it doesn't fight their tendency to over-generate - instead, it extracts exactly what's needed using robust pattern matching.

## Features

- **Field Types**: Supports `"string"` and `"number"` field types, plus nested objects
- **Pattern Extraction**: Robust regex-based field value extraction from over-generated content
- **Modern Model Support**: Works reliably with instruction-tuned models (Qwen, Phi-3.5, Gemma, etc.)
- **Automatic Validation**: Validates numeric fields and handles string quoting automatically
- **Error Handling**: Clear error messages for invalid field types or malformed values
- **VLLM Integration**: Seamless integration with VLLM using streaming approach
- **Compatibility Detection**: Automatic technical testing of model capabilities

## Installation

```bash
pip install -e .
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black .
isort .

# Type check
mypy driver/
```

## Field Schema Format

Each field is specified as a dictionary with exactly one key-value pair:
- **Key**: The field name (string)
- **Value**: The field type (`"string"` or `"number"`)

```python
fields = [
    {"username": "string"},
    {"score": "number"},
    {"active": "string"}  # booleans can be represented as strings
]
```
