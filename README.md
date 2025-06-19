# Prefilled JSON

A Python library that helps low-parameter LLMs generate valid JSON by controlling the generation process through iterative field-by-field completion.

Small/low-parameter LLMs struggle to generate valid JSON, this library fixes that by prefilling JSON field names and using pattern matching to extract clean field values.

What this does:

1. **Controls the generation process**: The library fills in JSON field names and structure
2. **Letting the LLM focus on values**: The LLM only generates field values
3. **Using pattern extraction**: Uses regex patterns to extract precise field values from model output
4. **Ensuring valid structure**: The library maintains proper JSON syntax throughout

## How It Works

The library uses a **streaming approach with pattern matching** for modern LLMs. Instead of relying on stop tokens (which modern instruction-tuned models often ignore), it allows models to over-generate content and then extracts precise field values using regex patterns. This approach works reliably with state-of-the-art models like Qwen, Phi-3.5, and Gemma.

## Architecture

### Core Components

- **StreamingJsonFieldDriver**: Pattern-matching based JSON generation that works with modern models
- **JsonFieldDriver**: Traditional stop-token based driver (for custom implementations)
- **VLLM Plugin**: Seamless integration with VLLM using the streaming approach

## VLLM Integration

The library includes a VLLM plugin with **intelligent model compatibility detection** that tests actual technical capabilities rather than relying on naming patterns.

### Model Compatibility

The plugin automatically detects compatible models by testing:
- Assistant message resumption capabilities
- Chat template flexibility  
- `continue_final_message` parameter support
- Custom template acceptance

#### ✅ **Highly Compatible Models (Under 8B Parameters)**

**Recommended Chat Models:**
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

**Base Models (Maximum Flexibility):**
```python
"meta-llama/Llama-3.2-1B"        # 1B - Latest Llama base
"meta-llama/Llama-3.2-3B"        # 3B - Balanced base model
"TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T" # 1.1B - Ultra efficient
"microsoft/DialoGPT-medium"      # 345M - Proven compatibility
```

#### ❌ **Incompatible Models**
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

The library attempts to fix up JSON structure by stripping whatever final token the LLM gave and fixing it up to be "correct". Right now that means `,`s and `}`s get fixed up.

## What it doesn't do

Because we are dealing with very small parameter models, a lot of things are not going to work:

1. Fancy JSON schema restrictions on field values
2. Types other than string and number
3. Optional fields.

Nested objects are now supported! See examples below.

## Usage

```python
from driver.json_driver import JsonFieldDriver

# Define your generation function (connects to your LLM)
def my_generate_func(prompt: str, stop_token: str = None) -> str:
    # Your LLM call here - prompt contains the partial JSON so far
    # stop_token tells the LLM when to stop (e.g., "," or None for final field)
    return llm_response

# Basic usage with flat fields
fields = [
    {"name": "string"},
    {"age": "number"},
    {"city": "string"}
]

driver = JsonFieldDriver(generate=my_generate_func)
result = driver.generate_json(fields)
# Returns: '{"name": "Alice", "age": 30, "city": "Seattle"}'

# Nested object usage
nested_fields = [
    {"name": "string"},
    {"address": {
        "street": "string",
        "city": "string",
        "zip": "number"
    }},
    {"age": "number"}
]

result = driver.generate_json(nested_fields)
# Returns: '{"name": "Alice", "address": {"street": "123 Main St", "city": "Seattle", "zip": 98101}, "age": 30}'
```

## Nested Objects

You can create arbitrarily deep nested structures:

```python
# Deeply nested example
complex_fields = [
    {"user": {
        "profile": {
            "name": "string",
            "contact": {
                "email": "string",
                "phone": "string"
            }
        },
        "settings": {
            "theme": "string",
            "notifications": "number"
        }
    }},
    {"timestamp": "number"}
]
```

## Generation Process

For the example above, the library works as follows:

1. **Step 1**: Sends `'{"name": '` to LLM with stop token `','`
   - LLM generates: `'"Alice",'` 
   - Library strips trailing comma, gets: `'"Alice"'`

2. **Step 2**: Sends `'{"name": "Alice", "age": '` to LLM with stop token `','`
   - LLM generates: `'30,'`
   - Library strips trailing comma, gets: `'30'`

3. **Step 3**: Sends `'{"name": "Alice", "age": 30, "city": '` to LLM with no stop token
   - LLM generates: `'"Seattle"}'`
   - Library strips trailing brace, gets: `'"Seattle"'`

4. **Final result**: `'{"name": "Alice", "age": 30, "city": "Seattle"}'`

## Features

- **Field Types**: Supports `"string"` and `"number"` field types
- **Automatic Quoting**: Automatically adds quotes to string values if missing
- **Validation**: Validates that number fields contain valid numeric values
- **Error Handling**: Clear error messages for invalid field types or malformed values
- **Flexible Integration**: Works with any LLM that accepts prompt + stop token

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

## Why This Approach Works

1. **Reduced Cognitive Load**: LLM only needs to generate individual values, not entire JSON structure
2. **Guaranteed Syntax**: Library ensures proper JSON formatting, quotes, commas, and brackets
3. **Stop Token Control**: Prevents LLM from over-generating or breaking JSON structure
4. **Incremental Building**: Each step builds on valid JSON, making it easier for LLM to understand context