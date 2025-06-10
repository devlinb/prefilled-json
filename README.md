# Prefilled JSON

A Python library that helps low-parameter LLMs generate valid JSON by controlling the generation process through iterative field-by-field completion.

Small/low-parameter LLMs struggle to generate valid JSON, this library fixes that by prefilling JSON field names and stopping generation after a single field has been filled out.

What this does:

1. **Controls the generation process**: The library fills in JSON field names and structure
2. **Letting the LLM focus on values**: The LLM only generates field values
3. **Using stop tokens**: Uses JSON delimiters (like `,`) as stop tokens to regain control after each field value
4. **Ensuring valid structure**: The library maintains proper JSON syntax throughout

## How It Works

I'll be adding VLLM and Langchain plugins soon.

Right now know that this library requires VLLM with prefix caching, and a model that doesn't adhere to a strict back and forth chat template. This only works with LLM models that support resumption of generation in the same assistant turn.

The library attempts to fix up JSON structure by stripping whatever final token the LLM gave and fixing it up to be "correct". Right now that means `,`s and `}`s get fixed up.

## What it doesn't do

Because we are dealing with very small parameter models, a lot of things are not going to work:

1. Fancy JSON schema restrictions on field values
2. Types other than string and number
3. Optional fields.

Currently nested objects are not supported, but I'll add support soon.

## Usage

```python
from driver.json_driver import JsonFieldDriver

# Define your generation function (connects to your LLM)
def my_generate_func(prompt: str, stop_token: str = None) -> str:
    # Your LLM call here - prompt contains the partial JSON so far
    # stop_token tells the LLM when to stop (e.g., "," or None for final field)
    return llm_response

# Define the JSON schema you want
fields = [
    {"name": "string"},
    {"age": "number"}, 
    {"city": "string"}
]

# Create driver and generate
driver = JsonFieldDriver(generate=my_generate_func)
result = driver.generate_json(fields)
# Returns: '{"name": "Alice", "age": 30, "city": "Seattle"}'
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