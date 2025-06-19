# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Python library called `prefilled-json` that helps low-parameter LLMs generate valid JSON by controlling the generation process through iterative field-by-field completion. The library fills in JSON field names and structure, letting the LLM focus only on generating field values while using stop tokens to maintain control.

## Development Commands

### Installation and Setup
```bash
pip install -e .
pip install -e ".[dev]"  # Install with dev dependencies
```

### Testing
```bash
pytest                    # Run all tests
pytest tests/unit/        # Run unit tests only
pytest -v                 # Verbose output
```

### Code Quality
```bash
black .                   # Format code
isort .                   # Sort imports
mypy driver/              # Type checking
flake8 .                  # Linting
```

## Architecture

### Core Components

- **`driver/json_driver.py`**: Main `JsonFieldDriver` class that orchestrates JSON generation
- **`vllm_plugin/json_prefilled_plugin.py`**: VLLM integration plugin for seamless JSON generation
- **`tests/unit/driver/test_json_driver.py`**: Comprehensive test suite for core driver
- **`tests/unit/vllm_plugin/test_json_prefilled_plugin.py`**: Test suite for VLLM plugin
- **`examples/vllm_plugin_example.py`**: Usage examples for VLLM integration

### Key Architecture Patterns

1. **Iterative Generation**: The driver builds JSON incrementally, prompting the LLM for each field value separately
2. **Stop Token Control**: Uses JSON delimiters (`,`) as stop tokens to regain control after each field
3. **Nested Object Support**: Recursively handles nested objects by treating them as separate generation contexts
4. **Value Cleanup**: Strips trailing punctuation (`,`, `}`) that small LLMs often add incorrectly

### Generation Flow

1. Constructs partial JSON prompt with field name: `{"name": `
2. Calls LLM with appropriate stop token (`,` for non-final fields, `None` for final)
3. Cleans up generated value (strips trailing punctuation, adds quotes to strings)
4. Validates number fields and handles type-specific formatting
5. Continues iteratively until all fields are complete

### Field Specifications

Fields are specified as dictionaries with single key-value pairs:
- `{"field_name": "string"}` - String field
- `{"field_name": "number"}` - Number field  
- `{"field_name": {...}}` - Nested object with sub-fields

### Error Handling

- Validates field specifications have exactly one key
- Validates generated numbers are valid numeric values
- Raises `ValueError` for invalid field types or malformed values
- Provides clear error messages for debugging

## VLLM Plugin

The `vllm_plugin` directory contains a complete VLLM integration that provides:

### Key Features
- **Model Compatibility Checking**: Automatically validates that the model supports assistant message resumption
- **Seamless API Integration**: Extends VLLM's generate method with `json_prefilled_fields` parameter
- **Graceful Fallback**: Falls back to standard VLLM generation if JSON prefilled mode fails
- **Session Management**: Tracks iterative generation state across multiple field completions

### Usage Pattern
```python
from vllm import LLM
from vllm_plugin import generate_with_json_prefilled

llm = LLM(model="meta-llama/Llama-2-7b-hf", enable_prefix_caching=True)

outputs = generate_with_json_prefilled(
    engine=llm,
    prompts=["Generate user data:"],
    json_prefilled_fields=[{"name": "string"}, {"age": "number"}]
)
```

### Compatible Models
- Base models (e.g., `meta-llama/Llama-2-7b-hf`)
- Models without strict chat templates
- Models that support resuming assistant message generation

### Incompatible Models
- Chat models (e.g., `-chat`, `-instruct` variants)
- Models with strict turn-taking chat templates
- Models that enforce conversation patterns

## Dependencies

- Runtime: No external dependencies for core driver (Python 3.8+)
- VLLM Plugin: Requires `vllm` package
- Development: pytest, black, isort, mypy, flake8, pytest-cov