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
- **`tests/unit/driver/test_json_driver.py`**: Comprehensive test suite covering all functionality

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

## Dependencies

- Runtime: No external dependencies (Python 3.8+)
- Development: pytest, black, isort, mypy, flake8, pytest-cov