"""VLLM Plugin for JSON Prefilled Generation

This plugin extends VLLM to support iterative JSON generation for small parameter models.
"""

from .json_prefilled_plugin import VLLMJSONPrefilledPlugin, generate_with_json_prefilled

__all__ = ['VLLMJSONPrefilledPlugin', 'generate_with_json_prefilled']