"""Type definitions for prefilled JSON generation."""

from typing import List, Literal, Callable, Optional, Dict, Union, Any

# Field type specification
FieldType = Union[Literal["string", "number"], Dict[str, Any]]

# Generate function type for LLM integration
GenerateFunc = Callable[[str, Optional[str]], str]