# JSON Generation Benchmark Results

## Overview

This benchmark compares different approaches for generating valid JSON with small language models using the Phi-3.5-mini GPTQ 4-bit quantized model. All tests were conducted on the same hardware with identical model configurations to ensure fair comparison.

## Final Comprehensive Results (December 2024)

**Realistic Conversation Benchmark** - Multi-turn conversations with ~1000 tokens of context:

| Approach | Success Rate | Average Time (s) | JSON Validity | Field Accuracy |
|----------|--------------|------------------|---------------|----------------|
| **Prefilled-JSON (Stop Tokens)** | 100.0% | 2.333 | 100.0% | 100.0% |
| **VLLM JSON Mode** | 50.0% | 1.667 | 50.0% | 50.0% |
| **Simple Prompting** | 0.0% | 2.250 | 0.0% | 0.0% |

## Legacy Test Results (Basic Scenarios)

| Approach | Validity Rate | Average Time (s) | Accuracy/Compliance | Overall Score |
|----------|---------------|------------------|---------------------|---------------|
| **VLLM Constrained Generation** | 100.0% | 0.247 | 100.0% | 0.975 |
| **Prefilled-JSON (Stop Tokens)** | 100.0% | 0.320 | 100.0% | 0.968 |
| **VLLM JSON Mode** | 75.0% | 0.332 | 75.0% | 0.717 |
| **Simple Prompting** | 75.0% | 0.791 | 0.0% | 0.296 |

## Detailed Performance by Scenario

### Prefilled-JSON (Stop Tokens)
| Scenario | Validity | Time (s) | Accuracy |
|----------|----------|----------|----------|
| Simple Fields | 3/3 PASS | 0.253 | 100% |
| Multiple Fields | 3/3 PASS | 0.313 | 100% |
| Complex Naming | 3/3 PASS | 0.280 | 100% |
| Nested Object | 3/3 PASS | 0.434 | 100% |

### VLLM JSON Mode (Guided Generation)
| Scenario | Validity | Time (s) | Compliance |
|----------|----------|----------|------------|
| Simple Fields | 3/3 PASS | 0.196 | 100% |
| Multiple Fields | 3/3 PASS | 0.284 | 100% |
| Complex Naming | 3/3 PASS | 0.337 | 100% |
| Nested Object | 0/3 FAIL | 0.509 | 0% |

### VLLM Constrained Generation (Regex)
| Scenario | Validity | Time (s) | Pattern Match |
|----------|----------|----------|---------------|
| Simple Fields | 3/3 PASS | 0.138 | 100% |
| Multiple Fields | 3/3 PASS | 0.356 | 100% |

### Simple Prompting
| Scenario | Validity | Time (s) | Over-generation |
|----------|----------|----------|-----------------|
| Simple Fields | 1/3 PARTIAL | 0.791 | 33% |
| Multiple Fields | 3/3 PASS | 0.791 | 100% |
| Complex Naming | 3/3 PASS | 0.791 | 100% |
| Nested Object | 2/3 PARTIAL | 0.790 | 67% |

## Key Findings

### Winner: Prefilled-JSON (Stop Tokens)
- **Best for**: All JSON generation scenarios, especially complex conversations
- **Strengths**: 100% success rate on realistic conversations, handles nested objects perfectly
- **Performance**: Reliable 2.333s average for complex multi-turn conversations
- **Robust**: Works with long context windows and maintains accuracy

### Runner-up: VLLM JSON Mode (Limited)
- **Best for**: Simple JSON schemas in basic scenarios
- **Limitations**: Only 50% success rate on realistic conversations
- **Issues**: Struggles with complex context and multi-turn conversations

### Poor Performance: Simple Prompting
- **Issues**: 0% success rate on realistic conversations
- **Not recommended**: Completely unreliable for production JSON generation

## Technical Details

### Model Configuration
- **Model**: `thesven/Phi-3.5-mini-instruct-GPTQ-4bit`
- **GPU Memory**: 40% utilization (~2.1GB)
- **Prefix Caching**: Enabled with `disable_sliding_window=True`
- **Quantization**: GPTQ 4-bit (70% memory reduction vs full model)

### Hardware
- **GPU**: NVIDIA L4 (23GB VRAM)
- **VLLM Version**: 0.9.1
- **Framework**: PyTorch with Flash Attention

## Recommendations

### For Production Use

1. **Fixed Schema Applications** → Use **VLLM Constrained Generation**
   - API responses with known structure
   - Data export formats
   - Configuration files

2. **Dynamic Schema Applications** → Use **Prefilled-JSON**
   - User-defined data structures
   - Nested object generation
   - Variable field requirements

3. **Avoid Simple Prompting** → Reliability issues make it unsuitable for production

### Performance vs Flexibility Trade-off

```
High Performance ←→ High Flexibility
Constrained Gen ← JSON Mode ← Prefilled-JSON ← Simple Prompting
     0.247s         0.332s       0.320s         0.791s
```

## Conclusion

The comprehensive benchmark reveals that **prefilled-JSON with stop tokens is the clear winner for production JSON generation**. While legacy benchmarks showed competitive performance across different approaches, realistic conversation testing demonstrates the superiority of prefilled-JSON:

- **100% success rate** on complex multi-turn conversations
- **Perfect JSON validity** across all test scenarios  
- **Robust handling** of long context and nested structures
- **Reliable performance** even with challenging conversational prompts

VLLM JSON mode, while faster in simple scenarios, fails 50% of the time on realistic conversations. Simple prompting is completely unreliable for production use.

**Recommendation**: Use prefilled-JSON for all production JSON generation needs requiring reliability and accuracy.