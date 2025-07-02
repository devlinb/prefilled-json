#!/usr/bin/env python3
"""
KV Cache Performance Benchmark

This benchmark tests VLLM's prefix caching performance under three scenarios:
1. Streaming with backup/resume (simulating prefilled JSON cleanup)
2. Back-and-forth conversation (typical chat usage)
3. Repeated conversation replay (same prefix reprocessing)
"""

import time
import statistics
from typing import List, Dict, Any
from dataclasses import dataclass

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: VLLM not available. Install with 'pip install vllm' to run benchmarks.")

@dataclass
class BenchmarkResult:
    """Results from a KV cache benchmark scenario."""
    scenario: str
    operation: str
    prompt_tokens: int
    generation_tokens: int
    total_time: float
    tokens_per_second: float
    cache_hit_ratio: float = 0.0  # If available from VLLM metrics

# Test data
BASE_CONVERSATION = """User: I'm working on a Python project and need help with async programming. Can you explain how asyncio works?

Assistant: I'd be happy to explain asyncio! Asyncio is Python's built-in library for writing concurrent code using the async/await syntax. Here's how it works:

1. **Event Loop**: The core of asyncio is the event loop, which manages and executes asynchronous tasks.

2. **Coroutines**: Functions defined with `async def` that can be paused and resumed.

3. **Awaitable Objects**: Objects that can be used with the `await` keyword.

Here's a simple example:

```python
import asyncio

async def fetch_data(url):
    # Simulate an async operation
    await asyncio.sleep(1)
    return f"Data from {url}"

async def main():
    # Run multiple operations concurrently
    tasks = [
        fetch_data("api1.com"),
        fetch_data("api2.com"),
        fetch_data("api3.com")
    ]
    results = await asyncio.gather(*tasks)
    print(results)

# Run the async function
asyncio.run(main())
```

User: That's helpful! Can you show me how to handle errors in async code?"""

JSON_GENERATION_CONTEXT = """Customer: I'm CS-789456. CloudSync Pro keeps crashing when syncing files. This is my third call about this! Very frustrated.

Agent: I see your previous 2 tickets. This is high priority - escalating to senior team. They resolve similar issues in 6 hours.

Customer: Perfect, call me on my phone please.

Extract information and generate JSON:"""

class KVCacheBenchmark:
    """Benchmark for testing KV cache performance under different scenarios."""
    
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model_name = model_name
        self.llm = None
        self.results: List[BenchmarkResult] = []
        
    def initialize_model(self):
        """Initialize the VLLM model with prefix caching enabled."""
        print(f"Initializing {self.model_name} with prefix caching...")
        self.llm = LLM(
            model=self.model_name,
            max_model_len=512,  # Reduced to avoid issues
            gpu_memory_utilization=0.6,
            enforce_eager=True,
            enable_prefix_caching=True,  # This is the key setting
            trust_remote_code=True,
            disable_custom_all_reduce=True,  # Additional stability
            use_v2_block_manager=False  # Use V1 block manager
        )
        print("Model initialized successfully.")
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count for a text string."""
        # Simple approximation: ~4 characters per token
        return len(text) // 4
    
    def generate_text(self, prompt: str, max_tokens: int = 50, temperature: float = 0.1) -> tuple[str, float]:
        """Generate text and return output with timing."""
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=["\n\n", "User:", "Assistant:"]
        )
        
        start_time = time.time()
        outputs = self.llm.generate([prompt], sampling_params)
        end_time = time.time()
        
        generated_text = outputs[0].outputs[0].text.strip()
        generation_time = end_time - start_time
        
        return generated_text, generation_time
    
    def scenario_1_streaming_backup_resume(self) -> List[BenchmarkResult]:
        """Scenario 1: Streaming with backup/resume (simulating prefilled JSON cleanup)."""
        print("\n=== SCENARIO 1: Streaming with Backup/Resume ===")
        print("Simulating how prefilled-json library works: generate, backup, clean, resume")
        
        results = []
        base_prompt = JSON_GENERATION_CONTEXT
        
        # Step 1: Initial generation
        print("\nStep 1: Initial JSON generation...")
        initial_prompt = base_prompt + '\n{"customer_id": "'
        generated, gen_time = self.generate_text(initial_prompt, max_tokens=20)
        
        result1 = BenchmarkResult(
            scenario="streaming_backup_resume",
            operation="initial_generation",
            prompt_tokens=self.count_tokens(initial_prompt),
            generation_tokens=self.count_tokens(generated),
            total_time=gen_time,
            tokens_per_second=self.count_tokens(generated) / gen_time if gen_time > 0 else 0
        )
        results.append(result1)
        print(f"Generated: {generated[:50]}...")
        print(f"Time: {gen_time:.3f}s, Speed: {result1.tokens_per_second:.1f} tok/s")
        
        # Step 2: Backup and clean (simulate removing unwanted text)
        print("\nStep 2: Backup and clean generated text...")
        full_generated = initial_prompt + generated
        # Simulate cleaning: remove trailing junk, keep only valid part
        cleaned_value = generated.split('"')[0] if '"' in generated else generated.split(',')[0]
        cleaned_prompt = base_prompt + f'\n{{"customer_id": "{cleaned_value}", "issue_category": "'
        
        # Step 3: Resume generation from cleaned state
        print("\nStep 3: Resume generation after cleanup...")
        resumed, resume_time = self.generate_text(cleaned_prompt, max_tokens=20)
        
        result2 = BenchmarkResult(
            scenario="streaming_backup_resume",
            operation="resume_after_cleanup",
            prompt_tokens=self.count_tokens(cleaned_prompt),
            generation_tokens=self.count_tokens(resumed),
            total_time=resume_time,
            tokens_per_second=self.count_tokens(resumed) / resume_time if resume_time > 0 else 0
        )
        results.append(result2)
        print(f"Resumed: {resumed[:50]}...")
        print(f"Time: {resume_time:.3f}s, Speed: {result2.tokens_per_second:.1f} tok/s")
        
        # Step 4: Continue building JSON iteratively
        current_prompt = cleaned_prompt + resumed.split('"')[0] + '", "priority_level": "'
        continued, continue_time = self.generate_text(current_prompt, max_tokens=20)
        
        result3 = BenchmarkResult(
            scenario="streaming_backup_resume",
            operation="iterative_continuation",
            prompt_tokens=self.count_tokens(current_prompt),
            generation_tokens=self.count_tokens(continued),
            total_time=continue_time,
            tokens_per_second=self.count_tokens(continued) / continue_time if continue_time > 0 else 0
        )
        results.append(result3)
        print(f"Continued: {continued[:50]}...")
        print(f"Time: {continue_time:.3f}s, Speed: {result3.tokens_per_second:.1f} tok/s")
        
        return results
    
    def scenario_2_back_and_forth_conversation(self) -> List[BenchmarkResult]:
        """Scenario 2: Back-and-forth conversation (typical chat usage)."""
        print("\n=== SCENARIO 2: Back-and-forth Conversation ===")
        print("Testing cache efficiency in typical chat patterns")
        
        results = []
        conversation = BASE_CONVERSATION
        
        # Round 1: First response
        print("\nRound 1: First user question...")
        prompt1 = conversation + "\n\nAssistant:"
        response1, time1 = self.generate_text(prompt1, max_tokens=100)
        
        result1 = BenchmarkResult(
            scenario="back_and_forth",
            operation="first_response",
            prompt_tokens=self.count_tokens(prompt1),
            generation_tokens=self.count_tokens(response1),
            total_time=time1,
            tokens_per_second=self.count_tokens(response1) / time1 if time1 > 0 else 0
        )
        results.append(result1)
        print(f"Response time: {time1:.3f}s, Speed: {result1.tokens_per_second:.1f} tok/s")
        
        # Round 2: Follow-up question (should benefit from cache)
        print("\nRound 2: Follow-up question (cache should help)...")
        conversation_r2 = conversation + "\n\nAssistant: " + response1 + "\n\nUser: Can you show me a more complex example with error handling?\n\nAssistant:"
        response2, time2 = self.generate_text(conversation_r2, max_tokens=100)
        
        result2 = BenchmarkResult(
            scenario="back_and_forth",
            operation="cached_followup",
            prompt_tokens=self.count_tokens(conversation_r2),
            generation_tokens=self.count_tokens(response2),
            total_time=time2,
            tokens_per_second=self.count_tokens(response2) / time2 if time2 > 0 else 0
        )
        results.append(result2)
        print(f"Response time: {time2:.3f}s, Speed: {result2.tokens_per_second:.1f} tok/s")
        
        # Round 3: Another follow-up (even more cache benefit)
        print("\nRound 3: Another follow-up (maximum cache benefit)...")
        conversation_r3 = conversation_r2 + response2 + "\n\nUser: What about asyncio.gather vs asyncio.as_completed?\n\nAssistant:"
        response3, time3 = self.generate_text(conversation_r3, max_tokens=100)
        
        result3 = BenchmarkResult(
            scenario="back_and_forth",
            operation="deep_cached_followup",
            prompt_tokens=self.count_tokens(conversation_r3),
            generation_tokens=self.count_tokens(response3),
            total_time=time3,
            tokens_per_second=self.count_tokens(response3) / time3 if time3 > 0 else 0
        )
        results.append(result3)
        print(f"Response time: {time3:.3f}s, Speed: {result3.tokens_per_second:.1f} tok/s")
        
        return results
    
    def scenario_3_repeated_conversation_replay(self) -> List[BenchmarkResult]:
        """Scenario 3: Repeated conversation replay (same prefix reprocessing)."""
        print("\n=== SCENARIO 3: Repeated Conversation Replay ===")
        print("Testing cache with repeated identical prefixes")
        
        results = []
        base_conversation = BASE_CONVERSATION[:400]  # Truncate to reasonable length
        
        # First run: No cache benefit (cold start)
        print("\nRun 1: Cold start (no cache)...")
        prompt = base_conversation + "\n\nAssistant:"
        response1, time1 = self.generate_text(prompt, max_tokens=80)
        
        result1 = BenchmarkResult(
            scenario="repeated_replay",
            operation="cold_start",
            prompt_tokens=self.count_tokens(prompt),
            generation_tokens=self.count_tokens(response1),
            total_time=time1,
            tokens_per_second=self.count_tokens(response1) / time1 if time1 > 0 else 0
        )
        results.append(result1)
        print(f"Cold start time: {time1:.3f}s, Speed: {result1.tokens_per_second:.1f} tok/s")
        
        # Subsequent runs: Should benefit from prefix cache
        for run in range(2, 6):  # Runs 2-5
            print(f"\nRun {run}: Repeated same prefix (cache should help)...")
            # Same prompt as before - prefix caching should kick in
            response, run_time = self.generate_text(prompt, max_tokens=80)
            
            result = BenchmarkResult(
                scenario="repeated_replay",
                operation=f"cached_run_{run}",
                prompt_tokens=self.count_tokens(prompt),
                generation_tokens=self.count_tokens(response),
                total_time=run_time,
                tokens_per_second=self.count_tokens(response) / run_time if run_time > 0 else 0
            )
            results.append(result)
            print(f"Cached run time: {run_time:.3f}s, Speed: {result.tokens_per_second:.1f} tok/s")
            print(f"Speedup vs cold start: {time1/run_time:.2f}x")
        
        return results
    
    def run_all_benchmarks(self) -> Dict[str, List[BenchmarkResult]]:
        """Run all benchmark scenarios and return results."""
        if not VLLM_AVAILABLE:
            print("VLLM not available. Cannot run benchmarks.")
            return {}
        
        self.initialize_model()
        
        all_results = {}
        
        # Run each scenario
        all_results["streaming_backup_resume"] = self.scenario_1_streaming_backup_resume()
        all_results["back_and_forth"] = self.scenario_2_back_and_forth_conversation()
        all_results["repeated_replay"] = self.scenario_3_repeated_conversation_replay()
        
        return all_results
    
    def print_summary(self, all_results: Dict[str, List[BenchmarkResult]]):
        """Print a summary of all benchmark results."""
        print("\n" + "="*80)
        print("KV CACHE BENCHMARK SUMMARY")
        print("="*80)
        
        for scenario_name, results in all_results.items():
            print(f"\n{scenario_name.upper().replace('_', ' ')}:")
            print("-" * 50)
            
            for result in results:
                print(f"  {result.operation:25} | {result.total_time:6.3f}s | {result.tokens_per_second:6.1f} tok/s")
            
            # Calculate averages and improvements
            if len(results) > 1:
                first_time = results[0].total_time
                avg_subsequent = statistics.mean([r.total_time for r in results[1:]])
                improvement = first_time / avg_subsequent if avg_subsequent > 0 else 1.0
                print(f"  Average speedup from caching: {improvement:.2f}x")
        
        print("\nKEY INSIGHTS:")
        print("- Scenario 1 tests iterative JSON generation with cleanup")
        print("- Scenario 2 tests typical chat conversation patterns")
        print("- Scenario 3 tests repeated identical prefix processing")
        print("- Higher speedup ratios indicate better prefix cache effectiveness")

def main():
    """Run the KV cache benchmark."""
    benchmark = KVCacheBenchmark()
    results = benchmark.run_all_benchmarks()
    benchmark.print_summary(results)

if __name__ == "__main__":
    main()