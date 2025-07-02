#!/usr/bin/env python3
"""
Realistic Conversation Benchmark

Tests JSON extraction from complex multi-turn conversations between agents and customers.
Uses proper chat templates for each model and realistic conversation scenarios.
"""

import time
import json
import statistics
import sys
import os
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_chat_template_conversation(tokenizer, conversation_turns):
    """Convert conversation turns to proper chat template format."""
    messages = []
    for i, (role, content) in enumerate(conversation_turns):
        messages.append({"role": role, "content": content})
    
    # Apply chat template
    formatted_conversation = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False
    )
    return formatted_conversation

def create_realistic_conversations():
    """Create realistic multi-turn conversations for testing."""
    
    conversations = {
        "customer_support_ticket": {
            "turns": [
                ("user", "Hi, I'm having trouble with my order #12345. It was supposed to arrive yesterday but I haven't received anything yet."),
                ("assistant", "I'm sorry to hear about the delay with your order. Let me look that up for you. Can you please confirm the email address associated with your account?"),
                ("user", "Sure, it's john.smith@email.com. I ordered a laptop and some accessories last week."),
                ("assistant", "Thank you John. I can see your order here. It looks like there was a shipping delay due to weather conditions. Your package is currently in transit and should arrive by tomorrow evening. Would you like me to upgrade you to express shipping at no charge for the inconvenience?"),
                ("user", "That would be great, thank you! Also, I noticed I was charged twice for the laptop case. Can you check that?"),
                ("assistant", "I see the duplicate charge - that was a processing error on our end. I'll refund the extra $45.99 immediately. You should see it back in your account within 2-3 business days. Is there anything else I can help you with today?"),
                ("user", "Perfect! One last thing - can I change the delivery address? I'll be at my office tomorrow instead of home."),
                ("assistant", "Absolutely! I can update that for you. What's the new delivery address you'd like to use?"),
                ("user", "123 Business Ave, Suite 400, New York, NY 10001. The building has a front desk that can receive packages."),
                ("assistant", "Perfect! I've updated your delivery address and notes about the front desk. Your order will now be delivered to 123 Business Ave, Suite 400, with express shipping. You'll receive tracking updates via email. Is there anything else I can assist you with?"),
                ("user", "That's everything, thank you so much for your help!")
            ],
            "extract_schema": {
                "type": "object",
                "properties": {
                    "customer_info": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "email": {"type": "string"},
                            "order_id": {"type": "string"}
                        },
                        "required": ["name", "email", "order_id"]
                    },
                    "primary_issue": {
                        "type": "object",
                        "properties": {
                            "issue_type": {"type": "string"},
                            "resolution": {"type": "string"},
                            "refund_amount": {"type": "number"}
                        },
                        "required": ["issue_type", "resolution", "refund_amount"]
                    },
                    "delivery_info": {
                        "type": "object",
                        "properties": {
                            "new_address": {"type": "string"},
                            "shipping_upgrade": {"type": "string"},
                            "special_instructions": {"type": "string"}
                        },
                        "required": ["new_address", "shipping_upgrade", "special_instructions"]
                    }
                },
                "required": ["customer_info", "primary_issue", "delivery_info"]
            },
            "prefilled_fields": [
                {"customer_info": {
                    "name": "string",
                    "email": "string", 
                    "order_id": "string"
                }},
                {"primary_issue": {
                    "issue_type": "string", 
                    "resolution": "string", 
                    "refund_amount": "number"
                }},
                {"delivery_info": {
                    "new_address": "string",
                    "shipping_upgrade": "string",
                    "special_instructions": "string"
                }}
            ]
        },
        
        "technical_support_session": {
            "turns": [
                ("user", "My website keeps going down and I'm losing customers. This is the third time this month!"),
                ("assistant", "I understand your frustration - website downtime is critical for business. Let me help you get this resolved quickly. Can you tell me your domain name and when you first noticed the issue?"),
                ("user", "It's mystore.com and it started around 2 PM today. Users are getting a 500 error when they try to checkout."),
                ("assistant", "Thanks for that information. I'm checking our server status now. It looks like your site is on server cluster US-East-2. Let me run some diagnostics. Can you tell me if you made any recent changes to your website or plugins?"),
                ("user", "Yes, we updated our payment plugin yesterday to version 3.2.1. Could that be causing it?"),
                ("assistant", "That's very likely the cause. Version 3.2.1 of that plugin has known compatibility issues with some server configurations. I'm going to temporarily roll back your plugin to version 3.1.8 which is stable. This should restore your checkout functionality immediately."),
                ("user", "Okay, please do that. How long will it take? And will this affect any existing orders?"),
                ("assistant", "The rollback will take about 5 minutes and won't affect any existing orders or customer data. I'm implementing it now. While I do this, I'll also check your error logs to see if there are any other related issues."),
                ("user", "Great, thank you. Also, can you set up monitoring so we get alerts if this happens again?"),
                ("assistant", "Absolutely! I'll configure real-time monitoring for your checkout process and payment system. You'll receive email and SMS alerts within 30 seconds of any downtime. I'm also recommending we schedule a maintenance window to properly update your plugins with compatibility testing."),
                ("user", "Perfect. When would be the best time for maintenance? We're busiest during business hours EST."),
                ("assistant", "I recommend Sunday 2-4 AM EST for minimal impact. I'll schedule it for this Sunday and send you a detailed maintenance plan. Your site is back online now - can you test the checkout process?"),
                ("user", "Just tested it and it's working perfectly! Thank you so much for the quick resolution.")
            ],
            "extract_schema": {
                "type": "object", 
                "properties": {
                    "incident_details": {
                        "type": "object",
                        "properties": {
                            "domain": {"type": "string"},
                            "error_type": {"type": "string"},
                            "start_time": {"type": "string"},
                            "affected_functionality": {"type": "string"}
                        },
                        "required": ["domain", "error_type", "start_time", "affected_functionality"]
                    },
                    "root_cause": {
                        "type": "object",
                        "properties": {
                            "component": {"type": "string"},
                            "version": {"type": "string"},
                            "issue_description": {"type": "string"}
                        },
                        "required": ["component", "version", "issue_description"]
                    },
                    "primary_resolution": {
                        "type": "object",
                        "properties": {
                            "action_taken": {"type": "string"},
                            "timeline": {"type": "string"},
                            "success": {"type": "string"}
                        },
                        "required": ["action_taken", "timeline", "success"]
                    },
                    "preventive_measures": {
                        "type": "object",
                        "properties": {
                            "monitoring_setup": {"type": "string"},
                            "maintenance_scheduled": {"type": "string"}, 
                            "maintenance_window": {"type": "string"}
                        },
                        "required": ["monitoring_setup", "maintenance_scheduled", "maintenance_window"]
                    }
                },
                "required": ["incident_details", "root_cause", "primary_resolution", "preventive_measures"]
            },
            "prefilled_fields": [
                {"incident_details": {
                    "domain": "string",
                    "error_type": "string", 
                    "start_time": "string",
                    "affected_functionality": "string"
                }},
                {"root_cause": {
                    "component": "string",
                    "version": "string",
                    "issue_description": "string"
                }},
                {"primary_resolution": {
                    "action_taken": "string", 
                    "timeline": "string",
                    "success": "string"
                }},
                {"preventive_measures": {
                    "monitoring_setup": "string",
                    "maintenance_scheduled": "string", 
                    "maintenance_window": "string"
                }}
            ]
        }
    }
    
    return conversations

def test_realistic_prefilled_json(llm, tokenizer):
    """Test prefilled JSON on realistic conversations."""
    
    print("🎭 Testing Prefilled-JSON on Realistic Conversations")
    print("=" * 65)
    
    try:
        from vllm import SamplingParams
        from driver.stop_token_json_driver import StopTokenJsonDriver
        
        def generate_func(prompt: str, stop_token: str = None) -> str:
            stop_list = [stop_token] if stop_token else None
            params = SamplingParams(
                temperature=0.3,
                max_tokens=50,
                stop=stop_list,
                skip_special_tokens=True
            )
            outputs = llm.generate([prompt], params)
            return outputs[0].outputs[0].text.strip()
        
        config = {"stop_tokens": [",", "}", "\n", "<|end|>"], "stop_reliable": True}
        driver = StopTokenJsonDriver(generate_func, config)
        
        conversations = create_realistic_conversations()
        results = {}
        
        for conv_name, conv_data in conversations.items():
            print(f"\n  Testing {conv_name.replace('_', ' ').title()}...")
            
            # Format conversation with chat template
            formatted_conv = get_chat_template_conversation(tokenizer, conv_data["turns"])
            
            # Create extraction prompt and set context
            extraction_prompt = f"""{formatted_conv}

Extract information from the conversation above. Return ONLY a JSON object with the relevant data. No explanation, no extra text:
{{"""
            
            print(f"    Conversation length: {len(formatted_conv)} chars")
            print(f"    Fields to extract: {len(conv_data['prefilled_fields'])}")
            
            # Create a context-aware generate function for this conversation
            def context_generate_func(prompt: str, stop_token: str = None) -> str:
                # For first field, prepend the conversation context
                if prompt.startswith('{"'):
                    full_prompt = extraction_prompt + prompt[1:]  # Remove the opening brace since we included it
                else:
                    full_prompt = prompt
                    
                stop_list = [stop_token] if stop_token else None
                params = SamplingParams(
                    temperature=0.3,
                    max_tokens=50,
                    stop=stop_list,
                    skip_special_tokens=True
                )
                outputs = llm.generate([full_prompt], params)
                return outputs[0].outputs[0].text.strip()
            
            # Create driver with context-aware generate function
            context_driver = StopTokenJsonDriver(context_generate_func, config)
            
            times = []
            valid_count = 0
            field_accuracy = []
            
            for run in range(2):  # Reduced runs due to complexity
                start = time.time()
                try:
                    result = context_driver.generate_json(conv_data["prefilled_fields"])
                    elapsed = time.time() - start
                    times.append(elapsed)
                    
                    # Validate JSON
                    parsed = json.loads(result)
                    valid_count += 1
                    
                    # Calculate field accuracy (check if major sections exist)
                    expected_sections = len(conv_data["prefilled_fields"])
                    actual_sections = len(parsed.keys())
                    accuracy = min(actual_sections / expected_sections, 1.0)
                    field_accuracy.append(accuracy)
                    
                    print(f"    Run {run+1}: {elapsed:.3f}s ✅ {len(result)} chars")
                    
                except Exception as e:
                    elapsed = time.time() - start
                    times.append(elapsed)
                    field_accuracy.append(0.0)
                    print(f"    Run {run+1}: {elapsed:.3f}s ❌ {str(e)[:50]}...")
            
            results[conv_name] = {
                "avg_time": statistics.mean(times),
                "validity_rate": valid_count / 2,
                "avg_accuracy": statistics.mean(field_accuracy),
                "conversation_length": len(formatted_conv)
            }
            
            print(f"    Summary: {valid_count}/2 valid, {statistics.mean(field_accuracy):.1%} accurate")
        
        return results
        
    except Exception as e:
        print(f"❌ Realistic prefilled-JSON test failed: {e}")
        return {}

def test_realistic_vllm_json_mode(llm, tokenizer):
    """Test VLLM JSON mode on realistic conversations."""
    
    print(f"\n🎯 Testing VLLM JSON Mode on Realistic Conversations")
    print("=" * 65)
    
    try:
        from vllm import SamplingParams
        from vllm.sampling_params import GuidedDecodingParams
        
        conversations = create_realistic_conversations()
        results = {}
        
        for conv_name, conv_data in conversations.items():
            print(f"\n  Testing {conv_name.replace('_', ' ').title()}...")
            
            # Format conversation with chat template
            formatted_conv = get_chat_template_conversation(tokenizer, conv_data["turns"])
            
            # Create extraction prompt
            extraction_prompt = f"""{formatted_conv}

Based on the conversation above, extract the following information in JSON format:"""
            
            print(f"    Conversation length: {len(formatted_conv)} chars")
            
            try:
                guided_decoding_params = GuidedDecodingParams(json=conv_data["extract_schema"])
                params = SamplingParams(
                    temperature=0.3,
                    max_tokens=200,
                    guided_decoding=guided_decoding_params,
                    skip_special_tokens=True
                )
                
                times = []
                valid_count = 0
                schema_compliance = []
                
                for run in range(2):
                    start = time.time()
                    outputs = llm.generate([extraction_prompt], params)
                    elapsed = time.time() - start
                    result = outputs[0].outputs[0].text.strip()
                    times.append(elapsed)
                    
                    print(f"    Run {run+1}: {elapsed:.3f}s → {len(result)} chars")
                    
                    try:
                        parsed = json.loads(result)
                        valid_count += 1
                        
                        # Check schema compliance
                        required_fields = conv_data["extract_schema"].get("required", [])
                        actual_fields = set(parsed.keys()) if isinstance(parsed, dict) else set()
                        if required_fields:
                            compliance = len(set(required_fields) & actual_fields) / len(required_fields)
                        else:
                            compliance = 1.0 if actual_fields else 0.0
                        schema_compliance.append(compliance)
                        
                    except json.JSONDecodeError:
                        schema_compliance.append(0.0)
                
                results[conv_name] = {
                    "avg_time": statistics.mean(times),
                    "validity_rate": valid_count / 2,
                    "schema_compliance": statistics.mean(schema_compliance),
                    "conversation_length": len(formatted_conv)
                }
                
                print(f"    Summary: {valid_count}/2 valid, {statistics.mean(schema_compliance):.1%} compliant")
                
            except Exception as e:
                print(f"    ❌ {conv_name}: {e}")
                results[conv_name] = {
                    "avg_time": 0,
                    "validity_rate": 0,
                    "schema_compliance": 0,
                    "conversation_length": len(formatted_conv)
                }
        
        return results
        
    except Exception as e:
        print(f"❌ VLLM JSON mode test failed: {e}")
        return {}

def test_realistic_simple_prompting(llm, tokenizer):
    """Test simple prompting on realistic conversations."""
    
    print(f"\n📝 Testing Simple Prompting on Realistic Conversations")
    print("=" * 65)
    
    try:
        from vllm import SamplingParams
        
        conversations = create_realistic_conversations()
        results = {}
        
        params = SamplingParams(
            temperature=0.3,
            max_tokens=300,
            skip_special_tokens=True
        )
        
        for conv_name, conv_data in conversations.items():
            print(f"\n  Testing {conv_name.replace('_', ' ').title()}...")
            
            # Format conversation with chat template
            formatted_conv = get_chat_template_conversation(tokenizer, conv_data["turns"])
            
            # Create extraction prompt
            extraction_prompt = f"""{formatted_conv}

Based on the conversation above, extract the key information and format it as a JSON object. Include customer details, issues discussed, and resolutions provided:"""
            
            print(f"    Conversation length: {len(formatted_conv)} chars")
            
            times = []
            valid_count = 0
            over_generation_count = 0
            
            for run in range(2):
                start = time.time()
                outputs = llm.generate([extraction_prompt], params)
                elapsed = time.time() - start
                result = outputs[0].outputs[0].text.strip()
                times.append(elapsed)
                
                print(f"    Run {run+1}: {elapsed:.3f}s → {len(result)} chars")
                
                try:
                    # Look for JSON in the response
                    json_start = result.find('{')
                    if json_start != -1:
                        json_part = result[json_start:]
                        json_end = json_part.rfind('}') + 1
                        if json_end > 0:
                            json_str = json_part[:json_end]
                            parsed = json.loads(json_str)
                            valid_count += 1
                            
                            # Check for over-generation
                            remaining = result[json_start + json_end:].strip()
                            if len(remaining) > 20:  # More lenient for complex responses
                                over_generation_count += 1
                                print(f"      ⚠️  Over-generation: {len(remaining)} extra chars")
                        
                except json.JSONDecodeError:
                    print(f"      ❌ Invalid JSON")
            
            results[conv_name] = {
                "avg_time": statistics.mean(times),
                "validity_rate": valid_count / 2,
                "over_generation_rate": over_generation_count / 2,
                "conversation_length": len(formatted_conv)
            }
            
            print(f"    Summary: {valid_count}/2 valid, {over_generation_count}/2 over-generated")
        
        return results
        
    except Exception as e:
        print(f"❌ Simple prompting test failed: {e}")
        return {}

def analyze_realistic_results(prefilled_results, json_mode_results, simple_results):
    """Analyze results from realistic conversation tests."""
    
    print(f"\n📊 REALISTIC CONVERSATION ANALYSIS")
    print("=" * 80)
    
    approaches = {
        "Prefilled-JSON (Stop Tokens)": prefilled_results,
        "VLLM JSON Mode": json_mode_results,
        "Simple Prompting": simple_results
    }
    
    # Calculate overall metrics
    approach_summaries = {}
    
    for approach_name, results in approaches.items():
        if not results:
            continue
            
        all_validity = []
        all_times = []
        all_accuracy = []
        total_conv_length = 0
        
        for conv_name, conv_data in results.items():
            validity = conv_data.get("validity_rate", 0)
            avg_time = conv_data.get("avg_time", 0)
            accuracy = conv_data.get("avg_accuracy", conv_data.get("schema_compliance", 0))
            conv_length = conv_data.get("conversation_length", 0)
            
            all_validity.append(validity)
            if avg_time > 0:
                all_times.append(avg_time)
            all_accuracy.append(accuracy)
            total_conv_length += conv_length
        
        if all_validity:
            approach_summaries[approach_name] = {
                "overall_validity": statistics.mean(all_validity),
                "overall_time": statistics.mean(all_times) if all_times else 0,
                "overall_accuracy": statistics.mean(all_accuracy),
                "avg_conversation_length": total_conv_length / len(all_validity),
                "scenario_count": len(all_validity)
            }
    
    # Display results
    print(f"\n📋 PERFORMANCE ON COMPLEX CONVERSATIONS:")
    
    for approach_name, summary in approach_summaries.items():
        validity = summary["overall_validity"]
        avg_time = summary["overall_time"]
        accuracy = summary["overall_accuracy"]
        avg_length = summary["avg_conversation_length"]
        count = summary["scenario_count"]
        
        print(f"\n✅ {approach_name}:")
        print(f"   Validity Rate: {validity:.1%}")
        print(f"   Average Time: {avg_time:.3f}s")
        print(f"   Accuracy/Compliance: {accuracy:.1%}")
        print(f"   Avg Conversation Length: {avg_length:.0f} chars")
        print(f"   Scenarios Tested: {count}")
    
    # Detailed breakdown
    print(f"\n📝 DETAILED BREAKDOWN BY CONVERSATION:")
    
    for conv_name in ["customer_support_ticket", "technical_support_session"]:
        print(f"\n  {conv_name.replace('_', ' ').title()}:")
        for approach_name, results in approaches.items():
            if conv_name in results:
                data = results[conv_name]
                validity = data.get("validity_rate", 0)
                time_taken = data.get("avg_time", 0)
                accuracy = data.get("avg_accuracy", data.get("schema_compliance", 0))
                print(f"    {approach_name}: {validity:.0%} valid, {time_taken:.2f}s, {accuracy:.0%} accurate")
    
    # Key insights
    print(f"\n💡 INSIGHTS FOR COMPLEX CONVERSATIONS:")
    
    if approach_summaries:
        # Find best approach
        best_overall = max(approach_summaries.items(), 
                          key=lambda x: (x[1]["overall_validity"] + x[1]["overall_accuracy"]) / 2)
        best_name, best_stats = best_overall
        
        print(f"  🎯 Best for Complex Extraction: {best_name}")
        print(f"  📊 Success Rate: {best_stats['overall_validity']:.1%}")
        print(f"  ⚡ Average Processing Time: {best_stats['overall_time']:.3f}s")
        
        # Check for conversation length impact
        avg_conv_length = statistics.mean([s["avg_conversation_length"] for s in approach_summaries.values()])
        print(f"  📏 Average Conversation Length: {avg_conv_length:.0f} characters")
        
        # Performance comparison
        if len(approach_summaries) > 1:
            sorted_approaches = sorted(approach_summaries.items(), 
                                     key=lambda x: (x[1]["overall_validity"] + x[1]["overall_accuracy"]) / 2, 
                                     reverse=True)
            
            print(f"\n🏆 RANKING FOR REALISTIC SCENARIOS:")
            for i, (name, stats) in enumerate(sorted_approaches):
                medal = ["🥇", "🥈", "🥉"][i] if i < 3 else "  "
                combined_score = (stats["overall_validity"] + stats["overall_accuracy"]) / 2
                print(f"  {medal} {name}: {combined_score:.1%} combined score")

def main():
    print("🎭 Realistic Conversation JSON Extraction Benchmark")
    print("Testing complex multi-turn conversations with proper chat templates")
    print("=" * 75)
    
    try:
        from vllm import LLM, SamplingParams
        
        print("Loading Phi-3.5 GPTQ 4-bit model...")
        llm = LLM(
            model="thesven/Phi-3.5-mini-instruct-GPTQ-4bit",
            max_model_len=2048,  # Increased for longer conversations
            gpu_memory_utilization=0.4,
            enable_prefix_caching=True,
            disable_sliding_window=True,
            trust_remote_code=True,
            dtype="float16"
        )
        
        # Get tokenizer for chat template
        tokenizer = llm.get_tokenizer()
        
        # Test all approaches
        prefilled_results = test_realistic_prefilled_json(llm, tokenizer)
        json_mode_results = test_realistic_vllm_json_mode(llm, tokenizer)
        simple_results = test_realistic_simple_prompting(llm, tokenizer)
        
        # Analyze results
        analyze_realistic_results(prefilled_results, json_mode_results, simple_results)
        
        print(f"\n🎉 Realistic Conversation Benchmark Complete!")
        print(f"📊 Results show performance on complex information extraction tasks")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()