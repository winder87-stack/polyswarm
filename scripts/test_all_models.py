#!/usr/bin/env python3
"""
Test All 5 AI Models Individually

Tests each AI model in the swarm with a simple mathematical prompt
to verify configuration and performance.

Usage:
    python scripts/test_all_models.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv
from termcolor import colored

# Load environment variables
load_dotenv()

from src.models.model_factory import model_factory


def print_header():
    """Print a colorful header."""
    print(colored("🌙 Polymarket AI - Model Testing Suite", "cyan", attrs=["bold"]))
    print(colored("=" * 50, "cyan"))
    print(colored("Testing all 5 AI models individually", "white"))
    print(colored(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "grey"))
    print()


def print_model_status(available_models):
    """Print which models are configured."""
    print(colored("🔑 API Key Status:", "yellow", attrs=["bold"]))

    model_names = {
        "claude": "Claude Sonnet 4.5",
        "google": "Gemini 3 Pro",
        "openai": "GPT-5.2",
        "deepseek": "DeepSeek V3.2",
        "perplexity": "Perplexity Sonar Pro"
    }

    configured_count = 0
    for model_key, is_available in available_models.items():
        status = "✅ Configured" if is_available else "❌ Missing API key"
        color = "green" if is_available else "red"
        name = model_names.get(model_key, model_key)
        print(colored(f"  {name}: {status}", color))
        if is_available:
            configured_count += 1

    print(colored(f"\n🤖 {configured_count}/5 models configured", "cyan"))
    print()


def test_single_model(model_key, model_name, prompt):
    """Test a single AI model."""
    try:
        # Create model instance
        model = model_factory.get_model(model_key, model_name)

        print(colored(f"🧪 Testing {model_key.upper()}...", "blue", attrs=["bold"]))

        # Test the model
        response = model.generate_response(
            system_prompt="You are a helpful AI assistant. Answer questions directly and concisely.",
            user_content=prompt,
            temperature=0.1,  # Low temperature for consistent answers
            max_tokens=50     # Short response expected
        )

        # Display results
        if response.success:
            print(colored("  ✅ Success!", "green"))
            print(colored(f"  📝 Response: {response.content.strip()}", "white"))
            print(colored(f"  ⏱️  Time: {response.response_time:.2f}s", "yellow"))
            print(colored(f"  💰 Cost: ${response.cost_estimate:.4f}", "magenta"))
            return True, response.response_time, response.cost_estimate
        else:
            print(colored(f"  ❌ Failed: {response.error}", "red"))
            return False, 0.0, 0.0

    except Exception as e:
        print(colored(f"  ❌ Error: {str(e)}", "red"))
        return False, 0.0, 0.0


def print_summary(results):
    """Print test summary."""
    print(colored("\n📊 TEST SUMMARY", "cyan", attrs=["bold"]))
    print(colored("=" * 40, "cyan"))

    successful = sum(1 for r in results.values() if r["success"])
    total_time = sum(r["time"] for r in results.values())
    total_cost = sum(r["cost"] for r in results.values())

    print(colored(f"✅ Successful tests: {successful}/5", "green" if successful > 0 else "red"))

    if successful > 0:
        avg_time = total_time / successful
        print(colored(f"⏱️  Average response time: {avg_time:.2f}s", "yellow"))
        print(colored(f"💰 Total cost: ${total_cost:.4f}", "magenta"))

    print(colored("\n🎯 Expected answer: '4' (or equivalent)", "cyan"))

    # Individual results
    print(colored("\n📋 Detailed Results:", "white", attrs=["bold"]))
    for model_key, result in results.items():
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        color = "green" if result["success"] else "red"
        time_str = f"{result['time']:.2f}s" if result["success"] else "N/A"
        cost_str = f"${result['cost']:.4f}" if result["success"] else "N/A"
        print(colored(f"  {model_key.upper()}: {status} | Time: {time_str} | Cost: {cost_str}", color))


def main() -> None:
    """Main test function."""
    print_header()

    # Check which models are available
    available_models = model_factory.check_available_models()
    print_model_status(available_models)

    # Test configuration
    models_to_test = {
        "claude": "claude-3-5-haiku-20241022",
        "google": "gemini-pro",
        "openai": "gpt-3.5-turbo",
        "openrouter": "deepseek/deepseek-v3.2",  # Changed from "deepseek" to "openrouter"
        "perplexity": "sonar-pro"
    }

    # Simple test prompt
    test_prompt = "What is 2+2? Reply with just the number."

    print(colored("🧪 RUNNING INDIVIDUAL MODEL TESTS", "cyan", attrs=["bold"]))
    print(colored("=" * 50, "cyan"))
    print(colored(f"Prompt: \"{test_prompt}\"", "white"))
    print()

    # Test each model
    results = {}
    for model_key, model_name in models_to_test.items():
        if available_models.get(model_key, False):
            success, time_taken, cost = test_single_model(model_key, model_name, test_prompt)
            results[model_key] = {
                "success": success,
                "time": time_taken,
                "cost": cost
            }
        else:
            print(colored(f"⏭️  Skipping {model_key.upper()} (no API key)", "yellow"))
            results[model_key] = {"success": False, "time": 0.0, "cost": 0.0}

        print()  # Empty line between tests

    # Print final summary
    print_summary(results)

    # Final message
    successful = sum(1 for r in results.values() if r["success"])
    if successful == 5:
        print(colored("\n🎉 ALL MODELS WORKING PERFECTLY!", "green", attrs=["bold"]))
        print(colored("🚀 Your AI swarm is ready for trading!", "green"))
    elif successful > 0:
        print(colored(f"\n✅ {successful} models working correctly", "green"))
        print(colored("💡 Add API keys for the remaining models to maximize accuracy", "yellow"))
    else:
        print(colored("\n❌ NO MODELS CONFIGURED", "red", attrs=["bold"]))
        print(colored("🔑 Add at least one API key to .env to get started", "yellow"))
        print(colored("📖 See .env.example for setup instructions", "cyan"))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(colored("\n\n👋 Testing interrupted by user", "yellow"))
    except Exception as e:
        print(colored(f"\n❌ Unexpected error: {e}", "red"))
        import traceback
        traceback.print_exc()
