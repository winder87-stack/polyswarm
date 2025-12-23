#!/usr/bin/env python3
"""
Test script for the AI Swarm functionality
"""

import os
import sys
import asyncio
from pathlib import Path

# Add project root to path so we can import src modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv
from termcolor import colored

# Load environment variables
load_dotenv()

def test_swarm_basic():
    """Test basic swarm initialization."""
    print(colored("🧪 Testing AI Swarm initialization...", "blue", attrs=["bold"]))

    try:
        from src.agents import SwarmAgent

        swarm = SwarmAgent()
        print(colored("✅ Swarm initialized successfully", "green"))
        print(f"🤖 Models loaded: {len(swarm.models)}")
        print(f"💰 Total API costs so far: ${swarm.total_cost:.4f}")

        # Show model status
        for name, model in swarm.models.items():
            status = "✅" if model else "❌"
            print(f"  {status} {name}")

        return swarm

    except Exception as e:
        print(colored(f"❌ Swarm initialization failed: {e}", "red"))
        import traceback
        traceback.print_exc()
        return None

async def test_swarm_query(swarm):
    """Test swarm query functionality."""
    if not swarm:
        print(colored("⏭️  Skipping query test (swarm not available)", "yellow"))
        return

    print(colored("\n🧪 Testing AI Swarm query...", "blue", attrs=["bold"]))

    test_prompt = "What is the probability that it will rain tomorrow in New York City?"
    system_prompt = "You are a weather expert. Provide probability estimates based on available data."

    try:
        result = await swarm.query(
            prompt=test_prompt,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=512
        )

        print(colored("✅ Swarm query successful", "green"))
        print(f"📊 Responses received: {len(result['responses'])}")
        print(f"💰 Query cost: ${result.get('cost', 0):.4f}")
        print(f"⏱️  Total response time: {result.get('total_time', 0):.2f}s")

        # Show consensus summary
        if 'consensus_summary' in result:
            print(colored("\n📝 Consensus Summary:", "cyan"))
            print(result['consensus_summary'][:200] + "..." if len(result['consensus_summary']) > 200 else result['consensus_summary'])

        # Show individual responses (brief)
        print(colored("\n🤖 Model Responses:", "cyan"))
        for provider, response_data in result['responses'].items():
            status = "✅" if response_data['success'] else "❌"
            content_preview = response_data.get('content', 'No content')[:50] + "..." if len(response_data.get('content', '')) > 50 else response_data.get('content', 'No content')
            print(f"  {status} {provider}: {content_preview}")

    except Exception as e:
        print(colored(f"❌ Swarm query failed: {e}", "red"))
        import traceback
        traceback.print_exc()

async def main():
    """Run all swarm tests."""
    print(colored("🌙 Polymarket AI Swarm Test", "cyan", attrs=["bold"]))
    print(colored("=" * 50, "cyan"))

    # Test basic initialization
    swarm = test_swarm_basic()

    # Test query functionality
    await test_swarm_query(swarm)

    # Summary
    print(colored("\n" + "=" * 50, "cyan"))
    if swarm:
        print(colored("🎉 Swarm tests completed successfully!", "green", attrs=["bold"]))
        print(f"💡 Ready for market analysis with {len(swarm.models)} AI models")
    else:
        print(colored("❌ Swarm tests failed - check API keys and configuration", "red", attrs=["bold"]))

    print(colored("\n📋 Next steps:", "yellow"))
    print("  1. Configure API keys in .env file")
    print("  2. Run: python main.py markets")
    print("  3. Run: python main.py analyze <market-slug>")
    print("  4. Run: python main.py trade --paper")

if __name__ == "__main__":
    asyncio.run(main())
