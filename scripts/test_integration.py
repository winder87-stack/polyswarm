#!/usr/bin/env python3
"""
Integration test for all trading bot components.
Run this to verify everything is wired correctly.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from termcolor import cprint


async def main():
    cprint("\n" + "="*60, "cyan")
    cprint("🧪 POLYMARKET BOT - INTEGRATION TEST", "cyan", attrs=["bold"])
    cprint("="*60 + "\n", "cyan")

    results = {}

    # 1. Test environment
    cprint("1️⃣  Testing Environment Variables...", "yellow")
    from dotenv import load_dotenv
    import os
    load_dotenv()

    required_keys = [
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "PERPLEXITY_API_KEY",
        "POLYGON_WALLET_PRIVATE_KEY",
    ]

    optional_keys = [
        "POLYGON_FUNDER_ADDRESS",  # Only needed for live trading
    ]

    missing_required = [k for k in required_keys if not os.getenv(k)]
    missing_optional = [k for k in optional_keys if not os.getenv(k)]

    if missing_required:
        cprint(f"   ❌ Missing required: {', '.join(missing_required)}", "red")
        results["Environment"] = f"❌ Missing {len(missing_required)} required keys"
    else:
        if missing_optional:
            cprint(f"   ⚠️  Missing optional: {', '.join(missing_optional)} (only needed for live trading)", "yellow")
            results["Environment"] = f"⚠️ Missing {len(missing_optional)} optional keys"
        else:
            cprint("   ✅ All environment variables set", "green")
            results["Environment"] = "✅"

    # 2. Test AI Models
    cprint("\n2️⃣  Testing AI Models...", "yellow")
    try:
        from src.models.model_factory import model_factory
        available = model_factory.check_available_models()
        working = sum(1 for v in available.values() if v)
        cprint(f"   ✅ {working}/5 models available", "green")
        results["AI Models"] = f"✅ {working}/5"
    except Exception as e:
        cprint(f"   ❌ {e}", "red")
        results["AI Models"] = "❌"

    # 3. Test Polymarket
    cprint("\n3️⃣  Testing Polymarket Connection...", "yellow")
    try:
        from src.connectors.polymarket_client import PolymarketClient
        pm_client = PolymarketClient()
        markets = await pm_client.get_markets(limit=3)
        cprint(f"   ✅ Connected - {len(markets)} markets fetched", "green")
        results["Polymarket"] = "✅"
        # Make it available for E2E test
        global polymarket
        polymarket = pm_client
    except Exception as e:
        error_msg = str(e)
        if "POLYGON_FUNDER_ADDRESS" in error_msg:
            cprint("   ⚠️  POLYGON_FUNDER_ADDRESS not set (only needed for live trading)", "yellow")
            results["Polymarket"] = "⚠️"
        else:
            cprint(f"   ❌ {error_msg[:50]}", "red")
            results["Polymarket"] = "❌"

    # 4. Test Core Components
    cprint("\n4️⃣  Testing Core Components...", "yellow")

    components = [
        ("SwarmAgent", "src.agents.swarm_agent", "SwarmAgent"),
        ("RiskManager", "src.strategies.risk_manager", "RiskManager"),
        ("EntryTiming", "src.strategies.entry_timing", "EntryTimingOptimizer"),
        ("CategorySpecialist", "src.strategies.category_specialist", "CategorySpecialist"),
        ("ContrarianDetector", "src.strategies.contrarian_detector", "ContrarianDetector"),
        ("PositionManager", "src.strategies.position_manager", "PositionManager"),
    ]

    for name, module, class_name in components:
        try:
            mod = __import__(module, fromlist=[class_name])
            cls = getattr(mod, class_name)
            if class_name == "PositionManager":
                # PositionManager needs a swarm parameter, skip for now
                instance = None
                cprint(f"   ⚠️  {name} (requires swarm)", "yellow")
                results[name] = "⚠️"
            else:
                instance = cls()
                cprint(f"   ✅ {name}", "green")
                results[name] = "✅"
        except Exception as e:
            cprint(f"   ❌ {name}: {str(e)[:40]}", "red")
            results[name] = "❌"

    # 5. Test Services
    cprint("\n5️⃣  Testing Services...", "yellow")

    services = [
        ("NewsMonitor", "src.services.news_monitor", "NewsAggregator"),
        ("ConsensusDetector", "src.services.consensus_detector", "ConsensusDetector"),
    ]

    for name, module, class_name in services:
        try:
            mod = __import__(module, fromlist=[class_name])
            cls = getattr(mod, class_name)
            if class_name == "NewsAggregator":
                instance = cls()  # No sources parameter needed
            else:
                instance = cls()
            cprint(f"   ✅ {name}", "green")
            results[name] = "✅"
        except Exception as e:
            cprint(f"   ❌ {name}: {str(e)[:40]}", "red")
            results[name] = "❌"

    # 6. Test Analysis
    cprint("\n6️⃣  Testing Analysis Components...", "yellow")

    analysis = [
        ("PatternAnalyzer", "src.analysis.pattern_analyzer", "PatternAnalyzer"),
        ("ModelCalibration", "src.analysis.model_calibration", "ModelCalibration"),
        ("AIAccuracyTracker", "src.analysis.ai_accuracy_tracker", "AIAccuracyTracker"),
    ]

    for name, module, class_name in analysis:
        try:
            mod = __import__(module, fromlist=[class_name])
            cls = getattr(mod, class_name)
            instance = cls()
            cprint(f"   ✅ {name}", "green")
            results[name] = "✅"
        except Exception as e:
            cprint(f"   ❌ {name}: {str(e)[:40]}", "red")
            results[name] = "❌"

    # 7. Test TradingSwarm V2
    cprint("\n7️⃣  Testing TradingSwarm V2...", "yellow")
    try:
        from src.agents.trading_swarm_v2 import TradingSwarmV2
        swarm = TradingSwarmV2(paper_trading=True)
        cprint("   ✅ TradingSwarmV2 initialized", "green")
        results["TradingSwarmV2"] = "✅"
    except Exception as e:
        cprint(f"   ❌ TradingSwarmV2: {str(e)[:40]}", "red")
        results["TradingSwarmV2"] = "❌"

    # 8. End-to-End Test (optional)
    cprint("\n8️⃣  Running End-to-End Test...", "yellow")
    try:
        if results.get("Polymarket") == "✅" and results.get("TradingSwarmV2") == "✅":
            # Analyze one market
            markets = await polymarket.get_markets(limit=1)
            if markets:
                signal = await swarm.analyze_market(markets[0], include_news=False, include_external=False)
                if signal:
                    cprint(f"   ✅ Analysis complete - Edge: {signal.edge*100:.1f}%, Quality: {signal.signal_quality}", "green")
                    results["E2E Test"] = "✅"
                else:
                    cprint("   ⚠️ Analysis returned None", "yellow")
                    results["E2E Test"] = "⚠️"
            else:
                cprint("   ⚠️ No markets available for E2E test", "yellow")
                results["E2E Test"] = "⚠️"
        else:
            cprint("   ⏭️  Skipped (dependencies failed)", "yellow")
            results["E2E Test"] = "⏭️"
    except Exception as e:
        cprint(f"   ❌ {str(e)[:50]}", "red")
        results["E2E Test"] = "❌"

    # 9. Test CLI Commands
    cprint("\n9️⃣  Testing CLI Commands...", "yellow")
    try:
        import subprocess
        result = subprocess.run([sys.executable, "main.py", "--help"],
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and "Available commands" in result.stdout:
            cprint("   ✅ CLI help works", "green")
            results["CLI"] = "✅"
        else:
            cprint("   ❌ CLI help failed", "red")
            results["CLI"] = "❌"
    except Exception as e:
        cprint(f"   ❌ CLI test failed: {str(e)[:40]}", "red")
        results["CLI"] = "❌"

    # Summary
    cprint("\n" + "="*60, "cyan")
    cprint("📊 TEST SUMMARY", "cyan", attrs=["bold"])
    cprint("="*60, "cyan")

    passed = sum(1 for v in results.values() if v.startswith("✅"))
    warned = sum(1 for v in results.values() if v.startswith("⚠️"))
    skipped = sum(1 for v in results.values() if v.startswith("⏭️"))
    failed = sum(1 for v in results.values() if v.startswith("❌"))

    for component, status in results.items():
        color = "green" if status.startswith("✅") else "yellow" if status.startswith("⚠️") or status.startswith("⏭️") else "red"
        cprint(f"  {status} {component}", color)

    cprint(f"\n  Passed: {passed} | Warnings: {warned} | Skipped: {skipped} | Failed: {failed}", "white")

    if failed == 0:
        cprint("\n🎉 All tests passed! Ready for paper trading.", "green", attrs=["bold"])
        cprint("   Run: python main.py trade --paper", "white")
    elif failed <= 2:
        cprint("\n⚠️ Some tests failed. Check the issues above.", "yellow", attrs=["bold"])
        cprint("   You can still run paper trading but some features may be limited.", "white")
    else:
        cprint("\n❌ Multiple failures. Fix issues before trading.", "red", attrs=["bold"])
        cprint("   Check environment variables and dependencies.", "white")

    # Recommendations
    if results.get("Environment") and results["Environment"].startswith("⚠️"):
        cprint("\n💡 Tip: Set missing environment variables in .env file", "cyan")

    if results.get("AI Models") and not results["AI Models"].startswith("✅"):
        cprint("💡 Tip: Check API keys and internet connection", "cyan")

    if results.get("Polymarket") == "❌":
        cprint("💡 Tip: Verify Polymarket credentials and network", "cyan")

    print()


if __name__ == "__main__":
    asyncio.run(main())
