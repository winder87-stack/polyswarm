#!/usr/bin/env python3
"""
Pre-Flight Check - Run ALL tests before paper trading
"""

import asyncio
import sys
import os
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

# Colors for terminal
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{text}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'='*60}{Colors.END}\n")

def print_pass(text):
    print(f"{Colors.GREEN}✅ PASS:{Colors.END} {text}")

def print_fail(text):
    print(f"{Colors.RED}❌ FAIL:{Colors.END} {text}")

def print_warn(text):
    print(f"{Colors.YELLOW}⚠️  WARN:{Colors.END} {text}")

def print_info(text):
    print(f"{Colors.BLUE}ℹ️  INFO:{Colors.END} {text}")


class PreFlightCheck:
    def __init__(self) -> None:
        self.results = {
            "passed": [],
            "failed": [],
            "warnings": [],
        }
        self.start_time = time.time()

    def record_pass(self, test_name, details=""):
        self.results["passed"].append((test_name, details))
        print_pass(f"{test_name} {details}")

    def record_fail(self, test_name, error):
        self.results["failed"].append((test_name, str(error)))
        print_fail(f"{test_name}: {error}")

    def record_warn(self, test_name, warning):
        self.results["warnings"].append((test_name, warning))
        print_warn(f"{test_name}: {warning}")

    # ==========================================
    # TEST 1: Environment Variables
    # ==========================================
    def test_environment(self):
        print_header("1️⃣  ENVIRONMENT VARIABLES")

        required = {
            "ANTHROPIC_API_KEY": "Claude AI",
            "GOOGLE_API_KEY": "Gemini AI",
            "OPENAI_API_KEY": "GPT AI",
            "OPENROUTER_API_KEY": "DeepSeek AI",
            "PERPLEXITY_API_KEY": "Perplexity AI",
        }

        optional = {
            "POLYGON_WALLET_PRIVATE_KEY": "Polymarket trading",
            "POLYGON_FUNDER_ADDRESS": "Polymarket trading",
            "POLYMARKET_API_KEY": "Polymarket API",
        }

        # Check required AI keys
        ai_keys_found = 0
        for key, desc in required.items():
            val = os.getenv(key)
            if val and len(val) > 10:
                self.record_pass(f"{key}", f"({desc})")
                ai_keys_found += 1
            else:
                self.record_fail(f"{key}", f"Missing or invalid ({desc})")

        if ai_keys_found < 3:
            self.record_fail("AI Models", "Need at least 3 AI API keys for swarm")
        else:
            self.record_pass("AI Models", f"{ai_keys_found}/5 available")

        # Check optional keys
        print()
        for key, desc in optional.items():
            val = os.getenv(key)
            if val and len(val) > 5:
                self.record_pass(f"{key}", f"({desc})")
            else:
                self.record_warn(f"{key}", f"Not set ({desc})")

        # Check trading config
        print()
        paper = os.getenv("PAPER_TRADING", "true").lower()
        if paper == "true":
            self.record_pass("PAPER_TRADING", "Enabled (safe mode)")
        else:
            self.record_warn("PAPER_TRADING", "DISABLED - Real money at risk!")

        bankroll = os.getenv("BANKROLL", "1000")
        self.record_pass("BANKROLL", f"${bankroll}")

    # ==========================================
    # TEST 2: Python Imports
    # ==========================================
    def test_imports(self):
        print_header("2️⃣  PYTHON IMPORTS")

        modules = [
            ("anthropic", "Anthropic SDK"),
            ("openai", "OpenAI SDK"),
            ("google.generativeai", "Google AI SDK"),
            ("httpx", "HTTP Client"),
            ("aiohttp", "Async HTTP"),
            ("pandas", "Data Analysis"),
            ("numpy", "Numerical Computing"),
            ("feedparser", "RSS Feeds"),
            ("fuzzywuzzy", "Fuzzy Matching"),
            ("loguru", "Logging"),
            ("dotenv", "Environment"),
        ]

        for module, desc in modules:
            try:
                __import__(module)
                self.record_pass(module, f"({desc})")
            except ImportError as e:
                self.record_fail(module, f"Not installed: {e}")

        # Check optional
        print()
        optional = [
            ("py_clob_client", "Polymarket Client"),
            ("web3", "Web3"),
            ("eth_account", "Ethereum Account"),
        ]

        for module, desc in optional:
            try:
                __import__(module)
                self.record_pass(module, f"({desc})")
            except ImportError:
                self.record_warn(module, f"Not installed ({desc}) - needed for live trading")

    # ==========================================
    # TEST 3: Project Structure
    # ==========================================
    def test_project_structure(self):
        print_header("3️⃣  PROJECT STRUCTURE")

        required_files = [
            "main.py",
            ".env",
            "requirements.txt",
            "src/agents/swarm_agent.py",
            "src/agents/trading_swarm.py",
            "src/models/model_factory.py",
            "src/connectors/polymarket_client.py",
            "src/strategies/risk_manager.py",
        ]

        optional_files = [
            "src/strategies/entry_timing.py",
            "src/strategies/position_manager.py",
            "src/strategies/category_specialist.py",
            "src/strategies/contrarian_detector.py",
            "src/services/news_monitor.py",
            "src/services/external_odds.py",
            "src/analysis/pattern_analyzer.py",
            "src/analysis/model_calibration.py",
            "scripts/paper_trading_24h.py",
            "config/categories.yaml",
        ]

        required_dirs = [
            "src",
            "src/agents",
            "src/models",
            "src/connectors",
            "src/strategies",
            "logs",
            "data",
        ]

        # Check directories
        for dir_path in required_dirs:
            if os.path.isdir(dir_path):
                self.record_pass(f"Directory: {dir_path}")
            else:
                self.record_fail(f"Directory: {dir_path}", "Missing")
                os.makedirs(dir_path, exist_ok=True)
                print_info(f"Created {dir_path}")

        print()

        # Check required files
        for file_path in required_files:
            if os.path.isfile(file_path):
                self.record_pass(f"File: {file_path}")
            else:
                self.record_fail(f"File: {file_path}", "Missing")

        print()

        # Check optional files
        for file_path in optional_files:
            if os.path.isfile(file_path):
                self.record_pass(f"File: {file_path}")
            else:
                self.record_warn(f"File: {file_path}", "Missing (optional)")

    # ==========================================
    # TEST 4: AI Models
    # ==========================================
    async def test_ai_models(self):
        print_header("4️⃣  AI MODEL CONNECTIVITY")

        working_models = 0
        total_cost = 0.0

        # Test Claude
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            start = time.time()
            response = client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=10,
                messages=[{"role": "user", "content": "Say 'test'"}]
            )
            elapsed = time.time() - start
            self.record_pass(f"Claude", f"({elapsed:.2f}s)")
            working_models += 1
        except Exception as e:
            self.record_fail("Claude", str(e)[:50])

        # Test Gemini
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            model = genai.GenerativeModel("gemini-pro")
            start = time.time()
            response = model.generate_content("Say 'test'")
            elapsed = time.time() - start
            self.record_pass(f"Gemini", f"({elapsed:.2f}s)")
            working_models += 1
        except Exception as e:
            self.record_fail("Gemini", str(e)[:50])

        # Test OpenAI
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            start = time.time()
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                max_tokens=10,
                messages=[{"role": "user", "content": "Say 'test'"}]
            )
            elapsed = time.time() - start
            self.record_pass(f"OpenAI GPT", f"({elapsed:.2f}s)")
            working_models += 1
        except Exception as e:
            self.record_fail("OpenAI GPT", str(e)[:50])

        # Test DeepSeek via OpenRouter
        try:
            import httpx
            start = time.time()
            response = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "deepseek/deepseek-chat",
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "Say 'test'"}]
                },
                timeout=30
            )
            if response.status_code == 200:
                elapsed = time.time() - start
                self.record_pass(f"DeepSeek (OpenRouter)", f"({elapsed:.2f}s)")
                working_models += 1
            else:
                self.record_fail("DeepSeek", f"HTTP {response.status_code}")
        except Exception as e:
            self.record_fail("DeepSeek", str(e)[:50])

        # Test Perplexity
        try:
            import httpx
            start = time.time()
            response = httpx.post(
                "https://api.perplexity.ai/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "sonar",
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "Say 'test'"}]
                },
                timeout=30
            )
            if response.status_code == 200:
                elapsed = time.time() - start
                self.record_pass(f"Perplexity", f"({elapsed:.2f}s)")
                working_models += 1
            else:
                self.record_fail("Perplexity", f"HTTP {response.status_code}")
        except Exception as e:
            self.record_fail("Perplexity", str(e)[:50])

        print()
        if working_models >= 3:
            self.record_pass(f"AI Swarm", f"{working_models}/5 models working")
        else:
            self.record_fail("AI Swarm", f"Only {working_models}/5 models working (need 3+)")

    # ==========================================
    # TEST 5: Polymarket Connection
    # ==========================================
    async def test_polymarket(self):
        print_header("5️⃣  POLYMARKET CONNECTION")

        # Test public API (no auth needed)
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                start = time.time()
                async with session.get(
                    "https://gamma-api.polymarket.com/markets?limit=5",
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        elapsed = time.time() - start
                        self.record_pass(f"Gamma API (Market Data)", f"({elapsed:.2f}s, {len(data)} markets)")

                        # Show sample markets
                        print()
                        print_info("Sample markets:")
                        for m in data[:3]:
                            q = m.get("question", "Unknown")[:50]
                            print(f"      • {q}...")
                    else:
                        self.record_fail("Gamma API", f"HTTP {response.status}")
        except Exception as e:
            self.record_fail("Gamma API", str(e)[:50])

        print()

        # Test CLOB API (auth needed)
        try:
            from src.connectors.polymarket_client import polymarket
            if hasattr(polymarket, 'test_connection'):
                result = await polymarket.test_connection()
                if result:
                    self.record_pass("CLOB API (Trading)", "Authenticated")
                else:
                    self.record_warn("CLOB API (Trading)", "Not authenticated (paper trading OK)")
            else:
                self.record_warn("CLOB API (Trading)", "test_connection not implemented")
        except Exception as e:
            self.record_warn("CLOB API (Trading)", f"Not available: {str(e)[:30]} (paper trading OK)")

    # ==========================================
    # TEST 6: Core Components
    # ==========================================
    async def test_core_components(self):
        print_header("6️⃣  CORE COMPONENTS")

        components = [
            ("src.agents.swarm_agent", "SwarmAgent", []),
            ("src.models.model_factory", "ModelFactory", []),
            ("src.strategies.risk_manager", "RiskManager", []),
        ]

        for module_path, class_name, init_args in components:
            try:
                module = __import__(module_path, fromlist=[class_name])
                cls = getattr(module, class_name)
                instance = cls(*init_args)
                self.record_pass(f"{class_name}")
            except Exception as e:
                self.record_fail(f"{class_name}", str(e)[:50])

        print()

        # Optional components
        optional_components = [
            ("src.strategies.entry_timing", "EntryTimingOptimizer"),
            ("src.strategies.category_specialist", "CategorySpecialist"),
            ("src.strategies.position_manager", "PositionManager"),
            ("src.strategies.contrarian_detector", "ContrarianDetector"),
            ("src.services.news_monitor", "NewsAggregator"),
            ("src.services.external_odds", "ConsensusDetector"),
            ("src.analysis.pattern_analyzer", "PatternAnalyzer"),
            ("src.analysis.model_calibration", "ModelCalibration"),
        ]

        for module_path, class_name in optional_components:
            try:
                module = __import__(module_path, fromlist=[class_name])
                cls = getattr(module, class_name)

                # Handle classes that need parameters
                if class_name == "PositionManager":
                    # PositionManager needs a TradingSwarm instance
                    try:
                        from src.agents.trading_swarm import TradingSwarm
                        instance = cls(TradingSwarm())
                    except Exception as e:
                        print(f"    ❌ {class_name}: {e}")
                        continue
                else:
                    instance = cls()
                self.record_pass(f"{class_name}")
            except Exception as e:
                self.record_warn(f"{class_name}", str(e)[:40])

    # ==========================================
    # TEST 7: News & External Data
    # ==========================================
    async def test_external_data(self):
        print_header("7️⃣  EXTERNAL DATA SOURCES (FREE)")

        # Test Google News RSS
        try:
            import feedparser
            import aiohttp

            url = "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        content = await response.text()
                        feed = feedparser.parse(content)
                        self.record_pass(f"Google News RSS", f"({len(feed.entries)} articles)")
                    else:
                        self.record_fail("Google News RSS", f"HTTP {response.status}")
        except Exception as e:
            self.record_fail("Google News RSS", str(e)[:50])

        # Test PredictIt
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://www.predictit.org/api/marketdata/all/",
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        markets = data.get("markets", [])
                        self.record_pass(f"PredictIt API", f"({len(markets)} markets)")
                    else:
                        self.record_warn("PredictIt API", f"HTTP {response.status}")
        except Exception as e:
            self.record_warn("PredictIt API", str(e)[:50])

        # Test Metaculus
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://www.metaculus.com/api2/questions/?limit=5",
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        questions = data.get("results", [])
                        self.record_pass(f"Metaculus API", f"({len(questions)} questions)")
                    else:
                        self.record_warn("Metaculus API", f"HTTP {response.status}")
        except Exception as e:
            self.record_warn("Metaculus API", str(e)[:50])

    # ==========================================
    # TEST 8: End-to-End Analysis
    # ==========================================
    async def test_e2e_analysis(self):
        print_header("8️⃣  END-TO-END ANALYSIS TEST")

        try:
            # Get a real market
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://gamma-api.polymarket.com/markets?limit=1&active=true",
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    if response.status != 200:
                        self.record_fail("E2E Test", "Could not fetch market")
                        return
                    markets = await response.json()
                    if not markets:
                        self.record_fail("E2E Test", "No markets found")
                        return
                    market_data = markets[0]

            print_info(f"Testing with: {market_data.get('question', 'Unknown')[:50]}...")
            print()

            # Try to import and run trading swarm
            try:
                # First try v2
                from src.agents.trading_swarm_v2 import TradingSwarmV2
                swarm = TradingSwarmV2(paper_trading=True)
                swarm_version = "V2"
            except ImportError:
                # Fall back to v1
                from src.agents.trading_swarm import TradingSwarm
                swarm = TradingSwarm(paper_trading=True)
                swarm_version = "V1"

            self.record_pass(f"TradingSwarm {swarm_version}", "Loaded")

            # Create market object
            from dataclasses import dataclass

            @dataclass
            class TestMarket:
                condition_id: str
                question: str
                yes_price: float
                no_price: float
                volume: float
                liquidity: float
                category: str
                slug: str
                end_date: str = ""
                yes_token_id: str = ""
                no_token_id: str = ""

            test_market = TestMarket(
                condition_id=market_data.get("conditionId", "test"),
                question=market_data.get("question", "Test"),
                yes_price=float(market_data.get("outcomePrices", [0.5, 0.5])[0]),
                no_price=float(market_data.get("outcomePrices", [0.5, 0.5])[1]),
                volume=float(market_data.get("volume", 10000)),
                liquidity=float(market_data.get("liquidity", 5000)),
                category=market_data.get("category", "politics"),
                slug=market_data.get("slug", "test"),
            )

            # Run analysis
            print_info("Running AI analysis (this may take 30-60 seconds)...")
            start = time.time()

            signal = await swarm.analyze_market(test_market)

            elapsed = time.time() - start

            if signal:
                self.record_pass(f"Market Analysis", f"({elapsed:.1f}s)")
                print()
                print_info(f"Results:")
                print(f"      Direction: {getattr(signal, 'direction', 'N/A')}")
                print(f"      Edge: {getattr(signal, 'edge', 0)*100:.1f}%")
                print(f"      Confidence: {getattr(signal, 'confidence', 0)*100:.0f}%")
                print(f"      Recommended Size: ${getattr(signal, 'recommended_size', 0):.2f}")
                print(f"      Actionable: {getattr(signal, 'is_actionable', False)}")
            else:
                self.record_warn("Market Analysis", f"No signal returned ({elapsed:.1f}s)")

        except Exception as e:
            import traceback
            self.record_fail("E2E Test", str(e))
            print()
            print_info("Traceback:")
            traceback.print_exc()

    # ==========================================
    # TEST 9: Paper Trading Script
    # ==========================================
    def test_paper_trading_script(self):
        print_header("9️⃣  PAPER TRADING SCRIPT")

        script_path = "scripts/paper_trading_24h.py"

        if not os.path.isfile(script_path):
            self.record_fail("Paper Trading Script", "File not found")
            return

        self.record_pass("Script exists", script_path)

        # Check syntax
        try:
            with open(script_path, 'r') as f:
                code = f.read()
            compile(code, script_path, 'exec')
            self.record_pass("Syntax check", "Valid Python")
        except SyntaxError as e:
            self.record_fail("Syntax check", f"Line {e.lineno}: {e.msg}")

        # Check logs directory is writable
        try:
            test_file = "logs/test_write.tmp"
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            self.record_pass("Logs directory", "Writable")
        except Exception as e:
            self.record_fail("Logs directory", f"Not writable: {e}")

        # Check data directory is writable
        try:
            test_file = "data/test_write.tmp"
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            self.record_pass("Data directory", "Writable")
        except Exception as e:
            self.record_fail("Data directory", f"Not writable: {e}")

    # ==========================================
    # TEST 10: Disk Space & Resources
    # ==========================================
    def test_resources(self):
        print_header("🔟  SYSTEM RESOURCES")

        import shutil

        # Check disk space
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024**3)

        if free_gb >= 1:
            self.record_pass(f"Disk Space", f"{free_gb:.1f} GB free")
        else:
            self.record_warn(f"Disk Space", f"Only {free_gb:.2f} GB free")

        # Check Python version
        import sys
        py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        if sys.version_info >= (3, 9):
            self.record_pass(f"Python Version", py_version)
        else:
            self.record_warn(f"Python Version", f"{py_version} (3.9+ recommended)")

        # Check if running in venv
        in_venv = sys.prefix != sys.base_prefix
        if in_venv:
            self.record_pass("Virtual Environment", "Active")
        else:
            self.record_warn("Virtual Environment", "Not active (recommended)")

    # ==========================================
    # SUMMARY
    # ==========================================
    def print_summary(self):
        elapsed = time.time() - self.start_time

        print_header("📊 TEST SUMMARY")

        passed = len(self.results["passed"])
        failed = len(self.results["failed"])
        warnings = len(self.results["warnings"])
        total = passed + failed + warnings

        print(f"  {Colors.GREEN}Passed:   {passed}{Colors.END}")
        print(f"  {Colors.RED}Failed:   {failed}{Colors.END}")
        print(f"  {Colors.YELLOW}Warnings: {warnings}{Colors.END}")
        print(f"  Total:    {total}")
        print(f"\n  Time: {elapsed:.1f} seconds")

        if failed > 0:
            print(f"\n{Colors.RED}{Colors.BOLD}❌ FAILURES:{Colors.END}")
            for name, error in self.results["failed"]:
                print(f"  • {name}: {error}")

        if warnings > 0:
            print(f"\n{Colors.YELLOW}⚠️  WARNINGS:{Colors.END}")
            for name, warning in self.results["warnings"]:
                print(f"  • {name}: {warning}")

        print()

        if failed == 0:
            print(f"{Colors.GREEN}{Colors.BOLD}✅ ALL CRITICAL TESTS PASSED!{Colors.END}")
            print(f"{Colors.GREEN}   Ready for 24-hour paper trading.{Colors.END}")
            return True
        elif failed <= 2:
            print(f"{Colors.YELLOW}{Colors.BOLD}⚠️  SOME ISSUES DETECTED{Colors.END}")
            print(f"{Colors.YELLOW}   Fix failures above before paper trading.{Colors.END}")
            return False
        else:
            print(f"{Colors.RED}{Colors.BOLD}❌ MULTIPLE FAILURES{Colors.END}")
            print(f"{Colors.RED}   Fix critical issues before proceeding.{Colors.END}")
            return False

    # ==========================================
    # RUN ALL TESTS
    # ==========================================
    async def run_all(self):
        print(f"\n{Colors.BOLD}🚀 POLYMARKET BOT - PRE-FLIGHT CHECK{Colors.END}")
        print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Sync tests
        self.test_environment()
        self.test_imports()
        self.test_project_structure()

        # Async tests
        await self.test_ai_models()
        await self.test_polymarket()
        await self.test_core_components()
        await self.test_external_data()
        await self.test_e2e_analysis()

        # More sync tests
        self.test_paper_trading_script()
        self.test_resources()

        # Summary
        return self.print_summary()


async def main():
    checker = PreFlightCheck()
    success = await checker.run_all()

    print()
    if success:
        print(f"{Colors.CYAN}Next step:{Colors.END}")
        print(f"  python scripts/paper_trading_24h.py --hours 1 --interval 10")
        print()

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
