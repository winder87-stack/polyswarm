#!/usr/bin/env python3
"""
Automated Project Audit
Run all checks and generate summary report.
"""

import os
import sys
import ast
import importlib
import subprocess
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

class ProjectAuditor:
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.passed = []

    def audit_syntax(self):
        """Check all Python files for syntax errors"""
        print("\n📝 Checking Syntax...")

        py_files = []
        for py_file in Path(".").rglob("*.py"):
            if "__pycache__" in str(py_file) or "venv" in str(py_file):
                continue
            py_files.append(py_file)

        print(f"  Found {len(py_files)} Python files to check...")

        for py_file in py_files:
            try:
                with open(py_file, encoding='utf-8') as f:
                    ast.parse(f.read(), str(py_file))
                self.passed.append(f"Syntax OK: {py_file}")
            except SyntaxError as e:
                self.issues.append(f"Syntax Error in {py_file}:{e.lineno}: {e.msg}")
            except UnicodeDecodeError:
                self.warnings.append(f"Encoding issue in {py_file} (may be binary)")

        print(f"  ✅ {len([p for p in self.passed if 'Syntax OK' in p])} files passed")

    def audit_imports(self):
        """Check for import issues"""
        print("\n📦 Checking Imports...")

        required_modules = [
            "anthropic", "openai",
            # Accept either the new or legacy Google SDK
            "google.genai", "google.generativeai",
            "aiohttp", "httpx", "pandas", "numpy",
            "feedparser", "fuzzywuzzy", "loguru"
        ]

        # Check dotenv separately
        try:
            importlib.import_module("dotenv")
            self.passed.append("Import OK: python-dotenv")
        except ImportError:
            self.issues.append("Missing required module: python-dotenv")

        optional_modules = [
            "matplotlib", "seaborn", "scikit-learn"
        ]

        print(f"  Checking {len(required_modules)} required modules...")

        for module in required_modules:
            try:
                importlib.import_module(module.replace("-", "."))
                self.passed.append(f"Import OK: {module}")
            except ImportError:
                # Don't require BOTH Google SDKs; at least one is enough.
                if module in {"google.genai", "google.generativeai"}:
                    self.warnings.append(f"Missing optional Google SDK: {module}")
                else:
                    self.issues.append(f"Missing required module: {module}")

        for module in optional_modules:
            try:
                importlib.import_module(module.replace("-", "."))
                self.passed.append(f"Import OK (optional): {module}")
            except ImportError:
                self.warnings.append(f"Missing optional module: {module}")

    def audit_env_vars(self):
        """Check environment variables"""
        print("\n🔑 Checking Environment Variables...")

        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            self.issues.append("Cannot load python-dotenv for env checking")
            return

        required = [
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "OPENAI_API_KEY",
            "OPENROUTER_API_KEY",
            "PERPLEXITY_API_KEY",
        ]

        optional = [
            "POLYGON_WALLET_PRIVATE_KEY",
            "POLYGON_FUNDER_ADDRESS",
            "PAPER_TRADING",
        ]

        print(f"  Checking {len(required)} required and {len(optional)} optional env vars...")

        for key in required:
            val = os.getenv(key)
            if val and len(val.strip()) > 10:
                self.passed.append(f"Env OK: {key}")
            else:
                self.issues.append(f"Missing/Invalid required env: {key}")

        for key in optional:
            val = os.getenv(key)
            if key == "PAPER_TRADING":
                if val and val.lower() in ["true", "false"]:
                    self.passed.append(f"Env OK: {key}={val}")
                else:
                    self.warnings.append(f"Optional env missing or invalid: {key} (should be 'true' or 'false')")
            elif val and len(val.strip()) > 5:
                self.passed.append(f"Env OK: {key}")
            else:
                self.warnings.append(f"Optional env missing: {key}")

    def audit_project_structure(self):
        """Check required files exist"""
        print("\n📁 Checking Project Structure...")

        required = [
            "main.py",
            ".env",
            "requirements.txt",
            "README.md",
            ".gitignore",
            "src/agents/swarm_agent.py",
            "src/agents/trading_swarm.py",
            "src/models/model_factory.py",
            "src/connectors/polymarket_client.py",
            "src/strategies/risk_manager.py",
            "scripts/paper_trading_24h.py",
            "scripts/pre_flight_check.py",
            "scripts/run_unit_tests.py",
        ]

        print(f"  Checking {len(required)} required files...")

        for filepath in required:
            if os.path.isfile(filepath):
                self.passed.append(f"File exists: {filepath}")
            else:
                self.issues.append(f"Missing required file: {filepath}")

    def audit_init_files(self):
        """Check __init__.py files exist"""
        print("\n📄 Checking __init__.py files...")

        dirs = [
            "src",
            "src/agents",
            "src/models",
            "src/connectors",
            "src/strategies",
            "src/services",
            "src/analysis",
            "scripts",
        ]

        print(f"  Checking {len(dirs)} directories for __init__.py files...")

        for dir_path in dirs:
            if os.path.isdir(dir_path):
                init_file = os.path.join(dir_path, "__init__.py")
                if os.path.isfile(init_file):
                    self.passed.append(f"Init OK: {init_file}")
                else:
                    self.warnings.append(f"Missing __init__.py: {dir_path}")
                    # Create it
                    try:
                        Path(init_file).touch()
                        print(f"  📝 Created: {init_file}")
                        self.passed.append(f"Init created: {init_file}")
                    except Exception as e:
                        self.issues.append(f"Cannot create __init__.py in {dir_path}: {e}")

    def audit_config_consistency(self):
        """Check configuration is consistent"""
        print("\n⚙️ Checking Configuration Consistency...")

        try:
            # Import the necessary modules
            import importlib.util

            # Load swarm_agent module
            spec = importlib.util.spec_from_file_location("swarm_agent", "src/agents/swarm_agent.py")
            swarm_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(swarm_module)

            # Load model_factory module
            spec = importlib.util.spec_from_file_location("model_factory", "src/models/model_factory.py")
            factory_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(factory_module)

            SWARM_MODELS = swarm_module.SWARM_MODELS
            MODEL_WEIGHTS = swarm_module.MODEL_WEIGHTS
            MODEL_CLASSES = factory_module.ModelFactory().MODEL_CLASSES

            # Check SWARM_MODELS providers match MODEL_CLASSES
            print(f"  Checking {len(SWARM_MODELS)} AI models configuration...")

            for name, (enabled, provider, model_id) in SWARM_MODELS.items():
                if provider in MODEL_CLASSES or provider == "openrouter":
                    self.passed.append(f"Config OK: {name} -> {provider} -> {model_id}")
                else:
                    self.issues.append(f"SWARM_MODELS '{name}' has provider '{provider}' not in MODEL_CLASSES")

            # Check MODEL_WEIGHTS has all enabled models
            for name, (enabled, provider, model_id) in SWARM_MODELS.items():
                if enabled:
                    if name in MODEL_WEIGHTS:
                        self.passed.append(f"Weight OK: {name} = {MODEL_WEIGHTS[name]}")
                    else:
                        self.warnings.append(f"Missing weight for enabled model: {name}")

            # Check we have at least some models
            enabled_count = sum(1 for _, (enabled, _, _) in SWARM_MODELS.items() if enabled)
            if enabled_count >= 1:
                self.passed.append(f"Enabled models: {enabled_count}")
            else:
                self.issues.append("No AI models enabled")

        except Exception as e:
            self.issues.append(f"Config check failed: {e}")

    def audit_gitignore(self):
        """Check .gitignore is properly configured"""
        print("\n🔒 Checking .gitignore Security...")

        if not os.path.isfile(".gitignore"):
            self.issues.append("Missing .gitignore file")
            return

        required_patterns = [
            ".env",
            "*.db",
            "*.sqlite*",
            "__pycache__/",
            "*.pyc",
            "logs/",
            "*.log"
        ]

        with open(".gitignore") as f:
            gitignore_content = f.read()

        for pattern in required_patterns:
            if pattern in gitignore_content:
                self.passed.append(f"Gitignore OK: {pattern}")
            else:
                self.issues.append(f"Missing from .gitignore: {pattern}")

    def audit_security(self):
        """Check for security issues"""
        print("\n🔐 Checking Security...")

        # Check for hardcoded API keys
        api_patterns = [
            r'sk-ant-', r'sk-or-', r'sk-proj-', r'AIzaSy', r'AIza',
            r'pk_', r'AKIAI', r'xoxb-', r'xoxp-', r'ghp_'
        ]

        py_files = []
        for py_file in Path(".").rglob("*.py"):
            if "__pycache__" in str(py_file) or "venv" in str(py_file):
                continue
            py_files.append(py_file)

        print(f"  Scanning {len(py_files)} files for hardcoded secrets...")

        for py_file in py_files:
            # Skip the audit script itself (contains example patterns)
            if "audit_summary.py" in str(py_file):
                continue

            try:
                with open(py_file, encoding='utf-8') as f:
                    content = f.read()

                for pattern in api_patterns:
                    # More thorough check - look for actual key patterns
                    import re
                    if re.search(pattern, content):
                        self.issues.append(f"Potential hardcoded secret in {py_file}: {pattern}")
            except Exception:
                pass  # Skip files we can't read

        if not any("hardcoded secret" in issue for issue in self.issues):
            self.passed.append("No hardcoded API keys found")

    def audit_trading_config(self):
        """Check trading configuration is safe"""
        print("\n📈 Checking Trading Configuration...")

        try:
            from dotenv import load_dotenv
            load_dotenv()

            # Check paper trading is enabled
            paper_trading = os.getenv("PAPER_TRADING", "").lower()
            if paper_trading == "true":
                self.passed.append("Paper trading enabled (safe for testing)")
            elif paper_trading == "false":
                self.warnings.append("Live trading enabled - ensure you understand the risks")
            else:
                self.issues.append("PAPER_TRADING not properly set (should be 'true' or 'false')")

            # Check risk limits are reasonable
            # Env var compatibility: prefer MAX_POSITION_SIZE, fallback to legacy MAX_POSITION
            max_position = float(os.getenv("MAX_POSITION_SIZE", os.getenv("MAX_POSITION", "100")))
            if max_position <= 50:
                self.passed.append(f"Very conservative position limit: ${max_position}")
            elif max_position <= 100:
                self.passed.append(f"Conservative position limit: ${max_position}")
            elif max_position <= 500:
                self.warnings.append(f"Moderate position limit: ${max_position}")
            else:
                self.issues.append(f"High position limit: ${max_position} (reduce for safety)")

        except Exception as e:
            self.warnings.append(f"Cannot check trading config: {e}")

    def generate_report(self):
        """Generate final report"""
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE PROJECT AUDIT REPORT")
        print("="*80)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Project: Polymarket AI Trading Bot")
        print()

        total_checks = len(self.passed) + len(self.issues) + len(self.warnings)
        success_rate = (len(self.passed) / total_checks * 100) if total_checks > 0 else 0

        print(f"📈 OVERALL STATUS:")
        print(f"   Total Checks: {total_checks}")
        print(f"   ✅ Passed: {len(self.passed)}")
        print(f"   ❌ Issues: {len(self.issues)}")
        print()

        if self.issues:
            print("❌ CRITICAL ISSUES (Must Fix Before Trading):")
            print("-" * 50)
            for i, issue in enumerate(self.issues, 1):
                print(f"   {i:2d}. {issue}")
            print()

        if self.warnings:
            print("⚠️ WARNINGS (Should Address):")
            print("-" * 50)
            for i, warning in enumerate(self.warnings, 1):
                print(f"   {i:2d}. {warning}")
            print()

        if len(self.issues) == 0:
            print("🎉 PROJECT AUDIT PASSED!")
            print("✅ Ready for paper trading")
            print()
            print("🚀 Next Steps:")
            print("   1. Run: python scripts/pre_flight_check.py")
            print("   2. Run: python scripts/run_unit_tests.py")
            print("   3. Run: python scripts/paper_trading_24h.py --hours 1")
            print("   4. Monitor logs and results")
            return True
        else:
            print("❌ PROJECT HAS ISSUES THAT NEED FIXING")
            print("🔧 Fix all critical issues before paper trading")
            return False

    def run_all(self):
        """Run all audits"""
        print("🔍 POLYMARKET AI TRADING BOT - COMPREHENSIVE AUDIT")
        print("="*80)
        print("This audit checks all critical aspects of the trading bot:")
        print("• Syntax & Imports • Configuration • Security • Structure")
        print("="*80)

        self.audit_syntax()
        self.audit_imports()
        self.audit_env_vars()
        self.audit_project_structure()
        self.audit_init_files()
        self.audit_config_consistency()
        self.audit_gitignore()
        self.audit_security()
        self.audit_trading_config()

        return self.generate_report()


if __name__ == "__main__":
    auditor = ProjectAuditor()
    success = auditor.run_all()
    sys.exit(0 if success else 1)
