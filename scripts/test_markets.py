#!/usr/bin/env python3
"""
🧪 Polymarket Connection Test

Tests connection to Polymarket APIs and displays top markets.
Verifies that your setup is working correctly.

Usage:
    python scripts/test_markets.py
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime, timezone

from termcolor import colored

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

try:
    from src.connectors import polymarket, Market
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


async def test_connection():
    """Test Polymarket connection and display top markets."""
    print(colored("="*70, "cyan"))
    print(colored("🧪 POLYMARKET CONNECTION TEST", "cyan", attrs=["bold"]))
    print(colored("="*70, "cyan"))

    # Check imports
    if not IMPORT_SUCCESS:
        print(colored("❌ Import failed:", "red"))
        print(colored(f"   {IMPORT_ERROR}", "red"))
        print(colored("\n🔧 Troubleshooting:", "yellow"))
        print("   1. Activate virtual environment: source venv/bin/activate")
        print("   2. Install dependencies: pip install -r requirements.txt")
        print("   3. Check .env file exists: cp .env.example .env")
        return False

    print(colored("✅ Imports successful", "green"))

    # Check if polymarket client initialized
    if polymarket is None:
        print(colored("❌ Polymarket client failed to initialize", "red"))
        print(colored("   This usually means missing environment variables.", "yellow"))
        print(colored("\n🔧 Required environment variables:", "yellow"))
        print("   • POLYGON_WALLET_PRIVATE_KEY (from wallet generator)")
        print("   • POLYGON_FUNDER_ADDRESS (your wallet address)")
        print("   • POLYGON_SIGNATURE_TYPE (usually 0)")
        print(colored("\n🔧 Setup steps:", "yellow"))
        print("   1. Generate wallet: python scripts/generate_wallet.py")
        print("   2. Add private key to .env file")
        print("   3. Add funder address to .env file")
        return False

    print(colored("✅ Polymarket client initialized", "green"))

    try:
        # Test connection by fetching markets
        print(colored("\n🌐 Testing connection to Polymarket...", "blue"))

        markets = await polymarket.get_markets(limit=10, active_only=True)

        if not markets:
            print(colored("❌ No markets returned", "red"))
            print(colored("   Check your internet connection", "yellow"))
            return False

        print(colored(f"✅ Connected! Retrieved {len(markets)} markets", "green"))

        # Display markets
        print(colored("\n📊 TOP MARKETS BY VOLUME", "cyan", attrs=["bold"]))
        print(colored("="*70, "cyan"))

        for i, market in enumerate(markets, 1):
            print(colored(f"\n{i}. {market.question}", "white", attrs=["bold"]))

            # Format prices
            yes_price = colored(f"{market.yes_price:.1%}", "green")
            no_price = colored(f"{market.no_price:.1%}", "red")

            print(colored(f"   YES: {yes_price} | NO: {no_price}", "white"))

            # Format volume and liquidity
            volume_str = f"${market.volume:,.0f}" if market.volume >= 1000 else f"${market.volume:.0f}"
            liquidity_str = f"${market.liquidity:,.0f}" if market.liquidity >= 1000 else f"${market.liquidity:.0f}"

            print(colored(f"   Volume: {volume_str} | Liquidity: {liquidity_str}", "white"))

            # Category and end date
            category = colored(market.category.upper(), "yellow")
            print(colored(f"   Category: {category}", "white"))

            if market.end_date:
                try:
                    # Parse string end_date to datetime
                    end_date = datetime.fromisoformat(market.end_date.replace('Z', '+00:00'))
                    now = datetime.now(timezone.utc)
                    if end_date.tzinfo is None:
                        end_date = end_date.replace(tzinfo=timezone.utc)

                    days_until = (end_date - now).days
                    if days_until > 0:
                        end_date_str = f"{end_date.strftime('%Y-%m-%d')} ({days_until} days)"
                    else:
                        end_date_str = f"{end_date.strftime('%Y-%m-%d')} (EXPIRED)"
                except Exception:
                    # Fallback if datetime parsing fails
                    end_date_str = market.end_date[:10] if len(market.end_date) >= 10 else market.end_date
                print(colored(f"   End Date: {end_date_str}", "white"))
            else:
                print(colored("   End Date: No end date", "white"))

            # Market slug (truncated)
            slug_display = market.slug[:50] + "..." if len(market.slug) > 50 else market.slug
            print(colored(f"   Slug: {slug_display}", "grey"))

        print(colored("\n" + "="*70, "cyan"))
        print(colored("🎉 Connection test successful!", "green", attrs=["bold"]))
        print(colored("   Your Polymarket setup is working correctly.", "white"))

        # Show some statistics
        total_volume = sum(m.volume for m in markets)
        total_liquidity = sum(m.liquidity for m in markets)
        avg_volume = total_volume / len(markets)
        avg_liquidity = total_liquidity / len(markets)

        print(colored("\n📈 Market Statistics:", "blue"))
        print(colored(f"   Total Volume (top 10): ${total_volume:,.0f}", "white"))
        print(colored(f"   Average Volume: ${avg_volume:,.0f}", "white"))
        print(colored(f"   Total Liquidity: ${total_liquidity:,.0f}", "white"))
        print(colored(f"   Average Liquidity: ${avg_liquidity:,.0f}", "white"))

        return True

    except Exception as e:
        error_msg = str(e)
        print(colored(f"\n❌ Connection test failed: {error_msg}", "red"))

        # Provide helpful troubleshooting
        print(colored("\n🔧 Troubleshooting:", "yellow"))
        if "connection" in error_msg.lower() or "network" in error_msg.lower():
            print("   • Check your internet connection")
            print("   • Verify Polymarket APIs are accessible")
        elif "timeout" in error_msg.lower():
            print("   • Request timed out - try again")
            print("   • Check if you're behind a firewall")
        elif "unauthorized" in error_msg.lower() or "auth" in error_msg.lower():
            print("   • API credentials may be needed for some endpoints")
            print("   • Run: python scripts/generate_api_keys.py")
        else:
            print("   • Check your .env file configuration")
            print("   • Ensure all dependencies are installed")
            print("   • Try running the wallet generator first")

        return False


async def main():
    """Main entry point."""
    try:
        success = await test_connection()
        if success:
            print(colored("\n🚀 Ready for trading! Run the bot with:", "green"))
            print(colored("   python -c 'from src.agents import run_trading_bot; run_trading_bot()'", "white"))
        else:
            print(colored("\n❌ Setup issues detected. Please fix the errors above.", "red"))
            sys.exit(1)

    except KeyboardInterrupt:
        print(colored("\n\n👋 Test cancelled by user", "yellow"))
    except Exception as e:
        print(colored(f"\n❌ Unexpected error: {e}", "red"))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
