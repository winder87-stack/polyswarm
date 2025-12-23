#!/usr/bin/env python3
"""
Test Polymarket Connection and Configuration

Verifies:
1. Gamma API (market data) - no auth needed
2. CLOB API (order book) - no auth needed
3. Trading client - requires wallet
4. API credentials - derived from wallet

Usage:
    python scripts/test_polymarket_connection.py
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


async def main():
    print()
    print("=" * 60)
    print("🧪 POLYMARKET CONNECTION TEST")
    print("=" * 60)
    print()

    results = {}

    # Test 1: Environment variables
    print("1️⃣  Checking environment variables...")
    env_vars = {
        "POLYGON_WALLET_PRIVATE_KEY": os.getenv("POLYGON_WALLET_PRIVATE_KEY"),
        "POLYGON_FUNDER_ADDRESS": os.getenv("POLYGON_FUNDER_ADDRESS"),
        "POLYMARKET_SIGNATURE_TYPE": os.getenv("POLYMARKET_SIGNATURE_TYPE", "0"),
        "POLYMARKET_API_KEY": os.getenv("POLYMARKET_API_KEY"),
    }

    for key, val in env_vars.items():
        if val:
            if "KEY" in key or "SECRET" in key:
                display = f"{val[:8]}...{val[-4:]}" if len(val) > 12 else "***"
            else:
                display = val[:20] + "..." if len(val) > 20 else val
            print(f"   ✅ {key}: {display}")
        else:
            print(f"   ⚠️  {key}: Not set")

    results["Environment"] = all(v for k, v in env_vars.items() if k != "POLYMARKET_API_KEY")
    print()

    # Test 2: Gamma API (no auth)
    print("2️⃣  Testing Gamma API (market data)...")
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            url = "https://gamma-api.polymarket.com/markets?limit=3&active=true"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"   ✅ Gamma API: {len(data)} markets fetched")
                    for m in data[:2]:
                        q = m.get("question", "?")[:40]
                        vol = float(m.get("volume", 0) or 0)
                        print(f"      • {q}... (${vol:,.0f} vol)")
                    results["Gamma API"] = True
                else:
                    print(f"   ❌ Gamma API: HTTP {resp.status}")
                    results["Gamma API"] = False
    except Exception as e:
        print(f"   ❌ Gamma API: {e}")
        results["Gamma API"] = False
    print()

    # Test 3: CLOB API read (no auth)
    print("3️⃣  Testing CLOB API (order book - no auth)...")
    try:
        from py_clob_client.client import ClobClient
        client = ClobClient("https://clob.polymarket.com")

        # Test basic connectivity
        ok = client.get_ok()
        print(f"   ✅ CLOB OK: {ok}")

        server_time = client.get_server_time()
        print(f"   ✅ Server time: {server_time}")

        results["CLOB Read"] = True
    except Exception as e:
        print(f"   ❌ CLOB API: {e}")
        results["CLOB Read"] = False
    print()

    # Test 4: Trading client (requires auth)
    print("4️⃣  Testing Trading Client (requires wallet)...")
    private_key = os.getenv("POLYGON_WALLET_PRIVATE_KEY", "")
    funder = os.getenv("POLYGON_FUNDER_ADDRESS", "")
    sig_type = int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "0"))

    if not private_key or not funder:
        print("   ⏭️  Skipped (no wallet credentials)")
        results["Trading Client"] = None
    else:
        try:
            if not private_key.startswith("0x"):
                private_key = "0x" + private_key

            client = ClobClient(
                "https://clob.polymarket.com",
                key=private_key,
                chain_id=137,
                signature_type=sig_type,
                funder=funder
            )

            # Test API credential derivation
            creds = client.create_or_derive_api_creds()
            client.set_api_creds(creds)

            print(f"   ✅ Trading client initialized")
            print(f"   ✅ API Key: {creds.api_key[:20]}...")
            print(f"   ✅ Funder: {funder[:10]}...{funder[-6:]}")
            print(f"   ✅ Sig type: {sig_type}")

            results["Trading Client"] = True

        except Exception as e:
            print(f"   ❌ Trading client: {e}")
            results["Trading Client"] = False
    print()

    # Test 5: Full client
    print("5️⃣  Testing Full PolymarketClient...")
    try:
        from src.connectors.polymarket_client import polymarket
        print(f"   ✅ PolymarketClient imported")

        # Test get_markets
        markets = await polymarket.get_markets(limit=3, min_volume=10000, min_liquidity=1000)
        print(f"   ✅ get_markets: {len(markets)} markets")

        if markets:
            m = markets[0]
            print(f"   ✅ Sample market:")
            print(f"      Question: {m.question[:40]}...")
            print(f"      YES token: {m.yes_token_id[:20]}..." if m.yes_token_id else "      ⚠️  No YES token")
            print(f"      NO token: {m.no_token_id[:20]}..." if m.no_token_id else "      ⚠️  No NO token")
            print(f"      Tick size: {m.tick_size}")
            print(f"      Neg risk: {m.neg_risk}")

        results["PolymarketClient"] = True

    except Exception as e:
        print(f"   ❌ PolymarketClient: {e}")
        import traceback
        traceback.print_exc()
        results["PolymarketClient"] = False
    print()

    # Summary
    print("=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    for test, passed in results.items():
        if passed is True:
            print(f"   ✅ {test}")
        elif passed is False:
            print(f"   ❌ {test}")
        else:
            print(f"   ⏭️  {test} (skipped)")

    failed = sum(1 for v in results.values() if v is False)

    print()
    if failed == 0:
        print("✅ All tests passed!")
        print()
        print("Next steps:")
        print("  1. python scripts/generate_api_keys.py  (if not done)")
        print("  2. python scripts/pre_flight_check.py")
        print("  3. python scripts/paper_trading_24h.py --hours 1 --interval 10")
    else:
        print(f"❌ {failed} test(s) failed - fix issues above")

    print()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
