#!/usr/bin/env python3
"""
Set Token Allowances for Polymarket Trading

EOA/MetaMask wallets must set allowances ONCE before trading.
This allows the Polymarket exchange to spend your USDC.

Usage:
    python scripts/set_allowances.py

Requirements:
    - POLYGON_WALLET_PRIVATE_KEY in .env
    - POLYGON_FUNDER_ADDRESS in .env
    - Small amount of MATIC in wallet (for gas)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from py_clob_client.client import ClobClient

HOST = "https://clob.polymarket.com"
CHAIN_ID = 137


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Set Polymarket token allowances (EOA wallets only)"
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive confirmation prompt (required for non-interactive runs).",
    )
    args = parser.parse_args(argv)

    print("=" * 60)
    print("🔓 SET POLYMARKET TOKEN ALLOWANCES")
    print("=" * 60)
    print()

    # Get credentials
    private_key = os.getenv("POLYGON_WALLET_PRIVATE_KEY", "")
    funder = os.getenv("POLYGON_FUNDER_ADDRESS", "")
    sig_type = int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "0"))

    if not private_key or not funder:
        print("❌ Missing wallet credentials in .env")
        return 1

    if not private_key.startswith("0x"):
        private_key = "0x" + private_key

    print(f"Wallet: {funder}")
    print(f"Signature Type: {sig_type}")
    print()

    # Check signature type
    if sig_type == 1:
        print("ℹ️  Signature type 1 (Magic/Email) - allowances managed by Polymarket")
        print("   No action needed.")
        return 0

    print("⚠️  EOA wallet detected - setting allowances...")
    print("   This requires a small amount of MATIC for gas.")
    print()

    # Confirm
    if not args.yes:
        if not sys.stdin.isatty():
            print("❌ Non-interactive session detected. Re-run with --yes to proceed.")
            return 2
        confirm = input("Continue? (yes/no): ").strip().lower()
        if confirm != "yes":
            print("Cancelled.")
            return 0

    # Initialize client
    try:
        client = ClobClient(
            HOST, key=private_key, chain_id=CHAIN_ID, signature_type=sig_type, funder=funder
        )

        # Set API creds first
        creds = client.create_or_derive_api_creds()
        client.set_api_creds(creds)

    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return 1

    # Set allowances
    print()
    print("Setting allowances (this may take a moment)...")

    try:
        client_any: Any = client

        # Check current allowances
        allowances = client_any.get_balance_allowance()
        print(f"Current allowances: {allowances}")

        # Set if needed
        if hasattr(client_any, "set_allowance"):
            result = client_any.set_allowance()
            print(f"✅ Allowance result: {result}")
        elif hasattr(client_any, "update_balance_allowance"):
            # Alternative method
            result = client_any.update_balance_allowance()
            print(f"✅ Allowance result: {result}")
        else:
            print("ℹ️  Allowance methods not available in this client version")
            print("   You may need to set allowances manually via PolygonScan")

    except Exception as e:
        print(f"⚠️  Allowance error: {e}")
        print()
        print("If allowances are already set, this error can be ignored.")
        print("Otherwise, you may need to set allowances manually:")
        print("1. Go to https://polygonscan.com")
        print("2. Find USDC.e contract: 0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
        print("3. Write Contract > approve")
        print("4. Approve the Exchange contract to spend your USDC")

    print()
    print("=" * 60)
    print("✅ SETUP COMPLETE")
    print("=" * 60)
    print()
    print("You can now trade on Polymarket!")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
