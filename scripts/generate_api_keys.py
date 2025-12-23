#!/usr/bin/env python3
"""
Generate Polymarket API Credentials

This script derives API credentials from your wallet.
The credentials are deterministic - running again gives same keys.

Usage:
    python scripts/generate_api_keys.py

Requirements:
    - POLYGON_WALLET_PRIVATE_KEY in .env
    - POLYGON_FUNDER_ADDRESS in .env
    - POLYMARKET_SIGNATURE_TYPE in .env (0=MetaMask, 1=Magic/Email)
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from py_clob_client.client import ClobClient

HOST = "https://clob.polymarket.com"
CHAIN_ID = 137


def main():
    print("=" * 60)
    print("🔑 POLYMARKET API KEY GENERATOR")
    print("=" * 60)
    print()

    # Get credentials from env
    private_key = os.getenv("POLYGON_WALLET_PRIVATE_KEY", "")
    funder = os.getenv("POLYGON_FUNDER_ADDRESS", "")
    sig_type = int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "0"))

    # Validate
    if not private_key:
        print("❌ ERROR: POLYGON_WALLET_PRIVATE_KEY not set in .env")
        print("   Add your wallet's private key to .env file")
        return 1

    if not funder:
        print("❌ ERROR: POLYGON_FUNDER_ADDRESS not set in .env")
        print("   Add your wallet's public address to .env file")
        return 1

    # Add 0x prefix if missing
    if not private_key.startswith("0x"):
        private_key = "0x" + private_key

    print(f"Funder Address: {funder}")
    print(f"Signature Type: {sig_type} ({'EOA/MetaMask' if sig_type == 0 else 'Magic/Email'})")
    print()

    # Initialize client
    try:
        print("Initializing client...")
        client = ClobClient(
            HOST,
            key=private_key,
            chain_id=CHAIN_ID,
            signature_type=sig_type,
            funder=funder
        )
    except Exception as e:
        print(f"❌ Failed to initialize client: {e}")
        return 1

    # Generate credentials
    try:
        print("Generating API credentials...")
        creds = client.create_or_derive_api_creds()

        print()
        print("=" * 60)
        print("✅ API CREDENTIALS GENERATED SUCCESSFULLY")
        print("=" * 60)
        print()
        print("Add these lines to your .env file:")
        print()
        print(f"POLYMARKET_API_KEY={creds.api_key}")
        print(f"POLYMARKET_API_SECRET={creds.api_secret}")
        print(f"POLYMARKET_API_PASSPHRASE={creds.api_passphrase}")
        print()
        print("=" * 60)
        print()
        print("💡 These credentials are derived from your wallet.")
        print("   Running this script again will generate the same keys.")
        print()

        return 0

    except Exception as e:
        print(f"❌ Failed to generate credentials: {e}")
        print()
        print("Common issues:")
        print("  - Invalid private key format")
        print("  - Network connectivity issues")
        print("  - Wrong signature type")
        return 1


if __name__ == "__main__":
    sys.exit(main())
