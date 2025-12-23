#!/usr/bin/env python3
"""
Generate a New Polygon Wallet for Polymarket Trading

Creates a fresh wallet that you can fund for trading.
This is safer than using your main wallet.

Usage:
    python scripts/generate_wallet.py
"""

import secrets


def main():
    try:
        from eth_account import Account
    except ImportError:
        print("❌ eth_account not installed")
        print("   Run: pip install eth-account")
        return 1

    print()
    print("=" * 60)
    print("🔐 NEW POLYGON WALLET GENERATOR")
    print("=" * 60)
    print()

    # Generate secure random private key
    private_key = "0x" + secrets.token_hex(32)

    # Derive account from private key
    account = Account.from_key(private_key)

    print("✅ NEW WALLET GENERATED")
    print()
    print(f"Public Address:  {account.address}")
    print(f"Private Key:     {private_key}")
    print()
    print("⚠️  IMPORTANT - SAVE YOUR PRIVATE KEY SECURELY!")
    print("   Anyone with this key can access your funds.")
    print()
    print("-" * 60)
    print()
    print("Add to your .env file:")
    print()
    print(f"POLYGON_WALLET_PRIVATE_KEY={private_key[2:]}")  # Remove 0x
    print(f"POLYGON_FUNDER_ADDRESS={account.address}")
    print("POLYMARKET_SIGNATURE_TYPE=0")
    print()
    print("-" * 60)
    print()
    print("NEXT STEPS:")
    print()
    print("1. Save the private key securely (password manager)")
    print()
    print("2. Fund your wallet:")
    print(f"   - Send MATIC to {account.address} (for gas, ~$1-5)")
    print(f"   - Send USDC.e to {account.address} (for trading)")
    print()
    print("3. Generate API keys:")
    print("   python scripts/generate_api_keys.py")
    print()
    print("4. Set allowances:")
    print("   python scripts/set_allowances.py")
    print()
    print("5. Start paper trading:")
    print("   python scripts/paper_trading_24h.py --hours 1")
    print()

    return 0


if __name__ == "__main__":
    exit(main())
