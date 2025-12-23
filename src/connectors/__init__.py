"""
API Connectors Package

Provides connections to external services including Polymarket APIs.
"""

from .polymarket_client import Market, PolymarketClient, polymarket

# The polymarket singleton is already created in polymarket_client.py
# Just re-export it here for convenience

__all__ = [
    "Market",
    "PolymarketClient",
    "polymarket"
]
