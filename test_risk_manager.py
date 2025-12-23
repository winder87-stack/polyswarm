#!/usr/bin/env python3
"""
Quick test of the Risk Manager functionality
"""

from src.strategies import RiskManager, RiskLimits

# Test basic functionality
print("🧪 Testing Risk Manager...")

# Create custom limits
limits = RiskLimits(
    max_daily_loss=500.0,
    max_position_size=200.0,
    min_edge=0.08
)

# Initialize risk manager
risk_mgr = RiskManager(limits)
print("✅ Risk Manager initialized")

# Test Kelly sizing
kelly_size = risk_mgr.get_kelly_size(0.10, 0.75, 1000.0)
print(f"🎯 Kelly size for 10% edge, 75% confidence: ${kelly_size:.2f}")

# Test correlation
correlation = risk_mgr.check_correlation("bitcoin", "crypto")
print(f"🔗 Correlation between bitcoin and crypto: {correlation:.2f}")

# Test daily stats
stats = risk_mgr.get_daily_stats()
print(f"📊 Daily P&L: ${stats['daily_pnl']:.2f}")

print("🎉 Risk Manager test completed successfully!")
