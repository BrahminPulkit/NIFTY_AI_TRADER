import unittest
from datetime import date

from src.risk_manager import RiskManager


class RiskManagerTests(unittest.TestCase):
    def test_position_size_obeys_notional_and_lot_limit(self):
        risk = RiskManager(capital=100000, risk_per_trade=0.01, max_notional_pct=0.10, lot_size=25)
        self.assertEqual(risk.position_size(100, 98), 100)

    def test_daily_loss_blocks_new_trade(self):
        session = date(2026, 7, 17)
        risk = RiskManager(capital=100000, max_daily_loss_pct=0.02)
        risk.register_close(-2000, session)
        allowed, reason = risk.can_open(session)
        self.assertFalse(allowed)
        self.assertEqual(reason, "daily loss limit reached")

    def test_trade_plan_requires_direction(self):
        risk = RiskManager()
        plan, reason = risk.create_plan(0, 100, 99, date(2026, 7, 17))
        self.assertIsNone(plan)
        self.assertEqual(reason, "flat signal")
