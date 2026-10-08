"""Risk limits for paper and broker execution layers."""

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class TradePlan:
    side: int
    quantity: int
    entry: float
    stop_loss: float
    risk_amount: float


class RiskManager:
    def __init__(
        self,
        capital=100000,
        risk_per_trade=0.005,
        max_daily_loss_pct=0.02,
        max_open_positions=1,
        max_trades_per_day=3,
        max_notional_pct=0.25,
        lot_size=1,
    ):
        if capital <= 0 or not 0 < risk_per_trade <= 1 or not 0 < max_daily_loss_pct <= 1:
            raise ValueError("capital and percentage limits must be positive")
        if max_open_positions < 1 or max_trades_per_day < 1:
            raise ValueError("position and trade limits must be positive integers")
        if not 0 < max_notional_pct <= 1 or lot_size < 1:
            raise ValueError("max_notional_pct must be in (0, 1] and lot_size must be positive")
        self.capital = float(capital)
        self.risk_per_trade = risk_per_trade
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_open_positions = max_open_positions
        self.max_trades_per_day = max_trades_per_day
        self.max_notional_pct = max_notional_pct
        self.lot_size = lot_size
        self._session = None
        self._session_start_capital = self.capital
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.open_positions = 0

    def _reset_session(self, session: date):
        if session != self._session:
            self._session = session
            self._session_start_capital = self.capital
            self.daily_pnl = 0.0
            self.trades_today = 0

    def risk_amount(self):
        return self.capital * self.risk_per_trade

    def can_open(self, session: date):
        self._reset_session(session)
        if self.daily_pnl <= -(self._session_start_capital * self.max_daily_loss_pct):
            return False, "daily loss limit reached"
        if self.open_positions >= self.max_open_positions:
            return False, "maximum open positions reached"
        if self.trades_today >= self.max_trades_per_day:
            return False, "maximum daily trades reached"
        return True, "approved"

    def position_size(self, entry, stop_loss):
        risk_per_unit = abs(entry - stop_loss)
        if entry <= 0 or risk_per_unit <= 0:
            return 0
        risk_limited = int(self.risk_amount() // risk_per_unit)
        notional_limited = int((self.capital * self.max_notional_pct) // entry)
        quantity = min(risk_limited, notional_limited)
        return (quantity // self.lot_size) * self.lot_size

    def create_plan(self, side, entry, stop_loss, session):
        if side not in (1, -1):
            return None, "flat signal"
        allowed, reason = self.can_open(session)
        if not allowed:
            return None, reason
        quantity = self.position_size(entry, stop_loss)
        if quantity < self.lot_size:
            return None, "position size below one lot"
        return TradePlan(side, quantity, entry, stop_loss, quantity * abs(entry - stop_loss)), "approved"

    def register_open(self, session):
        self._reset_session(session)
        self.open_positions += 1
        self.trades_today += 1

    def register_close(self, pnl, session):
        self._reset_session(session)
        self.open_positions = max(0, self.open_positions - 1)
        self.daily_pnl += pnl
        self.capital += pnl
        return self.capital
