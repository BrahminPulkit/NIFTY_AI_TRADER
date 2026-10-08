from ..core import FixedExecution, backtest
def run(signals): return backtest(signals, FixedExecution())
