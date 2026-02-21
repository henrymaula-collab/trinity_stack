"""
Supply-Chain Lead-Lag Strategy (Network Momentum).

Illiquid Nordic small-cap suppliers underreact to earnings surprises or price
surges of their global mega-cap customers due to information asymmetry and
lack of analyst coverage.

Modules:
    data_pipeline: Load and process supply-chain relationships, prices, earnings.
    signal_engine: Cross-asset signal generation from customer triggers.
    execution_logic: T+1 entry and dynamic exit rules.
    backtest_network: Vectorized backtest with geometric return and hit rate.
"""
