"""
Debt Wall Arbitrage Strategy (Micro-Cap Refinancing Risk).

Retail investors systematically misprice refinancing risk of illiquid small-caps.
Companies with negative FCF and impending debt maturity (6–9 months) will be
forced into highly dilutive rights issues, crushing the stock price.

Modules:
    data_pipeline: Load Capital IQ / Bloomberg debt maturity data.
    arbitrage_engine: Cash Runway Deficit and SHORT/EXCLUDE signal logic.
    cost_of_borrowing: Risk-free vs debt rate overlay for conviction.
    backtest_debt_wall: Short backtest with Omega Ratio and CAGR.
"""
