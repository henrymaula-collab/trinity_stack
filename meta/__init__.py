"""
Meta-Portfolio Layer — Master Allocator.

Orchestrates three strategies at a high abstraction level:
  1. Trinity Stack (Core): 80% — systematic base portfolio.
  2. Supply-Chain Lead-Lag (Tactical Spear): 20% — uncorrelated tactical bets.
  3. Debt Wall (Veto Shield): Blacklist — blocks dangerous long positions.

Does not merge source code; each strategy remains isolated.
"""
