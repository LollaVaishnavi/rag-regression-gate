"""ragscore: decide whether a retrieval change is a regression, an improvement, or unknowable.

The package exists because a pass or fail threshold on a golden set answers a question the set
cannot support: it reports a difference in a mean metric without any model of how much that mean
would move if a different sample of queries had been drawn. Every module here is organised around
restoring that missing model of variance, so the gate can return a third verdict, "inconclusive",
that a boolean threshold structurally cannot.
"""

from ragscore.gate import GateVerdict, Verdict, gate

__all__ = ["GateVerdict", "Verdict", "gate"]
