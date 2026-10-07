"""Views package for the European Soccer Match Outcome Predictor.

Group: 2026-AI-09 | SLIIT Machine Learning Module IT3091
"""

from .simulator import render_simulator
from .evaluator import render_evaluator
from .comparison import render_comparison

__all__ = ["render_simulator", "render_evaluator", "render_comparison"]
