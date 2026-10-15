"""European Soccer Match Outcome Predictor package.

Group: 2026-AI-09, SLIIT ML Module IT3091
"""

from .inference import SoccerInferenceEngine
from .features import (
    SIMULATOR_CORE_FEATURES,
    extract_simulator_inputs,
    get_simulator_default,
    get_simulator_slider_specs,
)

__all__ = [
    "SoccerInferenceEngine",
    "SIMULATOR_CORE_FEATURES",
    "extract_simulator_inputs",
    "get_simulator_default",
    "get_simulator_slider_specs",
]
