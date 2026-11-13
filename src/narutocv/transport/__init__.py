"""Local browser<->Python transport bridge for the NarutoCV recognition
runtime (game_implementation_plan.md, Phase A).

This package only carries frames and events between a browser client and the
existing `cv_model.inference` runtime. It never re-implements recognition,
thresholds, or decisions -- see server.py for the exact boundary.
"""
