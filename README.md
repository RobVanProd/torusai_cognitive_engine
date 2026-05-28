# torusai_cognitive_engine

TorusAI has 15 layer directories, 45 Python files, and a Streamlit dashboard, but its pytest collection failed locally on 2026-05-28 because the stress test uses a relative import path that pytest collected incorrectly.

## What It Is

TorusAI is a cognitive-engine prototype built around a layered concept-graph architecture. The code includes graph activation, Hebbian learning, symbolic tagging/reflection, metrics observation, affective state, reward/social influence modules, dream logic, a learning orchestrator, language generation, document-to-graph processing, ConceptNet loading scaffolds, persistence helpers, a CLI, a stress-test script, and a dashboard.

The interesting part is the explicit architecture: the repo maps cognitive behaviors into named layers and keeps most of them as inspectable Python modules.

## Current Status

The repo includes a `stress_test_profile.prof` artifact and a `tests/stress_test.py` file. A local `python -m pytest -q` run failed during collection with `ImportError: attempted relative import beyond top-level package`.

That is a packaging/test-discovery issue, not evidence that the engine works or fails. It should be fixed before publishing test claims.

## Tech Stack

- Python
- Streamlit
- PyTorch dependency
- NumPy, pandas, SciPy
- Plotly/matplotlib/seaborn
- PDF/DOCX parsing dependencies
- NLTK, spaCy, sentence-transformers dependencies

## Limitations

This is a research prototype. The README should avoid claiming a complete cognitive system. Current validation should focus on import hygiene, unit tests for each layer, stress-test reproducibility, and dashboard smoke tests.
