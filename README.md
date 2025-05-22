# TorusAI Cognitive Engine

This project contains the TorusAI Cognitive Engine, a system designed for dynamic knowledge representation and reasoning, along with a Streamlit-based interactive dashboard.

## Core Components

*   **`torusai_cognitive_engine/`**: The main Python package for the cognitive engine.
    *   Implements a 15-layer cognitive architecture (details can be found within the source code comments and original design specifications if available).
    *   Key modules include concept graph management, activation dynamics, Hebbian learning, symbolic reflection, language generation, affective modeling, and more.
    *   `engine/torus_engine.py`: Orchestrates the layers and manages the engine's state.
    *   `main.py`: Provides a command-line interface for interactive sessions with the engine.
    *   `tests/stress_test.py`: A script to test engine performance and behavior.
*   **`dashboard.py`**: A Streamlit application that provides a web-based graphical user interface to interact with the TorusAI engine. It allows users to send queries, trigger dream cycles, and view engine status in real-time.
*   **`requirements.txt`**: Lists the Python dependencies required to run the engine and the dashboard.
*   **`stress_test_results.txt`**: Example output from the stress test.

## Getting Started

### Prerequisites

*   Python 3.9+
*   Git

### Installation

1.  **Clone the repository:**
    ```bash
    # If you are setting this up in a new repository:
    # git clone https://github.com/RobVanProd/torusai_cognitive_engine.git
    # cd torusai_cognitive_engine
    ```

2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Interactive CLI

To interact with the engine via the command line:

```bash
python -m torusai_cognitive_engine.main
```
This will start an interactive session where you can type queries or commands like `dream`, `status`, or `quit`.

### Running the Dashboard

To use the web-based dashboard:

```bash
streamlit run dashboard.py
```
This will start a local web server, and your browser should open to the dashboard interface automatically.

## Overview of the Engine

The TorusAI Cognitive Engine is built upon a concept graph where nodes represent concepts and edges represent relationships. The engine processes information through a series of layers that handle:
*   Activation spread and decay
*   Symbolic tagging and typing
*   Feedback modulation
*   Hebbian learning (strengthening/weakening connections)
*   Identity tracking
*   Symbolic reflection (identifying salient patterns)
*   Metrics observation
*   Language generation for responses
*   Affective state modeling
*   Dreaming (offline processing and consolidation)

The `main.py` script provides an interactive command-line interface, while `dashboard.py` offers a more user-friendly Streamlit GUI for exploration.

## Contributing

Contributions are welcome. Please refer to `CONTRIBUTING.md` if available, or open an issue to discuss potential changes.

## License

(To be determined - consider adding an MIT License file if appropriate)
