# To run this script as part of the package, execute from the directory
# containing 'torusai_cognitive_engine':
# python -m torusai_cognitive_engine.main

def run_interactive_mode(engine, narrative_log_ref):
    """Runs the TorusEngine in an interactive command-line mode."""
    print("\n--- TorusAI Interactive Mode ---")
    print("Type your query, or one of the following commands:")
    print("  dream          : Run a dream cycle.")
    print("  status         : Show current engine status (metrics, valence, reflection).")
    print("  run_sequence N : Run N dream cycles (simulates continuous activity).")
    print("  quit           : Exit interactive mode.")
    print("---------------------------------")

    while True:
        try:
            user_input = input("TorusAI> ").strip()
            if not user_input:
                continue

            if user_input.lower() == "quit":
                print("Exiting interactive mode.")
                break
            elif user_input.lower() == "dream":
                print("Running a dream cycle...")
                engine.dream_cycle_execution()
                print(f"  Dream Log: {engine.state.get('dream_log')}")
                print(f"  Metrics: {engine.state.get('metrics')}")
            elif user_input.lower() == "status":
                print("--- Engine Status ---")
                print(f"  Metrics: {engine.state.get('metrics')}")
                print(f"  Valence: {engine.state.get('valence')}")
                print(f"  Reflection: {engine.state.get('reflection')}")
                print(f"  Last Response: {engine.state.get('last_response')}")
                print(f"  Narrative Log (last 5): {narrative_log_ref[-5:]}")
                print("---------------------")
            elif user_input.lower().startswith("run_sequence"):
                parts = user_input.split()
                if len(parts) == 2 and parts[1].isdigit():
                    num_cycles = int(parts[1])
                    print(f"Running {num_cycles} dream cycles...")
                    for i in range(num_cycles):
                        engine.dream_cycle_execution()
                        print(f"  Cycle {i+1}: Dream Log: {engine.state.get('dream_log')[-1 if engine.state.get('dream_log') else 'N/A']}")
                    print("Sequence finished.")
                else:
                    print("Invalid run_sequence command. Usage: run_sequence <number_of_cycles>")
            else: # Treat as a query
                print(f"Processing query: '{user_input}'")
                response = engine.standard_cycle(query=user_input)
                print(f"  Response: {response}")
                print(f"  Narrative Log: {narrative_log_ref[-1 if narrative_log_ref else 'N/A']}")
                print(f"  Valence: {engine.state.get('valence')}")
                print(f"  Reflection: {engine.state.get('reflection')}")
                print(f"  Metrics: {engine.state.get('metrics')}")
        except KeyboardInterrupt:
            print("\nExiting interactive mode (Ctrl+C).")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    try:
        from .layer_04_dcg.concept_graph import EnhancedConceptGraph
        from .layer_13_imr.language_generator import EnhancedLanguageGenerator
        from .engine.torus_engine import TorusEngine
        from .tests.stress_test import run_stress_test
    except ImportError as e:
        print(f"ImportError: {e}. Ensure you are running this as part of the torusai_cognitive_engine package.")
        print("For example, from the directory containing 'torusai_cognitive_engine', run: python -m torusai_cognitive_engine.main")
        exit(1)

    # --- Optional: Run Stress Test ---
    # print("--- Running TorusAI Cognitive Engine Stress Test (via main.py) ---")
    # stress_test_results = run_stress_test(
    #     cg_class=EnhancedConceptGraph,
    #     lg_class=EnhancedLanguageGenerator,
    #     engine_class=TorusEngine
    # )
    # print("\n--- Stress Test Results ---")
    # for key, value in stress_test_results.items():
    #     if isinstance(value, list) and key in ["dreams_logged", "narrative_samples"]:
    #         print(f"{key}:")
    #         for item_idx, item_val in enumerate(value):
    #             print(f"  - {item_val}")
    #     else:
    #         print(f"{key}: {value}")
    # print("\n--- End of Stress Test ---")


    # --- Setup for Interactive Mode ---
    print("\n--- Initializing Engine for Interactive Mode ---")
    cg_interactive = EnhancedConceptGraph()
    # Add some initial concepts for a richer interaction
    cg_interactive.addEnhancedNode("user", linguistics={"wordForms":{"base":"user"}}, properties={"type":"agent"})
    cg_interactive.addEnhancedNode("system", linguistics={"wordForms":{"base":"system"}}, properties={"type":"entity"})
    cg_interactive.addEnhancedNode("information", linguistics={"wordForms":{"base":"information"}}, properties={"type":"concept"})
    cg_interactive.addEnhancedNode("status", linguistics={"wordForms":{"base":"status"}}, properties={"type":"concept"})
    cg_interactive.addEnhancedNode("time", linguistics={"wordForms":{"base":"time"}}, properties={"type":"concept"})
    cg_interactive.addEnhancedNode("weather", linguistics={"wordForms":{"base":"weather"}}, properties={"type":"concept"})
    
    cg_interactive.addEdge("user", "system", weight=0.8, relation="interacts_with")
    cg_interactive.addEdge("system", "information", weight=0.9, relation="provides")
    cg_interactive.addEdge("system", "status", weight=0.9, relation="has_a")


    narrative_for_interactive_lg = [] # This list will be passed by reference
    lg_interactive = EnhancedLanguageGenerator(cg_interactive, narrative_for_interactive_lg)
    
    engine_params = {
        'decay_awake': 0.02,
        'decay_dream': 0.05,
        'hebbian_multiplier': 1.1,
        'prune_threshold': 0.2,
        'dream_n_candidates': 5,
        'dream_keep_top_k': 2
    }
    engine_interactive = TorusEngine(cg_instance=cg_interactive, lg_instance=lg_interactive, params=engine_params)

    print("Initial interactive CG nodes:", cg_interactive.nodes)
    print("Initial interactive CG edges:", cg_interactive.edges)
    print(f"Initial interactive engine valence: {engine_interactive.state.get('valence')}")
    
    run_interactive_mode(engine_interactive, narrative_for_interactive_lg)
