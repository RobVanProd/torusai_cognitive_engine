import random
import time
import cProfile
import pstats
import io # For pstats output to string

# Relative imports for classes used by the stress test
from ..layer_04_dcg.concept_graph import EnhancedConceptGraph
from ..layer_13_imr.language_generator import EnhancedLanguageGenerator
from ..engine.torus_engine import TorusEngine
# Note: Specific layer classes like GeometryLayer etc. are no longer directly needed here
# as TorusEngine instantiates them internally.

def run_stress_test_core_logic(cg_class, lg_class, engine_class):
    """The core logic of the stress test, to be profiled."""
    cg_instance = cg_class()
    node_ids = [f"node{i}" for i in range(500)]
    for i, node_id in enumerate(node_ids):
        role = 'subject' if i % 5 == 0 else ('verb' if i % 5 in (1, 2) else 'object')
        cg_instance.addEnhancedNode(node_id,
                                  properties={"type": role},
                                  linguistics={"wordForms": {"base": node_id}, "syntacticRoles": [role]})
        cg_instance.setActivation(node_id, random.random() * 0.2)

    for _ in range(1000):
        if len(node_ids) < 2: break
        s_node_id, t_node_id = random.sample(node_ids, 2)
        cg_instance.addEdge(s_node_id, t_node_id, weight=random.random())

    narrative_log = []
    lg_instance = lg_class(cg_instance, narrative_log)

    engine_params = {
        'decay_awake': 0.02,
        'decay_dream': 0.05,
        'hebbian_multiplier': 1.1,
        'prune_threshold': 0.2,
        'dream_n_candidates': 5,
        'dream_keep_top_k': 2
    }
    engine_instance = engine_class(cg_instance=cg_instance, lg_instance=lg_instance, params=engine_params)

    start_run_time = time.perf_counter()

    for i in range(min(10, len(node_ids))):
        engine_instance.standard_cycle(query=f"What does {node_ids[i]} do?")

    for _ in range(min(2, len(node_ids))):
        if not node_ids: break
        engine_instance.dream_cycle_execution()

    end_run_time = time.perf_counter()

    results = {
        "run_time_sec_total": round(end_run_time - start_run_time, 4),
        "node_count_final": len(cg_instance.nodes),
        "edge_count_final": len(cg_instance.edges),
        "dreams_logged": engine_instance.state.get("dream_log", []),
        "narrative_samples": narrative_log[-5:]
    }
    return results

def run_stress_test(cg_class, lg_class, engine_class):
    profiler = cProfile.Profile()
    profiler.enable()

    results = run_stress_test_core_logic(cg_class, lg_class, engine_class)

    profiler.disable()
    
    # Save profiling stats
    profile_file_path = "stress_test_profile.prof"
    profiler.dump_stats(profile_file_path)
    print(f"\nProfiling data saved to {profile_file_path}")

    # Optionally, print some stats to console
    s = io.StringIO()
    sortby = pstats.SortKey.CUMULATIVE # Can also use 'tottime'
    ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
    ps.print_stats(20) # Print top 20 lines
    print("\n--- Top 20 Profile Stats (Cumulative Time) ---")
    print(s.getvalue())
    print("---------------------------------------------")


    # Save results to a TXT file
    try:
        with open("stress_test_results.txt", "w") as f:
            f.write("--- Stress Test Results ---\n")
            for key, value in results.items():
                if isinstance(value, list) and key in ["dreams_logged", "narrative_samples"]:
                    f.write(f"{key}:\n")
                    for item_val in value:
                        f.write(f"  - {item_val}\n")
                else:
                    f.write(f"{key}: {value}\n")
            f.write("\n--- Profiling Summary (Top 20 Cumulative) ---\n")
            f.write(s.getvalue())

        print("\nStress test results and profiling summary saved to stress_test_results.txt")
    except Exception as e:
        print(f"\nError saving stress test results to file: {e}")

    return results
