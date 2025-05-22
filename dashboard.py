import streamlit as st
import sys
import os

# Adjust path to import from torusai_cognitive_engine
# Assuming dashboard.py is in Liquid-neural-network/
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

try:
    from torusai_cognitive_engine.layer_04_dcg.concept_graph import EnhancedConceptGraph
    from torusai_cognitive_engine.layer_13_imr.language_generator import EnhancedLanguageGenerator
    from torusai_cognitive_engine.engine.torus_engine import TorusEngine
except ImportError as e:
    st.error(f"ImportError: {e}. Ensure 'torusai_cognitive_engine' is in the Python path.")
    st.stop()

def initialize_engine():
    """Initializes and returns the TorusEngine and its narrative log."""
    cg = EnhancedConceptGraph()
    # Add some initial concepts for a richer interaction
    cg.addEnhancedNode("user", linguistics={"wordForms":{"base":"user"}}, properties={"type":"agent"})
    cg.addEnhancedNode("system", linguistics={"wordForms":{"base":"system"}}, properties={"type":"entity"})
    cg.addEnhancedNode("information", linguistics={"wordForms":{"base":"information"}}, properties={"type":"concept"})
    cg.addEnhancedNode("status", linguistics={"wordForms":{"base":"status"}}, properties={"type":"concept"})
    cg.addEnhancedNode("time", linguistics={"wordForms":{"base":"time"}}, properties={"type":"concept"})
    cg.addEnhancedNode("weather", linguistics={"wordForms":{"base":"weather"}}, properties={"type":"concept"})
    
    cg.addEdge("user", "system", weight=0.8, relation="interacts_with")
    cg.addEdge("system", "information", weight=0.9, relation="provides")
    cg.addEdge("system", "status", weight=0.9, relation="has_a")

    narrative_log = [] 
    lg = EnhancedLanguageGenerator(cg, narrative_log)
    
    engine_params = {
        'decay_awake': 0.02, 
        'decay_dream': 0.05,
        'hebbian_multiplier': 1.1,
        'prune_threshold': 0.2,
        'dream_n_candidates': 5,
        'dream_keep_top_k': 2
    }
    engine = TorusEngine(cg_instance=cg, lg_instance=lg, params=engine_params)
    return engine, narrative_log

st.set_page_config(layout="wide")
st.title("🧠 TorusAI Cognitive Engine Dashboard")

# Initialize engine and store in session state
if 'engine' not in st.session_state:
    engine, narrative_log = initialize_engine()
    st.session_state.engine = engine
    st.session_state.narrative_log = narrative_log
    st.session_state.chat_history = [] # To store query-response pairs

engine = st.session_state.engine
narrative_log = st.session_state.narrative_log
chat_history = st.session_state.chat_history

# --- Main Interaction Area ---
st.header("Interactive Query")
user_query = st.text_input("Enter your query for TorusAI:", key="query_input")

col_query_btn, col_dream_btn, col_status_btn = st.columns(3)

if col_query_btn.button("Send Query", key="send_query_btn"):
    if user_query:
        with st.spinner("TorusAI is thinking..."):
            response = engine.standard_cycle(query=user_query)
            chat_history.append({"query": user_query, "response": response, "type": "query"})
            st.rerun() # Rerun to update displays
    else:
        st.warning("Please enter a query.")

if col_dream_btn.button("Run Dream Cycle", key="dream_btn"):
    with st.spinner("TorusAI is dreaming..."):
        engine.dream_cycle_execution()
        latest_dream = engine.state.get('dream_log', [])
        if latest_dream:
             chat_history.append({"query": "Dream Cycle Initiated", "response": f"Latest Dream: {latest_dream[-1]}", "type": "dream"})
        else:
            chat_history.append({"query": "Dream Cycle Initiated", "response": "No dreams logged yet.", "type": "dream"})
        st.rerun()

if col_status_btn.button("Show Engine Status", key="status_btn"):
    metrics = engine.state.get('metrics', {})
    valence = engine.state.get('valence', 'N/A')
    reflection = engine.state.get('reflection', [])
    last_response = engine.state.get('last_response', 'N/A')
    
    status_md = f"""
    **Engine Status:**
    - **Metrics:** `{metrics}`
    - **Valence:** `{valence}`
    - **Reflection:** `{reflection}`
    - **Last Engine Response:** `{last_response}`
    - **Narrative Log (last 5):** `{narrative_log[-5:]}`
    """
    chat_history.append({"query": "Engine Status Requested", "response": status_md, "type": "status"})
    st.rerun()


# --- Display Areas ---
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Engine State")
    st.json(engine.state, expanded=False)
    
    st.subheader("📜 Narrative Log (Latest 5)")
    if narrative_log:
        for entry in reversed(narrative_log[-5:]):
            st.text(entry)
    else:
        st.text("Narrative log is empty.")

with col2:
    st.subheader("💬 Chat / Event Log")
    if not chat_history:
        st.info("No interactions yet. Send a query or run a dream cycle.")
    else:
        for entry in reversed(chat_history):
            if entry["type"] == "query":
                st.markdown(f"**You:** {entry['query']}")
                st.markdown(f"**TorusAI:** {entry['response']}")
            elif entry["type"] == "dream":
                st.markdown(f"*{entry['query']}*")
                st.markdown(f"*{entry['response']}*")
            elif entry["type"] == "status":
                 st.markdown(f"*{entry['query']}*")
                 st.markdown(entry['response'], unsafe_allow_html=True) # For markdown in status
            st.markdown("---")


st.sidebar.header("Engine Parameters")
st.sidebar.json(engine.params)

st.sidebar.header("Concept Graph")
st.sidebar.caption("Nodes:")
st.sidebar.json(engine.cg.nodes, expanded=False)
st.sidebar.caption("Edges:")
st.sidebar.json(engine.cg.edges, expanded=False)

st.sidebar.info("This dashboard provides a basic interface to interact with the TorusAI engine.")