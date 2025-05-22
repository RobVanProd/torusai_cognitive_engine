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
    from torusai_cognitive_engine.document_processor.parser import parse_document
    from torusai_cognitive_engine.document_processor.text_to_graph import populate_graph_from_text
    from torusai_cognitive_engine.document_processor.model_io import save_engine_state, load_engine_state
except ImportError as e:
    st.error(f"ImportError: {e}. Ensure 'torusai_cognitive_engine' and its submodules are in the Python path and all dependencies are installed (pdfplumber, python-docx, nltk).")
    st.stop()

# --- Engine Initialization ---
def initialize_engine():
    """Initializes and returns a new TorusEngine instance and its narrative log."""
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

engine: TorusEngine = st.session_state.engine
narrative_log: list = st.session_state.narrative_log
chat_history: list = st.session_state.chat_history

# --- Sidebar for Document Processing and Model I/O ---
st.sidebar.divider()
st.sidebar.header("📄 Document Processing")
uploaded_doc = st.sidebar.file_uploader("Upload Document (PDF, DOCX, TXT)", type=['pdf', 'docx', 'txt'], key="doc_uploader")
post_process_cycles = st.sidebar.number_input("Post-Processing Dream Cycles:", min_value=0, value=3, step=1, key="post_process_cycles",
                                             help="Number of dream cycles to run after document processing to help integrate new information.")

if st.sidebar.button("Process Uploaded Document", key="process_doc_btn"):
    if uploaded_doc is not None:
        with st.spinner(f"Processing {uploaded_doc.name}..."):
            try:
                document_text = parse_document(uploaded_doc, uploaded_doc.name)
                if document_text:
                    st.sidebar.success(f"Extracted {len(document_text)} characters from {uploaded_doc.name}.")
                    
                    with st.spinner("Populating concept graph from document..."):
                        populate_graph_from_text(document_text, engine.cg, engine.state)
                    st.sidebar.success(f"Document content processed. Graph: {len(engine.cg.nodes)} nodes, {len(engine.cg.edges)} edges.")
                    
                    chat_history.append({
                        "query": f"Processed Document: {uploaded_doc.name}",
                        "response": f"Extracted {len(document_text)} chars. Nodes: {len(engine.cg.nodes)}, Edges: {len(engine.cg.edges)}",
                        "type": "system"
                    })

                    if post_process_cycles > 0:
                        st.sidebar.info(f"Running {post_process_cycles} post-processing dream cycles...")
                        progress_bar = st.sidebar.progress(0)
                        for i in range(post_process_cycles):
                            engine.dream_cycle_execution()
                            progress_bar.progress((i + 1) / post_process_cycles)
                        st.sidebar.success(f"Finished {post_process_cycles} post-processing cycles.")
                        chat_history.append({
                            "query": "Post-Processing Complete",
                            "response": f"Ran {post_process_cycles} dream cycles. Final dream log entry: {engine.state.get('dream_log', ['N/A'])[-1]}",
                            "type": "system"
                        })
                    st.rerun()
                else:
                    st.sidebar.error(f"Could not extract text from {uploaded_doc.name}.")
            except Exception as e:
                st.sidebar.error(f"Error processing document: {e}")
                st.exception(e) # Show full traceback in sidebar for debugging
    else:
        st.sidebar.warning("Please upload a document first.")

st.sidebar.divider()
st.sidebar.header("💾 Engine State (Save/Load)")

# Save Engine State
# We use a button to trigger the save, then provide a download link for the generated file.
if st.sidebar.button("Prepare Engine State for Download", key="save_engine_btn"):
    with st.spinner("Saving engine state..."):
        # Define a temporary path or use BytesIO if st.download_button can handle it directly
        # For simplicity, saving to a fixed name then offering for download.
        # In a multi-user scenario, unique filenames or BytesIO would be better.
        save_file_name = "torusai_engine_state.json"
        if save_engine_state(engine, save_file_name):
            st.session_state.engine_state_to_download = save_file_name
            st.sidebar.success(f"Engine state prepared as {save_file_name}.")
        else:
            st.sidebar.error("Failed to save engine state.")

if 'engine_state_to_download' in st.session_state and st.session_state.engine_state_to_download:
    with open(st.session_state.engine_state_to_download, "rb") as fp:
        st.sidebar.download_button(
            label="Download Engine State",
            data=fp,
            file_name=st.session_state.engine_state_to_download,
            mime="application/json"
        )
    # Clean up session state var after offering download
    # st.session_state.engine_state_to_download = None # Keep it available until next action

# Load Engine State
uploaded_state_file = st.sidebar.file_uploader("Upload Engine State File (.json)", type=['json'], key="state_uploader")
if st.sidebar.button("Load Engine State", key="load_engine_btn"):
    if uploaded_state_file is not None:
        with st.spinner(f"Loading engine state from {uploaded_state_file.name}..."):
            try:
                # Create a new engine instance to load into, or modify current one
                # For simplicity, we'll re-initialize and then load.
                # This ensures clean slate before loading.
                new_engine, new_narrative_log = initialize_engine()
                
                # Save to a temporary file to pass its path to load_engine_state
                temp_load_path = "temp_loaded_state.json"
                with open(temp_load_path, "wb") as f:
                    f.write(uploaded_state_file.getvalue())

                if load_engine_state(temp_load_path, new_engine):
                    st.session_state.engine = new_engine
                    st.session_state.narrative_log = new_narrative_log # Reset narrative log with new engine
                    st.session_state.chat_history.append({
                        "query": f"Loaded Engine State: {uploaded_state_file.name}",
                        "response": f"Nodes: {len(new_engine.cg.nodes)}, Edges: {len(new_engine.cg.edges)}",
                        "type": "system"
                    })
                    st.sidebar.success("Engine state loaded successfully.")
                    if os.path.exists(temp_load_path):
                        os.remove(temp_load_path)
                    st.rerun()
                else:
                    st.sidebar.error("Failed to load engine state.")
                    if os.path.exists(temp_load_path):
                        os.remove(temp_load_path)
            except Exception as e:
                st.sidebar.error(f"Error loading engine state: {e}")
    else:
        st.sidebar.warning("Please upload an engine state file first.")


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
            elif entry["type"] == "system": # For system messages like doc processing, load/save
                 st.markdown(f"**System:** {entry['query']}")
                 st.markdown(f"{entry['response']}")
            st.markdown("---")

# Moved Engine Parameters and Concept Graph to main area for better visibility if sidebar is busy
st.divider()
col_params, col_cg_nodes, col_cg_edges = st.columns(3)
with col_params:
    st.subheader("⚙️ Engine Parameters")
    st.json(engine.params, expanded=False)
with col_cg_nodes:
    st.subheader("🕸️ Concept Graph Nodes")
    st.metric(label="Node Count", value=len(engine.cg.nodes))
    if st.checkbox("Show Nodes JSON", key="show_nodes_json", value=False):
        st.json(engine.cg.nodes, expanded=False)
with col_cg_edges:
    st.subheader("🔗 Concept Graph Edges")
    st.metric(label="Edge Count", value=len(engine.cg.edges))
    if st.checkbox("Show Edges JSON", key="show_edges_json", value=False):
        st.json({str(k): v for k, v in engine.cg.edges.items()}, expanded=False)


st.sidebar.info("This dashboard provides an interface to interact with and manage the TorusAI engine.")
st.sidebar.json(engine.params)

st.sidebar.header("Concept Graph")
st.sidebar.caption("Nodes:")
st.sidebar.json(engine.cg.nodes, expanded=False)
st.sidebar.caption("Edges:")
st.sidebar.json(engine.cg.edges, expanded=False)

st.sidebar.info("This dashboard provides a basic interface to interact with the TorusAI engine.")