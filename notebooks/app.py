"""
CrediTrust Financial - RAG Chatbot Interface
Streamlit Frontend for Complaint Analysis System
"""

import streamlit as st
import sys
from pathlib import Path
import json
import pandas as pd
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Import RAG system
try:
    from RAG import ImprovedRAG
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    st.error("❌ RAG system not found. Make sure fiastochroma.py is in the same directory.")

# Page configuration
st.set_page_config(
    page_title="CrediTrust Financial - Complaint Analyzer",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        color: #000000 !important;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f1f8e9;
        border-left: 4px solid #8bc34a;
    }
    .chat-message strong {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False

# Initialize RAG system
@st.cache_resource
def initialize_rag():
    """Initialize the RAG system (cached)."""
    if not RAG_AVAILABLE:
        return None
    
    with st.spinner("🔧 Initializing RAG system... This may take a few minutes on first load."):
        try:
            rag = ImprovedRAG(chroma_path="notebooks/vector_store")
            if rag.initialize():
                return rag
        except Exception as e:
            st.error(f"❌ Error initializing RAG: {str(e)}")
    return None

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/200x80/1f77b4/ffffff?text=CrediTrust", width=200)
    st.markdown("### 🏦 Complaint Analyzer")
    st.markdown("---")
    
    # System Status
    st.markdown("### 📊 System Status")
    if not st.session_state.system_initialized:
        if st.button("🚀 Initialize System", use_container_width=True):
            st.session_state.rag_system = initialize_rag()
            if st.session_state.rag_system:
                st.session_state.system_initialized = True
                st.success("✅ System ready!")
                st.rerun()
            else:
                st.error("❌ Initialization failed")
    else:
        st.success("✅ System Active")
        
        # Query Settings
        st.markdown("---")
        st.markdown("### ⚙️ Query Settings")
        k_value = st.slider("Number of sources", min_value=3, max_value=10, value=5, 
                           help="How many relevant complaints to retrieve")
        
        st.markdown("---")
        st.markdown("### 📈 Analytics")
        if st.session_state.chat_history:
            st.metric("Total Queries", len(st.session_state.chat_history))
            
        # Clear History
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        # Export Chat
        if st.session_state.chat_history:
            if st.button("💾 Export Chat", use_container_width=True):
                export_data = {
                    'timestamp': datetime.now().isoformat(),
                    'chat_history': st.session_state.chat_history
                }
                st.download_button(
                    "📥 Download JSON",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
    
    st.markdown("---")
    st.markdown("### 💡 Example Questions")
    example_questions = [
        "What are common credit card issues?",
        "Why are customers unhappy with loans?",
        "What billing problems are reported?",
        "Are there fraud issues?",
        "What are the main complaints?",
        "How long do disputes take?",
        "What fee issues exist?",
        "What payment problems occur?"
    ]
    
    for question in example_questions:
        if st.button(f"💬 {question}", use_container_width=True, key=question):
            st.session_state.current_question = question

# Main Content
st.markdown('<div class="main-header">🏦 CrediTrust Financial</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Intelligent Complaint Analysis System</div>', unsafe_allow_html=True)

# Check if system is initialized
if not st.session_state.system_initialized:
    st.info("👈 Click **Initialize System** in the sidebar to get started")
    
    # Show system info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📚 Vector Database</h3>
            <p>Chroma DB with 12,000+ complaint chunks</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 AI Model</h3>
            <p>Google Flan-T5 Base (FREE)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🔍 Retrieval</h3>
            <p>Semantic similarity search</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🎯 Features")
    features_col1, features_col2 = st.columns(2)
    
    with features_col1:
        st.markdown("""
        - ✅ **Natural Language Queries** - Ask questions in plain English
        - ✅ **Source Attribution** - See which complaints informed the answer
        - ✅ **Real-time Analysis** - Get instant insights
        """)
    
    with features_col2:
        st.markdown("""
        - ✅ **Product-Specific Insights** - Filter by financial product
        - ✅ **Pattern Detection** - Identify recurring issues
        - ✅ **100% Free** - No API costs
        """)

else:
    # Main chat interface
    st.markdown("### 💬 Chat with Your Complaint Data")
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 You:</strong><br>{message['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>🤖 Assistant:</strong><br>{message['content']}
            </div>
            """, unsafe_allow_html=True)
            
            # Show sources
            if 'sources' in message:
                with st.expander(f"📚 View {len(message['sources'])} Source Documents"):
                    for i, source in enumerate(message['sources'], 1):
                        product = source['metadata'].get('product', 'Unknown')
                        text = source['text'][:200]
                        st.markdown(f"""
                        <div class="source-card">
                            <strong>Source {i}: {product}</strong><br>
                            <small>{text}...</small>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Chat input
    if 'current_question' in st.session_state:
        user_question = st.session_state.current_question
        del st.session_state.current_question
    else:
        user_question = st.chat_input("Ask a question about customer complaints...")
    
    if user_question:
        # Add user message to history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_question,
            'timestamp': datetime.now().isoformat()
        })
        
        # Query RAG system
        with st.spinner("🔍 Analyzing complaints and generating answer..."):
            try:
                k_value = st.session_state.get('k_value', 5)
                result = st.session_state.rag_system.query(user_question, k=k_value)
                
                # Add assistant message to history
                sources_data = []
                for doc in result['sources']:
                    sources_data.append({
                        'text': doc.page_content,
                        'metadata': doc.metadata
                    })
                
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': result['answer'],
                    'sources': sources_data,
                    'num_sources': result['num_sources'],
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <small>Powered by LangChain 🦜 | Chroma DB | Streamlit | Flan-T5</small><br>
    <small>💯 100% Free & Open Source</small>
</div>
""", unsafe_allow_html=True)