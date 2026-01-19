# --- 1. SQLITE FIX (Sab se upar honi chahiye) ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# --- Imports ---
import streamlit as st
import os
import gc
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Tokenizers ko multitasking se rokna (Memory bachane ke liye)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="iPhone Expert Bot", layout="wide", page_icon="📱")
st.title("📱 iPhone Information System")
st.markdown("---")

# --- 2. CACHING LOGIC ---
@st.cache_resource
def get_embeddings():
    """HuggingFace Model ko memory mein aik baar load karta hai."""
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_resource
def setup_knowledge_base():
    """PDF process karke vector database banata hai."""
    pdf_path = "iphone_data.pdf"
    
    if not os.path.exists(pdf_path):
        return None, f"Error: '{pdf_path}' file nahi mili. Upload karein."
    
    try:
        # PDF Loading
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        # Text Splitting (chunk size 600, overlap 50)
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
        docs = splitter.split_documents(pages)
        
        # Database Creation (Chroma, persistent memory)
        embeddings = get_embeddings()
        vector_db = Chroma.from_documents(
            docs,
            embeddings,
            persist_directory="chroma_db"
        )
        vector_db.persist()
        
        # Memory cleanup
        gc.collect()
        return vector_db, None
    except Exception as e:
        return None, f"Database Error: {str(e)}"

# --- 3. MAIN APP LOGIC ---

# API Key Check (Secrets se ya Sidebar se)
if "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]
else:
    api_key = st.sidebar.text_input("Enter Groq API Key:", type="password")

if api_key:
    # Database Initialization (Sirf aik baar)
    with st.status("Initializing Knowledge Base...", expanded=False) as status:
        vector_db, err = setup_knowledge_base()
        if err:
            status.update(label="Setup Failed!", state="error")
            st.error(err)
        else:
            status.update(label="System Ready!", state="complete")

    # Safe check: vector_db exist karta hai ya nahi
    if vector_db is not None:
        try:
            # AI Model Setup (Groq Llama 3)
            llm = ChatGroq(
                groq_api_key=api_key,
                model_name="llama3-8b-8192",
                temperature=0.1
            )
            
            # Retrieval Chain
            chat_bot = RetrievalQA.from_chain_type(
                llm=llm, 
                retriever=vector_db.as_retriever(search_kwargs={"k": 3})
            )
            
            # User Interface
            st.subheader("Puchiye apne sawal:")
            query = st.text_input(
                "Example: iPhone 16 Pro Max ki specs aur Pakistan price kya hai?",
                placeholder="Yahan type karein..."
            )
            
            if query:
                with st.spinner("PDF se maloomat dhoondi ja rahi hain..."):
                    response = chat_bot.invoke(query)
                    st.markdown("### 🤖 AI Response:")
                    st.success(response["result"])
                    
        except Exception as e:
            st.error(f"Chat Error: {e}")
else:
    st.warning("Side bar mein Groq API Key enter karein ya Streamlit Secrets use karein.")

st.markdown("---")
st.caption("Powered by LangChain, Groq & Streamlit Cloud")
