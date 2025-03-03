import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
import json
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from pinecone import Pinecone, ServerlessSpec
import time

# Load environment variables from .env file
env_path = Path('.env')
load_dotenv(dotenv_path=env_path)

# Page configuration
st.set_page_config(
    page_title="JioPay FAQ Chatbot",
    page_icon="💬",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None

if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #f5f5f5;
    }
    .chat-container {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .user-message {
        background-color: #DCF8C6;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        text-align: right;
    }
    .bot-message {
        background-color: #ECECEC;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("JioPay FAQ Chatbot")
st.markdown("Ask questions about JioPay services and get instant answers from our knowledge base.")

# Sidebar for configuration and document upload
with st.sidebar:
    st.header("Configuration")
    
    # Add file uploader for JSON FAQ files
    uploaded_file = st.file_uploader("Upload JSON FAQ file", type=["json"])
    
    # Process documents button
    if uploaded_file is not None and st.button("Process Documents"):
        with st.spinner("Processing documents..."):
            try:
                # Read JSON data from uploaded file
                json_data = uploaded_file.read().decode("utf-8")
                
                # LangChain JSON FAQ Splitter Class
                class LangChainJSONFAQSplitter:
                    def __init__(self, chunk_size=1000, chunk_overlap=0):
                        self.text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            separators=["\n\n", "\n", ". ", " ", ""]
                        )
                    
                    def process_json_faqs(self, json_data):
                        faq_sections = json.loads(json_data)
                        documents = []
                        
                        for section in faq_sections:
                            source = section.get("source", "")
                            title = section.get("title", "")
                            content = section.get("content", "").strip()
                            
                            # Extract QA pairs from the content
                            pairs = []
                            lines = content.split("\n")
                            i = 0
                            
                            while i < len(lines):
                                if lines[i].strip():
                                    question = lines[i].strip()
                                    answer = lines[i+1].strip() if i+1 < len(lines) else ""
                                    pairs.append(f"Q: {question}\nA: {answer}")
                                    i += 2
                                else:
                                    i += 1
                            
                            # Join all QA pairs for this section
                            section_text = "\n\n".join(pairs)
                            
                            # Create metadata
                            metadata = {
                                "source": source,
                                "title": title,
                                "category": title.replace(" FAQ", "")
                            }
                            
                            # Create initial document
                            section_doc = Document(page_content=section_text, metadata=metadata)
                            
                            # Split into smaller chunks if needed
                            if len(section_text) > 1000:
                                split_docs = self.text_splitter.split_documents([section_doc])
                                documents.extend(split_docs)
                            else:
                                documents.append(section_doc)
                        
                        return documents
                
                # Create splitter and process the data
                splitter = LangChainJSONFAQSplitter(chunk_size=500, chunk_overlap=50)
                documents = splitter.process_json_faqs(json_data)
                
                # Pinecone setup
                model_name = 'multilingual-e5-large'
                embeddings = PineconeEmbeddings(
                    model=model_name,
                    pinecone_api_key=os.getenv("PINECONE_API_KEY")
                )
                
                cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
                region = os.environ.get('PINECONE_REGION') or 'us-east-1'
                spec = ServerlessSpec(cloud=cloud, region=region)
                
                index_name = "rag-getting-started"
                
                pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
                
                # Get the dimensions from your embedding model
                embedding_dimensions = 1024  # multilingual-e5-large has 1024 dimensions
                
                # Check if index exists and has correct dimensions
                if index_name in pc.list_indexes().names():
                    index_info = pc.describe_index(index_name)
                    existing_dimensions = index_info.dimension
                    
                    if existing_dimensions != embedding_dimensions:
                        st.warning(f"Existing index has {existing_dimensions} dimensions, but your model needs {embedding_dimensions}. Recreating index...")
                        pc.delete_index(index_name)
                        time.sleep(5)  # Give it time to delete
                        
                        # Create new index
                        pc.create_index(
                            name=index_name,
                            dimension=embedding_dimensions,
                            metric="cosine",
                            spec=ServerlessSpec(
                                cloud='aws',
                                region='us-east-1'
                            )
                        )
                        # Wait for index to be ready
                        while not pc.describe_index(index_name).status['ready']:
                            time.sleep(1)
                else:
                    # Create new index
                    st.info(f"Creating new index {index_name} with dimensions {embedding_dimensions}...")
                    pc.create_index(
                        name=index_name,
                        dimension=embedding_dimensions,
                        metric="cosine",
                        spec=ServerlessSpec(
                            cloud='aws',
                            region='us-east-1'
                        )
                    )
                    # Wait for index to be ready
                    while not pc.describe_index(index_name).status['ready']:
                        time.sleep(1)
                
                namespace = "wondervector5000"
                
                # Store documents in Pinecone
                docsearch = PineconeVectorStore.from_documents(
                    documents=documents,
                    index_name=index_name,
                    embedding=embeddings,
                    namespace=namespace,
                )
                time.sleep(2)
                
                # Create retrieval chain
                retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
                retriever = docsearch.as_retriever()
                
                llm = ChatGroq(
                    groq_api_key=os.getenv("GROQ_API_KEY"),
                    model="mixtral-8x7b-32768",
                    temperature=0,
                    max_tokens=None,
                    timeout=None,
                )
                
                combine_docs_chain = create_stuff_documents_chain(
                    llm, retrieval_qa_chat_prompt
                )
                st.session_state.retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
                st.session_state.documents_processed = True
                
                st.success(f"Successfully processed {len(documents)} document chunks!")
                
            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")
    
    # Display API key status
    st.subheader("API Keys Status")
    pinecone_key = "✓ Connected" if os.getenv("PINECONE_API_KEY") else "❌ Missing"
    groq_key = "✓ Connected" if os.getenv("GROQ_API_KEY") else "❌ Missing"
    
    st.info(f"Pinecone API: {pinecone_key}")
    st.info(f"Groq API: {groq_key}")
    
    # Instructions
    st.markdown("---")
    st.markdown("""
    ### How to use
    1. Upload your JSON FAQ file
    2. Click "Process Documents"
    3. Start chatting in the main panel
    """)

# Main chat interface
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"<div class='user-message'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-message'>{message['content']}</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Chat input
    if query := st.chat_input("Ask a question about JioPay..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Get bot response
        if st.session_state.retrieval_chain:
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.retrieval_chain.invoke({"input": query})
                    answer = response['answer']
                    
                    # Extract sources if available
                    sources = []
                    if 'context' in response:
                        sources = [doc.metadata.get("source", "") for doc in response['context']]
                        sources = list(set([s for s in sources if s]))  # Remove duplicates and empty strings
                    
                    # Add source info if available
                    if sources:
                        source_text = "\n\nSources: " + ", ".join(sources)
                        answer += source_text
                    
                    # Add bot response to chat
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.rerun()
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.rerun()
        else:
            message = "Please upload and process FAQ documents first using the sidebar options."
            st.session_state.messages.append({"role": "assistant", "content": message})
            st.rerun()

with col2:
    st.markdown("### Recently Asked Questions")
    sample_questions = [
        "What is JioPay?",
        "How do I reset my password?",
        "Is JioPay secure?",
        "What payment methods are supported?",
        "How to contact customer support?"
    ]
    
    for question in sample_questions:
        if st.button(question):
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": question})
            
            # Get bot response
            if st.session_state.retrieval_chain:
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.retrieval_chain.invoke({"input": question})
                        answer = response['answer']
                        
                        # Extract sources if available
                        sources = []
                        if 'context' in response:
                            sources = [doc.metadata.get("source", "") for doc in response['context']]
                            sources = list(set([s for s in sources if s]))  # Remove duplicates and empty strings
                        
                        # Add source info if available
                        if sources:
                            source_text = "\n\nSources: " + ", ".join(sources)
                            answer += source_text
                        
                        # Add bot response to chat
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        st.rerun()
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        st.rerun()
            else:
                message = "Please upload and process FAQ documents first using the sidebar options."
                st.session_state.messages.append({"role": "assistant", "content": message})
                st.rerun()