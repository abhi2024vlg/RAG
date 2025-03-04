import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
import json
from langchain_core.documents import Document
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from pinecone import Pinecone, ServerlessSpec
import time
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import EnsembleRetriever

# Load environment variables from .env file
env_path = Path('.env')
load_dotenv(dotenv_path=env_path)

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

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
    .stButton>button {
        width: 100%;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

def semantic_chunking(json_data):
    try:
        faq_sections = json.loads(json_data)
        documents = []
        
        for section in faq_sections:
            source = section.get("source", "")
            title = section.get("title", "")
            content = section.get("content", "").strip()
            
            # Split content by questions (assuming questions end with ?)
            qa_pairs = []
            current_question = None
            current_answer = []
            
            for line in content.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                # Check if this line is a question (ends with ?)
                if '?' in line:
                    # If we already have a question, save the previous Q&A pair
                    if current_question is not None:
                        qa_pairs.append((current_question, '\n'.join(current_answer)))
                    
                    current_question = line
                    current_answer = []
                else:
                    # This is part of the answer
                    if current_question is not None:
                        current_answer.append(line)
            
            # Don't forget the last Q&A pair
            if current_question is not None and (current_answer or len(qa_pairs) == 0):
                qa_pairs.append((current_question, '\n'.join(current_answer)))
            
            # Create documents from Q&A pairs
            for question, answer in qa_pairs:
                doc = Document(
                    page_content=f"Question: {question}\nAnswer: {answer}",
                    metadata={
                        "source": source,
                        "title": title,
                        "category": title.replace(" FAQ", "") if title else "",
                        "question": question
                    }
                )
                documents.append(doc)
        
        return documents
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON data: {str(e)}")
    except Exception as e:
        raise Exception(f"Error in semantic chunking: {str(e)}")

def process_faq_data(json_data):
    try:
        print("Starting to process FAQ data...")
        documents = semantic_chunking(json_data)
        print(f"Successfully created {len(documents)} documents")
        
        if not documents:
            return None, False, "No documents were extracted from the JSON data"
        
        # Sample the first document to verify structure
        print(f"Sample document: {documents[0].page_content[:100]}...")
        print(f"Sample metadata: {documents[0].metadata}")
        
        # Create BM25 retriever
        print("Creating BM25 retriever...")
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 5
        print("BM25 retriever created successfully")

        # Pinecone setup
        print("Setting up Pinecone embeddings...")
        embeddings = PineconeEmbeddings(
            model='intfloat/multilingual-e5-large-instruct',
            pinecone_api_key=os.getenv("PINECONE_API_KEY")
        )
        print("Embeddings initialized successfully")

        index_name = "jiopay-faq-improved"
        embedding_dimensions = 1024

        # Create or verify index
        print(f"Checking for Pinecone index: {index_name}")
        existing_indexes = pc.list_indexes().names()
        print(f"Existing indexes: {existing_indexes}")
        
        if index_name not in existing_indexes:
            print(f"Creating new index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=embedding_dimensions,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print("Waiting for index to initialize...")
            time.sleep(20)
            print("Index creation wait complete")

        namespace = "jiopay_improved"
        
        # Store documents in Pinecone
        print(f"Storing {len(documents)} documents in Pinecone...")
        vector_store = PineconeVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            index_name=index_name,
            namespace=namespace
        )
        print("Documents stored in Pinecone successfully")

        # Create Pinecone retriever
        print("Creating Pinecone retriever...")
        pinecone_retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.7}
        )
        print("Pinecone retriever created successfully")

        # Create ensemble retriever
        print("Creating ensemble retriever...")
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, pinecone_retriever],
            weights=[0.4, 0.6]
        )
        print("Ensemble retriever created successfully")

        # Create RAG prompt
        print("Creating RAG prompt...")
        rag_prompt = ChatPromptTemplate.from_template("""
        You are a customer service AI assistant for JioPay. Answer questions based ONLY on the context.
        If unsure, say "I don't have enough information. Contact JioPay support at merchant.support@jiopay.in".

        Context:
        {context}

        Question: {input}

        Answer professionally using ONLY context. Include specific details if available.
        """)
        print("RAG prompt created successfully")

        # Setup LLM
        print("Setting up LLM...")
        llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model="mixtral-8x7b-32768",
            temperature=0,
            max_tokens=512,
        )
        print("LLM setup complete")

        # Create chain
        print("Creating document chain...")
        combine_docs_chain = create_stuff_documents_chain(llm, rag_prompt)
        print("Creating retrieval chain...")
        retrieval_chain = create_retrieval_chain(ensemble_retriever, combine_docs_chain)
        print("Chains created successfully")

        return retrieval_chain, True, f"Processed {len(documents)} documents successfully"
    
    except Exception as e:
        import traceback
        stack_trace = traceback.format_exc()
        print(f"ERROR: {str(e)}\n{stack_trace}")
        return None, False, f"Error processing documents: {str(e)}"

# Sidebar for configuration and status
with st.sidebar:
    st.header("Status")
    st.subheader("API Keys Status")
    pinecone_key = "✓ Connected" if os.getenv("PINECONE_API_KEY") else "❌ Missing"
    groq_key = "✓ Connected" if os.getenv("GROQ_API_KEY") else "❌ Missing"
    
    st.info(f"Pinecone API: {pinecone_key}")
    st.info(f"Groq API: {groq_key}")
    
    st.subheader("Chatbot Status")
    if st.session_state.documents_processed:
        st.success("✓ Chatbot is ready")
    else:
        st.warning("⚠️ Chatbot initializing...")

# Main chat interface
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    
    for message in st.session_state.messages:
        role_class = "user-message" if message["role"] == "user" else "bot-message"
        st.markdown(f"<div class='{role_class}'>{message['content']}</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    if query := st.chat_input("Ask a question about JioPay..."):
        st.session_state.messages.append({"role": "user", "content": query})
        
        if st.session_state.retrieval_chain:
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.retrieval_chain.invoke({"input": query})
                    answer = response['answer']
                    
                    sources = list(set(
                        doc.metadata.get("source", "") 
                        for doc in response["context"] 
                        if doc.metadata.get("source")
                    ))
                    
                    if sources:
                        answer += f"\n\nSources: {', '.join(sources)}"
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.rerun()
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.rerun()

with col2:
    st.markdown("### Suggested Questions")
    sample_questions = [
        "How do I reset my JioPay password?",
        "What are the transaction limits on JioPay?",
        "How to contact JioPay customer support?",
        "Is JioPay available internationally?",
        "What security features does JioPay have?"
    ]
    
    for question in sample_questions:
        if st.button(question):
            st.session_state.messages.append({"role": "user", "content": question})
            
            if st.session_state.retrieval_chain:
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.retrieval_chain.invoke({"input": question})
                        answer = response['answer']
                        
                        sources = list(set(
                            doc.metadata.get("source", "") 
                            for doc in response["context"] 
                            if doc.metadata.get("source")
                        ))
                        
                        if sources:
                            answer += f"\n\nSources: {', '.join(sources)}"
                        
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        st.rerun()
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        st.rerun()

# Initialization logic
if not st.session_state.documents_processed:
    with st.spinner("Initializing JioPay FAQ Chatbot..."):
        try:
            faq_file_path = "dummy_jiopay_faqs.json"
            if os.path.exists(faq_file_path):
                with open(faq_file_path, "r") as f:
                    json_data = f.read()
                
                chain, success, message = process_faq_data(json_data)
                if success:
                    st.session_state.retrieval_chain = chain
                    st.session_state.documents_processed = True
                    st.success("Chatbot initialized successfully!")
                else:
                    st.error(f"Initialization failed: {message}")
            else:
                st.error("FAQ file not found. Please check the file path.")
        except Exception as e:
            st.error(f"Initialization error: {str(e)}")