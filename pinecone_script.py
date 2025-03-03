from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
from typing import List, Dict, Any
from langchain_pinecone import PineconeEmbeddings
import os
from pinecone import Pinecone, ServerlessSpec
import time

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub

import pinecone
from langchain_pinecone import PineconeVectorStore

class LangChainJSONFAQSplitter:
    """
    A class for processing JSON FAQ data using LangChain components.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 0):
        """
        Initialize the splitter with custom chunk size and overlap.
        
        Args:
            chunk_size (int): The size of each text chunk
            chunk_overlap (int): The overlap between chunks
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def process_json_faqs(self, json_data: str) -> List[Document]:
        """
        Process JSON FAQ data into LangChain Documents.
        
        Args:
            json_data (str): JSON string containing FAQ data
            
        Returns:
            List[Document]: A list of LangChain Document objects
        """
        # Parse the JSON data
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

def main():
    # Read JSON from file (or use the provided string)
    with open('/Users/abhinavkumar/Desktop/RAG/data/dummy_jiopay_faqs.json', 'r') as file:
        json_data = file.read()
    
    # Create splitter and process the data
    splitter = LangChainJSONFAQSplitter(chunk_size=500, chunk_overlap=50)
    documents = splitter.process_json_faqs(json_data)
    
    # Print results
    print(f"Total documents created: {len(documents)}")
    
    for i, doc in enumerate(documents[:2]):  # Print first 2 for demonstration
        print(f"\nDocument {i+1}:")
        print(f"Metadata: {doc.metadata}")
        print(f"Content (first 150 chars): {doc.page_content[:150]}...")
        print(f"Content length: {len(doc.page_content)} characters")
    
    return documents

documents = main()

model_name = 'multilingual-e5-large'
embeddings = PineconeEmbeddings(
    model=model_name,
    pinecone_api_key="pcsk_25wNg6_KHi3Sv9nxUk228sbyJdDH4AM17qqgZVKAqcjvstkUTEBVCmWwnoYE99VpzEkL1R"
)



# pinecone.init(api_key="pcsk_25wNg6_KHi3Sv9nxUk228sbyJdDH4AM17qqgZVKAqcjvstkUTEBVCmWwnoYE99VpzEkL1R")

cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'
spec = ServerlessSpec(cloud=cloud, region=region)

index_name = "rag-getting-started"

pc = Pinecone(api_key="pcsk_25wNg6_KHi3Sv9nxUk228sbyJdDH4AM17qqgZVKAqcjvstkUTEBVCmWwnoYE99VpzEkL1R")

# Get the dimensions from your embedding model
model_name = 'multilingual-e5-large'
embedding_dimensions = 1024  # multilingual-e5-large has 1024 dimensions

# Check if index exists and has correct dimensions
if index_name in pc.list_indexes().names():
    index_info = pc.describe_index(index_name)
    existing_dimensions = index_info.dimension
    
    if existing_dimensions != embedding_dimensions:
        print(f"Existing index has {existing_dimensions} dimensions, but your model needs {embedding_dimensions}.")
        print("Deleting and recreating index with correct dimensions...")
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
        print(f"Index {index_name} exists with correct dimensions ({embedding_dimensions}).")
else:
    # Create new index
    print(f"Creating new index {index_name} with dimensions {embedding_dimensions}...")
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
pc = Pinecone(api_key="pcsk_25wNg6_KHi3Sv9nxUk228sbyJdDH4AM17qqgZVKAqcjvstkUTEBVCmWwnoYE99VpzEkL1R")

docsearch = PineconeVectorStore.from_documents(
    documents=documents,
    index_name=index_name,
    embedding=embeddings,
    namespace=namespace,
)
time.sleep(2)


retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
retriever=docsearch.as_retriever()

llm = ChatGroq(
    groq_api_key="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",
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=None,
    timeout=None,
)

combine_docs_chain = create_stuff_documents_chain(
    llm, retrieval_qa_chat_prompt
)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)