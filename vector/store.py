from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever


# TODO: Where is this being used?
# Post-processing
#def format_docs(docs):
#    return "\n\n".join(doc.page_content for doc in docs)

def index(docs):
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # Add to vectorDB
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=GPT4AllEmbeddings(
            model_name="nomic-embed-text-v1.5.f16.gguf", 
            gpt4all_kwargs={'allow_download': False}
        ),
    )

    retriever = vectorstore.as_retriever()
    
    return retriever

def get(question, retriever:(VectorStoreRetriever)):
    docs = retriever._get_relevant_documents(query=question, run_manager=None)
    return docs
