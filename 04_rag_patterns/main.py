from langchain_openai.embeddings.base import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "investing.txt")
    
    # Load the document
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    # Split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=0)

    chunks = text_splitter.split_documents(documents)
    # Print the chunks
    print(f"Number of chunks: {len(chunks)}")
    for chunk in chunks:
        print(chunk.page_content)
        print("--------------------------------------------------")
    
    # create embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    PineconeVectorStore.from_documents(chunks, embeddings, index_name=os.environ.get("PINECONE_INDEX_NAME"))

    print("Vector store created successfully")



if __name__ == "__main__":
    main()
