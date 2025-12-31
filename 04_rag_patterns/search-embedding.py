from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import os

load_dotenv()
embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
llm = ChatOpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"), model="gpt-5.2")
vectorstore = PineconeVectorStore(
    index_name=os.environ.get("PINECONE_INDEX_NAME"), embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant that answers questions about stocks and the stock market only based on the provided context.
{context}
Question: {question}

Provide a detailed answer.
"""
)


def format_docs(docs):
    """
    Formats a list of documents into a single string.
    """
    return "\n".join([doc.page_content for doc in docs])

def create_rag_chain_with_lcel():
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return retrieval_chain


query = "Enlist key characteristics of bonds ? Provide bullet points"
messages = [HumanMessage(content=query)]

print("\n" + "=" * 70)
print("Implementation 0 without RAG")
print("=" * 70)

result_raw = llm.invoke([HumanMessage(content=query)])
print("\nAnswer")
print(result_raw.content)

print("\n" + "=" * 70)
print("Implementation 1 with RAG")
print("=" * 70)

chain_with_lcel = create_rag_chain_with_lcel()  
result_with_lcel = chain_with_lcel.invoke({"question": query})
print("\nAnswer")
print(result_with_lcel)
