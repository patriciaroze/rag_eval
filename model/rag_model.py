import dotenv
import pandas as pd
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

dotenv.load_dotenv()


class RagModel:
    """Rag Model."""

    def __init__(self, document_path):
        self.document_path = document_path
        self.model = None

    def initialize_rag_chain(self):
        """Returns a Q&A RAG chain."""
        loader = TextLoader(self.document_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=[" ", ",", "\n"],
            length_function=len,
            add_start_index=True,
        )
        docs = text_splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(
            documents=docs, embedding=OpenAIEmbeddings()
        )

        retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 2}
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=OpenAI(),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            max_tokens=3500,
        )
        self.model = qa_chain

    def predict(self, df: pd.DataFrame) -> list[str]:
        """Predict function of the model."""
        if not self.model:
            self.initialize_rag_chain()
        return [
            self.model({"query": question})["result"] for question in df["question"]
        ]
