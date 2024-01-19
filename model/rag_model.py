import logging

import dotenv
import pandas as pd
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.llms import OpenLLM
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

logger = logging.getLogger(__name__)
dotenv.load_dotenv()


class RagModel:
    """Rag Model."""

    def __init__(self, document_path, llm_model):
        self.document_path = document_path
        self.llm_model = llm_model
        self.model = None

    def initialize_rag_chain(self):
        """Returns a Q&A RAG chain."""
        logger.info("Initializing RAG model")

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

        if self.llm_model == "openai":
            llm = OpenAI()
        elif self.llm_model == "dolly":
            llm = OpenLLM(
                model_name="dolly-v2",
                model_id="databricks/dolly-v2-3b",
                temperature=0,
                repetition_penalty=1.2,
            )
        else:
            raise ValueError

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )
        self.model = qa_chain

    def predict(self, df: pd.DataFrame) -> list[str]:
        """Predict function of the model."""
        logger.info("Running model predict")
        if not self.model:
            self.initialize_rag_chain()
        return [
            self.model({"query": question})["result"] for question in df["question"]
        ]
