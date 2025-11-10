from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from typing import Optional

load_dotenv()

class VectorStoreBuilder:
    def __init__(self, csv_path: str, persist_directory: Optional[str] = None):
        self.csv_path = csv_path
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    def build_and_save_vector_store(self):
        loader = CSVLoader(
            file_path=self.csv_path,
            encoding="utf-8",
            metadata_columns=[]
        )
        
        data = loader.load()
        
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = splitter.split_documents(data)
        
        db = Chroma.from_documents(texts, self.embeddings, persist_directory=self.persist_directory)
        db.persist()
        return db
    
    def load_vector_store(self):
        db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        return db
