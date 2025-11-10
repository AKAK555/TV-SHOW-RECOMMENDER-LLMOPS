# pipeline.py
from src.vector_store import VectorStoreBuilder
from src.recommender import Recommender
from src.prompt_template import get_tvshow_prompt
from config.config import GROQ_API_KEY, MODEL_NAME
from utils.logger import get_logger
from utils.custom_exception import CustomException

logger = get_logger(__name__)

# lazy import ChatGroq (so pipeline still loads even if langchain_groq not installed)
try:
    from langchain_groq import ChatGroq  # type: ignore
except Exception:
    ChatGroq = None

class TVShowRecommendationPipeline:
    def __init__(self, persist_directory="chroma_db"):
        try:
            logger.info("Initializing Recommendation Pipeline")

            vector_builder = VectorStoreBuilder(csv_path="", persist_directory=persist_directory)
            retriever = vector_builder.load_vector_store().as_retriever()

            # instantiate the LLM (ChatGroq) with your API key + model
            if ChatGroq is None:
                raise RuntimeError("langchain_groq.ChatGroq is not available. Install langchain-groq to use the Groq LLM.")
            llm = ChatGroq(api_key=GROQ_API_KEY, model=MODEL_NAME, temperature=0)

            # build your prompt template using the helper
            prompt = get_tvshow_prompt()  # should return a PromptTemplate

            self.recommender = Recommender(retriever, llm, prompt)

            logger.info("Pipeline initialized successfully...")

        except Exception as e:
            logger.error(f"Failed to initialize pipeline {e}")
            raise CustomException("Error during pipeline initialization", e)

    def recommend(self, query: str, context: str = "") -> str:
        try:
            logger.info(f"Received a query: {query}")
            recommendation = self.recommender.get_recommendation(query, context)
            logger.info("Recommendation generated successfully...")
            return recommendation
        except Exception as e:
            logger.error(f"Failed to get recommendation: {e}")
            raise CustomException("Error during getting recommendation", e)
