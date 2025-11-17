from dotenv import load_dotenv
from pydantic import BaseModel
import os

load_dotenv()


class Settings(BaseModel):
    DEFAULT_MAX_RESULTS: int = int(os.getenv("DEFAULT_MAX_RESULTS", "20"))

    GITHUB_TOKEN: str | None = os.getenv("GITHUB_TOKEN")

    S2_API_KEY: str | None = os.getenv("S2_API_KEY")
    S2_MAX_RETRIES: int = int(os.getenv("S2_MAX_RETRIES", "2"))
    S2_BACKOFF_SECONDS: float = float(os.getenv("S2_BACKOFF_SECONDS", "1.0"))

    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    CLUSTERS_K: int = int(os.getenv("CLUSTERS_K", "1"))
    RECENCY_HALFLIFE_DAYS: int = int(os.getenv("RECENCY_HALFLIFE_DAYS", "180"))

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    VLLM_BASE_URL: str = os.getenv("VLLM_BASE_URL")
    VLLM_API_KEY: str = os.getenv("VLLM_API_KEY")

    DEFAULT_OPENAI_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    DEFAULT_VLLM_MODEL: str = os.getenv("VLLM_MODEL", os.getenv("LLM_MODEL", "qwen2.5-7b-instruct"))


settings = Settings()
