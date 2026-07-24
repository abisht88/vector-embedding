from typing import List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from langchain_core.embeddings import Embeddings


class JinaEmbeddings(Embeddings):
    """
    LangChain-compatible Jina AI Embeddings.

    Features:
    - Connection pooling
    - Automatic batching
    - Automatic retries
    - Compatible with Chroma, FAISS and LangChain
    """

    def __init__(
        self,
        api_key: str,
        model: str = "jina-embeddings-v5",
        base_url: str = "https://api.jina.ai/v1/embeddings",
        timeout: int = 60,
        batch_size: int = 128,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.batch_size = batch_size

        # Reuse HTTP connections
        self.session = requests.Session()

        retries = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
        )

        adapter = HTTPAdapter(
            max_retries=retries,
            pool_connections=10,
            pool_maxsize=20,
        )

        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _request_embeddings(self, texts: List[str]) -> List[List[float]]:
        response = self.session.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": texts,
            },
            timeout=self.timeout,
        )

        response.raise_for_status()

        data = response.json()["data"]

        # Preserve original ordering
        data.sort(key=lambda x: x["index"])

        return [item["embedding"] for item in data]

    def _embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            embeddings.extend(self._request_embeddings(batch))

        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]