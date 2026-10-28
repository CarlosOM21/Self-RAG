"""Generador de embeddings con Sentence Transformers (E5)."""

import hashlib
import logging

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import normalize_embeddings

from ..config.settings import get_settings
from ..core.exceptions import ModelNotFoundError

logger = logging.getLogger(__name__)


class Embedder:
    """Wrapper para modelo de embeddings Sentence Transformers con prefijos E5."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.embedding_settings = self.settings.embedding
        self._model: SentenceTransformer | None = None
        self._device: str | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Carga lazy del modelo."""
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self) -> None:
        """Carga el modelo de embeddings."""
        try:
            # Determinar dispositivo
            if self.embedding_settings.device == "auto":
                if torch.cuda.is_available():
                    device = "cuda"
                elif torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            else:
                device = self.embedding_settings.device

            self._device = device
            logger.info(f"Cargando modelo de embeddings: {self.embedding_settings.model_name} en {device}")

            self._model = SentenceTransformer(
                self.embedding_settings.model_name,
                device=device,
            )

            # Verificar dimensión
            test_emb = self._model.encode(["test"], convert_to_numpy=True)
            actual_dim = test_emb.shape[1]
            expected_dim = self.embedding_settings.dimension

            if actual_dim != expected_dim:
                logger.warning(
                    f"Dimensión del modelo ({actual_dim}) no coincide con configuración ({expected_dim}). "
                    f"Usando dimensión real: {actual_dim}"
                )
                self.embedding_settings.dimension = actual_dim

            logger.info(f"Modelo de embeddings cargado correctamente (dim={actual_dim})")

        except Exception as e:
            logger.error(f"Error cargando modelo de embeddings: {e}")
            raise ModelNotFoundError(f"No se pudo cargar el modelo de embeddings: {e}")

    def encode_queries(self, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
        """Codifica consultas con prefijo 'query: ' y normaliza."""
        prefixed = [f"{self.embedding_settings.query_prefix}{t}" for t in texts]
        return self._encode(prefixed, batch_size)

    def encode_passages(self, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
        """Codifica pasajes con prefijo 'passage: ' y normaliza."""
        prefixed = [f"{self.embedding_settings.passage_prefix}{t}" for t in texts]
        return self._encode(prefixed, batch_size)

    def _encode(self, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
        """Codifica textos y normaliza a vectores unitarios."""
        if not texts:
            return []

        batch_size = batch_size or self.embedding_settings.batch_size

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.embedding_settings.normalize,
            show_progress_bar=False,
        )

        # Asegurar normalización L2 (coseno = producto punto en vectores unitarios)
        if self.embedding_settings.normalize:
            embeddings = normalize_embeddings(torch.from_numpy(embeddings)).numpy()

        return embeddings.tolist()

    def encode_single_query(self, text: str) -> list[float]:
        """Codifica una sola consulta."""
        return self.encode_queries([text])[0]

    def encode_single_passage(self, text: str) -> list[float]:
        """Codifica un solo pasaje."""
        return self.encode_passages([text])[0]

    @property
    def dimension(self) -> int:
        """Dimensión del embedding."""
        return self.embedding_settings.dimension

    @property
    def model_name(self) -> str:
        """Nombre del modelo."""
        return self.embedding_settings.model_name

    def get_model_hash(self) -> str:
        """Hash del modelo para versionado (basado en nombre y configuración)."""
        config_str = f"{self.model_name}:{self.dimension}:{self.embedding_settings.normalize}"
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]


_embedder_instance: Embedder | None = None


def get_embedder(settings=None) -> Embedder:
    """Obtiene la instancia singleton del embedder."""
    global _embedder_instance
    if _embedder_instance is None or settings is not None:
        _embedder_instance = Embedder(settings)
    return _embedder_instance


def reset_embedder() -> None:
    """Resetea el embedder (útil para tests)."""
    global _embedder_instance
    _embedder_instance = None
