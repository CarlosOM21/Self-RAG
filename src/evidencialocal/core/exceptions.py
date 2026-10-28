"""Excepciones personalizadas para EvidenciaLocal."""


class EvidenciaLocalError(Exception):
    """Excepción base para todos los errores de EvidenciaLocal."""
    pass


class ConfigurationError(EvidenciaLocalError):
    """Error de configuración inválida o faltante."""
    pass


class ModelNotFoundError(EvidenciaLocalError):
    """Modelo requerido no encontrado (embedding, LLM, etc.)."""
    pass


class IndexingError(EvidenciaLocalError):
    """Error durante la indexación de documentos."""
    pass


class RetrievalError(EvidenciaLocalError):
    """Error durante la recuperación de chunks."""
    pass


class GenerationError(EvidenciaLocalError):
    """Error durante la generación de respuesta."""
    pass


class EvaluationError(EvidenciaLocalError):
    """Error durante la evaluación."""
    pass


class ValidationError(EvidenciaLocalError):
    """Error de validación de datos de entrada/salida."""
    pass
