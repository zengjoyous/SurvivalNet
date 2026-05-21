"""Custom exceptions used across survivalnet."""


class SurvivalNetError(Exception):
    """Base exception for survivalnet."""


class DataValidationError(SurvivalNetError):
    """Raised when survival data is missing required columns or invalid."""


class ModelNotFittedError(SurvivalNetError):
    """Raised when a model-dependent method is used before fitting."""
