"""User-facing error taxonomy for Terra Point.

Every failure mode in the pipeline raises an :class:`InfraError` subclass that
carries a *hint* describing the likely cause and a concrete remedy, so users are
never left with an obscure Python traceback.
"""
from __future__ import annotations


class InfraError(RuntimeError):
    """Base class for all Terra Point failures.

    Attributes:
        hint: Plain-language suggestion for resolving the failure.
    """

    hint = "See the documentation for the recommended fix."

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        if hint is not None:
            self.hint = hint

    def user_message(self) -> str:
        return f"{self.__class__.__name__}: {self}\n  Suggested fix: {self.hint}"


class InputNotFoundError(InfraError):
    def __init__(self, path: str) -> None:
        super().__init__(
            f"Input file not found: {path}",
            "Check that the path is correct and the file exists.",
        )


class UnsupportedLasVersionError(InfraError):
    def __init__(self, version: str, strict: bool) -> None:
        super().__init__(
            f"Unsupported LAS version: {version} (pipeline requires LAS 1.4).",
            "The competition source data is LAS 1.4. Re-export the file as LAS 1.4 "
            "(e.g. with CloudCompare or laspy) and retry.",
        )


class UnsupportedPointFormatError(InfraError):
    def __init__(self, point_format: int) -> None:
        super().__init__(
            f"Unsupported Point Data Record Format: {point_format} (expected 7).",
            "Re-export the file as Point Data Record Format 7 (with RGB and returns). "
            "Formats 0-5 lack RGB; formats 6-10 are LAS 1.4 native.",
        )


class MalformedLasError(InfraError):
    def __init__(self, message: str) -> None:
        super().__init__(
            f"Malformed LAS file: {message}",
            "Verify the file with laspy/lasinfo before processing. A truncated or "
            "corrupt download will fail here.",
        )


class LazBackendMissingError(InfraError):
    """Raised when a LAZ (compressed) file is opened without a decompression backend."""

    def __init__(self) -> None:
        super().__init__(
            "This is a LAZ-compressed point cloud, but no LAZ decoder is installed.",
            "Install the decoder with `pip install lazrs` (and reinstall laspy) so "
            "compressed .laz files can be read, then upload the file again.",
        )


class EmptyPointCloudError(InfraError):
    def __init__(self) -> None:
        super().__init__(
            "The LAS file contains zero points.",
            "Point the pipeline at a populated LAS file.",
        )


class InvalidCoordinateDataError(InfraError):
    def __init__(self, message: str) -> None:
        super().__init__(
            f"Invalid coordinate data: {message}",
            "Re-export the LAS with sane scales/offsets. All points must be finite "
            "and the cloud must have non-zero spatial extent.",
        )


class MissingDimensionError(InfraError):
    """Raised when a dimension required by an explicitly requested stage is absent."""

    def __init__(self, dimension: str, requested_by: str) -> None:
        super().__init__(
            f"LAS file does not expose '{dimension}', which is required by {requested_by}.",
            "Re-export the file with the dimension enabled (e.g. intensity, RGB). "
            "The default geometry backend only needs XYZ + intensity.",
        )


class ModelWeightsUnavailableError(InfraError):
    def __init__(self, weight: str) -> None:
        super().__init__(
            f"Model weights not found: {weight}",
            "Run `python -m infra_inventory download-models` (or scripts/download_models.py) "
            "to fetch the documented pretrained weights, then pass --pointcept-weight.",
        )


class CudaUnavailableError(InfraError):
    def __init__(self, message: str) -> None:
        super().__init__(
            f"CUDA/GPU unavailable: {message}",
            "Pointcept/PTv3 inference needs a CUDA-capable GPU and the Pointcept "
            "environment. The default `geometry` backend is fully CPU-safe; run "
            "`python -m infra_inventory backends` to inspect the environment.",
        )


class BackendConfigurationError(InfraError):
    def __init__(self, message: str, hint: str) -> None:
        super().__init__(message, hint)


class RoadMarkingConfigurationError(BackendConfigurationError):
    pass


class InsufficientMemoryError(InfraError):
    def __init__(self, message: str) -> None:
        super().__init__(
            f"Insufficient memory: {message}",
            "Reduce --tile-size, increase --chunk-size streaming, or process the LAS "
            "on a machine with more RAM. Tiles are processed one at a time.",
        )


class InvalidConfigError(InfraError):
    def __init__(self, message: str) -> None:
        super().__init__(
            f"Invalid configuration: {message}",
            "Validate the YAML config against configs/processing.yaml in this repository.",
        )