from .capture import Capture


def load_mission(mission_dir: str, strict: bool = True) -> list[Capture]:
    """
    Main entry point for ingestion.

    Scans mission_dir/rgb/ and mission_dir/multi/,
    groups files by capture ID, validates completeness.

    Returns sorted list of complete Capture objects.
    """
    from .grouper import group_captures
    from .validator import validate_captures

    captures = group_captures(mission_dir)
    return validate_captures(captures, strict=strict)


def __getattr__(name: str):
    if name == "group_captures":
        from .grouper import group_captures
        return group_captures
    if name in {"validate_captures", "ValidationError"}:
        from .validator import ValidationError, validate_captures
        return {"validate_captures": validate_captures, "ValidationError": ValidationError}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Capture",
    "load_mission",
    "group_captures",
    "validate_captures",
    "ValidationError",
]
