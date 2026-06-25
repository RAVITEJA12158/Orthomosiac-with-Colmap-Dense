from .capture import Capture


class ValidationError(Exception):
    pass


def validate_captures(captures: dict[str, Capture], strict: bool = True) -> list[Capture]:
    """
    Validate all captures.

    strict=True  → raise ValidationError if any capture is incomplete
    strict=False → skip incomplete captures with a warning, return only complete ones

    Returns list of complete Capture objects sorted by capture_id.
    """
    if not captures:
        raise ValidationError("No captures found. Check your mission folder and filenames.")

    complete   = []
    incomplete = []

    # Sort numerically by capture_id, not lexicographically — capture_id is
    # a string of digits (e.g. "1", "2", ..., "10"), and a plain string sort
    # would order "10" before "2". int() is safe here because capture_id is
    # always produced by grouper.py's FILENAME_PATTERN, which only matches
    # all-digit capture IDs.
    for capture_id, capture in sorted(captures.items(), key=lambda kv: (0, int(kv[0])) if str(kv[0]).isdigit() else (1, kv[0])):
        if capture.is_complete():
            complete.append(capture)
        else:
            incomplete.append(capture)

    if incomplete:
        lines = [f"  Capture {c.capture_id}: missing {c.missing_bands()}" for c in incomplete]
        message = "Incomplete captures found:\n" + "\n".join(lines)

        if strict:
            raise ValidationError(message)
        else:
            print(f"[validator] Warning: {message}")
            print(f"[validator] Skipping {len(incomplete)} incomplete capture(s).")

    print(f"[validator] {len(complete)} complete captures ready for processing.")
    return complete