"""Pure geometry and aspect-ratio helpers for datasets.

These are stateless utility functions — no I/O, no DB, no events.
"""

from __future__ import annotations

from fractions import Fraction


def calculate_target_dims(
    long_side: int, majority_ar: float, orientation: str
) -> tuple[int, int]:
    """Calculate target (width, height) where both dimensions are divisible by 32.

    Args:
        long_side: Reference long side in pixels.
        majority_ar: Width / Height ratio.
        orientation: 'landscape', 'portrait', or 'squared'.

    Returns:
        (target_width, target_height) tuple, both multiples of 32.
    """
    def _closest_32(val: float) -> int:
        return max(32, round(val / 32) * 32)

    target_long = _closest_32(long_side)

    if orientation == "portrait":
        # For portrait, AR = Width / Height (< 1)
        # Long side is Height. Width = Height * AR
        raw_short = target_long * majority_ar
        return (_closest_32(raw_short), target_long)
    else:
        # For landscape/squared, AR = Width / Height (>= 1)
        # Long side is Width. Height = Width / AR
        raw_short = target_long / majority_ar
        return (target_long, _closest_32(raw_short))


def ar_to_display(ar: float, orientation: str) -> str:
    """Convert a float aspect ratio to a human-friendly string like '3:2'.

    Args:
        ar: Width / Height ratio.
        orientation: 'landscape', 'portrait', or 'squared'.

    Returns:
        A colon-separated ratio string.
    """
    if abs(ar - 1.0) < 0.01:
        return "1:1"

    frac = Fraction(ar).limit_denominator(32)
    w_part, h_part = frac.numerator, frac.denominator

    # For portrait the stored AR is < 1 (W/H), so display as H:W feels more natural
    if orientation == "portrait":
        return f"{h_part}:{w_part}"
    return f"{w_part}:{h_part}"
