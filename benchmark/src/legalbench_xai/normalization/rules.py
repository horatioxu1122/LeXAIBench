from ..types import FeatureSpec


TEMPORAL_GROUP_PREFIXES = {
    "PAY_": "recent late payment history",
    "BILL_AMT": "recent bill balances",
    "PAY_AMT": "recent payment amounts",
}

# Correction reason_grouping_fix (v2): match the longest prefix first, so PAY_AMT1-6
# (payment amounts) group under "recent payment amounts" instead of matching "PAY_"
# ("recent late payment history"). The v1 order above is kept for the default path.
_LONGEST_PREFIX_FIRST = tuple(
    sorted(TEMPORAL_GROUP_PREFIXES.items(), key=lambda item: -len(item[0]))
)


def grouped_reason_label(feature_name: str, *, fix_prefix_order: bool = False) -> str | None:
    """Temporal reason group for a feature name, or None.

    fix_prefix_order=False (v1) checks prefixes in TEMPORAL_GROUP_PREFIXES order, so
    "PAY_" also captures "PAY_AMT*". fix_prefix_order=True (correction
    reason_grouping_fix) checks the longest prefix first.
    """
    prefixes = _LONGEST_PREFIX_FIRST if fix_prefix_order else TEMPORAL_GROUP_PREFIXES.items()
    for prefix, label in prefixes:
        if feature_name.startswith(prefix):
            return label
    return None


def signed_direction_from_weight(weight: float | None) -> str:
    if weight is None:
        return "neutral"
    if weight > 0:
        return "increases_denial"
    if weight < 0:
        return "decreases_denial"
    return "neutral"


def change_direction(from_value: object, to_value: object) -> str:
    if isinstance(from_value, (int, float)) and isinstance(to_value, (int, float)):
        if to_value > from_value:
            return "increase"
        if to_value < from_value:
            return "decrease"
    return "category_change"


def default_reason_text(feature: FeatureSpec, signed_direction: str) -> str:
    plain = feature.plain_meaning.replace("_", " ")
    if signed_direction == "increases_denial":
        return f"{plain} contributed to the denial"
    if signed_direction == "decreases_denial":
        return f"{plain} supported approval"
    return plain


def specificity_level(reason_text: str) -> str:
    lowered = reason_text.lower()
    if any(token in lowered for token in ("contributed", "recent", "high", "low", "missed")):
        return "specific"
    if len(lowered.split()) >= 3:
        return "partial"
    return "generic"
