from .scorer import ExplanationScorer
from ..normalization import ExplanationCanonicalizer
from ..normalization.examples import sample_payload
from ..types import DatasetBundle


def score_sample(dataset_bundle: DatasetBundle) -> dict:
    canonical = ExplanationCanonicalizer(dataset_bundle).canonicalize(
        sample_payload(dataset_bundle)
    )
    return {
        "canonical_explanation": canonical.to_dict(),
        "scores": ExplanationScorer().score(canonical),
    }
