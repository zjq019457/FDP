import evaluate
import datasets

_DESCRIPTION = """Exact match metric.
Returns 1.0 if prediction exactly matches reference (string equality), else 0.0.
If references are a list-of-lists, counts match if prediction equals any reference in the list.
"""

_CITATION = ""

_KWARGS_DESCRIPTION = """
Args:
    predictions: list of predicted strings
    references: list of reference strings OR list of list of reference strings
Returns:
    exact_match: float in [0, 1]
"""

class ExactMatch(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                    # allow either string or sequence on runtime; datasets Features requires a type
                    "references": datasets.Value("string"),
                }
            ),
        )

    def _compute(self, predictions, references):
        if len(predictions) == 0:
            return {"exact_match": 0.0}

        # If references is list-of-lists, match any
        # (Some tasks provide multiple gold answers)
        if len(references) > 0 and isinstance(references[0], (list, tuple)):
            correct = 0
            for p, rs in zip(predictions, references):
                correct += 1 if any(p == r for r in rs) else 0
            return {"exact_match": correct / len(predictions)}

        # Normal 1-1 case
        correct = sum(1 for p, r in zip(predictions, references) if p == r)
        return {"exact_match": correct / len(predictions)}
