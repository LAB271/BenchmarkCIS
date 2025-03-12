# we are going to create a dataclass that subclasses `MetricWithLLM` and `SingleTurnMetric`
from dataclasses import dataclass, field

# import the base classes
from ragas.metrics.base import MetricWithLLM, SingleTurnMetric, MetricType
from ragas.metrics import NonLLMStringSimilarity, SemanticSimilarity, BleuScore, RougeScore, DistanceMeasure

# import types
import typing as t
from ragas.callbacks import Callbacks
from ragas.dataset_schema import SingleTurnSample


@dataclass
class ConsistencyMetric(MetricWithLLM, SingleTurnMetric):
    name: str = "consistency_metric"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {MetricType.SINGLE_TURN: {"response", "reference"}}
    )

    def __post_init__(self):
        # init the faithfulness metric
        self.semantic_metric = SemanticSimilarity(embeddings=self.eval_embeddings)
        self.string_similarity_metric = NonLLMStringSimilarity(distance_measure=DistanceMeasure.LEVENSHTEIN)
        self.blue_metric = BleuScore()
        self.rouge_metric = RougeScore(rouge_type='rougeL')
        self.scorer_metrics = [
            # Don't use hamming distance because we are looking at strings of differing lengths
            (self.string_similarity_metric),
            (self.blue_metric),
            (self.rouge_metric),
            (self.semantic_metric)
        ]

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        avg_score = 0
        for scorer in self.scorer_metrics:
            avg_score += await scorer._single_turn_ascore(sample)
        avg_score = avg_score / len(self.scorer_metrics)
        return avg_score