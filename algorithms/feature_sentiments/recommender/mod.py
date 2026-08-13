from functools import reduce
from operator import concat
from typing import Callable
from mlxtend.frequent_patterns import apriori  # type: ignore
from mlxtend.preprocessing import TransactionEncoder  # type: ignore
import pandas as pd
import copy

from algorithms.feature_sentiments.types import (
    CategoryDetail, ItemSentiment, PreferenceConfigure,
    RecommendationResult, TradeoffDirection, TradeoffPair, UtilityPreference
)


class FSRecommender():
    #: Bounds of the user-facing preference scale that PreferenceConfigure targets live on.
    MIN_PREFERENCE_SCORE = 1.0
    MAX_PREFERENCE_SCORE = 5.0

    #: Eq. (6) tradeoff values, for an improved (↑) and a compromised (↓) attribute.
    IMPROVED_TRADEOFF_VALUE = 0.75
    COMPROMISED_TRADEOFF_VALUE = 0.25

    #: Eq. (8) weights, applied to the attributes named by the endorsed category.
    IMPROVED_WEIGHT = 5.0
    COMPROMISED_WEIGHT = 1.0

    def __init__(
        self,
        attributes: list[str],
        item_id_to_item_sentiments: dict[str, list[ItemSentiment]],
        attribute_to_preference_configure: dict[str, PreferenceConfigure],
        total_categories_to_recommend: int,
    ):
        self.attributes = attributes
        self.item_id_to_item_sentiments = item_id_to_item_sentiments
        self.attribute_to_preference_configure = attribute_to_preference_configure
        self.total_recommended_categories = total_categories_to_recommend

    def recommend(self) -> RecommendationResult:
        item_ids = list(self.item_id_to_item_sentiments.keys())

        attribute_to_utility_preference = self._build_attribute_to_utility_preference()

        item_id_to_utility = {
            item_id: self._compute_utility_of_item(item_id, attribute_to_utility_preference)
            for item_id in item_ids
        }

        top_item_id = max(item_id_to_utility, key=item_id_to_utility.__getitem__)

        item_id_to_tradeoff_vector = {
            item_id: self._get_tradeoff_vector_of_item(item_id, top_item_id, attribute_to_utility_preference)
            for item_id in item_ids
            if item_id != top_item_id
        }

        frequent_tradeoff_subsets = self._build_frequent_tradeoff_subsets(item_id_to_tradeoff_vector)

        recommended_categories = self._recommend_categories(
            item_ids,
            attribute_to_utility_preference,
            item_id_to_utility,
            item_id_to_tradeoff_vector,
            frequent_tradeoff_subsets,
        )

        return RecommendationResult(
            top_item_id=top_item_id,
            category_details=[
                CategoryDetail(
                    category=category,
                    item_ids=self._get_items_of_category(category, item_ids, item_id_to_tradeoff_vector)
                )
                for category in recommended_categories
            ],
            item_id_to_tradeoff_vector=item_id_to_tradeoff_vector,
        )

    @classmethod
    def generate_refined_preference_configure(
        cls,
        old_attribute_to_preference_configure: dict[str, PreferenceConfigure],
        category: list[TradeoffPair],
        item_tradeoff_vector: list[TradeoffPair],
    ) -> dict[str, PreferenceConfigure]:
        """Eq. (8): fold "show me items like this one" back into the preference model.

        Weight up the attributes the endorsed category improved on and weight down
        the ones it compromised. Since ↑ means the item sits closer to the user's
        target on that axis, chasing that target harder does pull the next round
        towards items like the endorsed one - which is what the paper's rule buys.

        The α term of Eq. (8) has no counterpart here: our attributes carry a
        sentiment score only, never a static specification value.
        """
        result = copy.deepcopy(old_attribute_to_preference_configure)

        if not set(category).issubset(item_tradeoff_vector):
            raise ValueError("The provided tradeoff vector does not belong to the category")

        for tradeoff_pair in category:
            result[tradeoff_pair.attribute].weight = (
                cls.IMPROVED_WEIGHT if tradeoff_pair.is_improved()
                else cls.COMPROMISED_WEIGHT
            )

        return result

    @classmethod
    def _sentiment_score_range(
        cls,
        item_id_to_item_sentiments: dict[str, list[ItemSentiment]],
    ) -> tuple[float, float]:
        all_sentiments = reduce(concat, item_id_to_item_sentiments.values())

        return (
            min(x.sentiment_score for x in all_sentiments),
            max(x.sentiment_score for x in all_sentiments),
        )

    @classmethod
    def _scale_preference_score(cls, score: float, min_sentiment: float, max_sentiment: float) -> float:
        if score < cls.MIN_PREFERENCE_SCORE or score > cls.MAX_PREFERENCE_SCORE:
            raise ValueError(
                f"score must be in the closed interval "
                f"[{cls.MIN_PREFERENCE_SCORE}, {cls.MAX_PREFERENCE_SCORE}]"
            )

        return min_sentiment + (score - cls.MIN_PREFERENCE_SCORE) * (max_sentiment - min_sentiment) / (
            cls.MAX_PREFERENCE_SCORE - cls.MIN_PREFERENCE_SCORE
        )

    def _build_attribute_to_utility_preference(self) -> dict[str, UtilityPreference]:
        min_sentiment_score, max_sentiment_score = self._sentiment_score_range(self.item_id_to_item_sentiments)

        result: dict[str, UtilityPreference] = {}

        for attribute in self.attributes:
            target_sentiment_score = self._scale_preference_score(
                self.attribute_to_preference_configure[attribute].target_sentiment_score,
                min_sentiment_score,
                max_sentiment_score,
            )

            result[attribute] = UtilityPreference(
                weight=self.attribute_to_preference_configure[attribute].weight,
                preference_function=self._generate_preference_function(target_sentiment_score),
                target_sentiment_score=target_sentiment_score,
            )

        return result

    def _generate_preference_function(self, target_sentiment_score: float) -> Callable[[float], float]:
        def preference(x: float) -> float:
            return 1.0 - abs(x - target_sentiment_score) / (5 - 1)
        return preference

    def _compute_utility_of_item(
        self,
        item_id: str,
        attribute_to_utility_preference: dict[str, UtilityPreference],
    ) -> float:
        return sum(
            attribute_to_utility_preference[s.attribute].weight
            * attribute_to_utility_preference[s.attribute].preference_function(s.sentiment_score)
            for s in self.item_id_to_item_sentiments[item_id]
        )

    def _get_sentiments_of_item(self, item_id: str) -> dict[str, float]:
        return {s.attribute: s.sentiment_score for s in self.item_id_to_item_sentiments[item_id]}

    def _get_tradeoff_vector_of_item(
        self,
        item_id: str,
        top_item_id: str,
        attribute_to_utility_preference: dict[str, UtilityPreference],
    ) -> list[TradeoffPair]:
        """Eq. (4): ↑ where the item is better than the top candidate, ↓ where worse.

        The paper compares raw sentiments because its ↑ rides on a "the higher, the
        better" default. Ours is a bipolar axis, so better means a higher preference
        value V(senti) = 1 - |senti - target| / 4. That function is strictly
        decreasing in the distance to the target, so V(senti(p')) > V(senti(p)) is
        exactly |senti(p') - target| < |senti(p) - target| - which is the comparison
        made here, and which correctly reads an item that overshoots far past the
        target as compromised rather than improved.
        """
        top_sentiments = self._get_sentiments_of_item(top_item_id)

        result: list[TradeoffPair] = []

        for s in self.item_id_to_item_sentiments[item_id]:
            if s.attribute not in top_sentiments:
                continue

            target_sentiment_score = attribute_to_utility_preference[s.attribute].target_sentiment_score
            item_gap = abs(s.sentiment_score - target_sentiment_score)
            top_gap = abs(top_sentiments[s.attribute] - target_sentiment_score)

            # Eq. (4) leaves the tie undefined, and it is a genuine tie here: two
            # scores sitting either side of the target at the same distance match it
            # equally well. Calling that an improvement would put a claim in the
            # category's explanation that the item does not back up.
            if item_gap == top_gap:
                continue

            result.append(TradeoffPair(
                attribute=s.attribute,
                direction=TradeoffDirection.O_UP if item_gap < top_gap else TradeoffDirection.O_DOWN,
            ))

        return result

    def _build_frequent_tradeoff_subsets(
        self,
        item_id_to_tradeoff_vector: dict[str, list[TradeoffPair]],
    ) -> list[list[TradeoffPair]]:
        encoded = [
            [TradeoffPair.encode(x) for x in vector]
            for vector in item_id_to_tradeoff_vector.values()
        ]

        te = TransactionEncoder()
        transaction_df = pd.DataFrame(
            te.fit(encoded).transform(encoded),  # type: ignore
            columns=te.columns_,  # type: ignore
        )

        subsets: list[list[str]] = list(
            apriori(transaction_df, min_support=0.01, use_colnames=True, max_len=3)["itemsets"]  # type: ignore
        )

        return [[TradeoffPair.decode(x) for x in subset] for subset in subsets]

    def _is_item_belong_to_category(
        self,
        item_id: str,
        category: list[TradeoffPair],
        item_id_to_tradeoff_vector: dict[str, list[TradeoffPair]],
    ) -> bool:
        tradeoff_vector = item_id_to_tradeoff_vector.get(item_id)
        return tradeoff_vector is not None and set(category).issubset(tradeoff_vector)

    def _get_items_of_category(
        self,
        category: list[TradeoffPair],
        item_ids: list[str],
        item_id_to_tradeoff_vector: dict[str, list[TradeoffPair]],
    ) -> list[str]:
        return [
            item_id for item_id in item_ids
            if self._is_item_belong_to_category(item_id, category, item_id_to_tradeoff_vector)
        ]

    def _tradeoff_value(self, tradeoff_pair: TradeoffPair) -> float:
        """Eq. (6)'s tradeoff_i: whether this attribute counts as a pro or a con."""
        return (
            self.IMPROVED_TRADEOFF_VALUE if tradeoff_pair.is_improved()
            else self.COMPROMISED_TRADEOFF_VALUE
        )

    def _compute_tradeoff_benefit_of_category(
        self,
        category: list[TradeoffPair],
        item_ids: list[str],
        attribute_to_utility_preference: dict[str, UtilityPreference],
        item_id_to_utility: dict[str, float],
        item_id_to_tradeoff_vector: dict[str, list[TradeoffPair]],
    ) -> float:
        first_term = sum(
            attribute_to_utility_preference[tp.attribute].weight * self._tradeoff_value(tp)
            for tp in category
        )

        sr = self._get_items_of_category(category, item_ids, item_id_to_tradeoff_vector)
        second_term = sum(item_id_to_utility[item_id] for item_id in sr) / len(sr)

        return first_term * second_term

    def _compute_diversity_of_category(
        self,
        category: list[TradeoffPair],
        selected_categories: list[list[TradeoffPair]],
        item_ids: list[str],
        item_id_to_tradeoff_vector: dict[str, list[TradeoffPair]],
    ) -> float:
        sr_c = set(self._get_items_of_category(category, item_ids, item_id_to_tradeoff_vector))

        return min(
            (1 - len(set(category) & set(ci)) / len(category))
            * (1 - len(sr_c & set(self._get_items_of_category(ci, item_ids, item_id_to_tradeoff_vector))) / len(sr_c))
            for ci in selected_categories
        )

    def _compute_fc(
        self,
        category: list[TradeoffPair],
        selected_categories: list[list[TradeoffPair]],
        item_ids: list[str],
        attribute_to_utility_preference: dict[str, UtilityPreference],
        item_id_to_utility: dict[str, float],
        item_id_to_tradeoff_vector: dict[str, list[TradeoffPair]],
    ) -> float:
        return self._compute_tradeoff_benefit_of_category(
            category, item_ids, attribute_to_utility_preference, item_id_to_utility, item_id_to_tradeoff_vector
        ) * self._compute_diversity_of_category(
            category, selected_categories, item_ids, item_id_to_tradeoff_vector
        )

    def _recommend_categories(
        self,
        item_ids: list[str],
        attribute_to_utility_preference: dict[str, UtilityPreference],
        item_id_to_utility: dict[str, float],
        item_id_to_tradeoff_vector: dict[str, list[TradeoffPair]],
        frequent_tradeoff_subsets: list[list[TradeoffPair]],
    ) -> list[list[TradeoffPair]]:
        result: list[list[TradeoffPair]] = []

        while len(result) < self.total_recommended_categories - 1:
            candidates = [c for c in frequent_tradeoff_subsets if c not in result]

            if not candidates:
                break

            if not result:
                best = max(candidates, key=lambda c: self._compute_tradeoff_benefit_of_category(
                    c, item_ids, attribute_to_utility_preference, item_id_to_utility, item_id_to_tradeoff_vector
                ))
            else:
                best = max(candidates, key=lambda c: self._compute_fc(
                    c, result, item_ids, attribute_to_utility_preference, item_id_to_utility, item_id_to_tradeoff_vector
                ))

            result.append(best)

        return result