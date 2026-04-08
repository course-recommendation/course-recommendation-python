from functools import reduce
import math
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
            item_id: self._get_tradeoff_vector_of_item(item_id, top_item_id)
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

    @staticmethod
    def generate_refined_preference_weights(
        old_attribute_to_preference_configure: dict[str, PreferenceConfigure],
        category: list[TradeoffPair],
        item_tradeoff_vector: list[TradeoffPair],
    ) -> dict[str, PreferenceConfigure]:
        result = copy.deepcopy(old_attribute_to_preference_configure)

        if not set(category).issubset(item_tradeoff_vector):
            raise ValueError("The provided tradeoff vector does not belong to the category")

        for tradeoff_pair in item_tradeoff_vector:
            if tradeoff_pair in category and tradeoff_pair.improved():
                result[tradeoff_pair.attribute].weight = 5
            elif tradeoff_pair in category and tradeoff_pair.compromised():
                result[tradeoff_pair.attribute].weight = 1

        return result

    def _scale_sentiment_score(self, sentiment: int, N: float = 5) -> float:
        return 1.0 + (N - 1) / (1 + math.exp(-sentiment))

    def _scale_preference_score(self, score: float, min_sentiment: float, max_sentiment: float) -> float:
        if score < 1 or score > 5:
            raise ValueError(f"score must be in the closed interval [{1}, {5}]")
        return min_sentiment + (score - 1) * (max_sentiment - min_sentiment) / (5 - 1)

    def _build_attribute_to_utility_preference(self) -> dict[str, UtilityPreference]:
        all_sentiments = reduce(concat, self.item_id_to_item_sentiments.values())
        min_sentiment_score = min(x.sentiment_score for x in all_sentiments)
        max_sentiment_score = max(x.sentiment_score for x in all_sentiments)

        return {
            attribute: UtilityPreference(
                self.attribute_to_preference_configure[attribute].weight,
                self._generate_preference_function(
                    self._scale_preference_score(
                        self.attribute_to_preference_configure[attribute].target_sentiment_score,
                        min_sentiment_score,
                        max_sentiment_score,
                    )
                ),
            )
            for attribute in self.attributes
        }

    def _generate_preference_function(self, target_sentiment_score: float) -> Callable[[float], float]:
        def preference(x: float) -> float:
            if x >= target_sentiment_score:
                return 1.0
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

    def _get_tradeoff_vector_of_item(self, item_id: str, top_item_id: str) -> list[TradeoffPair]:
        top_sentiments = {s.attribute: s.sentiment_score for s in self.item_id_to_item_sentiments[top_item_id]}

        return [
            TradeoffPair(
                attribute=s.attribute,
                direction=TradeoffDirection.O_UP if s.sentiment_score > top_sentiments[s.attribute] else TradeoffDirection.O_DOWN,
            )
            for s in self.item_id_to_item_sentiments[item_id]
            if s.attribute in top_sentiments
        ]

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

    def _compute_tradeoff_benefit_of_category(
        self,
        category: list[TradeoffPair],
        item_ids: list[str],
        attribute_to_utility_preference: dict[str, UtilityPreference],
        item_id_to_utility: dict[str, float],
        item_id_to_tradeoff_vector: dict[str, list[TradeoffPair]],
    ) -> float:
        first_term = sum(
            attribute_to_utility_preference[tp.attribute].weight * (0.75 if tp.improved() else 0.25)
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