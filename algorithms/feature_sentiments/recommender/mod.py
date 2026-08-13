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

    #: Eq. (6) tradeoff values. The paper assigns the higher one to an improved (↑)
    #: attribute, but our attributes are bipolar axes where neither pole is better,
    #: so a lean is only a pro when it moves towards the user's target.
    TOWARDS_TARGET_TRADEOFF_VALUE = 0.75
    AWAY_FROM_TARGET_TRADEOFF_VALUE = 0.25

    #: Weight given to an attribute the user endorsed by asking for similar items.
    ENDORSED_WEIGHT = 5.0

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
            self._get_sentiments_of_item(top_item_id),
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
        item_id: str,
        item_id_to_item_sentiments: dict[str, list[ItemSentiment]],
    ) -> dict[str, PreferenceConfigure]:
        """Eq. (8): fold "show me items like this one" back into the preference model.

        The paper only raises the weight of the category's improved attributes and
        drops that of the compromised ones, which works because there ↑ already means
        better, so a heavier weight is enough to chase it. On a bipolar axis the
        weight carries no direction at all - the preference function peaks at the
        user's target whichever way the item leaned - so raising it alone would make
        the recommender pursue the *old* target harder, the opposite of the lean the
        user just endorsed.

        Instead we move the target onto the endorsed item's own position on every axis
        named by the category, which is what "similar to this one" actually means, and
        treat all of those attributes as endorsed rather than demoting the
        low-leaning half of them.
        """
        result = copy.deepcopy(old_attribute_to_preference_configure)

        if not set(category).issubset(item_tradeoff_vector):
            raise ValueError("The provided tradeoff vector does not belong to the category")

        item_sentiments = {s.attribute: s.sentiment_score for s in item_id_to_item_sentiments[item_id]}
        min_sentiment_score, max_sentiment_score = cls._sentiment_score_range(item_id_to_item_sentiments)
        # A flat sentiment range leaves no axis to move a target along.
        can_retarget = max_sentiment_score != min_sentiment_score

        for tradeoff_pair in category:
            preference_configure = result[tradeoff_pair.attribute]
            preference_configure.weight = cls.ENDORSED_WEIGHT

            if can_retarget and tradeoff_pair.attribute in item_sentiments:
                preference_configure.target_sentiment_score = cls._unscale_sentiment_score(
                    item_sentiments[tradeoff_pair.attribute],
                    min_sentiment_score,
                    max_sentiment_score,
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

    @classmethod
    def _unscale_sentiment_score(cls, sentiment: float, min_sentiment: float, max_sentiment: float) -> float:
        """Inverse of _scale_preference_score.

        Targets are stored on the user-facing [1, 5] preference scale, so a sentiment
        read off a real item has to be mapped back before it can become a target.
        """
        if max_sentiment == min_sentiment:
            raise ValueError("cannot unscale a sentiment when every item scores the same")

        score = cls.MIN_PREFERENCE_SCORE + (sentiment - min_sentiment) * (
            cls.MAX_PREFERENCE_SCORE - cls.MIN_PREFERENCE_SCORE
        ) / (max_sentiment - min_sentiment)

        return min(cls.MAX_PREFERENCE_SCORE, max(cls.MIN_PREFERENCE_SCORE, score))

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

    def _get_tradeoff_vector_of_item(self, item_id: str, top_item_id: str) -> list[TradeoffPair]:
        top_sentiments = self._get_sentiments_of_item(top_item_id)

        return [
            TradeoffPair(
                attribute=s.attribute,
                direction=TradeoffDirection.O_UP if s.sentiment_score > top_sentiments[s.attribute] else TradeoffDirection.O_DOWN,
            )
            for s in self.item_id_to_item_sentiments[item_id]
            # Eq. (4) only defines a tradeoff for a strict difference. Reading an equal
            # score as a lean would claim a pole the item does not lean towards, which
            # the explanation then states outright.
            if s.attribute in top_sentiments and s.sentiment_score != top_sentiments[s.attribute]
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

    def _tradeoff_value(
        self,
        tradeoff_pair: TradeoffPair,
        top_sentiments: dict[str, float],
        attribute_to_utility_preference: dict[str, UtilityPreference],
    ) -> float:
        """Eq. (6)'s tradeoff_i: whether this lean counts as a pro or a con.

        The paper reads ↑ as improved, which only holds on an axis that has a good
        end. Our axes are bipolar, so leaning high is not a benefit in itself; what
        makes a lean a pro is that it moves off the top candidate *towards* the user's
        target on that axis, and a con that it moves away.
        """
        target_sentiment_score = attribute_to_utility_preference[tradeoff_pair.attribute].target_sentiment_score
        top_sentiment_score = top_sentiments[tradeoff_pair.attribute]

        towards_target = (
            target_sentiment_score > top_sentiment_score if tradeoff_pair.leans_high()
            else target_sentiment_score < top_sentiment_score
        )

        return (
            self.TOWARDS_TARGET_TRADEOFF_VALUE if towards_target
            else self.AWAY_FROM_TARGET_TRADEOFF_VALUE
        )

    def _compute_tradeoff_benefit_of_category(
        self,
        category: list[TradeoffPair],
        item_ids: list[str],
        attribute_to_utility_preference: dict[str, UtilityPreference],
        item_id_to_utility: dict[str, float],
        item_id_to_tradeoff_vector: dict[str, list[TradeoffPair]],
        top_sentiments: dict[str, float],
    ) -> float:
        first_term = sum(
            attribute_to_utility_preference[tp.attribute].weight
            * self._tradeoff_value(tp, top_sentiments, attribute_to_utility_preference)
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
        top_sentiments: dict[str, float],
    ) -> float:
        return self._compute_tradeoff_benefit_of_category(
            category, item_ids, attribute_to_utility_preference, item_id_to_utility, item_id_to_tradeoff_vector,
            top_sentiments
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
        top_sentiments: dict[str, float],
    ) -> list[list[TradeoffPair]]:
        result: list[list[TradeoffPair]] = []

        while len(result) < self.total_recommended_categories - 1:
            candidates = [c for c in frequent_tradeoff_subsets if c not in result]

            if not candidates:
                break

            if not result:
                best = max(candidates, key=lambda c: self._compute_tradeoff_benefit_of_category(
                    c, item_ids, attribute_to_utility_preference, item_id_to_utility, item_id_to_tradeoff_vector,
                    top_sentiments
                ))
            else:
                best = max(candidates, key=lambda c: self._compute_fc(
                    c, result, item_ids, attribute_to_utility_preference, item_id_to_utility, item_id_to_tradeoff_vector,
                    top_sentiments
                ))

            result.append(best)

        return result