from functools import reduce
import math
from operator import concat
from typing import Callable
from mlxtend.frequent_patterns import apriori  # type: ignore
from mlxtend.preprocessing import TransactionEncoder # type: ignore
import pandas as pd
import copy

from algorithms.feature_sentiments.types import CategoryDetail, ItemSentiment, PreferenceConfigure, RecommendationResult, TradeoffDirection, TradeoffPair, UtilityPreference

class FeatureSentimentsRecommender():
  def __init__(
      self,
      attributes: list[str],
      item_id_to_item_sentiments: dict[str, list[ItemSentiment]],
      attribute_to_preference_configure: dict[str, PreferenceConfigure],
      total_categories_to_recommend: int,
  ):
    self.attributes = attributes
    self.item_id_to_item_sentiments = item_id_to_item_sentiments
    self.attribute_to_preference_configure: dict[str, PreferenceConfigure] = attribute_to_preference_configure
    self.total_recommended_categories = total_categories_to_recommend
    
    # Context for algorithm
    self.item_ids: list[str] = []
    self.attribute_to_utility_preference: dict[str, UtilityPreference] = {}
    self.item_id_to_utility: dict[str, float] = {}
    self.item_id_to_tradeoff_vector: dict[str, list[TradeoffPair]] = {
    }
    self.frequent_tradeoff_subsets: list[list[TradeoffPair]] = []
    self.top_item_id: str = ""
    
    # Result
    self.recommendation_result: RecommendationResult | None = None

  def recommend(self) -> RecommendationResult:
    self._populate_algorithm_context()
    
    recommended_categories = self._recommend_categories()
    recommendation_result = RecommendationResult(
        top_item_id=self.top_item_id,
        category_details=[CategoryDetail(category=category, item_ids=self._get_items_of_category(category)) for category in recommended_categories],
        item_id_to_tradeoff_vector=self.item_id_to_tradeoff_vector
    )
    
    self.recommendation_result = recommendation_result
    
    return recommendation_result
 
  def _populate_algorithm_context(self):
    self.item_ids = list(self.item_id_to_item_sentiments.keys())
    self._populate_attribute_to_utility_preference()
    self._populate_item_id_to_utility()
    self._populate_item_id_to_tradeoff_vector()
    self._populate_frequent_tradeoff_subsets()  
    
  @staticmethod
  def generate_refined_preference_weights(old_attribute_to_preference_configure: dict[str, PreferenceConfigure] , category: list[TradeoffPair], item_tradeoff_vector: list[TradeoffPair]) -> dict[str, PreferenceConfigure]:
    result: dict[str, PreferenceConfigure] = copy.deepcopy(old_attribute_to_preference_configure)
    
    is_item_belong_to_category = set(category).issubset(item_tradeoff_vector)
    
    if not is_item_belong_to_category:
      raise ValueError("The provided tradeoff vector does not belong to the category")

    for tradeoff_pair in item_tradeoff_vector:
      if tradeoff_pair in category and tradeoff_pair.improved():
        result[tradeoff_pair.attribute].weight = 5
      elif tradeoff_pair in category and tradeoff_pair.compromised():
        result[tradeoff_pair.attribute].weight = 1
    
    return result

  def _scale_sentiment_score(self, sentiment: int, N: float=5) -> float:
    return 1.0 + (N - 1) / (
        1 + math.exp(-sentiment)
    )

  def _scale_preference_score(self, score: float, min_sentiment: float, max_sentiment: float) -> float:
    if score < 1 or score > 5:
      raise ValueError(
          f"score must be in the closed interval [{1}, {5}]")

    return min_sentiment + (score - 1) * (max_sentiment - min_sentiment) / (5 - 1)
    
  def _populate_attribute_to_utility_preference(self) -> None:
    result: dict[str, UtilityPreference] = {}
    
    min_sentiment_score = min([x.sentiment_score for x in reduce(concat, self.item_id_to_item_sentiments.values())])
    max_sentiment_score = max([x.sentiment_score for x in reduce(concat, self.item_id_to_item_sentiments.values())])

    for attribute in self.attributes:
      preference = self.attribute_to_preference_configure[attribute]
      
      scaled_preference_score = self._scale_preference_score(
          preference.target_sentiment_score, min_sentiment_score, max_sentiment_score)
      
      result[attribute] = UtilityPreference(preference.weight, self._generate_preference_function(scaled_preference_score))

    self.attribute_to_utility_preference = result

  def _generate_preference_function(
      self, target_sentiment_score: float
  ) -> Callable[[float], float]:

    def preference(x: float) -> float:
      if x >= target_sentiment_score:
        return 1.0
      return 1.0 - abs(x - target_sentiment_score) / (5 - 1)

    return preference

  def _compute_utility_of_item(self, item_id: str) -> float:
    total = 0.0

    for item_sentiment in self.item_id_to_item_sentiments[item_id]:
      utility_preference = self.attribute_to_utility_preference[
          item_sentiment.attribute]

      weight = utility_preference.weight
      preference_function = utility_preference.preference_function

      total += weight * preference_function(item_sentiment.sentiment_score)

    return total

  def _populate_item_id_to_utility(
      self,
  ) -> None:

    result: dict[str, float] = {}

    for item_id in self.item_ids:
      result[item_id] = self._compute_utility_of_item(item_id)

    self.item_id_to_utility = result

  def _get_tradeoff_vector_of_item(self, item_id: str, top_item_id: str) -> list[TradeoffPair]:
    result: list[TradeoffPair] = []

    item_sentiments = self.item_id_to_item_sentiments[item_id]
    top_item_sentiments = self.item_id_to_item_sentiments[top_item_id]

    for item_sentiment in item_sentiments:
      for top_item_sentiment in top_item_sentiments:
        if item_sentiment.attribute == top_item_sentiment.attribute:
          direction = TradeoffDirection.O_UP if item_sentiment.sentiment_score > top_item_sentiment.sentiment_score else TradeoffDirection.O_DOWN

          result.append(TradeoffPair(
              attribute=item_sentiment.attribute, direction=direction))

    return result

  def _populate_item_id_to_tradeoff_vector(self) -> None:
    result: dict[str, list[TradeoffPair]] = {}

    max_utility_item_id = max(
        self.item_id_to_utility.items(), key=lambda item: item[1])[0]

    self.top_item_id = max_utility_item_id

    for item_id in self.item_ids:
      if item_id != max_utility_item_id:
        result[item_id] = self._get_tradeoff_vector_of_item(
            item_id, max_utility_item_id)

    self.item_id_to_tradeoff_vector = result

  def _populate_frequent_tradeoff_subsets(self) -> None:

    tradeoff_vectors = list(self.item_id_to_tradeoff_vector.values())
    encoded_tradeoff_vectors = [[TradeoffPair.encode(
        x) for x in vector] for vector in tradeoff_vectors]

    te = TransactionEncoder()
    transaction_array = te.fit( # type: ignore
        encoded_tradeoff_vectors).transform(encoded_tradeoff_vectors)
    transaction_df = pd.DataFrame(
        transaction_array, columns=te.columns_)  # type: ignore

    subsets: list[list[str]] = list(apriori(transaction_df, min_support=0.01,
                   use_colnames=True, max_len=3)['itemsets']) # type: ignore
    decoded_subsets = [[TradeoffPair.decode(
        x) for x in vector] for vector in subsets]

    self.frequent_tradeoff_subsets = decoded_subsets

  def _is_item_belong_to_category(self, item_id: str, category: list[TradeoffPair]):
    tradeoff_vector = self.item_id_to_tradeoff_vector.get(item_id)
    return (tradeoff_vector is not None) and set(category).issubset(tradeoff_vector)

  def _get_items_of_category(self, category: list[TradeoffPair]) -> list[str]:
    return [item_id for item_id in self.item_ids if self._is_item_belong_to_category(
        item_id, category)]

  def _compute_tradeoff_benefit_of_category(self, category: list[TradeoffPair]) -> float:
    first_term = 0.0

    for tradeoff_pair in category:
      W = self.attribute_to_utility_preference[tradeoff_pair.attribute].weight
      tradeoff_value = 0.75 if tradeoff_pair.improved() else 0.25

      first_term += W * tradeoff_value

    SR = self._get_items_of_category(category)
    second_term = 1.0 / len(SR) * sum([self.item_id_to_utility[item_id] for item_id in SR])

    return first_term * second_term

  def _compute_diversity_of_category(self, category: list[TradeoffPair], selected_categories: list[list[TradeoffPair]]) -> float:
    product_list: list[float] = []

    # to match with paper's notation
    c = category
    sc = selected_categories

    sr_c = self._get_items_of_category(c)

    for ci in sc:
      sr_ci = self._get_items_of_category(ci)

      first_term = 1 - len(set(c) & set(ci)) / len(c)
      second_term = 1 - len(set(sr_c) & set(sr_ci)) / len(sr_c)

      product_list.append(first_term * second_term)

    return min(product_list)

  def _compute_fc(self, category: list[TradeoffPair], selected_categories: list[list[TradeoffPair]]) -> float:
    return self._compute_tradeoff_benefit_of_category(category) * self._compute_diversity_of_category(category, selected_categories)

  def _recommend_categories(self) -> list[list[TradeoffPair]]:
    result: list[list[TradeoffPair]] = []

    while len(result) < self.total_recommended_categories - 1:
      if len(result) == 0:
        result.append(max(
            self.frequent_tradeoff_subsets,
            key=lambda category: self._compute_tradeoff_benefit_of_category(
                category)
        ))

      else:
        result.append(max(
            [category for category in self.frequent_tradeoff_subsets if category not in result],
            key=lambda category: self._compute_fc(category, result)
        ))

    return result
