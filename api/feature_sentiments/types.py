from dataclasses import dataclass

from pydantic import BaseModel

from algorithms.feature_sentiments.types import CategoryDetail, ItemSentiment, PreferenceConfigure, TradeoffPair


@dataclass
class ServerRecommendationResult:
  top_item_id: str
  category_details: list[CategoryDetail]
  item_id_to_tradeoff_vector: dict[str, list[TradeoffPair]]
  attribute_to_preference_configure: dict[str, PreferenceConfigure]

class RecommendationRequest(BaseModel):
  attributes: list[str]
  item_id_to_item_sentiments: dict[str, list[ItemSentiment]]
  attribute_to_preference_configure: dict[str, PreferenceConfigure]
  
class RefinedRecommendationRequest(BaseModel):
  attributes: list[str]
  item_id_to_item_sentiments: dict[str, list[ItemSentiment]]
  #: The item the user endorsed, whose sentiments become the new targets.
  item_id: str
  item_tradeoff_vector: list[TradeoffPair]
  category: list[TradeoffPair]
  old_attribute_to_preference_configure: dict[str, PreferenceConfigure]