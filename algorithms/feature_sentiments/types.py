from dataclasses import dataclass
from enum import Enum
from typing import Callable


@dataclass
class ItemSentiment:
  attribute: str
  sentiment_score: float


@dataclass
class UtilityPreference:
  weight: float
  preference_function: Callable[[float], float]


class TradeoffDirection(Enum):
  V_UP = "V_UP"
  V_DOWN = "V_DOWN"
  O_UP = "O_UP"
  O_DOWN = "O_DOWN"

@dataclass(frozen=True)
class TradeoffPair:
  attribute: str
  direction: TradeoffDirection

  def __repr__(self) -> str:
    better = self.direction in (
        TradeoffDirection.V_UP,
        TradeoffDirection.O_UP,
    )
    return f'{"better" if better else "worse"} {self.attribute}'

  @staticmethod
  def encode(tradeoff: 'TradeoffPair') -> str:
    return f"{tradeoff.attribute}:{tradeoff.direction.value}"

  @staticmethod
  def decode(encoded: str) -> "TradeoffPair":
    try:
      attribute, direction_value = encoded.split(":")
      direction = TradeoffDirection(str(direction_value))
    except (ValueError, KeyError) as e:
      raise ValueError(
          f"Invalid TradeoffResult encoding: {encoded}") from e

    return TradeoffPair(
        attribute=attribute,
        direction=direction
    )

  def improved(self):
    return self.direction == TradeoffDirection.O_UP or self.direction == TradeoffDirection.V_UP

  def compromised(self):
    return not self.improved()

@dataclass
class PreferenceConfigure:
  weight: float
  target_sentiment_score: float
  
  def __init__(self, weight: float = 3.0, target_sentiment_score: float = 3.0) -> None:
    self.weight = weight
    self.target_sentiment_score = target_sentiment_score

@dataclass
class CategoryDetail:
  category: list[TradeoffPair]
  item_ids: list[str]

@dataclass
class RecommendationResult:
  top_item_id: str
  category_details: list[CategoryDetail]
  item_id_to_tradeoff_vector: dict[str, list[TradeoffPair]]
  
@dataclass
class FSItemReview:
  item_id: str
  user_id: str
  review_text: str
  
@dataclass
class FSExtractSentimentsRequest:
  reviews: list[FSItemReview]
  attributes: list[str]
@dataclass
class FSExtractSentimentsResult:
  item_id: str
  item_sentiments: list[ItemSentiment]