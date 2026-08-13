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
  #: The target the preference function peaks at, in sentiment space (i.e. after
  #: FSRecommender._scale_preference_score).
  target_sentiment_score: float


class TradeoffDirection(Enum):
  V_UP = "V_UP"
  V_DOWN = "V_DOWN"
  O_UP = "O_UP"
  O_DOWN = "O_DOWN"

@dataclass(frozen=True)
class TradeoffPair:
  """Where an item sits on one attribute axis relative to the top candidate.

  The attributes are bipolar descriptive axes ("Lý thuyết" at score 1, "Thực
  hành" at score 5): neither pole is better, so a direction only says which pole
  the item leans towards, never that it improved. What makes a lean a pro or a
  con is whether it moves towards the user's target - see
  FSRecommender._tradeoff_value.
  """
  attribute: str
  direction: TradeoffDirection

  def __repr__(self) -> str:
    return f'{self.attribute} {"higher" if self.leans_high() else "lower"}'

  @staticmethod
  def encode(tradeoff: 'TradeoffPair') -> str:
    return f"{tradeoff.attribute}:{tradeoff.direction.value}"

  @staticmethod
  def decode(encoded: str) -> "TradeoffPair":
    try:
      attribute, direction_value = encoded.rsplit(":", 1)
      direction = TradeoffDirection(str(direction_value))
    except (ValueError, KeyError) as e:
      raise ValueError(
          f"Invalid TradeoffResult encoding: {encoded}") from e

    return TradeoffPair(
        attribute=attribute,
        direction=direction
    )

  def leans_high(self):
    """Whether the item sits closer to the score-5 pole than the top candidate."""
    return self.direction == TradeoffDirection.O_UP or self.direction == TradeoffDirection.V_UP

  def leans_low(self):
    return not self.leans_high()

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