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
  """Eq. (4)'s ↑ / ↓, i.e. improved / compromised against the top candidate.

  The "UP"/"DOWN" wording is historical: these values are stored verbatim in
  persisted recommendation results, so they keep their old spelling even though
  they no longer mean "leans towards the high pole". See TradeoffPair.
  """
  V_UP = "V_UP"
  V_DOWN = "V_DOWN"
  O_UP = "O_UP"
  O_DOWN = "O_DOWN"

@dataclass(frozen=True)
class TradeoffPair:
  """Whether an item is better or worse than the top candidate on one attribute.

  Eq. (4) reads ↑ off `senti_i(p') > senti_i(p)`, which is only "better" because
  the paper's attributes default to "the higher, the better". Ours are bipolar
  descriptive axes ("Lý thuyết" at score 1, "Thực hành" at score 5) whose
  preference function V peaks at the user's target, so better means a higher
  V - equivalently, a smaller distance to that target. See
  FSRecommender._get_tradeoff_vector_of_item.
  """
  attribute: str
  direction: TradeoffDirection

  def __repr__(self) -> str:
    return f'{self.attribute} {"improved" if self.is_improved() else "compromised"}'

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

  def is_improved(self):
    """Whether the item sits closer to the user's target than the top candidate."""
    return self.direction == TradeoffDirection.O_UP or self.direction == TradeoffDirection.V_UP

  def is_compromised(self):
    return not self.is_improved()

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