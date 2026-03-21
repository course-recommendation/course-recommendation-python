


from collections import defaultdict

from algorithms.feature_sentiments.extract_sentiments.sentires import sentires_extract_sentiments
from algorithms.feature_sentiments.extract_sentiments.translate.mod import translate_reviews
from algorithms.feature_sentiments.types import FSExtractSentimentsRequest, FSExtractSentimentsResult, ItemSentiment


def fs_extract_sentiments(request: FSExtractSentimentsRequest) -> list[FSExtractSentimentsResult]:
  translated_reviews = translate_reviews(request.reviews)
  
  sentires_sentiments = sentires_extract_sentiments(translated_reviews)
  print(sentires_sentiments)
  
  # item_id -> attribute -> list[sentiment]
  sentiment_map: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))

  for result in sentires_sentiments:
      for triplet in result.aspects:
          if triplet.feature in request.attributes:
              sentiment_map[result.item_id][triplet.feature].append(triplet.sentiment)

  results: list[FSExtractSentimentsResult] = []

  item_ids = {r.item_id for r in sentires_sentiments}

  for item_id in item_ids:
      item_sentiments: list[ItemSentiment] = []

      for attribute in request.attributes:
          sentiments = sentiment_map[item_id].get(attribute, [])

          if sentiments:
              sentiment_score = sum(sentiments) / len(sentiments)
          else:
              sentiment_score = 3

          item_sentiments.append(
              ItemSentiment(
                  attribute=attribute,
                  sentiment_score=sentiment_score
              )
          )

      results.append(
          FSExtractSentimentsResult(
              item_id=item_id,
              item_sentiments=item_sentiments
          )
      )

  print(results)
  return results