
from algorithms.feature_sentiments.extract_sentiments.mod import fs_extract_sentiments
from algorithms.feature_sentiments.extract_sentiments.translate.mod import translate_reviews
from algorithms.feature_sentiments.types import FSExtractSentimentsRequest, FSExtractSentimentsResult, FSItemReview
from api.mod import app
from algorithms.feature_sentiments.recommender.mod import FSRecommender
from algorithms.feature_sentiments.extract_sentiments.mod import fs_extract_sentiments
from api.feature_sentiments.types import RecommendationRequest, RefinedRecommendationRequest, ServerRecommendationResult

@app.post("/fs/recommendation")
async def get_recommendation(payload: RecommendationRequest) -> ServerRecommendationResult:
  recommender = FSRecommender(
  attributes=payload.attributes,
  item_id_to_item_sentiments=payload.item_id_to_item_sentiments,
  attribute_to_preference_configure=payload.attribute_to_preference_configure,
  total_categories_to_recommend=5
  )
  
  result = recommender.recommend()
  
  return ServerRecommendationResult(
    top_item_id=result.top_item_id,
    category_details=result.category_details,
    item_id_to_tradeoff_vector=result.item_id_to_tradeoff_vector,
    attribute_to_preference_configure=payload.attribute_to_preference_configure
  )

@app.post("/fs/recommendation/refined")
async def get_refined_recommendation(payload: RefinedRecommendationRequest) -> ServerRecommendationResult:
  
  attribute_to_preference_configure = FSRecommender.generate_refined_preference_configure(
    payload.old_attribute_to_preference_configure,
    payload.category,
    payload.item_tradeoff_vector,
    payload.item_id,
    payload.item_id_to_item_sentiments
  )
  
  recommender = FSRecommender(
    attributes=payload.attributes,
    item_id_to_item_sentiments=payload.item_id_to_item_sentiments,
    attribute_to_preference_configure=attribute_to_preference_configure,
    total_categories_to_recommend=5
  )
  
  result = recommender.recommend()
  
  return ServerRecommendationResult(
    top_item_id=result.top_item_id,
    category_details=result.category_details,
    item_id_to_tradeoff_vector=result.item_id_to_tradeoff_vector,
    attribute_to_preference_configure=attribute_to_preference_configure
  )
  
@app.post("/fs/translate-reviews")
async def translate_reviews_api(payload: list[FSItemReview]) -> list[FSItemReview]:
  return translate_reviews(payload)

@app.post("/fs/extract-sentiments")
async def extract_sentiments(payload: FSExtractSentimentsRequest) -> list[FSExtractSentimentsResult]:
  return fs_extract_sentiments(payload)