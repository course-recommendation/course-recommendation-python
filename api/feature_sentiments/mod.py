
from api.mod import app
from algorithms.feature_sentiments.mod import FeatureSentimentsRecommender
from api.feature_sentiments.types import RecommendationRequest, RefinedRecommendationRequest, ServerRecommendationResult

@app.post("/fs/recommendation")
async def get_recommendation(payload: RecommendationRequest) -> ServerRecommendationResult:
  recommender = FeatureSentimentsRecommender(
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
  
  attribute_to_preference_configure = FeatureSentimentsRecommender.generate_refined_preference_weights(
    payload.old_attribute_to_preference_configure,
    payload.category,
    payload.item_tradeoff_vector
  )
  
  recommender = FeatureSentimentsRecommender(
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