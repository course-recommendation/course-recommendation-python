import os

from dotenv import load_dotenv
from openai import OpenAI
import json

from algorithms.feature_sentiments.types import FSItemReview

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def translate_reviews(reviews: list[FSItemReview]) -> list[FSItemReview]:
  input_json = json.dumps([review.__dict__ for review in reviews], ensure_ascii=False)

  response = openai_client.responses.create(
      model="gpt-4.1-mini",
      input=f"Translate all review fields in the following JSON to English. Keep the JSON structure unchanged and only output valid JSON, dont format:\n{input_json}"
  )

  translated_json = response.output_text

  try:
    translated_data = json.loads(translated_json)
    return [FSItemReview(**item) for item in translated_data]
  except json.JSONDecodeError as e:
    print("Failed to decode JSON:", e)
    return []