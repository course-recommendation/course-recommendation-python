import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI

# Initialize OpenAI client (set your API key in env: OPENAI_API_KEY)
client = OpenAI()

# Your course IDs
course_ids = [
   "BAA00003",
  "BAA00004",
  "BAA00005",
  "BAA00006",
  "BAA00007",
  "BAA00011",
  "BAA00012",
  "BAA00013",
  "BAA00014",
  "BAA00021",
  "BAA00022",
  "BAA00101",
  "BAA00102",
  "BAA00103",
  "BAA00104",
  "BIO00001",
  "BIO00002",
  "BIO00081",
  "BIO00082",
  "CHE00001",
  "CHE00002",
  "CHE00081",
  "CHE00082",
  "CSC00004",
  "CSC00009",
  "CSC10001",
  "CSC10002",
  "CSC10003",
  "CSC10004",
  "CSC10006",
  "CSC10007",
  "CSC10008",
  "CSC10009",
  "CSC10010",
  "CSC10011",
  "CSC10012",
  "CSC10013",
  "CSC10102",
  "CSC10103",
  "CSC10104",
  "CSC10105",
  "CSC10106",
  "CSC10107",
  "CSC10108",
  "CSC10121",
  "CSC10204",
  "CSC10251",
  "CSC10252",
  "CSC11002",
  "CSC11003",
  "CSC11004",
  "CSC11005",
  "CSC11103",
  "CSC11106",
  "CSC11107",
  "CSC11111",
  "CSC11112",
  "CSC11113",
  "CSC11115",
  "CSC12001",
  "CSC12002",
  "CSC12003",
  "CSC12004",
  "CSC12005",
  "CSC12102",
  "CSC12103",
  "CSC12105",
  "CSC12106",
  "CSC12107",
  "CSC12108",
  "CSC12109",
  "CSC12110",
  "CSC12111",
  "CSC13001",
  "CSC13002",
  "CSC13003",
  "CSC13005",
  "CSC13006",
  "CSC13007",
  "CSC13008",
  "CSC13009",
  "CSC13010",
  "CSC13101",
  "CSC13102",
  "CSC13103",
  "CSC13106",
  "CSC13107",
  "CSC13108",
  "CSC13112",
  "CSC13114",
  "CSC13115",
  "CSC13116",
  "CSC13117",
  "CSC13118",
  "CSC14001",
  "CSC14002",
  "CSC14003",
  "CSC14004",
  "CSC14005",
  "CSC14006",
  "CSC14007",
  "CSC14008",
  "CSC14101",
  "CSC14105",
  "CSC14109",
  "CSC14111",
  "CSC14112",
  "CSC14113",
  "CSC14114",
  "CSC14115",
  "CSC14116",
  "CSC14117",
  "CSC14118",
  "CSC14119",
  "CSC14120",
  "CSC15001",
  "CSC15002",
  "CSC15003",
  "CSC15004",
  "CSC15005",
  "CSC15006",
  "CSC15007",
  "CSC15008",
  "CSC15009",
  "CSC15010",
  "CSC15011",
  "CSC15102",
  "CSC15103",
  "CSC15104",
  "CSC15105",
  "CSC15106",
  "CSC15107",
  "CSC15201",
  "CSC15202",
  "CSC16001",
  "CSC16002",
  "CSC16003",
  "CSC16004",
  "CSC16005",
  "CSC16101",
  "CSC16102",
  "CSC16104",
  "CSC16105",
  "CSC16106",
  "CSC16107",
  "CSC16109",
  "CSC16110",
  "CSC16111",
  "CSC16112",
  "CSC17001",
  "CSC17101",
  "CSC17102",
  "CSC17103",
  "CSC17104",
  "CSC17105",
  "CSC17106",
  "CSC17107",
  "CSC18001",
  "CSC18101",
  "CSC18102",
  "CSC18103",
  "CSC18104",
  "CSC18105",
  "ENV00001",
  "ENV00003",
  "GEO00002",
  "MTH00021",
  "MTH00022",
  "MTH00035",
  "MTH00044",
  "MTH00045",
  "MTH00050",
  "MTH00051",
  "MTH00052",
  "MTH00053",
  "MTH00056",
  "PHY00001",
  "PHY00002",
  "PHY00081"
]

API_URL = "https://www.fit.hcmus.edu.vn/dai-hoc/thong-tin-mon-hoc?handler=courseDetails&mamh={}"

def fetch_course_description(course_id):
    try:
        url = API_URL.format(course_id)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://www.fit.hcmus.edu.vn/",
            "X-Requested-With": "XMLHttpRequest"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Adjust based on actual API response structure
        description = data.get("course", {}).get("description", "")

        return course_id, description
    except Exception as e:
        print(f"Error fetching {course_id}: {e}")
        return course_id, ""

def translate_text(text):
    if not text:
        return ""

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=f"Translate the following text to Vietnamese:\n\n{text}"
        )
        return response.output[0].content[0].text.strip()
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # fallback to original

def process_course(course_id):
    course_id, description = fetch_course_description(course_id)
    translated = translate_text(description)

    return {
        "id": course_id,
        "description": translated
    }

def main():
    results = []

    # Adjust max_workers depending on your machine / API limits
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_course, cid) for cid in course_ids]

        for future in tqdm(as_completed(futures), total=len(futures)):
            results.append(future.result())

    # Save to JSON file
    with open("output/course_description.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("Done! Output saved to output/course_description.json")

if __name__ == "__main__":
    main()