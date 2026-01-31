import json

INPUT_FILE = "item_sentiments.json"
LIMIT = 200

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

values = []

for i, (course_id, sentiments) in enumerate(data.items()):
    if i >= LIMIT:
        break

    extra_data = { # type: ignore
        "algorithm": "FS",
        "itemSentiments": [
            {
                "attribute": s["attribute"],
                "sentimentScore": s["sentiment_score"]
            }
            for s in sentiments
        ]
    }

    # Convert to compact JSON string
    extra_data_json = json.dumps(extra_data, ensure_ascii=False)

    # Escape single quotes for MySQL
    extra_data_json = extra_data_json.replace("'", "''")

    values.append(f"('{course_id}', '{extra_data_json}')") # type: ignore

sql = (
    "INSERT INTO course (id, extra_data)\nVALUES\n  "
    + ",\n  ".join(values) # type: ignore
    + ";"
)

with open("insert_courses_2.sql", "w", encoding="utf-8") as f:
    f.write(sql)

print("Generated insert_courses_2.sql")
