import requests
import random
import json

URL = "https://www.fit.hcmus.edu.vn/dai-hoc/thong-tin-mon-hoc?handler=courses"

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Referer": "https://www.fit.hcmus.edu.vn/",
}

ATTRIBUTES = [
    "content",
    "instructor",
    "workload",
    "difficulty",
    "practicality",
    "fairness",
    "materials",
    "interaction",
    "organization",
    "value",
]

response = requests.get(URL, headers=headers)
response.raise_for_status()

courses = response.json()

sql_lines: list[str] = []
sql_lines.append("INSERT INTO course (id, name, extra_data) VALUES")

values: list[str] = []
for course in courses:
    ma_mh = course.get("maMH")
    name_vn = course.get("nameVN")

    if ma_mh and name_vn:
        # escape single quotes for SQL
        name_vn = name_vn.replace("'", "''")

        extra_data = { # type: ignore
            "source": "FS",
            "itemSentiments": [
                {
                    "attribute": attr,
                    "sentimentScore": round(random.uniform(1.0, 5.0), 1),
                }
                for attr in ATTRIBUTES
            ],
        }

        extra_data_json = json.dumps(extra_data, ensure_ascii=False)
        extra_data_json = extra_data_json.replace("'", "''")

        values.append(
            f"('{ma_mh}', '{name_vn}', '{extra_data_json}')"
        )

sql_lines.append(",\n".join(values) + ";")

sql_script = "\n".join(sql_lines)

with open("insert_courses.sql", "w", encoding="utf-8") as f:
    f.write(sql_script)

print("SQL script written to insert_courses.sql")
