import json

with open("output/course_description.json", "r", encoding="utf-8") as f:
    data = json.load(f)

with open("output/course_description.sql", "w", encoding="utf-8") as f:
    for item in data:
        course_id = item["id"]
        description = item["description"].replace("'", "''")  # escape quotes

        sql = f"UPDATE course SET description = '{description}' WHERE code = '{course_id}';\n"
        f.write(sql)

print("SQL file generated: output/course_description.sql")