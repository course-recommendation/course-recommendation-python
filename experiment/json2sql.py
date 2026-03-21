from calendar import c
import json
import argparse

USER_ID = "bceba397-16c3-4e92-ba57-56d44c87b805"
POST_ID = 19


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input JSON file")
    parser.add_argument("-o", "--output", required=True, help="Output SQL file")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    values = []
    for item in data:
        review = item["review"].replace("'", "''")
        course_id = item["course_id"]
        values.append(f"({POST_ID}, '{course_id}', '{USER_ID}', '{review}')")

    sql = "INSERT INTO post_comment(post_id, course_id, user_id, content) VALUES\n"
    sql += ",\n".join(values) + ";"

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(sql)


if __name__ == "__main__":
    main()