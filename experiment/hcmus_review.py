import json
import sys
import os
import pandas as pd


def xlsx_to_json(xlsx_file, json_file):
    df = pd.read_excel(xlsx_file, sheet_name='clean')
    data = json.loads(df.to_json(orient='records', force_ascii=False))

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Converted {xlsx_file} -> {json_file}")


def json_to_xlsx(json_file, xlsx_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # If file exists → append, otherwise create new
    if os.path.exists(xlsx_file):
        with pd.ExcelWriter(
            xlsx_file,
            engine="openpyxl",
            mode="a",
            if_sheet_exists="replace"
        ) as writer:
            df.to_excel(writer, sheet_name="english", index=False)
    else:
        with pd.ExcelWriter(xlsx_file, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="english", index=False)

    print(f"Converted {json_file} -> {xlsx_file} (sheet: english)")


def main():
    if len(sys.argv) != 4:
        print("Usage:")
        print("  python convert.py xlsx2json input.xlsx output.json")
        print("  python convert.py json2xlsx input.json output.xlsx")
        sys.exit(1)

    mode = sys.argv[1]
    input_file = sys.argv[2]
    output_file = sys.argv[3]

    if mode == "xlsx2json":
        xlsx_to_json(input_file, output_file)
    elif mode == "json2xlsx":
        json_to_xlsx(input_file, output_file)
    else:
        print("Invalid mode. Use 'xlsx2json' or 'json2xlsx'")


if __name__ == "__main__":
    main()