import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
from cossim import cossim, svd_recommend
import re
import ast

# ROOT_PATH for linking with all your files.
# Feel free to use a config.py or settings.py with a global export variable
os.environ["ROOT_PATH"] = os.path.abspath(os.path.join("..", os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, "init.json")

# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, "r", encoding="utf-8") as file:
    data = json.load(file)
drinks_df = pd.DataFrame(data)


def extract_caffeine(row):
    text = " ".join(
        [
            str(row.get("description", "")),
            str(row.get("bullet_points", "")),
            str(row.get("reviews", "")),
            str(row.get("title", "")),
        ]
    )
    match = re.search(r"(\d+)\s*mg(?:\s*of)?\s*caffeine", text, re.IGNORECASE)
    if match:
        return f"{match.group(1)} mg caffeine"
    new_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:cups?|cup)\s+of\s+coffee", text, re.IGNORECASE
    )
    if new_match:
        return f"{new_match.group(1)} cups of coffee"
    return ""


drinks_df["caffeine_mg"] = drinks_df.apply(extract_caffeine, axis=1)


app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return render_template("base.html", title="sample html")


# TEST
def process_results(results):
    processed = []
    for _, d in results.iterrows():
        try:
            flavors = ast.literal_eval(d["flavors"])
        except ValueError:
            flavors = []
        extra_info = d["whole_title"].replace(d["title"], "").strip(", ")
        processed.append(
            {
                "title": d["title"],
                "whole_title": d.get("whole_title", ""),
                "extra_info": extra_info,
                "url": d["url"],
                "caffeine_mg": d["caffeine_mg"],
                "flavors": flavors,
            }
        )

    return processed


@app.route("/recommendations")
def episodes_search():
    text = request.args.get("query")
    method = request.args.get("method", "svd")
    ingredient = request.args.get("ingredient")
    if method == "svd":
        result_df = svd_recommend(text, ingredient, drinks_df)
    else:
        result_df = cossim(text, drinks_df)

    res = process_results(result_df)
    return res


if "DB_NAME" not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)
