import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
from cossim import cossim

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

app = Flask(__name__)
CORS(app)



@app.route("/")
def home():
    return render_template("base.html", title="sample html")


# TEST

def process_results(results):
    processed = []
    for _,d in results.iterrows():
        try:
            flavors = ast.literal_eval(d["flavors"])
        except (ValueError):
            flavors = []
        extra_info = d["whole_title"].replace(d["title"], "").strip(", ")
        processed.append({"title": d["title"], "whole_title": d.get("whole_title", ""), "extra_info": extra_info, "url": d["url"], "caffeine_mg": d["caffeine_mg"], "flavors": flavors, "rating": d["rating"]})
    return processed


@app.route("/recommendations")
def episodes_search():
    text = request.args.get("query")
    res = process_results(cossim(text, drinks_df))
    return res


if "DB_NAME" not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)
