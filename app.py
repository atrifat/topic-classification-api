import os
from dotenv import load_dotenv
import pandas as pd
from flask import Flask, request, jsonify
import datetime
import logging
import torch
from transformers import pipeline

load_dotenv()

APP_ENV = os.getenv("APP_ENV", "production")
LISTEN_HOST = os.getenv("LISTEN_HOST", "0.0.0.0")
LISTEN_PORT = os.getenv("LISTEN_PORT", "5000")
TOPIC_CLASSIFICATION_MODEL = os.getenv(
    "TOPIC_CLASSIFICATION_MODEL",
    "cardiffnlp/twitter-roberta-base-dec2021-tweet-topic-multi-all",
)

APP_VERSION = "0.0.1"

# Setup logging configuration
LOGGING_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOGGING_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
if APP_ENV == "production":
    logging.basicConfig(
        level=logging.INFO,
        datefmt=LOGGING_DATE_FORMAT,
        format=LOGGING_FORMAT,
    )
else:
    logging.basicConfig(
        level=logging.DEBUG,
        datefmt=LOGGING_DATE_FORMAT,
        format=LOGGING_FORMAT,
    )

app = Flask(__name__)

topic_classification_task = pipeline(
    "text-classification",
    model=TOPIC_CLASSIFICATION_MODEL,
    tokenizer=TOPIC_CLASSIFICATION_MODEL,
)


def perform_topic_classification(query):
    tokenizer_kwargs = {"truncation": True, "max_length": 512}

    result = []

    try:
        result = topic_classification_task(query, top_k=3, **tokenizer_kwargs)
    except Exception as e:
        logging.error(e)

    return result


@app.errorhandler(Exception)
def handle_exception(error):
    res = {"error": str(error)}
    return jsonify(res)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    q = data["q"]
    start_time = datetime.datetime.now()
    result = perform_topic_classification(q)
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    logging.debug("elapsed detection time: %s", str(elapsed_time))
    return jsonify(result)


@app.route("/", methods=["GET"])
def index():
    response = {"message": "Use /predict route to get prediction result"}
    return jsonify(response)


@app.route("/app_version", methods=["GET"])
def app_version():
    response = {"message": "This app version is ".APP_VERSION}
    return jsonify(response)


if __name__ == "__main__":
    app.run(host=LISTEN_HOST, port=LISTEN_PORT)
