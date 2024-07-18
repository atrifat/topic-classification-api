import os
from dotenv import load_dotenv
import pandas as pd
from flask import Flask, request, jsonify
from flask_caching import Cache
import datetime
import logging
import functools
import torch
from transformers import pipeline

load_dotenv()

ENABLE_API_TOKEN = os.getenv("ENABLE_API_TOKEN", "false") == "true"
API_TOKEN = os.getenv("API_TOKEN", "")
APP_ENV = os.getenv("APP_ENV", "production")
LISTEN_HOST = os.getenv("LISTEN_HOST", "0.0.0.0")
LISTEN_PORT = os.getenv("LISTEN_PORT", "5000")
TOPIC_CLASSIFICATION_MODEL = os.getenv(
    "TOPIC_CLASSIFICATION_MODEL",
    "cardiffnlp/twitter-roberta-base-dec2021-tweet-topic-multi-all",
)

CACHE_DURATION_SECONDS = int(os.getenv("CACHE_DURATION_SECONDS", 60))
ENABLE_CACHE = os.getenv("ENABLE_CACHE", "false") == "true"

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

if ENABLE_API_TOKEN and API_TOKEN == "":
    raise Exception("API_TOKEN is required if ENABLE_API_TOKEN is enabled")

app = Flask(__name__)

cache_config = {
    "DEBUG": True if APP_ENV != "production" else False,
    "CACHE_TYPE": "SimpleCache" if ENABLE_CACHE else "NullCache",
    "CACHE_DEFAULT_TIMEOUT": CACHE_DURATION_SECONDS,  # Cache duration in seconds
}
cache = Cache(config=cache_config)
cache.init_app(app)

topic_classification_task = pipeline(
    "text-classification",
    model=TOPIC_CLASSIFICATION_MODEL,
    tokenizer=TOPIC_CLASSIFICATION_MODEL,
)


def is_valid_api_key(api_key):
    if api_key == API_TOKEN:
        return True
    else:
        return False


def api_required(func):
    @functools.wraps(func)
    def decorator(*args, **kwargs):
        if ENABLE_API_TOKEN:
            if request.json:
                api_key = request.json.get("api_key")
            else:
                return {"message": "Please provide an API key"}, 400
            # Check if API key is correct and valid
            if request.method == "POST" and is_valid_api_key(api_key):
                return func(*args, **kwargs)
            else:
                return {"message": "The provided API key is not valid"}, 403
        else:
            return func(*args, **kwargs)

    return decorator


def make_key_fn():
    """A function which is called to derive the key for a computed value.
       The key in this case is the concat value of all the json request
       parameters. Other strategy could to use any hashing function.
    :returns: unique string for which the value should be cached.
    """
    user_data = request.get_json()
    return ",".join([f"{key}={value}" for key, value in user_data.items()])


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
@api_required
@cache.cached(make_cache_key=make_key_fn)
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
