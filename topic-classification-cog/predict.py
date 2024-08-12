# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import torch
from transformers import pipeline
import pandas as pd
import datetime
import json


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        model_path = "cardiffnlp/twitter-roberta-base-dec2021-tweet-topic-multi-all"
        self.device = 0 if torch.cuda.is_available() else -1
        self.model = pipeline(
            "text-classification", model=model_path, tokenizer=model_path
        )

    def predict(
        self,
        query: str = Input(description="Text input"),
    ) -> str:
        """Run a single prediction on the model"""
        all_result = []
        request_type = type(query)
        data = []
        try:
            data = json.loads(query)
            if type(data) is not list:
                data = [query]
            else:
                request_type = type(data)
        except Exception as e:
            print(e)
            data = [query]
            pass

        start_time = datetime.datetime.now()

        tokenizer_kwargs = {"truncation": True, "max_length": 512}
        all_result = self.model(data, batch_size=128,
                                top_k=3, **tokenizer_kwargs)

        end_time = datetime.datetime.now()
        elapsed_time = end_time - start_time

        output = {}
        output["time"] = str(elapsed_time)
        output["device"] = self.device
        output["result"] = all_result

        return json.dumps(all_result[0]) if request_type is str else json.dumps(output)
