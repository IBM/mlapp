import json


class RecencyModel:
    def __init__(self, data, recency, *args, **kwargs):
        self.recency = recency

    def fit(self):
        pass

    def predict(self):
        return list(json.loads(self.recency.to_json()).values())
