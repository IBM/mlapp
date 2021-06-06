import json


class PopularityModel:
    def __init__(self, data, popularity, *args, **kwargs):
        self.popularity = popularity

    def fit(self):
        pass

    def predict(self):
        return list(json.loads(self.popularity.to_json()).values())
