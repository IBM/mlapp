import pandas as pd
import numpy as np
import json


class AssociationRuleModel:
    def __init__(self, data, association_rules, *args, **kwargs):
        self.association_rules = association_rules
        self.purchases = data
        self.predictions = None

    def fit(self):
        helper_df = pd.DataFrame()
        helper_df['items_purchased'] = self.purchases.apply(
            lambda x: [idx for idx, val in enumerate(x) if pd.notnull(val)], axis=1)

        helper_df['relevant_rows'] = helper_df['items_purchased'].apply(
            lambda x: list(
                self.association_rules[
                    self.association_rules['item_x'].isin(x) & ~self.association_rules['item_y'].isin(x)
                ].index)
        )

        helper_df['ar_sorted'] = helper_df['relevant_rows'].apply(
            lambda x: list(json.loads(
                (self.association_rules.iloc[x].pivot_table(index='item_y', values='value', aggfunc=np.sum)
                 .sort_values(by='value', ascending=False)).to_json()
            ).values())[0])

        helper_df['max_value'] = helper_df['ar_sorted'].apply(lambda x: np.max(list(x.values())))

        # TODO: rethink how to score purchased items
        helper_df.apply(
            lambda x: x['ar_sorted'].update({str(item): x['max_value'] * 10 for item in x['items_purchased']}), axis=1)

        self.predictions = helper_df['ar_sorted']

    def predict(self):
        if self.predictions is None:
            raise Exception("Must fit model first before fitting model.")
        else:
            return self.predictions
