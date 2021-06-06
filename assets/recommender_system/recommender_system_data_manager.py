from mlapp.managers import DataManager, pipeline
from helpers.model_utilities import print_time
import helpers.recommender_system_feature_engineering as feature_engineering
import pandas as pd
import time


class RecommenderSystemDataManager(DataManager):
    @pipeline
    def load_train_data(self):
        # Load content based recommender data
        data = self._load_data()
        return data

    @pipeline
    def clean_train_data(self, data):
        # implicit/explicit ratings data
        data['orders'].rename(columns={
            'placed_on': 'date',
            'remote_consumer_id_thirdparty': 'user',
            'valid_id': 'item',
            'remote_order_id': 'order_id',
            'product_quantity': 'quantity'
        }, inplace=True)
        data['orders'] = data['orders'][['date', 'user', 'item', 'order_id', 'quantity']]

        # attributes data
        data['items_friendly_name'] = data['attributes'][['valid_id', 'product_friendlyname']].copy()\
            .drop_duplicates('valid_id').rename(columns={
                'valid_id': 'item',
                'product_friendlyname': 'item_name'
            }
        )
        data['attributes'].drop(['product_friendlyname'], axis=1, inplace=True)

        data['attributes']['category_product'] = data['attributes']['category_product'].apply(
            lambda x: 'Cappucino_and_Latte' if x == 'Cappucino&Latte' else x
        )
        data['attributes'] = pd.get_dummies(data['attributes'].drop_duplicates('valid_id').set_index(['valid_id']))

        return data

    @pipeline
    def transform_train_data(self, data):
        t = time.time()  # init current time

        data['orders'], data['dicts'] = \
            feature_engineering.create_user_and_item_dicts_and_replace_values(data['orders'])

        data['attributes'].index = data['attributes'].index.map(data['dicts']['items_to_id'])
        data['attributes'].sort_index(inplace=True)

        t = print_time("Creating dictionaries and replacing values in DataFrame", t)

        # changing types to numeric
        data['orders'] = data['orders'].apply(pd.to_numeric)
        # copying date column
        data['orders']['date_min'] = data['orders']['date']
        data['orders']['date_max'] = data['orders']['date']

        # Pivoting table (Grouping by user and item)
        user_item_index_df = feature_engineering.pivot_table_group_by_user_and_item(data['orders'])
        t = print_time("Pivoting table (Grouping by user and item)", t)

        # Pivoting table (Users x Items)
        user_x_item_df = feature_engineering.pivot_table_user_x_item(user_item_index_df.reset_index())
        t = print_time("Pivoting table (Users x Items)", t)

        # Calculating most popular items (Users x Items)
        popularity_df = feature_engineering.calc_popularity_data_frame(user_item_index_df)
        t = print_time("Calculating most popular items", t)

        # Calculating recency of items by age
        recency_df = feature_engineering.calc_recency_data_frame(data['dicts']['id_to_item'])
        t = print_time("Calculating recency items", t)

        # Calculating association rule
        association_rule_df = feature_engineering\
            .calc_association_rule_data_frame(data['orders'], len(data['orders']['order_id'].unique()))
        t = print_time("Calculating association rules", t)

        # TODO: create only df transformations required by models and by user_x_item df
        ratings_df = feature_engineering.transform_data_from_implicit_to_explicit_ratings(
            user_x_item_df['order_id'],
            transformation_type=self.data_handling.get('implicit_explicit_transformation_type', 0))
        t = print_time("Transforming data to unary ratings", t)

        # Data transformations to return
        data['purchases'] = user_x_item_df['quantity']
        data['frequencies'] = user_x_item_df['order_id']
        if self.data_handling.get('timestamps', False):
            data['timestamps'] = {
                'first': user_x_item_df['date_min'],
                'last': user_x_item_df['date_max']
            }
        data['popularity'] = popularity_df
        data['association_rule'] = association_rule_df
        data['recency'] = recency_df
        data['ratings'] = ratings_df
        data['black_list'] = list(map(
            lambda x: data['dicts']['items_to_id'][x], self.data_handling['black_list_items']))

        return data

    @pipeline
    def load_forecast_data(self):
        # Load content based recommender data
        data = self._load_data()

        return data

    @pipeline
    def clean_forecast_data(self, data):
        return data

    @pipeline
    def transform_forecast_data(self, data):
        return data

    # Loading data
    def _load_data(self):

        orders_query = """
            SELECT  t.* FROM (
            SELECT DISTINCT o.remote_consumer_id_thirdparty, o.remote_order_id, o.payment_status,
            o.placed_on, b.product_code, b.remote_basket_id, b.payment_amount,
            b.product_quantity, pcb.l1_code, pcb.valid_id
            FROM Orders o
            JOIN orderPayment b
            ON o.remote_order_id = b.remote_order_id
            JOIN
            (SELECT DISTINCT l1_code,valid_id,product_code
            FROM ProductCatalogBasketExtended_V4) AS pcb
            ON b.product_code = pcb.product_code
            ) AS t
            WHERE t.payment_status  = 'complete' AND t.l1_code = 'Capsules' ORDER BY remote_basket_id;
        """

        attributes_query = """
            SELECT valid_id, product_friendlyname, category_product, beverage_type, coffee_intensity,
            milk_intensity, cup_size, preparation_temperature, dietary_impact_fat,
            dietary_impact_sugar, kid_product, flavored, pack_size_beverages_per_pack
            FROM ProductCatalogBasketExtended_V4
            WHERE
            active_in_recommendation = 'Yes' AND
            l1_code = 'Capsules' AND
            valid_name IS NOT NULL
        """

        data = {
            'orders': self.db_handler.get_db_from_query_safely(orders_query),
            'attributes': self.db_handler.get_db_from_query_safely(attributes_query)
        }

        return data
