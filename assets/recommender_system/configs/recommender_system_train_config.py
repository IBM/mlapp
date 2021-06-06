recommender_system_config = {
    # analysis results properties to save
    "results_output": {
        "models_objects": [
            "models",
            "user_mapping",
            "item_mapping"
        ],
        "analysis_results": [
            "models",
        ]
    },
    # train data config
    "train_config": {
        # data sources configurations
        "data_sources": {
            "db": {
                "orders_query":
                    "SELECT  t.* FROM ("
                    "SELECT DISTINCT o.remote_consumer_id_thirdparty, o.remote_order_id, o.payment_status, "
                    "o.placed_on, b.product_code, b.remote_basket_id, b.payment_amount, "
                    "b.product_quantity, pcb.l1_code, pcb.valid_id "
                    "FROM Orders o "
                    "JOIN orderPayment b "
                    "ON o.remote_order_id = b.remote_order_id "
                    "JOIN "
                    "(SELECT DISTINCT l1_code,valid_id,product_code"
                    " FROM ProductCatalogBasketExtended_V4) AS pcb"
                    " ON b.product_code = pcb.product_code"
                    ") AS t "
                    "WHERE t.payment_status  = 'complete' AND t.l1_code = 'Capsules' ORDER BY remote_basket_id;",

                "attributes_query":
                    "SELECT valid_id, product_friendlyname, category_product, beverage_type, coffee_intensity, "
                    "milk_intensity, cup_size, preparation_temperature, dietary_impact_fat, "
                    "dietary_impact_sugar, kid_product, flavored, pack_size_beverages_per_pack "
                    "FROM ProductCatalogBasketExtended_V4 "
                    "WHERE "
                    "active_in_recommendation = 'Yes' AND "
                    "l1_code = 'Capsules' AND "
                    "valid_name IS NOT NULL"
                }
        },
        # features handling configurations
        "data_handling": {
            # type 0: bought => 1, not bought => 0
            # type 1: len(bought) > 1 => 1, len(bought) == 1 => 0
            # type 2: number of purchases => scale transformation to range 0-5
            # type 3: np.logp1(number of purchases)
            "implicit_explicit_transformation_type": 0,
            "black_list_items": [
                "caramel-latte-macchiato", "cappuccino-ice", "espresso-ristretto", "lungo",
                "espresso-decaf", "nestea-peach", "espresso", "soy-cappuccino", "catuai-do-brasil"
            ],
            "timestamps": True
        },
        "evaluation": {
            "second_best_item": True,
            "test_start_date": 20180601,
            "items_for_test": 1,
            "cv_splits": 5,
            "user_test_percent": 0.20,
            "seed": 42,
            "grid_search_measure": "top_1"
        },
        "models": {
            "popularity": {
                "params": {},
                "hyper_params": {},
                "score_function": "ranking",
                "score_kwargs": {},
                "data_frame": "purchases",
                "data_for_init": "popularity"
            },
            "implicit_bpr": {
                "params": {
                    "iterations": 15,
                    "model": 'bpr'
                },
                "hyper_params": {
                    "factors": [40],
                    "regularization": [0.1],
                    "alpha_val": [1]
                },
                "score_function": "ranking",
                "score_kwargs": {},
                "data_frame": "purchases"
            },
            "content_based": {
                "params": {},
                "hyper_params": {},
                "score_function": "ranking",
                "score_kwargs": {},
                "data_frame": "ratings",
                "data_for_init": "attributes",
                "params_for_init": {
                    "user_ratings_fill_na_value": 0,
                    "item_attributes_fill_na_value": 0
                }
            }
        },
        "hybrid": {
            "weights_algorithm": "accuracy",  # accuracy/regression/optimization/(None)
            "measure": "top_1",
            "minimum_models_num": 2
        }
    },
    # forecast config
    "forecast_config": {
        "model_id": "id of model used for prediction",
        "data_sources": {
            "local": {
                "files": {
                    "attributes": {"path": "./data/attributes.csv", "sep": ","},
                    "users": {"path": "./data/users.csv", "sep": ","},
                    "ratings": {"path": "./data/ratings.backup.csv", "sep": "|"},
                }
            },
            "item_item_collaborative_filtering": {
                "ratings": "ratings.csv",
                "normalized_ratings": "normalized_ratings.csv",
                "item_mean_ratings": "item_mean_ratings.csv",
                "user_ratings_mean": "user_ratings_mean.csv",
                "cosine_similarity": "cosine_similarity.csv"
            }
        },
    },
    # task settings
    "task_settings": {
        "model_name": "recommender_system",
    },
    # model settings
    "model_settings": {

    }
}
