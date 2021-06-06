from mlapp.managers import ModelManager, pipeline
import time
from model_utilities import print_time
import helpers.recommender_system_evaluator as evaluator
import helpers.recommender_system_model_utilities as utilities
import pandas as pd


class RecommenderSystemModelManager(ModelManager):
    @pipeline
    def train_model(self, data, model_id):
        """
        Training the recommender system model
        :param data: data received from data manager
        :param model_id: the model id
        :return: results manager object containing relevant results from the model
        """
        t = time.time()  # init current time
        models = {}  # init models dict

        # evaluation configurations
        items_for_test = self.train_config['evaluation'].get('items_for_test', 1)
        second_best_item = self.train_config['evaluation'].get('second_best_item', False)
        test_start_date = self.train_config['evaluation'].get('test_start_date')
        seed = self.train_config['evaluation'].get('seed', None)

        # split train test
        train_data, test_data = evaluator.split_train_test(
            data['purchases'],
            timestamps=data.get('timestamps'),
            users_percent=self.train_config['evaluation'].get('user_test_percent', 0.25),
            test_start_date=test_start_date,
            items_for_test=items_for_test,
            second_best_item=second_best_item,
            seed=seed)
        t = print_time('Splitting train and test', t)

        # create cross validation splits
        num_of_cv_splits = self.train_config['evaluation'].get('cv_splits', 5)
        cv_splits = evaluator.cv_splits(
            train_data,
            timestamps=data.get('timestamps'),
            n_splits=num_of_cv_splits,
            random_state=seed,
            test_start_date=test_start_date,
            items_for_test=items_for_test,
            second_best_item=second_best_item
        )
        t = print_time('Creating cross validation splits', t)

        # models grid search for best params by cross validation
        for model_key in self.train_config["models"]:
            models[model_key] = utilities.grid_search_model_on_cv(
                model_key, cv_splits, data, self.train_config["models"], self.train_config.get('evaluation', {})
            )
            t = print_time(
                'Finished grid search on model "' + model_key + '". ' +
                'CV mean score '
                '(' + self.train_config.get('evaluation', {}).get('grid_search_measure', 'auc') + ') ' +
                str(models[model_key]['cv_mean_score']), t)

        # training models with train set using best params and hyper params
        for model_key in models:
            models[model_key].update(
                utilities.run_model(
                    model_key, data, train_data, test_data, self.train_config["models"],
                    data['black_list'], models[model_key]["params"]
                )
            )

        t = print_time('Calculated test validation accuracy for all models', t)

        # printing results
        print("> Models results: ", {key: (val['accuracy'], val['params']) for key, val in models.items()})

        # hybrid model
        utilities.run_hybrid_model(data, models, train_data, test_data, cv_splits, self.train_config.get('hybrid', {}))
        t = print_time('Calculated hybrid model', t)
        if 'hybrid_model' in models:
            print("> Hybrid model, accuracy: ", models['hybrid_model']['accuracy'], ", ",
                  "combination: ", models['hybrid_model']['combination'], ", "
                  "weights: ", models['hybrid_model']['weights'], ", ")

            # store results in custom table
            output_df = utilities.hybrid_output_for_client(
                models['hybrid_model']['model'], data['purchases'], data['dicts'], black_list=data['black_list'])
            merged_df = pd.merge(output_df, data['items_friendly_name'], how='left', left_on='item', right_on='item')
            merged_df = merged_df[['user', 'item', 'item_name', 'score']]
            self.results_manager.add_custom_table('recommendations', merged_df)

        # storing analysis in results manager
        self.results_manager.set_value('models_objects', 'models', models)
        self.results_manager.set_value('models_objects', 'user_mapping', data['dicts']['id_to_user'])
        self.results_manager.set_value('models_objects', 'item_mapping', data['dicts']['id_to_item'])
        self.results_manager.set_value('analysis_results', 'models', {
            key: {
                'accuracy': val.get('accuracy', ''),
                'params': val.get('params', {}),
                'cv_mean_score': val.get('cv_mean_score', ''),
                'combination': val.get('combination', ''),
                'weights': val.get('weights', '')
            } for key, val in models.items()
        })
        return self.results_manager

    @pipeline
    def forecast(self, data, train_results):
        """
        Forecasting new users/items
        :param data: data received from data manager
        :param train_results: train results that were stored during the model's training
        :return: TODO
        """
        print("done forecast")


