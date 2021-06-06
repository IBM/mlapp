from common.data_science.model_utilities import powerset
import models.recommender_system.helpers.recommender_system_evaluator as evaluator
from models.recommender_system.models.hybrid import HybridModel
from models.recommender_system.models.popularity import PopularityModel
from models.recommender_system.models.recency import RecencyModel
from models.recommender_system.models.association_rule import AssociationRuleModel
from models.recommender_system.models.content_based import ContentBasedModel
from models.recommender_system.models.implicit import ImplicitModel
from models.recommender_system.models.item_item_collaborative_filtering import ItemItemCollaborativeFilteringModel
from sklearn.model_selection import ParameterGrid
import numpy as np
import pandas as pd


MODELS = {
    'popularity': PopularityModel,
    'recency': RecencyModel,
    'association_rule': AssociationRuleModel,
    'content_based': ContentBasedModel,
    'implicit_als': ImplicitModel,
    'implicit_bpr': ImplicitModel,
    'item_item_collaborative_filtering': ItemItemCollaborativeFilteringModel
}


def grid_search_model_on_cv(model_key, cv_splits, data, models_config, evaluation_config):
    """
    Running grid search of hyper params using cross validation splits
    :param model_key: recommender model name
    :param cv_splits: cross validation splits
    :param data: data received from data manager
    :param models_config: configuration of models
    :param evaluation_config: configuration of evaluation
    :return: best params from hyper params search and the cross validation mean score
    """
    # init best values
    best_score = -1
    best_params = None
    cv_predictions = None

    # iterating hyper params
    for hyper_params in list(ParameterGrid(models_config[model_key]["hyper_params"])):

        # initializing params
        curr_accuracies = []
        curr_predictions = []
        curr_params = models_config[model_key]["params"].copy()
        curr_params.update(hyper_params)

        # getting accuracy score for each split
        for split_train, split_test in cv_splits:
            model_result = run_model(
                model_key, data, split_train, split_test, models_config, data['black_list'], curr_params
            )
            # saving accuracy
            curr_accuracies.append(model_result['accuracy'])
            # saving predictions for later to be used by the hybrid model
            curr_predictions.append(model_result['predictions'])

        # calculating cv mean score and selecting best params
        cv_mean_score = np.mean(
            [acc[evaluation_config.get('grid_search_measure', 'auc')]
             for acc in curr_accuracies]
        )
        if cv_mean_score > best_score:
            best_score = cv_mean_score
            best_params = curr_params.copy()
            cv_predictions = curr_predictions

    return {
        "params": best_params,
        "cv_mean_score": best_score,
        "cv_predictions": cv_predictions
    }


def run_model(model_key, data, train, test, models_config, black_list=None, params=None):
    """
    Runs one of the recommender models
    :param model_key: recommender model name
    :param data: data received from data manager
    :param train: train data
    :param test: test data
    :param models_config: configuration of models
    :param black_list: list of black listed items
    :param params: params for model
    :return: model results: model object, test predictions, accuracy of test
    """
    if params is None:
        params = {}

    model_class = MODELS.get(model_key)
    if model_class is None:
        raise ValueError("Unsupported model name: '" + model_key + "'")

    # init current model
    model = model_class(
        data[models_config[model_key]["data_frame"]][pd.notnull(train)],
        data.get(models_config[model_key].get("data_for_init")),
        **models_config[model_key].get("params_for_init", {})
    )

    # fit current model
    model.fit(**params)

    # predict
    predictions = model.predict()
    if isinstance(predictions, pd.DataFrame) and hasattr(predictions, 'reindex_like'):
        predictions = predictions.reindex_like(test)

    # save current accuracy score
    accuracy = evaluator.score_function(
        models_config[model_key]["score_function"],
        predictions,
        data[models_config[model_key]["data_frame"]][pd.notnull(train)],
        data[models_config[model_key]["data_frame"]][pd.notnull(test)],
        black_list,
        **models_config[model_key]["score_kwargs"])

    return {
        'model': model,
        'predictions': predictions,
        'accuracy': accuracy
    }


def run_hybrid_model(data, models, train, test, cv, hybrid_config):
    """
    Receives all recommender models and creates a hybrid out of them. Updates the model dictionary object.
    :param data: data received from the data manager
    :param models: dictionary containing different models
    :param train: train implicit/explicit data
    :param test: test implicit/explicit data
    :param cv: cross validation splits
    :param hybrid_config: configuration of hybrid
    :return: None
    """
    # less than 2 models
    if len(models) < hybrid_config.get('minimum_models_num', 2):
        print("> Not enough models for creating a hybrid model!")
        return

    # accuracy measure for comparison
    hybrid_measured_accuracy = hybrid_config.get('measure', 'auc')

    # init values
    best_accuracy, best_combination, best_weights, best_scaler = 0, None, None, None

    # find best hybrid combination
    for models_combination in powerset(list(models.keys())):
        # only combinations with more than one models are relevant
        if len(models_combination) < hybrid_config.get('minimum_models_num', 2):
            continue

        # init model
        hybrid_model = HybridModel(
            num_of_users=len(data['purchases']),
            measure=hybrid_config.get('measure', 'auc'),
            weights_algorithm=hybrid_config.get('weights_algorithm')
        )

        # collect accuracies from all CV
        curr_accuracies = []
        curr_weights = []

        for cv_index in range(len(cv)):
            cv_train = cv[cv_index][0]
            cv_test = cv[cv_index][1]

            # fit and predict on current cv_index
            hybrid_model.fit({k: models[k] for k in models.keys() if k in models_combination},
                             train=cv_train, test=cv_test, cv_index=cv_index,
                             reset_weights=True, black_list=data['black_list'])
            hybrid_predictions = hybrid_model.predict()

            # calculating ranking score
            hybrid_accuracy = evaluator.score_function(
                "ranking",
                hybrid_predictions,
                data["purchases"][pd.notnull(cv_train)],
                data["purchases"][pd.notnull(cv_test)],
                data['black_list'],
                **hybrid_config.get("score_kwargs", {})
            )

            # hybrid accuracy for cv_index
            curr_accuracies.append(hybrid_accuracy[hybrid_measured_accuracy])
            # hybrid weights for cv_index
            curr_weights.append(hybrid_model.weights)

        # mean score for all CVs
        cv_mean_score = np.mean(curr_accuracies)
        weights_mean = [np.mean([w[i] for w in curr_weights]) for i in range(len(curr_weights[0]))]
        print(">>>>>>>> Hybrid Combination: ", models_combination, " <<<<<<<<<<<")
        print("> CV mean score: ", cv_mean_score)
        print("> Weights: ", weights_mean)

        # comparing with previous accuracies
        if cv_mean_score >= best_accuracy:
            best_accuracy = cv_mean_score
            best_combination = models_combination
            best_weights = weights_mean
            best_scaler = hybrid_model.scaler

    # running best hybrid model using train data
    hybrid_model = HybridModel(
        num_of_users=len(data['purchases']),
        measure=hybrid_config.get('measure', 'auc'),
        weights=best_weights,
        scaler=best_scaler,
        weights_algorithm=hybrid_config.get('weights_algorithm')
    )
    hybrid_model.fit({k: models[k] for k in models.keys() if k in best_combination})
    hybrid_predictions = hybrid_model.predict()

    # calculating ranking score
    hybrid_accuracy = evaluator.score_function(
        "ranking",
        hybrid_predictions,
        data["purchases"][pd.notnull(train)],
        data["purchases"][pd.notnull(test)],
        data['black_list'],
        **hybrid_config.get("score_kwargs", {})
    )

    # updating hybrid model into models dictionary object
    models['hybrid_model'] = {
        'model': hybrid_model,
        'cv_mean_score': best_accuracy,
        'accuracy': hybrid_accuracy,
        'combination': best_combination,
        'weights': best_weights,
        'predictions': hybrid_predictions,
        'params': {}
    }


def hybrid_output_for_client(model, purchases, mapping_dicts, black_list=None):
    if black_list is None:
        black_list = []

    null_items = purchases.apply(
        lambda x: [idx for idx, val in enumerate(x) if pd.isnull(val) and idx not in black_list], axis=1)

    output = pd.DataFrame([x for x in model.predictions['ranking']['buckets'].values])

    items_length = len(mapping_dicts['id_to_item'])
    for index in range(len(output)):
        output.iloc[index].loc[[x for x in range(items_length) if x not in null_items.iloc[index]]] = np.nan

    output.columns = output.columns.map(mapping_dicts['id_to_item'])
    output.index = output.index.map(mapping_dicts['id_to_user'])

    results = output.reset_index().rename(columns={'index': 'user'}).melt(
        id_vars=['user'], var_name='item', value_name='score')

    results = results[pd.notnull(results['score'])]
    # results.to_csv("./recommender_output.csv", index=False)  # TODO: remove
    return results
