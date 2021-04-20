import os
import traceback
import pandas as pd
from config import settings
from mlapp import MLApp
from mlapp.integrations.aml.utils.deploy import preprocess_deployment, get_predictions_path, insert_model_id
from mlapp.integrations.aml.utils.run_class import load_config_from_string


def init():
    global run_id
    model_path = os.path.join(os.getcwd(), os.getenv('AZUREML_MODEL_DIR'))
    os.chdir('<LOCAL_ROOT_DIRECTORY_NAME>')
    output_dir = os.path.join(os.getcwd(), 'output')
    os.makedirs(output_dir, exist_ok=True)
    run_id = preprocess_deployment(model_path)


def run(raw_config):
    try:
        # loading config from string into dict object
        config = load_config_from_string(raw_config)

        # inserting deployed model id to config
        insert_model_id(config, run_id)

        # running config
        mlapp = MLApp(settings)
        _, ids, outputs = mlapp.run_flow_from_config(config)

        # loading predictions and returning it
        file_path = get_predictions_path(ids)
        predictions_df = pd.read_csv(file_path)
        return predictions_df.to_json(orient='records')

    except Exception as e:
        # printing error
        traceback.print_exc()
        return str(e)
