import sagemaker
from sagemaker import get_execution_role
from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner
from mlapp.integrations.sm.scripts.utils import create_metrics_definitions


def convert_hyperparameters_ranges(input_dict):
    # - example: {'MLAPP_reg_alpha': 'IntegerParameter(1, 3)'}
    for k, v in input_dict.items():
        input_dict[k] = eval(v)
    return input_dict


def run_tuning_job(base_job_name, image_name, config, metrics, objective_metric_name, hyperparameter_ranges,
                   instance_type, instance_count, max_jobs, max_parallel_jobs):

    metrics_definitions = create_metrics_definitions(metrics)

    est = sagemaker.estimator.Estimator(
        image_name,
        get_execution_role(),
        instance_count=instance_count,
        instance_type=instance_type,
        base_job_name=base_job_name,
        sagemaker_session=sagemaker.Session(),
        metric_definitions=metrics_definitions
    )
    est.set_hyperparameters(config=config)
    tuner = HyperparameterTuner(est,
                                objective_metric_name,
                                convert_hyperparameters_ranges(hyperparameter_ranges),
                                metric_definitions=metrics_definitions,
                                max_jobs=max_jobs,
                                max_parallel_jobs=max_parallel_jobs)
    tuner.fit()

