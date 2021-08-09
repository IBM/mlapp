import sagemaker
from sagemaker import get_execution_role

from mlapp.integrations.sm.scripts.utils import create_metrics_definitions


def run_training_job(base_job_name, image_name, config, metrics, instance_type, instance_count):
    est = sagemaker.estimator.Estimator(
        image_name,
        get_execution_role(),
        instance_count=instance_count,
        instance_type=instance_type,
        base_job_name=base_job_name,
        sagemaker_session=sagemaker.Session(),
        metric_definitions=create_metrics_definitions(metrics)
    )

    est.set_hyperparameters(config=config)

    est.fit()
