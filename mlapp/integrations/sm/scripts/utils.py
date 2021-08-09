def create_metrics_definitions(metrics):
    return [
        {'Name': metric, 'Regex': f'{metric}.*=\D*(.*?)$'} for metric in metrics
    ]
