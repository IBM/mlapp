from azureml.core import ComputeTarget
from azureml.core.compute import AmlCompute


def get_or_create_compute_target(workspace, compute_name, vm_size='STANDARD_D2_V2', min_nodes=0, max_nodes=4,
                                 idle_sec=120):
    if compute_name in workspace.compute_targets:
        compute_target = workspace.compute_targets[compute_name]
        if compute_target and type(compute_target) is AmlCompute:
            print('Found compute target: ' + compute_name)
    else:
        print('Creating a new compute target...')
        provisioning_config = AmlCompute.provisioning_configuration(vm_size=vm_size,  # STANDARD_NC6 is GPU-enabled
                                                                    min_nodes=min_nodes,
                                                                    max_nodes=max_nodes,
                                                                    idle_seconds_before_scaledown=idle_sec)
        # create the compute target
        compute_target = ComputeTarget.create(
            workspace, compute_name, provisioning_config)

        # Can poll for a minimum number of nodes and for a specific timeout.
        # If no min node count is provided it will use the scale settings for the cluster
        compute_target.wait_for_completion(show_output=True)

        # For a more detailed view of current cluster status, use the 'status' property
        print(compute_target.status.serialize())

    return compute_target
