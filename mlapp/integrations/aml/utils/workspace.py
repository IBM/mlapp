from azureml.core import Workspace
from azureml.core.authentication import InteractiveLoginAuthentication


def init_workspace(tenant_id, subscription_id, resource_group, workspace_name):
    if tenant_id is not None:
        interactive_auth = InteractiveLoginAuthentication(tenant_id=tenant_id)
        workspace = Workspace(
            subscription_id=subscription_id,
            resource_group=resource_group,
            workspace_name=workspace_name,
            auth=interactive_auth)
    else:
        workspace = Workspace(
            subscription_id=subscription_id,
            resource_group=resource_group,
            workspace_name=workspace_name)
    return workspace
