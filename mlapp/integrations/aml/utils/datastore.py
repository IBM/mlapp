from azureml.core import Datastore


def get_datastore(workspace, name='workspaceblobstore'):
    return Datastore.get(workspace, datastore_name=name)


def register_blob_datastore(subscription_id, resource_group, workspace, datastore_name, container_name, account_name,
                       account_key, set_as_default=True):

    datastore = Datastore.register_azure_blob_container(
        workspace=workspace,
        datastore_name=datastore_name,
        grant_workspace_access=True,
        container_name=container_name,
        account_name=account_name,
        account_key=account_key,
        subscription_id=subscription_id,
        resource_group=resource_group
    )
    if set_as_default:
        datastore.set_as_default()

    return datastore
