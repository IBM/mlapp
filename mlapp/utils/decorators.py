AVAILABLE_STAGES = {}
BASE_CLASS_NAME = ''
MANAGER_TYPES = {
    'data_manager': 'DataManager',
    'model_manager': 'ModelManager'
}


class pipeline:
    def __init__(self, fn):
        self.fn = fn

    def __set_name__(self, owner, name):
        asset_name = owner.__name__

        manager_type = None
        for manager_type_key in MANAGER_TYPES:
            if MANAGER_TYPES[manager_type_key] in asset_name:
                manager_type = manager_type_key
        if manager_type is None:
            raise Exception("Wrong class name or placement of decorator! ('{}')".format(asset_name))

        asset_name = asset_name.replace('DataManager', '').replace('ModelManager', '')

        if asset_name not in AVAILABLE_STAGES:
            AVAILABLE_STAGES[asset_name] = {}

        if name in AVAILABLE_STAGES[asset_name]:
            raise Exception("Duplicate stage name '{}' for pipelines found in asset '{}'"
                            .format(asset_name, name))

        AVAILABLE_STAGES[asset_name][name] = {
            'function': self.fn,
            'manager': manager_type
        }

        return self.fn

