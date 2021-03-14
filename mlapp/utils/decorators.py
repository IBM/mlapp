class pipeline:
    MANAGER_TYPES = {
        'data_manager': 'DataManager',
        'model_manager': 'ModelManager'
    }
    AVAILABLE_STAGES = {}

    def __init__(self, fn):
        print(">>>>>>>> PIPELINE INIT: ")
        print(fn)
        self.fn = fn

    def __set_name__(self, owner, name):
        print(">>>>>>>> PIPELINE SET NAME: ")
        print(name)
        asset_name = owner.__name__

        manager_type = None
        for manager_type_key in self.MANAGER_TYPES:
            if self.MANAGER_TYPES[manager_type_key] in asset_name:
                manager_type = manager_type_key
        if manager_type is None:
            raise Exception("Wrong class name or placement of decorator! ('{}')".format(asset_name))

        asset_name = asset_name.replace('DataManager', '').replace('ModelManager', '')

        if asset_name not in self.AVAILABLE_STAGES:
            self.AVAILABLE_STAGES[asset_name] = {}

        if name in self.AVAILABLE_STAGES[asset_name]:
            raise Exception("Duplicate stage name '{}' for pipelines found in asset '{}'"
                            .format(asset_name, name))

        self.AVAILABLE_STAGES[asset_name][name] = {
            'function': self.fn,
            'manager': manager_type
        }

        return self.fn
