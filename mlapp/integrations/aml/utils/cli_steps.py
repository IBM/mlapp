from mlapp.mlapp_cli.common.cli_utilities import to_lower, is_int, to_int, to_upper, is_positive, clean_spaces

steps = {
    'number_of_steps': {
        "display_name": "Choose how many pipeline steps to deploy",
        "short_description": 'Press enter to set number of steps to 2',
        "validations": [is_int, is_positive],
        "transformations": [to_int],
        "error_msg": "Number of steps should b a valid positive integer.",
        "default": 2,
        "required": True,
        "nested_dependency": {
            "name": {
                "display_name": "compute target",
                "short_description": 'Press enter compute-target name',
                "transformations": [to_lower],
                "error_msg": "compute_target should be exists in your Workspace.",
                "required": True
            },
            "type": {
                "display_name": "Is it databricks machine?",
                "short_description": 'Y/N, default No',
                "transformations": [to_lower, clean_spaces],
                "values": {
                    'y': 'databricks',
                    'n': 'aml_compute_target',
                    'yes': 'databricks',
                    'no': 'aml_compute_target'
                },
                "error_msg": "Possible values should be 'y', 'n', 'yes' or 'no'.",
                "default": 'aml_compute_target',
                "required": True
            },
            "set_advanced": {
                "display_name": "Do you want to set compute-target advanced settings?",
                "short_description": 'Y/N, default no will use compute-target default settings',
                "transformations": [to_lower, clean_spaces],
                "values": {
                    'y': True,
                    'n': False,
                    'yes': True,
                    'no': False
                },
                "error_msg": "Possible values should be 'y', 'n', 'yes' or 'no'.",
                "default": 'false',
                "required": True
            },
            "vm_size": {
                "display_name": "vm size",
                "short_description": 'Press enter to set vm size to \'STANDARD_D2_V2\'',
                "transformations": [to_upper],
                "error_msg": "Port should contain digits only.",
                "default": 'STANDARD_D2_V2',
                "required": False
            },
            "min_nodes": {
                "display_name": "min nodes",
                "short_description": 'Press enter to set minimum nodes to 0',
                "validations": [is_int, is_positive],
                "transformations": [to_int],
                "error_msg": "Min nodes should be a valid non negative integer.",
                "default": 0,
                "required": False
            },
            "max_nodes": {
                "display_name": "max nodes",
                "short_description": 'Press enter to set maximum nodes to 4',
                "validations": [is_int, is_positive],
                "transformations": [to_int],
                "error_msg": "Max nodes should be a valid non negative integer.",
                "default": 4,
                "required": False
            },
            "idle_seconds_before_scale_down": {
                "display_name": "idle seconds before scale down",
                "short_description": 'Press enter to idle seconds before scale down to 120',
                "validations": [is_int, is_positive],
                "transformations": [to_int],
                "error_msg": "idle seconds before scale down should be a valid non negative integer.",
                "default": 120,
                "required": False
            }
        }
    }
}
