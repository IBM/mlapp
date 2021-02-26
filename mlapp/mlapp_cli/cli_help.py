cli_assets_help = {
    "create": \
'''Usage:

`mlapp assets create ASSET_NAME` - creates an asset named ASSET_NAME\n
`mlapp assets create ASSET_NAME -f` - forcing asset creation named ASSET_NAME (caution: can override existing asset)\n
''',
    "show": '''Show all your project assets.''',
    "rename": \
'''Usage:

`mlapp assets rename PREVIOUS_NAME NEW_NAME` - duplicates existing asset named PREVIOUS_NAME and renamed it NEW_NAME \n
`mlapp assets rename PREVIOUS_NAME NEW_NAME -d` - rename existing asset named PREVIOUS_NAME and renamed it NEW_NAME (deletes PREVIOUS_NAME asset) \n
'''

}

cli_boilerplates_help = {
    "install": \
'''Usage:

`mlapp boilerplates install BOILERPLATE_NAME` - install mlapp boilerplate named BOILERPLATE_NAME\n
`mlapp boilerplates install BOILERPLATE_NAME -f` - forcing boilerplate installation named BOILERPLATE_NAME (can override existing boilerplate)\n
`mlapp boilerplates install BOILERPLATE_NAME -r NEW_NAME` - install mlapp boilerplate named BOILERPLATE_NAME and rename it to NEW_NAME.
''',
    "show": '''Show all ML App's available boilerplates. ''',
    "rename": \
'''Usage:

`mlapp assets rename PREVIOUS_NAME NEW_NAME` - duplicates existing asset named PREVIOUS_NAME and renamed it NEW_NAME \n
`mlapp assets rename PREVIOUS_NAME NEW_NAME -d` - rename existing asset named PREVIOUS_NAME and renamed it NEW_NAME (deletes PREVIOUS_NAME asset) \n
'''

}

cli_services_help = {
    "add": \
'''Usage:

`mlapp services add SERVICE_NAME` - register a new service to ML App (hint: use `mlapp services show-types`
to see all ML App's available services).
''',
    "delete": \
'''Usage:

`mlapp services delete SERVICE_NAME` - deletes a registered service (hint: use `mlapp services show`
to see all your registered services).
''',
    "show": \
'''Usage:

`mlapp services show` - shows all your registered services.
''',
    "show_types": \
'''Usage:

`mlapp services show-types` - shows all ML App's available services.
''',

}

cli_environment_help = {
    "init": \
'''Usage:

`mlapp environment init` - creates an empty env file with default name '.env'.\n
`mlapp environment init NAME` - creates an empty env file named NAME.
''',
    "set": \
'''Usage:

`mlapp environment set NAME` - sets ML App to point on NAME environment file, (this command modifies config.py file).
''',
}

cli_mlcp_help = {
    "setup": \
'''Usgae:

`mlapp mlcp setup` - setup mlcp on your machine.
''',
    "start": \
'''Usgae:

`mlapp mlcp start` - starts mlcp docker container, open chrome browser with the address 'http://localhost:8081'.
''',
    "stop": \
'''Usgae:

`mlapp mlcp stop` - stops mlcp docker container.
''',
}