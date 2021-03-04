import click
import os
from tabulate import tabulate
from mlapp.main import MLApp
from mlapp.mlapp_cli.common.services_options import add_services_options
from mlapp.mlapp_cli.common.cli_utilities import create_directory, validate_str, clean_spaces, \
    format_service_dictto_txt, write_to_file, check_for_service_uniqueness_name, get_env
from mlapp.mlapp_cli.cli_help import cli_services_help

# constants
EMPTY_INPUT = ""


@click.group("services")
def commands():
    """
    ML App Services Command

    """
    pass


@commands.command("add", help=cli_services_help.get('add', 'add service'))
@click.argument("service", required=True)
def add(service):
    # creates the env directory if not exists.
    create_directory(directory_name='env', include_init=False)

    # find env file
    try:
        env_filename = get_env()
    except Exception as e:
        click.secho("ERROR: Please run 'mlapp init' first or set environment file.", fg='red')
        return

    env_full_path = os.path.join(os.getcwd(), env_filename)
    credentials = {}
    if os.path.exists(env_full_path):
        service_keys = add_services_options.get(service, False)
        if service_keys:

            # get service name from user
            while True:
                service_name = validate_str(input("Please name your service (to access the service in the code): "))
                is_unique = check_for_service_uniqueness_name(service_name, env_filename)
                if clean_spaces(service_name) != '' and is_unique:
                    break
                else:
                    click.secho("ERROR: Service name is required and must be unique, please try again.", fg='red')

            for key in service_keys:
                while True:
                    try:
                        body = service_keys[key]
                        new_key = service_name + '_' + key
                        if isinstance(body, str):
                            credentials[new_key] = body
                        else:
                            display_name = body.get('display_name', key)
                            short_description = body.get('short_description', '')

                            # create message to display on the terminal.
                            message = display_name + " (" + short_description + "): " if short_description != '' else key + ": "

                            # get user input
                            user_input = clean_spaces(input(message))

                            # check for default and required values
                            default_value = body.get('default', None)
                            is_required = body.get('required', False)

                            if default_value is not None and user_input == EMPTY_INPUT:
                                credentials[new_key] = default_value
                            elif user_input == EMPTY_INPUT and is_required:
                                raise Exception("'" + display_name + "' is required, please enter a valid value.")
                            else:

                                # validates user input
                                validations_methods = body.get('validations', [])
                                ans = True
                                for validate in validations_methods:
                                    ans &= validate(user_input)
                                    if not ans:
                                        raise Exception(body.get('error_msg', ''))

                                # transforms user input
                                transformations_methods = body.get('transformations', [])
                                for trans in transformations_methods:
                                    user_input = trans(user_input)

                                body_values = body.get('values', False)
                                if body_values:
                                    body_values_keys = body['values'].keys()
                                    if user_input in body_values_keys:
                                        credentials[new_key] = body['values'][user_input]
                                    else:
                                        raise Exception(body.get('error_msg', 'Oops something bad happened.'))
                                else:
                                    credentials[new_key] = user_input
                        break
                    except Exception as e:
                        if e == "":
                            click.secho("Invalid value please try again.", fg='red')
                        else:
                            click.secho(str(e), fg='red')

            comment = "\n# " + service_name.lower() + " " + service.lower() + " service\n"
            write_to_file(env_full_path, content=comment + format_service_dictto_txt(credentials))
            click.secho(
                "\nSuccess: " + service.capitalize() + " service was added to your project under the name " + service_name.upper() + ".\n",
                fg='green')
        else:
            click.secho("Service not found.", fg='red')
    else:
        # todo create a default env file?.
        click.secho('Env file not found, please create it before!', fg='red')


@commands.command("delete", help=cli_services_help.get('delete', 'delete service'))
@click.argument("service", required=True)
def delete(service):
    try:

        # find env file
        try:
            env_filename = get_env()
        except Exception as e:
            click.secho("ERROR: Please run 'mlapp init' first or set environment file.", fg='red')
            return

        env_full_path = os.path.join(os.getcwd(), env_filename)
        tmp_filename = "tmp_env_file_to_be_renamed.env"
        full_tmp_file_path = os.path.join(os.getcwd(), os.path.join('env', tmp_filename))
        is_service_name_to_del_found = False

        if os.path.exists(env_full_path):

            is_unique = check_for_service_uniqueness_name(service, env_filename)
            if is_unique:
                click.secho("ERROR: service not found.\nHint: type 'mlapp services' to see all available services.",
                            fg='red')
            else:
                with open(env_full_path) as f:
                    lines = f.readlines()

                    # creates new file with temporary name
                    with open(full_tmp_file_path, "w") as output:
                        for line in lines:
                            line_striped = line.strip()

                            if len(line_striped) == 0:
                                if is_service_name_to_del_found:
                                    continue
                                else:
                                    output.write(line)
                                    is_service_name_to_del_found = False
                                    continue

                            line_split = line_striped.split('_')

                            if len(line_split) > 0 and line_striped[0] != '#':
                                current_service_name = line_split[0]
                                # check if we should keep current service
                                if current_service_name.lower() != service.lower():
                                    output.write(line)
                                else:
                                    is_service_name_to_del_found = True
                            else:
                                if line_striped != '\n' and line_striped[0] == '#' and service.lower() in line_striped.lower():
                                    continue
                                else:
                                    output.write(line)

                # deletes original file
                os.remove(env_full_path)

                # renames tmp file to original file
                os.rename(full_tmp_file_path, env_full_path)

                click.secho(
                    "\nSuccess: " + service.upper() + " service was deleted from your project.\n", fg='green')

        else:
            click.secho('ERROR: environment file not found, please create it before!', fg='red')
    except Exception as e:
        raise e


@commands.command("show", help=cli_services_help.get('show', 'show services'))
def show():
    # find env file
    try:
        env_filename = get_env()
    except Exception as e:
        click.secho("ERROR: Please run 'mlapp init' first or set environment file.", fg='red')
        return

    env_full_path = os.path.join(os.getcwd(), env_filename)

    if os.path.exists(env_full_path):
        services = set()
        with open(env_full_path) as f:
            lines = f.readlines()
            for line in lines:
                line_striped = line.rstrip()
                line_split = line_striped.split('_')
                if len(line_split) > 1:
                    service_name = line_split[0].lstrip()
                    service_type = '_'.join(line_split[1:])
                    if service_name != '' and service_name[0] != '#' and MLApp.MLAPP_SERVICE_TYPE[1:] in service_type:
                        service_type = service_type.split('=')[1]
                        services.add(tuple([service_name, service_type]))

        if len(services):
            click.echo(tabulate(list(services), headers=['Service Name', 'Service Type']))
            click.echo()
        else:
            click.echo("No services registered")
    else:
        click.secho('Env file not found, please create it before!', fg='red')


@commands.command("show-types", help=cli_services_help.get('show_types', 'show services types'))
def show_types():
    services_keys = add_services_options.keys()
    click.echo('Available services types:')
    for key in services_keys:
        click.echo('[*] ' + key)
