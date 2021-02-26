from mlapp.mlapp_cli.common.cli_utilities import create_directory, str_to_camelcase, str_capitalize
from mlapp.mlapp_cli.common.files import data_manager_file, model_manager_file, train_config_file, \
    forecast_config_file, train_config_file_with_flow, forecast_config_file_with_flow
import os, click, shutil, uuid, re
import numpy as np
from mlapp.mlapp_cli.common.model_rename_dictionary import rename_dictionary
from mlapp.mlapp_cli.common.cli_utilities import create_file, copy_files
from mlapp.mlapp_cli.cli_help import cli_assets_help


@click.group("assets")
def commands():
    """
    ML App Assets Command

    Use it to install or create your Assets.
    type --help on each assets commands to get more information.

    """
    pass


@commands.command("show", help=cli_assets_help['show'])
def show():
    assets_folder_path = os.path.join(os.getcwd(), 'assets')
    if os.path.exists(assets_folder_path):
        your_assets = list(filter(lambda asset_name: os.path.isdir(os.path.join(assets_folder_path, asset_name)),
                                  os.listdir(assets_folder_path)))
        if len(your_assets) == 0:
            click.secho(
                "Your project not contains any assets at the moment.\nHint: use mlapp assets 'mlapp boilerplates install BOILERPLATE_NAME' or 'mlapp assets create ASSET_NAME' command for generating your first asset." ,fg='red')
        else:
            click.echo('Your project assets:')
            for asset in your_assets:
                if os.path.isdir(os.path.join(assets_folder_path, asset)) and asset != '__pycache__':
                    click.echo('[*] ' + asset)
    else:
        click.secho(
            "ERROR: your project is not empty.\nHint: you can use 'mlapp init --force' option to force init (caution: force may override exsiting files).", fg='red')


@commands.command("create", help=cli_assets_help['create'])
@click.argument("name", required=True)
@click.option("-f", "--force", is_flag=True, default=False,
              help="Flag force will override existing asset_name file.")
@click.option("-w", "--with-flow", is_flag=True, default=False,
              help="Creates asset train and forecast configs with flow settings.")
def create(name, force, with_flow):
    # model name validations and transformations.
    asset_name = name.lower()
    asset_name = asset_name.replace(" ", "_")
    asset_name = asset_name.replace("-", "_")

    # creates the assets directory if not exists.
    create_directory(directory_name='assets')

    full_directory_path = os.path.join(os.getcwd(), os.path.join('assets', asset_name))
    if os.path.exists(full_directory_path) and not force:
        click.secho(
            'Error: ' + asset_name + ' asset already exists.\nHint: please select a unique name to your asset or use --force option to override asset folder.', fg='red')
        return

    # create the necessary folders for the new asset, `asset_name` directory and asset configs directory.
    create_directory(directory_name=asset_name, path='assets')
    create_directory(directory_name='configs', path=os.path.join('assets', asset_name), include_init=False)

    model_name_capitalized = str_to_camelcase(asset_name
                                              )
    # generates model manager file template
    model_manager_file_content = model_manager_file.format(model_name_capitalized)

    # generates data manager file template
    data_manager_file_content = data_manager_file.format(model_name_capitalized)

    if with_flow:
        # generates train config file template with flow
        train_config_file_content = train_config_file_with_flow.replace("<ASSET_NAME>", asset_name)

        # generates forecast config file template with flow
        forecast_config_file_content = forecast_config_file_with_flow.replace("<ASSET_NAME>", asset_name)
    else:
        # generates train config file template
        train_config_file_content = train_config_file.replace("<ASSET_NAME>", asset_name)

        # generates forecast config file template
        forecast_config_file_content = forecast_config_file.replace("<ASSET_NAME>", asset_name)

    # create all managers templates
    create_file(file_name=asset_name + '_model_manager.py', path=os.path.join('assets', asset_name), permissions='w+',
                content=model_manager_file_content)
    create_file(file_name=asset_name + '_data_manager.py', path=os.path.join('assets', asset_name), permissions='w+',
                content=data_manager_file_content)

    # create all configs templates
    create_file(file_name=asset_name + '_train_config.json',
                path=os.path.join(os.path.join('assets/', asset_name), 'configs'),
                permissions='w+',
                content=train_config_file_content)
    create_file(file_name=asset_name + '_forecast_config.json',
                path=os.path.join(os.path.join('assets/', asset_name), 'configs'),
                permissions='w+',
                content=forecast_config_file_content)


@commands.command("rename", help=cli_assets_help['rename'])
@click.argument("name", required=True)
@click.argument("new_name", required=True)
@click.option('-d', '--delete', is_flag=True, default=False,
              help="Use it to delete previous asset directory on renaming.")
def rename(name, new_name, dir_name=None, show_success_msg=True, delete=False, force=False):
    if name is None or new_name is None:
        click.secho(
            "ERROR: asset old name and new name are required.\nHint: 'mlapp assets rename OLD_NAME NEW_NAME' is the right command format.", fg='red')
        return True

    if name == new_name:
        click.secho("ERROR: asset old name and new name are equal. please use a different name.", fg='red')
        return True

    except_list = []
    if dir_name is None:
        old_name_path = os.path.join(os.getcwd(), 'assets', name)
    else:
        old_name_path = os.path.join(os.getcwd(), 'assets', dir_name)
    new_name_path = os.path.join(os.getcwd(), 'assets', new_name)

    if not os.path.exists(old_name_path):
        click.secho(
            f"ERROR: asset \"{name}\" is not exists.\nHint: use 'mlapp assets show' command to view all available assets.", fg='red')
    elif os.path.exists(new_name_path) and not force:
        click.secho(
            f"ERROR: asset \"{new_name}\" is already exists.", fg='red')
    else:

        # creates the new model directory
        os.mkdir(new_name_path)

        # base renaming
        if name in rename_dictionary.keys():
            model_dict = rename_dictionary.get(name, {})
            replace_files = model_dict.keys()
        else:
            model_dict = rename_dictionary.get('base', {})
            replace_files = model_dict.keys()

        # rename model name file by file.
        for rf in replace_files:
            try:
                file_additional_path = model_dict[rf].get('inner_path', '')

                except_list.append(name + rf)
                # files path
                if dir_name is None:
                    full_old_filename = os.path.join(os.getcwd(), "assets", name, file_additional_path,
                                                     name + rf)
                else:
                    full_old_filename = os.path.join(os.getcwd(), "assets", dir_name, file_additional_path,
                                                     name + rf)
                full_new_filename = os.path.join(os.getcwd(), "assets", new_name, file_additional_path, new_name + rf)

                # checks if needs to create sub directories.
                if not os.path.exists(os.path.join(os.getcwd(), "assets", new_name, file_additional_path)):
                    os.mkdir(os.path.join(os.getcwd(), "assets", new_name, file_additional_path))

                words_to_replace = model_dict[rf].get('words', [])

                # read file content
                file_content = ''''''
                with open(full_old_filename, 'r') as file:
                    lines = file.readlines()
                    for line in lines:
                        # line_striped = line.strip()

                        # replace new file path
                        for wr_d in words_to_replace:
                            word = wr_d.get('word', None)
                            word_type = wr_d.get('word_type', 'append-left')
                            word_format = wr_d.get('word_format', 'str.lower')
                            word_pattern = wr_d.get('word_pattern', None)
                            if word is not None:
                                # check word format and type
                                word_format_func = eval(word_format)
                                old_name_formatted = word_format_func(name)
                                new_name_formatted = word_format_func(new_name)

                                if word_type == 'append-left':
                                    if word_pattern is not None and isinstance(word_pattern, str):
                                        word_pattern = old_name_formatted + word_pattern
                                    elif word_pattern is not None and (
                                            isinstance(word_pattern, list) or isinstance(word_pattern, np.array)):
                                        wp_res = ''
                                        for wp in word_pattern:
                                            wp_res += old_name_formatted + wp + '|'
                                        word_pattern = wp_res[:-1]

                                    old_name_formatted = old_name_formatted + word
                                    new_name_formatted = new_name_formatted + word
                                else:
                                    # 'append-right'

                                    # check for patterns type
                                    if word_pattern is not None and isinstance(word_pattern, str):
                                        word_pattern += old_name_formatted
                                    elif word_pattern is not None and (
                                            isinstance(word_pattern, list) or isinstance(word_pattern, np.array)):
                                        wp_res = ''
                                        for wp in word_pattern:
                                            wp_res += wp + old_name_formatted + '|'
                                        word_pattern = wp_res[:-1]

                                    old_name_formatted = word + old_name_formatted
                                    new_name_formatted = word + new_name_formatted

                                # checking and replacing model names or using regex
                                if word_pattern is not None:
                                    line = re.sub(word_pattern, new_name_formatted, line)
                                elif old_name_formatted in line:
                                    line = line.replace(old_name_formatted, new_name_formatted)

                        file_content += line

                # creates the new file with the new name.
                create_file(file_name=full_new_filename, content=file_content)

            except Exception as e:
                shutil.rmtree(new_name_path)
                click.echo(e)
                # click.echo('ERROR: Oops something bad happened.')
                return True

        # copy the rest of the files
        copy_files(old_name_path, new_name_path, except_list)

        if delete:
            shutil.rmtree(old_name_path)

        if show_success_msg:
            click.secho('Success: asset "' + name + '" was renamed to "' + new_name + '".', fg='green')