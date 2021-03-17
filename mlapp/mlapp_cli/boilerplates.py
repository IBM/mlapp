import os, click, shutil, uuid, re
import numpy as np
from mlapp.utils.general import get_project_root
from mlapp.mlapp_cli.common.model_rename_dictionary import rename_dictionary
from mlapp.mlapp_cli.common.cli_utilities import create_file, copy_files, str_capitalize
from mlapp.mlapp_cli.cli_help import cli_boilerplates_help
try:
    from mlapp.mlapp_cli.cli_test_env import env_test
except ModuleNotFoundError:
    env_test = None


@click.group("boilerplates")
def commands():
    """
    MLApp Boilerplates Command

    Use it to install MLApp boilerplates.
    type --help on each boilerplates command to get more information.

    """
    pass


@commands.command("install", help=cli_boilerplates_help['install'])
@click.argument("name", required=True)
@click.option("-f", "--force", is_flag=True, default=False, help="Flag force will override existing asset_name file.")
@click.option('-r', '--new-name', default=None, help="Use it to rename an asset name on installation.")
def install(name, force, new_name):
    if env_test:
        available_assets = os.listdir(os.path.join(os.getcwd(), '..', 'assets'))
    else:
        available_assets = os.listdir(os.path.join(get_project_root(), 'assets'))

    if name is None:
        click.secho("Asset name is required, please try again.", fg='red')
    elif name in available_assets:
        src = os.path.join(get_project_root(), 'assets', name)
        dst = os.path.join(os.getcwd(), 'assets', name)
        try:
            if new_name is not None and isinstance(new_name, str):
                if os.path.exists(os.path.join(os.getcwd(), 'assets', new_name)) and not force:
                    click.secho('ERROR: asset "' + new_name + '" is already exists.', fg='red')
                    return
                token = str(uuid.uuid4())
                tmp_name = name + '_' + token
                dst = dst + '_' + token

            if force and new_name is None:
                shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copytree(src, dst)

            if new_name is not None and isinstance(new_name, str):
                res = __rename__(name=name, new_name=new_name, dir_name=tmp_name, show_success_msg=False, force=force)
                shutil.rmtree(dst)
                if res:
                    return

            # add data files
            data_dest = os.path.join(os.getcwd(), 'data')
            include_list = {
                'crash_course': ['glass.csv', 'glass_forecast.csv'],
                'classification': ['breast_cancer.csv'],
                'spark_classification': ['breast_cancer.csv'],
                'basic_regression': ['diabetes.csv'],
                'spark_regression': ['diabetes.csv']
            }
            if name in include_list:
                os.makedirs(data_dest, exist_ok=True)
                copy_files(os.path.join(get_project_root(), 'data'), data_dest, include_list=include_list[name])

            click.secho(
                'Success: asset was installed successfully under the name: "' + (
                    new_name if new_name is not None and isinstance(
                        new_name, str) else name) + '"', fg='green')

        except Exception as e:
            click.secho(
                f"ERROR: asset \"{name}\" already exists in your project. \nHint: you can use the -r flag followed by NAME to rename asset or -f flag to force override (caution: -f flag may override existing project).", fg='red')
    else:
        click.secho(
            f"ERROR: couldn't find the asset \"{name}\".\nHint: run 'mlapp boilerplates show' to list all available assets.", fg='red')


@commands.command("show", help=cli_boilerplates_help['show'])
def show():
    if env_test:
        available_boilerplates = os.listdir(os.path.join(os.getcwd(), '..', 'assets'))
    else:
        available_boilerplates = os.listdir(os.path.join(get_project_root(), 'assets'))
    click.echo("Available boilerplates:")
    for boilerplate in available_boilerplates:
        if not boilerplate[0] == '_':
            click.echo('[*] ' + boilerplate)


# todo: duplicated for internal use. find a better solution
def __rename__(name, new_name, dir_name=None, show_success_msg=True, delete=False, force=False):
    if name is None or new_name is None:
        click.secho(
            "ERROR: asset old name and new name are required.\nHint: 'mlapp assets rename OLD_NAME NEW_NAME' is the right command format.", fg='red')
        return True

    if name == new_name:
        click.secho("ERROR: mlapp boilerplate old name and your asset new name are equal. please use a different name or dont use rename option.", fg='red')
        return True

    except_list = ["__pycache__"]
    if dir_name is None:
        old_name_path = os.path.join(os.getcwd(), 'assets', name)
    else:
        old_name_path = os.path.join(os.getcwd(), 'assets', dir_name)
    new_name_path = os.path.join(os.getcwd(), 'assets', new_name)

    if not os.path.exists(old_name_path):
        click.secho(
            f"ERROR: asset \"{name}\" is not exists.\nHint: use 'mlapp assets show' command to view all available assets.", fg='red')
    elif os.path.exists(new_name_path) and not force:
        click.escho(
            f"ERROR: asset \"{new_name}\" is already exists.", fg='red')
    else:

        if os.path.exists(new_name_path) and not force:
            # shutil.rmtree(new_name_path)
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
                click.secho('ERROR: Oops something bad happened.', fg='red')
                return True

        # copy the rest of the files
        copy_files(old_name_path, new_name_path, except_list)

        if delete:
            shutil.rmtree(old_name_path)

        if show_success_msg:
            click.secho('Success: asset "' + name + '" was renamed to "' + new_name + '".', fg='green')
