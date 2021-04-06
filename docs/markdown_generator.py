import os
import re

# This script will go over the modules listed in libraries and generates an md file with all docstrings for each one of
# modules using the third party library - pydocmd.
# It then does some editing on the file so it shows nicely in github's wiki.
# notice that pydoc will only list the outermost definitions, so docstrings from functions nested under a class will NOT
# be documented in the markdown.
#
# this requires the following setup:
# pip install pydoc-markdown mkdocs mkdocs-material
# pip install pygments (required for codehilite)
#
# docstrings should be in a a reStructuredTxt format (pycharm's default).
# This script expects Code examples (within the docstring to be written between @@@ tags @@@@
# i.e (@@@word@@@@= > ```word```).

# ------------------------------------ Generate markdown -----------------------------------
docs_folder = './'
libraries = [
    # api
    # {
    #     'src': 'mlapp.utils.automl ',
    #     'dest': 'api/utils.automl.md'
    # },
    # {
    #     'src': 'mlapp.utils.features.pandas ',
    #     'dest': 'api/utils.features.pandas.md'
    # },
    # {
    #     'src': 'mlapp.utils.features.spark ',
    #     'dest': 'api/utils.features.spark.md'
    # },
    # {
    #     'src': 'mlapp.utils.metrics.pandas ',
    #     'dest': 'api/utils.metrics.pandas.md'
    # },
    # {
    #     'src': 'mlapp.utils.metrics.spark ',
    #     'dest': 'api/utils.metrics.spark.md'
    # },
    # managers
    # {
    #     'src': 'mlapp.managers.user_managers',
    #     'dest': 'api/managers.md',
    # },
    # # handlers
    # {
    #     'src': 'mlapp.handlers.databases.database_interface',
    #     'dest': 'api/handlers.database.md',
    #     'exclude': ['update_job_running']
    # },
    # {
    #     'src': 'mlapp.handlers.file_storages.file_storage_interface',
    #     'dest': 'api/handlers.file_storage.md'
    # },
    # {
    #     'src': 'mlapp.handlers.message_queues.message_queue_interface',
    #     'dest': 'api/handlers.message_queue.md'
    # },
    # {
    #     'src': 'mlapp.handlers.spark.spark_interface',
    #     'dest': 'api/handlers.spark.md'
    # }
]


def create_documentation(file_name, create_main_table_of_contents=False, mkdocs_format=False, exclusion_dict=None):
    if not exclusion_dict:
        exclusion_dict = {}

    with open(file_name, 'r') as f:
        line = True
        temp = ''
        module_exclude_flag = False
        function_exclude_flag = False
        param_section = False
        output = ''
        while line:
            line = f.readline()

            # exclude functions from the docs:
            if function_exclude_flag:
                if line.startswith('#') and line[3:-1] not in list(exclusion_dict.values())[0]:
                    function_exclude_flag = False
                else:
                    continue
            elif module_exclude_flag:
                if (line.startswith('# ')) and (line[:-1] not in list(exclusion_dict.keys())):
                    module_exclude_flag = False  # made it to a new module in the file. turn flag off.
                elif line[3:-1] in list(exclusion_dict.values())[0]:  # this function needs to be excluded
                    function_exclude_flag = True
                    continue
            elif exclusion_dict:  # there are functions to exclude in one of these files.
                if line[2:-1] in list(exclusion_dict.keys()):  # this module has functions to exclude!
                    module_exclude_flag = True

            # if line starts with a lot of spaces, turn them to &ensp; to preseve indentation:
            if re.search(r'[^\s]', line):
                if re.search(r'[^\s]', line).start() > 5:
                    numSpaces = re.search(r'[^\s]', line).start()
                    line = '&ensp;'*(numSpaces-3) + line[numSpaces:]

            # turn the param section into a beautiful html table:
            if re.search(r'(:{1}\w*)\s*(\**\w*:{1})', line):  # match ":param: x"
                if not re.search(r'^:return:', line):
                    temp += line
                    param_section = True
                else:
                    temp = re.sub(r'(:{1}\w*)\s*(\**\w*:{1})',  r'<br/><b><i>\2</b></i>', temp)
                    temp = temp[5:] # remove the leading <br/> I added..
                    temp2 = re.sub(r'(:return:)', r'', line)
                    line = '<table style="width:100%"><tr><td valign="top"><b><i>Parameters:</b></i></td>' \
                           '<td valign="top">'+temp+'</td></tr><tr><td valign="top"><b><i>Returns:</b></i>' \
                                                    '</td><td valign="top">'+temp2+'</td></tr></table>'
                    output = output + '\n<br/>' + line + '\n<br/>\n'
                    temp = ''
                    param_section = False
            else:
                if param_section:
                    temp += line
                else:
                    output = output + line

        # convert examples to valid markdown.
        if mkdocs_format:
            output = re.sub('@{4}', '', output)
            output = re.sub('@{3}', '\n!!! note "Reference:"', output)
        else:
            output = re.sub('@{3,4}', '\n ```', output)

        return output


def run(mkdocs=False):
    # os.chdir('..')
    for lib in libraries:
        script = lib['src']
        md = docs_folder + lib['dest']
        command = 'pydoc-markdown -m' + ''.join(script) + ' > ' + md
        os.system(command)
        exclusion_dict = {lib['src'][:-2]: lib['exclude']} if 'exclude' in lib else {}
        output = create_documentation(md, False, mkdocs, exclusion_dict)

        # Save changes to file
        with open(md, 'w') as f:
            f.write(output)


if __name__ == '__main__':
    run(mkdocs=False)

