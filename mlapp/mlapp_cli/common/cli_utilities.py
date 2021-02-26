import os, json, socket, shutil, re


def check_port_in_use(port, ip='127.0.0.1'):
    '''
    This function check if port is in use under some if
    :param port: int
    :param ip: string
    :return: bool
    '''
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((ip, int(port)))
        s.shutdown(1)
        return True
    except:
        return False


def get_available_port(port='8081', ip='127.0.0.1'):
    '''
    This function return free port
    :param port: int
    :param ip: string
    :return: int
    '''
    while True:
        is_free = check_port_in_use(port, ip)
        if is_free:
            break
        else:
            port += 1
    return port


def str_to_camelcase(s):
    '''
    This function takes model_names, that contains underscore and transforms them to camelcase.
    :param s: string
    :return: string (camelcase)
    '''
    res = ''.join(x for x in s.title() if not x.isspace())
    res = res.replace("_", "")
    return res


def validate_str(s):
    '''
    This function validate a string by the rules:
    * turn each underscore to dash.
    * transform all string characters to uppercase.
    * remove spaces
    :param s: string
    :return: string
    '''
    new_s = s
    new_s = new_s.replace("_", "-")
    new_s = ''.join(x for x in new_s.title() if not x.isspace())
    new_s = new_s.upper()
    return new_s


def create_file(file_name, path='', permissions='w+', content=''''''):
    try:

        # concat the path and filename.
        full_file_path = os.path.join(path, file_name)
        full_file_path = os.path.join(os.getcwd(), full_file_path)

        # create file according to parameters.
        f = open(full_file_path, permissions)

        # write content in file.
        if content != '''''':
            f.write(content)

        # close file.
        f.close()

    except Exception as e:
        raise e


def write_to_file(file_name, path='', content=''''''):
    try:

        # concat the path and filename.
        full_file_path = os.path.join(path, file_name)
        full_file_path = os.path.join(os.getcwd(), full_file_path)

        # write content into file.
        with open(full_file_path, "a") as f:
            f.write(content)

    except Exception as e:
        raise e


def create_directory(directory_name, path='', include_init=True):
    try:

        # concat the path and directory name.
        full_directory_path = os.path.join(path, directory_name)
        full_directory_path = os.path.join(os.getcwd(), full_directory_path)

        # checks if directory already exists, if not then creating it.
        if not os.path.exists(full_directory_path):
            os.makedirs(full_directory_path)

        if include_init is True:
            create_file(file_name='__init__.py', path=full_directory_path)

    except Exception as e:
        raise e


def set_env(env_filename, env_path='env'):
    try:
        with open(os.path.join(os.getcwd(), 'config.py'), 'r') as f:
            config_content = f.read()
            exec(config_content)
            d = eval("settings")

        if env_filename == '':
            d['env_file_path'] = env_filename
        else:
            d['env_file_path'] = os.path.join(env_path, env_filename)

        config_file_content = '''settings = ''' + json.dumps(d, indent=2)
        create_file(file_name='config.py', content=config_file_content)

    except Exception as e:
        raise e


def get_env():
    try:
        with open(os.path.join(os.getcwd(), 'config.py'), 'r') as f:
            config_content = f.read()
            exec(config_content)
            d = eval("settings")

        env_filename = d.get('env_file_path', None)

        if env_filename is not None:
            return env_filename
        else:
            raise Exception("Env file not found, please create it before!")

    except Exception as e:
        raise e


def format_service_dictto_txt(service_dict):
    '''

    :param service_dict:
    :return:
    '''
    text = ''''''
    for key, value in service_dict.items():
        str_val = str(value)
        if str_val != "":
            text += key + '''=''' + str_val + '''\n'''

    return text


def check_for_service_uniqueness_name(service_name, env_filename):
    '''
    checks is service_name not exists in services
    :param service_name: string
    :return: bool
    '''

    try:
        # find env file
        env_full_path = os.path.join(os.getcwd(), env_filename)

        services = {}
        with open(env_full_path) as f:
            lines = f.readlines()
            for line in lines:
                line_striped = line.strip()

                if len(line_striped) == 0:
                    continue

                line_split = line_striped.split('_')
                if len(line_split) > 0:
                    current_service_name = line_split[0]
                    if current_service_name != '' and current_service_name[0] != '#':
                        services[current_service_name.lower()] = True

        return not service_name.lower() in services.keys()

    except Exception as e:
        raise e


def to_lower(s):
    return s.lower()


def to_upper(s):
    return s.upper()


def clean_spaces(s):
    return s.replace(" ", "")


def is_int(s):
    try:
        int(s)
        return True
    except:
        return False


def is_positive(s):
    try:
        return int(s) > 0
    except:
        return False


def to_int(s):
    return int(s)


def list_one_check(s):
    l = s.split(',')
    l = list(filter(lambda x: x != '', l))
    res = ''
    for x in l:
        res += x + ','
    return res[:-1]


def copy_files(src, dest, except_list=None, include_list=None):
    if except_list is None:
        except_list = []
    try:
        if not os.path.exists(src) or not os.path.exists(dest):
            raise Exception("src folder and destination folder must be exists")

        src_files = os.listdir(src)

        if include_list is not None:
            src_files = [file for file in src_files if file in include_list]

        for file_name in src_files:
            if file_name in except_list:
                continue
            else:
                full_filename = os.path.join(src, file_name)
                if os.path.isfile(full_filename):
                    shutil.copy(full_filename, dest)
                elif os.path.isdir(full_filename):
                    copy_files(os.path.join(src, file_name), os.path.join(dest, file_name), except_list)

    except Exception as e:
        raise e


def check_model_name(s):
    underscore = '_'
    dash = '-'

    if underscore in s and dash in s:
        return [underscore, dash]
    elif underscore in s:
        return [underscore]
    elif dash in s:
        return [dash]
    else:
        raise Exception('Name can include only letters, number, underscore and dash character.')


def str_capitalize(s):
    try:
        words = re.split(r"[, \-!#$%^&*@?_:]+", s)
        words = map(lambda x: x.capitalize(), words)
        res = "".join(words)
        return res
    except Exception as e:
        raise e
