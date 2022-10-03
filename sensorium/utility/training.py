import datetime

def read_config(config_yaml_file):
    """Read given yaml file and return dictionary with entries
    
    Requires ruamel.yaml to be installed """

    try:
        import ruamel.yaml as yaml   # install the package with: pip install ruamel.yaml
    except ImportError:
        raise Exception("Install missing package with 'pip install ruamel.yaml'")
        
    # TODO: add handling of file not found error
    yaml_config = yaml.YAML()
    with open(config_yaml_file, 'r') as file:
        config_dict = yaml_config.load(file)

    return config_dict

def print_t(message):
    """Print a message with a timestamp """
    print( '{}: {}'.format(datetime.datetime.now(), message) )
    
