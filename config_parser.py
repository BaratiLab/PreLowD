import argparse
import yaml
from collections import OrderedDict


def load_config(yaml_file_path: str) -> dict:
    """
    Load configuration from a YAML file
    arguments:
        yaml_file_path: str, path to the YAML file
    returns:
        config: dict, configuration information
    """
    with open(yaml_file_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    config_dict = OrderedDict(config_dict)
    for arg, params in config_dict.items():
        if isinstance(params, dict):
            assert 'default' in params and 'type' in params, f'Format not correct for {arg}. Should have type and default!'
            params['type'] = eval(params.get('type', 'str'))
            if not isinstance(params['default'], str): continue
            params['default'] = params['type'](params['default'])
    return dict(config_dict)


def save_config(config_dict: dict, filename: str, as_lists=False) -> None:
    """
    Save configuration to a YAML file
    arguments:
        config: dict, configuration to save
        filename: str, path to the YAML file
    """
    to_save_config_dict = {arg: params['default'] if isinstance(params, dict) else params for arg, params in config_dict.items()}
    if as_lists:
        to_save_config_dict = {arg: [params] for arg, params in to_save_config_dict.items()}

    with open(filename, 'w') as file:
        yaml.dump(to_save_config_dict, file, sort_keys=False)


def parse_args(base_config_path):
    """
    Parses arguments from command line
    First, a default configuration is loaded from default_config.yaml file.
    This file contains all the information about the arguments that can be parsed.
    The loaded configuration is used to define the arguments in the parser.
    The user has the option to load a configuration from a YAML file using the --from_yaml argument.

    The value for each argument is first defined from the default configuration.
    Then, it is overwritten by the yaml file specified by the --from_yaml argument.
    Finally, it is overwritten by the user-defined arguments.
    """
    base_config_dict = load_config(base_config_path)

    parser = argparse.ArgumentParser(description='Testing using yaml for argparse')

    # An optional argument to define the name of the experiment. Can be most useful
    # for testing and debugging. Later, an automated name can be generated based on the
    # configuration. It is most reasonable to define the name from the non-default arguments.
    parser.add_argument('--name', help='Name of the experiment, may be used for saving results')

    # The option to load the config file from a YAML file defined by the user:
    parser.add_argument('--default_config', default='', help='yaml with default configuration. Use this for data loading config')
    # This file does not need to include all the arguments.

    # These ones will be loaded later.
    parser.add_argument('--model_configs', default='', help='yaml with experiment configurations, containing a non-empty list for each argument')
    parser.add_argument('--transfer_configs', default='', help='yaml with transfer configurations, containing a non-empty list for each argument')

    for arg, params in base_config_dict.items():

        # Popping 'default' because we do not define default values for the parser arguments
        # This is becuase later, we check if they are None, meaning the user did not pass any value
        # If a default is specified for the parser, we cannot tell  whether the user passed the value or not
        default = params.pop('default')
        parser.add_argument(f'--{arg}', **params)
        
        # Adding it back to default_config_dict for later
        params['default'] = default

    parsed_args = parser.parse_args()
    if parsed_args.name:
        assert not parsed_args.name.startswith('_'), 'Name cannot start with _'


    # Starting with the default values from base_config_dict
    config_dict = {arg: base_config_dict[arg]['default'] for arg in base_config_dict}

    # If the user specifies a yaml file to load the configurations from.
    input_config_dict = load_config(parsed_args.default_config) if parsed_args.default_config else {}
    
    # We define another config_dict (to later save it maybe) for better readability
    # This will contain only the arguments that are different from the default values

    for arg in config_dict:

        # First, we overwrite the default values from the input yaml file
        if arg in input_config_dict:
            input_param = input_config_dict[arg]

            if isinstance(input_param, dict):
                if 'default' not in input_param: 
                    continue
                input_param = input_param['default']

            if isinstance(input_param, str):
                input_param = base_config_dict[arg]['type'](input_param)
            
            config_dict[arg] = input_param

        # Next, we check what the user passed for the argument:
        parsed_value = getattr(parsed_args, arg)

        # If the user did not pass anything, it will be None.
        # So we manually assign it from the either the default or the loaded yaml file:
        if parsed_value is None:
            setattr(parsed_args, arg, config_dict[arg])

        # If the user passed a value, we overwrite the config_dict with the user_defined value
        else:
            config_dict[arg] = parsed_value
        
    config_dict = dict(config_dict)

    return parsed_args, config_dict


if __name__ == '__main__':

    # use this for testing and creating your template to fill in for model or transfer configs

    args, config_dict = parse_args('configs_FFNO/_base.yaml') 

    save_config(config_dict, 'configs_FFNO/_base_template.yaml')
    save_config(config_dict, 'configs_FFNO/_base_template_lists.yaml', as_lists=True)

    # print args:
    for arg, value in vars(args).items():
        print(f'{arg}: {value}')
