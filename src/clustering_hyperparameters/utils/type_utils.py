type_map = { 'int': int,
             'float': float,
             'str': str,
             'bool': bool }


def get_type_from_str(typ):
    """ Get python data type from str

    Args:
        typ (str): Python data type in string

    Returns:
        [type]: Python data type described by the string object
    """
    return type_map[typ]
