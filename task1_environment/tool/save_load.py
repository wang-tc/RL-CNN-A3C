# https://stackoverflow.com/questions/56403013/how-to-save-the-dictionary-that-have-tuple-keys
from collections import defaultdict


def save_q_table(table, file_name):
    with open(file_name, 'w+') as f:
        f.write(str(table))


def load_q_table(filename):
    with open(filename, 'r') as f:
        dict_string = f.readlines()[0]
    dictionary = eval('(' + dict_string[27:])
    return dictionary


def save_default_dict(table, file_name):
    table = dict(table)
    with open(file_name, 'w+') as f:
        f.write(str(table))


def load_default_dict(filename):
    with open(filename, 'r') as f:
        dict_string = f.readlines()[0]
    dictionary = eval(dict_string)
    return defaultdict(lambda: 1, dictionary)
