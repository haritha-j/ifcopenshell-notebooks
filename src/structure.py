# parse a file containing the structure fo a design file to extract system information and branch information

import json
import collections

#  aggrgegation relationships


# flatten info dictionary
def flatten(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        # print(type(v))
        if isinstance(v, list):
            for i, el in enumerate(v):
                new_key_2 = new_key + sep + str(i)
                if isinstance(el, collections.MutableMapping):
                    items.extend(flatten(el, new_key_2, sep=sep))
                else:
                    items.append(el)
        elif isinstance(v, collections.MutableMapping):
            # print(v)
            items.extend(flatten(v, new_key, sep=sep))
        else:
            print(type(v))
            items.append(v)
    return items


# retrieve elements belonging to each system
def get_systems(system_dict_file):
    f = open(system_dict_file)
    system_dict = json.load(f)
    root = list(system_dict.keys())[0]
    out_dict = {}

    for sys in system_dict[root]:
        sys_name = list(sys.keys())[0]

        out_dict[sys_name] = flatten(sys)
        print(sys_name, len(out_dict[sys_name]))
    return out_dict


# Topological (connection) relationships


# flatten info dictionary to branch level
def flatten_branch(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        # print(type(v))
        if isinstance(v, list):
            for i, el in enumerate(v):
                new_key_2 = new_key + sep + str(i)
                if isinstance(el, collections.MutableMapping):
                    items.extend(flatten_branch(el, new_key_2, sep=sep))
                else:
                    items.append((k, el))
        elif isinstance(v, collections.MutableMapping):
            # print(v)
            items.extend(flatten_branch(v, new_key, sep=sep))
        else:
            # print(type(v))
            items.append((k, v))
    return items


# get components of each branch
def get_branches(system_dict_file):
    f = open(system_dict_file)
    system_dict = json.load(f)
    root = list(system_dict.keys())[0]
    out_dict = {}

    for sys in system_dict[root]:
        sys_name = list(sys.keys())[0]

        items = flatten_branch(sys)
        # print(items)
        d = {}
        for branch, e in items:
            if not branch in d:
                d[branch] = []
            d[branch] += [e]
        out_dict[sys_name] = d

        # out_dict[sys_name] = flatten_branch(sys)
        print(sys_name, len(out_dict[sys_name]))
    return out_dict
