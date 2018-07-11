#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import yaml

def load_configuration_file(path):
    with open(path) as f:
        data = yaml.safe_load(f)

    return data

def save_configuration_file(path, data):

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)