#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
utilities 

"""


import os


def get_url(string):
    return string.partition(".png")[0] + ".png"


def convert_to_path(name):
    return os.path.normpath(name)


def create_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
    return


def absolute_file_paths(directory):
    path = os.path.abspath(directory)
    return [entry.path for entry in os.scandir(path) if entry.is_file()]
