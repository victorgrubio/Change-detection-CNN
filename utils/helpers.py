# -*- coding: utf-8 -*-
# @Author: Victor Garcia
# @Date:   2018-10-02 12:20:12
# @Last Modified by:   Victor Garcia
# @Last Modified time: 2018-10-02 13:06:02
from pathlib import Path

"""
This class contains methods used in multiple scripts in order to compact
the code and avoid method repetition.
"""

"""
INPUT
"""


def parse_input(path):
    """
    Parse home folder if specified in input
    """
    if '~' in path:
        path = path.replace('~', str(Path.home()))
    return path
