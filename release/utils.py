# -*- coding: utf-8 -*-
# @Author: Victor Garcia
# @Date:   2018-10-02 12:20:12
# @Last Modified by:   Victor Garcia
# @Last Modified time: 2018-10-02 13:06:02

from pathlib  import Path

"""
This class contains methods used in multiple scripts in order to compact the code
and above repetition.
"""

"""
INPUT
"""

"""
Parse home folder if specified in input
"""
def parseInput(path):
	if '~' in path:
		path = path.replace('~',str(Path.home()))
	return path

