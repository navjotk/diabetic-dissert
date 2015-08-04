'''
Created on 4 Aug 2015

@author: navjotkukreja
'''
import platform
import sys

if platform.system()=='Windows' or not sys.stdout.isatty():
    GLOBAL_WINDOWS = True
else:
    GLOBAL_WINDOWS = False