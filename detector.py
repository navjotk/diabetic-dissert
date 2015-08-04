'''
Created on 4 Aug 2015

@author: navjotkukreja
'''
import platform

if platform.system()=='Windows':
    GLOBAL_WINDOWS = True
else:
    GLOBAL_WINDOWS = False