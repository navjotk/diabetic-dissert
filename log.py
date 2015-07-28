'''
Created on 27 Jul 2015

@author: navjotkukreja
'''
import sys

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  
    
    def flush(self):
        self.terminal.flush()