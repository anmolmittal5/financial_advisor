import sys
import os
sys.path.insert(0, os.getcwd())
from configparser import ConfigParser

class ConfigReader:
    def __init__(self) -> None:
        self.cfg = ConfigParser()
        self.cfg.read("config/config.ini")

    def get_config(self):
        return self.cfg