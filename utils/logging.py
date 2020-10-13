__all__ = ["FileHandler"]

import logging
import pathlib

from .shell import ShellUtils


class FileHandler(logging.FileHandler):

    def __init__(self, filename, mode='a', encoding=None, delay=False):
        ShellUtils().mkdir(pathlib.Path(filename).parent, True)
        super().__init__(filename, mode=mode, encoding=encoding, delay=delay)
