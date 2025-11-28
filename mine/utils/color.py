import sys

"""
from utils import color
print(color.RED + "Hello World" + color.RESET)
a = 114514
print(f"a = {color.RED}{a}{color.RESET}")
"""

IS_TTY = sys.stdout.isatty()
RED   = "\033[31m" if IS_TTY else ""
GREEN = "\033[32m" if IS_TTY else ""
YELLOW = "\033[33m" if IS_TTY else ""
RESET = "\033[0m" if IS_TTY else ""
