#! /usr/src/python3
from UserInterface import UserInterface
from Validate import Validate
import sys


def main(args):
    if len(args) > 1 and args[1] == "gui":
        x = UserInterface()
        x.startUI()
    else:
        Validate.test()


if __name__ == '__main__':
    main(sys.argv)
