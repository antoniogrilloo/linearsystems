#! /usr/src/python3
from ui.UserInterface import UserInterface
from methods.validation.Validate import Validate
import sys


def main(args):
    length = len(args)
    if length == 1:
        Validate.test()
        return
    if args[1] == "--gui":
        x = UserInterface()
        x.startUI()
    elif args[1] == "--matrix" and args[3] == "--tol":
        filename = args[2]
        tol = float(args[4])
        Validate.test_filename(matrix_file=filename, tol=tol)
    else:
        print("Error parameters")


if __name__ == '__main__':
    main(sys.argv)
