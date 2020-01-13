# -*- coding: utf-8 -*-
import sys
import os
import pdb

def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100,
                     fill='=', empty=' ', tip='>', begin='[', end=']', done="[DONE]", clear=True):
    """
    Print iterations progress.
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required : current iteration                          [int]
        total       - Required : total iterations                           [int]
        prefix      - Optional : prefix string                              [str]
        suffix      - Optional : suffix string                              [str]
        decimals    - Optional : positive number of decimals in percent     [int]
        length      - Optional : character length of bar                    [int]
        fill        - Optional : bar fill character                         [str] (ex: 'â– ', 'â–ˆ', '#', '=')
        empty       - Optional : not filled bar character                   [str] (ex: '-', ' ', 'â€¢')
        tip         - Optional : character at the end of the fill bar       [str] (ex: '>', '')
        begin       - Optional : starting bar character                     [str] (ex: '|', 'â–•', '[')
        end         - Optional : ending bar character                       [str] (ex: '|', 'â–', ']')
        done        - Optional : display message when 100% is reached       [str] (ex: "[DONE]")
        clear       - Optional : display completion message or leave as is  [str]
    """
    #pdb.set_trace()
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength
    if iteration != total:
         bar = bar + tip
    bar = bar + empty * (length - filledLength - len(tip))
    display = '\r{prefix}{begin}{bar}{end} {percent}%{suffix}' \
              .format(prefix=prefix, begin=begin, bar=bar, end=end, percent=percent, suffix=suffix)
    print(display, end=''),   # comma after print() required for python 2
    if iteration == total:      # print with newline on complete
        if clear:               # display given complete message with spaces to 'erase' previous progress bar
            finish = '\r{prefix}{done}'.format(prefix=prefix, done=done)
            if hasattr(str, 'decode'):   # handle python 2 non-unicode strings for proper length measure
                finish = finish.decode('utf-8')
                display = display.decode('utf-8')
            clear = ' ' * max(len(display) - len(finish), 0)
            print(finish + clear)
        else:
            print('')


def verbose(verboseLevel, requiredLevel, printFunc=print, *printArgs, **kwPrintArgs):
    """
    Calls `printFunc` passing it `printArgs` and `kwPrintArgs`
    only if `verboseLevel` meets the `requiredLevel` of verbosity.

    Following forms are supported:

        > verbose(1, 0, "message")

            >> message

        > verbose(1, 0, "message1", "message2")

            >> message1 message2

        > verbose(1, 2, "message")

            >>          <nothing since verbosity level not high enough>

        > verbose(1, 1, lambda x: print('MSG: ' + x), 'message')

            >> MSG: message

        > def myprint(x, y="msg_y", z=True): print('MSG_Y: ' + y) if z else print('MSG_X: ' + x)
        > verbose(1, 1, myprint, "msg_x", "msg_y")

            >> MSG_Y: msg_y

        > verbose(1, 1, myprint, "msg_x", "msg_Y!", z=True)

            >> MSG_Y: msg_Y!

        > verbose(1, 1, myprint, "msg_x", z=False)

            >> MSG_X: msg_x

        > verbose(1, 1, myprint, "msg_x", z=True)

            >> MSG_Y: msg_y
    """
    if verboseLevel >= requiredLevel:
        # handle cases when no additional arguments are provided (default print nothing)
        printArgs = printArgs if printArgs is not None else tuple([''])
        # handle cases when verbose is called directly with the object (ex: str) to print
        if not hasattr(printFunc, '__call__'):
            printArgs = tuple([printFunc]) + printArgs
            printFunc = print
        printFunc(*printArgs, **kwPrintArgs)


def print_flush(txt=''):
    print(txt)
    sys.stdout.flush()


if os.name == 'nt':
    import msvcrt
    import ctypes

    class _CursorInfo(ctypes.Structure):
        _fields_ = [("size", ctypes.c_int),
                    ("visible", ctypes.c_byte)]


def hide_cursor():
    if os.name == 'nt':
        ci = _CursorInfo()
        handle = ctypes.windll.kernel32.GetStdHandle(-11)
        ctypes.windll.kernel32.GetConsoleCursorInfo(handle, ctypes.byref(ci))
        ci.visible = False
        ctypes.windll.kernel32.SetConsoleCursorInfo(handle, ctypes.byref(ci))
    elif os.name == 'posix':
        sys.stdout.write("\033[?25l")
        sys.stdout.flush()


def show_cursor():
    if os.name == 'nt':
        ci = _CursorInfo()
        handle = ctypes.windll.kernel32.GetStdHandle(-11)
        ctypes.windll.kernel32.GetConsoleCursorInfo(handle, ctypes.byref(ci))
        ci.visible = True
        ctypes.windll.kernel32.SetConsoleCursorInfo(handle, ctypes.byref(ci))
    elif os.name == 'posix':
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()
