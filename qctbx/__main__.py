import sys
from .cli import modes, use_unselected_parser

if __name__ == '__main__':
    if len(sys.argv) == 1 or sys.argv[1] not in modes:
        use_unselected_parser([sys.argv[0], '--help'])
    else:
        parser_func = modes[sys.argv[1].lower()]
        if len(sys.argv) == 2:
            parser_func([sys.argv[0], '--help'])
        else:
            parser_func(list(sys.argv[2:]))
            