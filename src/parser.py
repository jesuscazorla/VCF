import argparse

def int_or_str(text):
    '''Helper function for argument parsing.'''
    try:
        return int(text)
    except ValueError:
        return text

# A way of converting a call to a object's method to a plain function
def encode(codec):
    return codec.encode()

def decode(codec):
    return codec.decode()

class CustomArgumentParser(argparse.ArgumentParser):
    def exit(self, status=0, message=None):
        if message:
            self._print_message(message, None)
        #self.print_help()
        #print("\nThis help only shows information about the top-level parameters.\nTo get information about lower-level parameters, use:\n\"python lower-level_module.py {encode|decode} -h\".\nSee docs/README.md to discover the available modules and their usage level.")
        exit(status)

# Main parameter of the arguments parser: "encode" or "decode"
parser = CustomArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, exit_on_error=False)
parser.add_argument("-g", "--debug", action="store_true", help=f"Output debug information")
subparser = parser.add_subparsers(help="You must specify one of the following subcomands:", dest="subparser_name")
parser_encode = subparser.add_parser("encode", help="Encode an image")
parser_decode = subparser.add_parser("decode", help="Decode an image")
parser_encode.set_defaults(func=encode)
parser_decode.set_defaults(func=decode)
