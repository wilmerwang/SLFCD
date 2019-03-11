import json
import os, sys, argparse
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

from camelyon16.data.annotation import Formatter # noqa

parser = argparse.ArgumentParser(description='Convert My json format to'
                                             'ASAP json format')
parser.add_argument('json_path', default=None, metavar='JSON_PATH', type=str,
                    help='Path to the input annotation in json format')
parser.add_argument('xml_path', default=None, metavar='XML_PATH', type=str,
                    help='Path to the output ASAP xml annotation')
parser.add_argument('color', default=None, metavar='COLOR', nargs='+', type=str,
                    help='The polygon part of color')


def run(args):
    with open(args.json_path) as f:
        dict = json.load(f)
    group_color = args.color
    gen = Formatter()
    gen.json2camelyon16xml(dict, args.xml_path, group_color)

def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()
