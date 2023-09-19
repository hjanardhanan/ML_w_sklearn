import sys
import argparse

parser = argparse.ArgumentParser(description='Toy problems for machine learning',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--log_reg', help='Invoke logistic regression script')

args = parser.parse_args()
arg_map = vars(args)

print(arg_map)