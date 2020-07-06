#===============================================================================
#     Tags Runner
#===============================================================================
import argparse

parser = argparse.ArgumentParser(description='Methods for Model Training, Evaluating and Predicting')
parser.add_argument("command", metavar="<command>", help="'train', 'evaluate', or 'predict'",)
args = parser.parse_args()

assert args.command in ['train', 'evaluate', 'predict'], "invalid parsing 'command'"

if args.command == "train":
        #train
        int = 1
elif args.command == 'evaluate':
        int = 2        
else:
        int = 3
