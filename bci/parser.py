import argparse
import sys

def parser():
    parser = argparse.ArgumentParser(description='Entry for Kaggle BCI competition', add_help=True)
    parser.add_argument('-c', '--continued', action="store_true", default=False, help='Use old training/test data')
    parser.add_argument('-v', '--validate', action="store_true", default=False, help='Test results via cross validation')
    parser.add_argument('-a', '--append', action="store_true", default=False, help='Append sum and diff data to dataframe')
    parser.add_argument('-d', '--dropout', action="store_true", default=False, help='Drop columns from dataframe')
    parser.add_argument('-m', '--model', action='store', default='GBClassifier', help='Choice of regression/classification model')
    parser.add_argument('--train', action='store', default='train_cz.csv', help='Input train file for use with option continue')
    parser.add_argument('--test', action='store', default='test_cz.csv', help='Input test file for use with option continue')

    return parser.parse_args()
    
if __name__ == '__main__':
    sys.argv.append('-h')    
    parser()
