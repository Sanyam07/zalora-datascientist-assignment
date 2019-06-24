import argparse
import sys
import os
wording_directory = os.getcwd()
sys.path.insert(0, wording_directory)

from src.test_model import test_model
from src.train_model import model_generating
from src.performance import performance_view

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Content based profile recommendation')
    parser.add_argument('--run_type', type=str, required=True, dest='run_type'
                        , help='''The type of this application, either view perfomance of model on available data (performance_view)
                        or train new model on available data (model_generating)
                        or run generated model on test data (test_model)
                        ''')
    args = parser.parse_args()
    
    if args.run_type == 'test_model':    
        test_model()
    elif args.run_type == 'model_generating':    
        model_generating()
    elif args.run_type == 'performance_view':    
        performance_view()
    else:
        print("This argument {} does not work".format(args.run_type))
        