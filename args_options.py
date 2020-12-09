import argparse

def parse_train_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name",
                        "-mn",
                        type = str,
                        default = None,
                        help = 'Name of model to load or create')
    parser.add_argument("--list",
                        "-l",
                        type = str,
                        choices = ["model","train","test","net"],
                        default = None,
                        help = "List all available values")
    parser.add_argument("--net_config",
                        "-nc",
                        type = str,
                        metavar = "FILENAME",
                        default = None,
                        help = 'Name of network config file')
    parser.add_argument("--train_config",
                        "-tc",
                        type = str,
                        metavar = "FILENAME",
                        default = None,
                        help = 'Name of training config file')
    parser.add_argument("--epochs",
                        "-e",
                        type = int,
                        metavar = "NUM",
                        default = None,
                        help = "Amount of epochs to train, overrides config file")
    parser.add_argument("--batch_size",
                        "-b",
                        type = int,
                        metavar = "NUM",
                        default = None,
                        help = "Size of batch during training, overrides config file")
    parser.add_argument("--load_weights",
                        "-lw",
                        type = str,
                        choices = ["last","best"],
                        default = "last",
                        help = "Which model checkpoint to load")
    parser.add_argument("--verbose",
                        "-v",
                        type = bool,
                        default = False,
                        help = "More logs")
    parser.add_argument("--new_model",
                        "-new",
                        action = 'store_true',
                        default = False,
                        help='Create new model or replace existing with that name')
    
    return vars(parser.parse_args())

def parse_test_args():
    #TODO
    pass