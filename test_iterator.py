from config import Config
from args_options import parse_train_args
from dataset_iterator import *
from index_dataset import *

if __name__ == "__main__":
    config = Config()    
    config.load_train_config('default_train')
    config.batch_size = 8

        
    save_dataset_to_file(config)
    
    train_set, valid_set, test_set = get_dataset_split(config)
    print('============================')
    print('SPLIT DATA')
    print('============================')
    print(len(train_set))
    print(len(valid_set))
    print(len(test_set))
    
    iterator = DatasetIterator(config, train_set)
    
    print('============================')
    print('ITERATOR LENGHT')
    print(len(iterator))
    for i in range(len(iterator)):
        print('============================')

        x,y = iterator[i] 
        print(x[0].shape)
        print(x[1].shape)
        print(x[2].shape)
        print(y.shape) 

#     print(y)
    

    
