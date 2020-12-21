from config import Config
from args_options import parse_train_args
from net_builder import DeepICPBuilder
import numpy as np
import tensorflow as tf
from tf_utils import  init_tf
from model import create_model, load_model
from dataset_iterator import DatasetIterator, get_dataset_split
from tensorflow.keras.callbacks import ModelCheckpoint
from history_callback import *


def console_info(args):
    #here go utils triggered by console arguments that dump info and exit immediately
    if args['list'] is not None:
        print("TODO, this will print available values")
        exit()
        


if __name__ == "__main__":
    config = Config()
    
    args = parse_train_args()
    console_info(args)
    
    config.apply_train_args(args)
    init_tf()
    
    
#     from loss import get_loss
    
#     loss = get_loss(0.5)
    
#     x1 = np.random.random((2,32,3)).astype('float32')
#     x2 = np.random.random((2,32,3)).astype('float32')
#     t = np.zeros((2,4,4)).astype('float32')
#     t[0] = np.eye(4).astype('float32')
    
#     res = loss(t,[x1,x2])
#     print("====================")
#     print(res)

    
    model = None
    if(config.new_model):
        model = load_model(config)
    else:
        model = create_model(config)
        
    config.dump()
    model.summary()
    
    train_split, val_split, test_split = get_dataset_split(config)
    
    train_iterator = DatasetIterator(config,train_split)
    val_iterator = DatasetIterator(config,val_split)
    train_iterator = DatasetIterator(config,test_split)
    
    checkpoint_path = f'{config.project_config["model_checkpoint_path"]}/{config.model_name}'
    mc1 = ModelCheckpoint(f'{checkpoint_path}/weights_last.h5', save_best_only=False, save_weights_only=True)
    mc2 = ModelCheckpoint(f'{checkpoint_path}/weights_best.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
    mc3 = HistoryCallback(config)
    callbacks = [mc1,mc2,mc3]
    
    history = model.fit(train_iterator,validation_data = val_iterator, epochs = config.train_config["epochs"], callbacks = callbacks)
    
    

    
    


def debug(model):
    seed = 40
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.compat.v1.set_random_seed(seed)
    
    PC_source_input = np.random.random((2,5000,3))
    PC_target_input = np.random.random((2,4000,3))
    transform = np.zeros((2,4,4))
    transform[0] = np.eye(4)
    transform[1] = np.eye(4)
    transform[0,0:3,0:3] = np.array([[1,0,0],[0,0,-1],[0,1,0]])
    
    
    
    out = model.predict([PC_source_input,PC_target_input, transform],batch_size=PC_source_input.shape[0])
    
    print("-------NET INPUT-------")
    print(f'source PC shape {PC_source_input.shape}')
    print(f'target PC shape {PC_target_input.shape}')
    print(f'prev transform shape {transform.shape}')
    print("-------NET OUTPUT-------")
    for i,part in enumerate(out):
        print("===========================")
        print(f' out{i} shape {part.shape}')
        




    


    
    
    
    

        
    
 
