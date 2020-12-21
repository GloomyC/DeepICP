import os, shutil, sys
from net_builder import DeepICPBuilder
from tensorflow.keras.models import load_model as lm
from tensorflow.keras.optimizers import Adam
import pickle
from loss import get_loss

def create_model(config):
    model_checkpoint_path = config.project_config['model_checkpoint_path']
    model_name = config.model_name

    model_path = f'{model_checkpoint_path}/{model_name}'
    
    model, loss_inp = DeepICPBuilder(config.net_config).build()
    loss = get_loss(config.train_config["loss_alpha"])
    source_pts, target_pts, GT = loss_inp
    model.add_loss(loss(source_pts,target_pts,GT))
    
    optimizer = Adam(learning_rate = config.train_config['learning_rate'])
    model.compile(optimizer = optimizer)
    
    save_model(model,config)
    config.save_current_model_net_config()
    
    return model

def delete_model(config):
    model_checkpoint_path = config.project_config['model_checkpoint_path']
    model_name = config.model_name
    model_path = f'{model_checkpoint_path}/{model_name}'
    shutil.rmtree(model_path)
    

def load_model(config):
    model_checkpoint_path = config.project_config['model_checkpoint_path']
    model_name = config.model_name
    weights = config.weights
    
   
    
    optimizer = Adam(learning_rate = config.train_config['learning_rate'])
    optimizer_weights_path = f'{model_path}/optimizer_{checkpoint}.pkl'
    with open(optimizer_weights_path, 'rb') as f:
        weight_values = pickle.load(f)
    optimizer.set_weights(weight_values)
    
    model, loss_inp = DeepICPBuilder(config.net_config).build()
    loss = get_loss(config.train_config["loss_alpha"])
    source_pts, target_pts, GT = loss_inp
    model.add_loss(loss(source_pts,target_pts,GT))
    
    optimizer = Adam(learning_rate = config.train_config['learning_rate'])
    model.compile(optimizer = optimizer)
    
    model.load_weights(model_weights)
    
    return model
    
    
    
#     assert os.path.exists()
    
def save_model(model,config,best = False):
    model_checkpoint_path = config.project_config['model_checkpoint_path']
    model_name = config.model_name
    
    model_path = f'{model_checkpoint_path}/{model_name}'
    
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    
    
    checkpoint = None
    if(best):
        checkpoint = 'best'
    else:
        checkpoint = 'last'
        
    model_weights_path = f'{model_path}/weights_{checkpoint}.h5'
    optimizer_weights_path = f'{model_path}/optimizer_{checkpoint}.pkl'
        
    optimizer_weights = getattr(model.optimizer, 'weights')
    
    model.save_weights(model_weights_path)
    with open(optimizer_weights_path, 'wb') as f:
        pickle.dump(optimizer_weights, f)
    
    if(config.train_config != None):
        config.save_current_model_train_config()
    
    