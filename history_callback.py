
import json
import os
import shutil
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
import matplotlib




def load_history(config, best = False):
    
    checkpoint_path = config.project_config["model_checkpoint_path"]
    network_name = config.model_name
    
    if best:
        with open(f'{checkpoint_path}/{network_name}/history_best.json','r') as json_file:
            history = json.load(json_file)
    else:
        with open(f'{checkpoint_path}/{network_name}/history_last.json','r') as json_file:
            history = json.load(json_file)
            
    return history


def save_history(config, history, best = False):
    checkpoint_path = config.project_config["model_checkpoint_path"]
    network_name = config.model_name
    
    if len(history["loss"]) > 2:
        matplotlib.use('Agg')
        x = [i+1 for i in range(len(history["loss"]))]
        plt.close('all')
        plt.figure()
        plt.plot(x[2:],history["loss"][2:])
        plt.plot(x[2:],history["val_loss"][2:])
        plt.legend(["loss","val_loss"])
        if best:
            plt.savefig(f'{checkpoint_path}/{network_name}/history_best_plot.png')
        else:
            plt.savefig(f'{checkpoint_path}/{network_name}/history_last_plot.png')
    
    if best:
        with open(f'{checkpoint_path}/{network_name}/history_best.json','w') as json_file:
            json.dump(history,json_file)
    else:
        with open(f'{checkpoint_path}/{network_name}/history_last.json','w') as json_file:
            json.dump(history,json_file)
            
def new_history():
    history = {"val_loss" : [],
               "loss" : [],
               "learning_rate": [],
               "batch_size" :[],
              }
    return history


class HistoryCallback(Callback):
    
    def __init__(self,config):
        self.config = config
    
    def on_epoch_end(self,epoch,logs = None):
        try:
            history = load_history(self.config)
        except:
            history = new_history()
            
        history["val_loss"].append(logs["val_loss"])
        history["loss"].append(logs["loss"])
        history["learning_rate"].append(self.config.train_config['learning_rate'])
        history["batch_size"].append(self.config.batch_size)
        
        save_history(self.config,history)