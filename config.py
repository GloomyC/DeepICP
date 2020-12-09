from os import listdir
from os.path import exists, isfile
import json

class Config:
    
    def __init__(self):
        
        self.project_config = None
        self.net_config = None
        self.train_config = None
        self.test_config = None
        self.new_model = False
        self.model_name = None
        self.verbose = False
        
        #other objects should not be reaching into args to configure anything
        self.args = None
        
        self.load_project_config()
        
    def load_project_config(self):
        if not isfile(f'./configs/project_config.json'):
            print(f"project config was not found in subpath ./configs/project_config.json\ncreate it from ./configs/project_config_template.json")
            exit()
            
        with open(f'./configs/project_config.json','r') as json_file:
            self.project_config = json.load(json_file)
    
    #LOAD NEW CONFIGS
    #========================
    def load_net_config(self, config_name):
        with open(f'./configs/net/{config_name}.json','r') as json_file:
            self.net_config = json.load(json_file)

    def load_train_config(self, config_name):
         with open(f'./configs/train/{config_name}.json','r') as json_file:
            self.train_config = json.load(json_file)
    
    def load_test_config(self, config_name):
         with open(f'./configs/test/{config_name}.json','r') as json_file:
            self.test_config = json.load(json_file)

    #LOAD EXISTING MODEL CONFIGS
    #========================
    def load_existing_model_net_config(self, model_name):
        checkpoint_path = self.project_config['model_checkpoint_path']
        with open(f'{checkpoint_path}/{model_name}/net_config.json','r') as json_file:
            self.net_config = json.load(json_file)

    def load_existing_model_train_config(self, model_name):
        checkpoint_path = self.project_config['model_checkpoint_path']
        with open(f'{checkpoint_path}/{model_name}/train_config.json','r') as json_file:
            self.train_config = json.load(json_file)

    def load_existing_model_test_config(self, model_name):
        checkpoint_path = self.project_config['model_checkpoint_path']
        with open(f'{checkpoint_path}/{model_name}/test_config.json','r') as json_file:
            self.test_config = json.load(json_file)

    #SAVE CURRENT MODEL CONFIGS
    #========================
    def save_current_model_net_config(self):
        checkpoint_path = self.project_config['model_checkpoint_path']
        with open(f'{checkpoint_path}/{self.model_name}/net_config.json','w') as json_file:
            json.dump(self.net_config,json_file)

    def save_current_model_train_config(self):
        checkpoint_path = self.project_config['model_checkpoint_path']
        with open(f'{checkpoint_path}/{self.model_name}/train_config.json','w') as json_file:
            json.dump(self.net_config,json_file)

    def save_current_model_test_config(self):
        checkpoint_path = self.project_config['model_checkpoint_path']
        with open(f'{checkpoint_path}/{self.model_name}/test_config.json','w') as json_file:
            json.dump(self.net_config,json_file)
    
    #APPLY ARGS
    #========================
    def apply_train_args(self,args):
        self.args = args
        if args['model_name'] is None:
            print("no model name was provided, use --model_name or -mn, use --help for full description")
            exit()
        self.model_name = args['model_name']
        self.verbose = args['verbose']
        
        if args['new_model']:
            
            if args['net_config'] is not None:
                self.load_net_config(args['net_config'])
            else:
                print(f"No net config specified, using default file \"{self.project_config['default_net_config']}\"")
                self.load_net_config(self.project_config['default_net_config'])
                
            if args['train_config'] is not None:
                self.load_train_config(args['train_config'])
            else:
                print(f"No train config specified, using default file \"{self.project_config['default_train_config']}\"")
                self.load_train_config(self.project_config['default_train_config'])
        else:
            if not exists(f"{self.project_config['model_checkpoint_path']}/{self.model_name}"):
                print(f"model with name \"{self.model_name}\" could not be found, to create new model use -new")
                exit()
            self.load_existing_model_net_config(self.model_name)
            self.load_existing_model_train_config(self.model_name)
           
        #override train_config with args
        if args['epochs'] is not None:
            self.train_config['epochs'] = args['epochs']
        if args['batch_size'] is not None:
            self.train_config['batch_size'] = args['batch_size']
            
    def apply_test_args(self,args):
        #TODO
        pass
    
    def dump(self):
        print("========Dumping config content========")
        print("args:")
        print(self.args)
        print("--------------------------------------")
        print("project_config:")
        print(self.project_config)
        print("--------------------------------------")
        print("net_config:")
        print(self.net_config)
        print("--------------------------------------")
        print("train_config:")
        print(self.train_config)
        print("--------------------------------------")
        print("test_config:")
        print(self.test_config)
        print("======================================")
        
        
            

