from config import Config
from args_options import parse_train_args
import json

def save_dataset_to_file(config):
        print("INDEXING DATASET")
        dataset_path = config.project_config['dataset_path']
        poses_file_path = f'{dataset_path}/poses/gt_poses.txt' 
        pc_path = f'{dataset_path}/pcds'
        json_path = f'{dataset_path}/indexed_data.json'
        
        data = []
        counter = 1
        file = open(poses_file_path, 'r') 
        first_line = file.readline() 
        second_line = file.readline() 

        while True: 
            third_line = file.readline() 
            if not third_line: 
                break
            split_data_from_lines(data, pc_path, counter, first_line, second_line, third_line)
            first_line = second_line
            second_line = third_line
            counter += 1
        file.close()
        
        with open(json_path, 'w') as outfile:
            json.dump(data, outfile)
            
        print('============================')
        print("       FINISHED SAVING")
        print('============================')
        
def split_data_from_lines(data, pc_path, counter, first_line, second_line, third_line):
    splited_first_line = first_line.split(' ')
    splited_second_line = second_line.split(' ')
    splited_third_line = third_line.split(' ')
    data.append({ 
        'start': pc_path + '/' + splited_first_line[0] + '.pcd',
        'end':  pc_path + '/' + splited_second_line[0] + '.pcd',
        'gt_prev': {
            	"x1": splited_first_line[2],
				"y1": splited_first_line[3],
				"z1": splited_first_line[4],
				"q_x1": splited_first_line[5],
				"q_y1": splited_first_line[6],
				"q_z1": splited_first_line[7],
				"q_w1": splited_first_line[8],
            
                "x2": splited_second_line[2],
				"y2": splited_second_line[3],
				"z2": splited_second_line[4],
				"q_x2": splited_second_line[5],
				"q_y2": splited_second_line[6],
				"q_z2": splited_second_line[7],
				"q_w2": splited_second_line[8],
        },
        'gt_curr': {
            	"x1": splited_second_line[2],
				"y1": splited_second_line[3],
				"z1": splited_second_line[4],
				"q_x1": splited_second_line[5],
				"q_y1": splited_second_line[6],
				"q_z1": splited_second_line[7],
				"q_w1": splited_second_line[8],
            
                "x2": splited_third_line[2],
				"y2": splited_third_line[3],
				"z2": splited_third_line[4],
				"q_x2": splited_third_line[5],
				"q_y2": splited_third_line[6],
				"q_z2": splited_third_line[7],
				"q_w2": splited_third_line[8],
        }
    })
    
if __name__ == "__main__":
    config = Config()    
    config.load_train_config('default_train')
    save_dataset_to_file(config)