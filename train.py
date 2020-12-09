from config import Config
from args_options import parse_train_args
from net_builder import DeepICPBuilder
import numpy as np
import tensorflow as tf
from tf_utils import  init_tf


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
    net_builder = DeepICPBuilder(config.net_config)
    net = net_builder.build()
    
    config.dump()
    net.summary()
    net.compile()
    
    seed = 40
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.compat.v1.set_random_seed(seed)
    
    PC_source_input = np.random.random((2,5000,3))
    PC_target_input = np.random.random((2,4000,3))
    transform = np.zeros((2,4,4))
    transform[0] = np.eye(4)
    transform[1] = np.eye(4)
    transform[0,3,0] = 10
    transform[1,3,0] = 10
    
    print("-------NET INPUT-------")
    print(f'source PC shape {PC_source_input.shape}')
    print(f'target PC shape {PC_target_input.shape}')
    print(f'prev transform shape {transform.shape}')
    
    out = net.predict([PC_source_input,PC_target_input, transform],batch_size=PC_source_input.shape[0])

    print("-------NET OUTPUT-------")
    for i,part in enumerate(out):
        print("===========================")
        print(f' out{i} shape {part.shape}')
#         print(part)
    
        
    #================
    
#     from pointconv.utils import knn_limited_r, grouping, grouping_limited_ball, knn_kdtree
    
#     x = tf.cast(tf.convert_to_tensor(np.random.random((5,100,3))),tf.double)
#     f = tf.cast(tf.convert_to_tensor(np.random.random((5,100,32))),tf.double)
#     nx = tf.cast(tf.convert_to_tensor(np.random.random((5,10,3))),tf.double)
# #     print(nx.get_shape())
    
#     y = grouping(f,4,x,nx,use_xyz=True)[1]
    
# #     print(y.to_numpy().shape)
#     print(tf.shape(y))
#     print(y)
        
    
    
    
    #================
#     source_features,selected_points_indices,selected_points_coordinates = pred
#     sel_idx = selected_points_indices[0][0]
#     origin_coord = inp[0][sel_idx]
#     sel_coord = selected_points_coordinates[0][0]
    
#     print(f"selected idx {sel_idx}")
#     print(f"origin coord {origin_coord}")
#     print(f"selected coord { sel_coord}")
        




    


    
    
    
    

        
    
 
