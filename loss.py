import tensorflow as tf
import numpy as np

def get_loss(alpha):
    def append_coulmn_to_tensor(y_source_net_selected):
        batch_size = tf.shape(y_source_net_selected)[0]
        point_count = tf.shape(y_source_net_selected)[1]
        ones = tf.ones([batch_size, point_count, 1])
        
        print(y_source_net_selected.shape)
        
        return tf.concat([y_source_net_selected, ones], -1)

    def calculate_target_using_transformation(transform, y_source):
        return tf.transpose(tf.linalg.matmul(transform, y_source, transpose_b=True),[0,2,1])[...,0:3]

    def create_result_transform_matrix(R, T):
        lastRow = tf.reshape(tf.constant(np.array([0,0,0,1]), tf.float32), (1,1,4))
        lastRow = tf.repeat(lastRow,tf.shape(R)[0],0)
        return tf.concat([tf.concat([R, tf.transpose(T,[0,2,1])], axis = -1), lastRow], axis = -2)

    def calculate_transform_pred(y_source_net_selected, y_target_net_calculated):
        mean_d = tf.math.reduce_mean(y_target_net_calculated, axis = 1, keepdims = True)
        mean_m = tf.math.reduce_mean(y_source_net_selected, axis = 1, keepdims = True)
        diff_d = tf.math.subtract(y_target_net_calculated, mean_d)
        diff_m = tf.math.subtract(y_source_net_selected, mean_m)

        ## diff_m ^ T * diff_d
        H = tf.linalg.matmul(tf.transpose(diff_m, [0, 2, 1]), diff_d)
        
        H = tf.reshape(H,[-1,3,3])

        # SVD
        S, U, V = tf.linalg.svd(H, full_matrices = True, compute_uv = True)

        ## V * U^T
        R = tf.linalg.matmul(V, tf.transpose(U, [0, 2, 1]))
        
        T = tf.math.subtract(mean_d, mean_m)

        return create_result_transform_matrix(R, T)

    def calculate_distance(y1, y2):
        return tf.math.sqrt(tf.math.reduce_sum((y1 - y2)**2, axis=-1))
    
    def loss1(transform_true, y_source_net_selected, y_target_net_calculated):
        y_source_net_selected = append_coulmn_to_tensor(y_source_net_selected)
        y1 = calculate_target_using_transformation(transform_true, y_source_net_selected)
        y2 = y_target_net_calculated
        return tf.math.reduce_mean(calculate_distance(y1, y2), axis=-1)

    def loss2(transform_true, y_source_net_selected, y_target_net_calculated): 
        transform_pred = calculate_transform_pred(y_source_net_selected, y_target_net_calculated)
        y_source_net_selected_appended = append_coulmn_to_tensor(y_source_net_selected)
        y1 = calculate_target_using_transformation(transform_true, y_source_net_selected_appended)
        y2 = calculate_target_using_transformation(transform_pred, y_source_net_selected_appended)
        return tf.math.reduce_mean(calculate_distance(y1, y2), axis=-1)
        
    def loss(y_source_net_selected, y_target_net_calculated, transform_true):

        return alpha * loss1(transform_true, y_source_net_selected, y_target_net_calculated) + (1 - alpha) * loss2(transform_true, y_source_net_selected, y_target_net_calculated)
    
    return loss