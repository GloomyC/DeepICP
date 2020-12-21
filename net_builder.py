from tensorflow.keras.layers import Conv2D, Input, MaxPool2D, Concatenate, Lambda, Dense, TimeDistributed, Dropout
from tensorflow.keras.models import Model
from functools import partial
import copy
import tensorflow as tf
import numpy as np
from pointconv.layers import PointConvSA, PointConvFP
from pointconv.utils import grouping

class DeepICPBuilder:
    def __init__(self, net_config):
        self.net_config = net_config
        
    def build(self):
        
        source_xyz = Input(shape = (None,3), name = 'source_xyz')
        target_xyz = Input(shape = (None,3), name = 'target_xyz')
        prev_transform_input = Input(shape = (4,4), name = 'prev_transform_input')
        GT_transform_target = Input(shape = (4,4), name = 'GT_transform_target')
        
        DFEX = self.get_DFEX_submodel()
        DFEB = self.get_DFEB_submodel()
        
        source_features = DFEX(source_xyz)
        target_features = DFEX(target_xyz)
        source_xyzfeatures = Lambda(lambda x: tf.concat([x[0], x[1]],axis = -1),name='append_xyz_to_source_features')([source_xyz,source_features])        
        selected_source_points_indices = self.weighting_layer(source_features)
        selected_source_points_xyzfeatures = self.select_points_by_indices(source_xyzfeatures, selected_source_points_indices)
        selected_source_points_xyz = Lambda(lambda x : x[:,:,0:3],name ="get_xyz_from_features")(selected_source_points_xyzfeatures)

        selected_source_xyz_transformed = self.apply_transform(selected_source_points_xyz,prev_transform_input)
        target_voxels_xyz = self.calculate_voxel_coordinates(selected_source_xyz_transformed)
        
        target_voxels_grouped_features = self.normalize_grouped_voxels(self.group_to_voxels(target_xyz,target_features,target_voxels_xyz))
        selected_source_points_grouped_features = self.normalize_grouped_balls(self.radius_grouping(source_xyz,source_features,selected_source_points_xyz))
        
        target_voxels_FEB = self.dfeb_grouped_voxels(DFEB,target_voxels_grouped_features)
        source_points_FEB = DFEB(selected_source_points_grouped_features)
        
        generated_target_xyz = self.CPG(target_voxels_xyz, target_voxels_FEB, source_points_FEB)
        
        
        model = Model([source_xyz,target_xyz, prev_transform_input, GT_transform_target], [selected_source_points_xyz ,generated_target_xyz])
        loss_args = (selected_source_points_xyz ,generated_target_xyz, GT_transform_target)

        return model, loss_args
        
        #=============
        

        
    def get_DFEX_submodel(self):
        # submodel parameters
        # points - (B,N,3) batched point coordinates
        #
        # returns - (B,N,F) batched points, each with F features
        
        d_group_sizes = self.net_config['DFEX_downsampling_group_sizes']
        d_group_nums = self.net_config['DFEX_downsampling_group_nums']
        sigma = self.net_config['DFEX_sigma']
        radius = self.net_config['DFEX_radius']
        mlp_d = self.net_config['DFEX_downsampling_mlps']
        mlp_u = self.net_config['DFEX_upsampling_mlps']
        full_conn = self.net_config['DFEX_full_conn']
        dropout = self.net_config['DFEX_dropout']
        
        DFE_input = Input(shape = (None,3))
        
        xyz_d1, features_d1 = PointConvSA(npoint=d_group_nums[0], radius=radius, sigma=sigma, K=d_group_sizes[0], mlp=mlp_d[0], bn=True)(DFE_input,DFE_input)
        xyz_d2, features_d2 = PointConvSA(npoint=d_group_nums[1], radius=radius, sigma=sigma, K=d_group_sizes[1], mlp=mlp_d[1], bn=True)(xyz_d1,features_d1)
        xyz_d3, features_d3 = PointConvSA(npoint=d_group_nums[2], radius=radius, sigma=sigma, K=d_group_sizes[2], mlp=mlp_d[2], bn=True)(xyz_d2,features_d2)
        
        features_u3 = PointConvFP(radius=radius, sigma=sigma, K=d_group_sizes[2], mlp = mlp_u[0])(xyz_d2, xyz_d3, features_d2, features_d3)
        features_u2 = PointConvFP(radius=radius, sigma=sigma, K=d_group_sizes[1], mlp = mlp_u[1])(xyz_d1, xyz_d2, features_d1, features_u3)
        features_u1 = PointConvFP(radius=radius, sigma=sigma, K=d_group_sizes[0], mlp = mlp_u[2])(DFE_input, xyz_d1, DFE_input, features_u2)
        
        features_fc = Dense(full_conn)(features_u1)
        features_dropped = TimeDistributed(Dropout(dropout))(features_fc)

        
        return Model(DFE_input,features_dropped,name='DFEX_submodel')
    
    
    
    
    def select_points_by_indices(self,points,indices):
        # points - (B,N,F) tensor N batched points each with F features
        # indices - (B,I) tensor of batched indexes of points that should be selected
        #
        # returns - (B,I,F) tensor of points selected by indices
        
        assert len(points.shape) == 3 and len(indices.shape) == 2
        
        #nothing in tf does this straight forward for each batch independantly, spaghetti incoming
        def fun(pt,idx):
            nb = tf.shape(pt)[0]     #batch_size
            np = tf.shape(idx)[1]    #selected_points_size
            nf = tf.shape(pt)[-1]    #features_size
            
            b_idxs = tf.range(0,nb)
            repeat_count = tf.ones(nb,dtype=tf.dtypes.int32)*np
            b_range = tf.reshape(tf.repeat(b_idxs, repeat_count),[-1,1])
            reshaped_idx = tf.reshape(idx,[-1,1])
            batched_idx = tf.concat([b_range,reshaped_idx],axis =1)
            selected = tf.gather_nd(pt,batched_idx)
            selected_rebatched = tf.reshape(selected,[nb,np,nf])
            return selected_rebatched
        
        return Lambda(lambda x: fun(x[0],x[1]), name = 'select_indices')([points,indices])
        
    def weighting_layer(self,inp):
        # inp - (B,N,F) batched N points each with F features
        #
        # returns - (B,I) indexes of I points rated highest
        
        mlp_shape = self.net_config['weighting_mlp']
        k = self.net_config['weighting_select_top_k']
        
        MLP = Dense(mlp_shape[0], name = f'weighting_mlp_0')(inp)
        for i in range(1, len(mlp_shape)):
            MLP = Dense(mlp_shape[i], name = f'weighting_mlp_{i}')(MLP)
        
        ratings = Lambda(lambda x: tf.math.softplus(x),name="weighting_activation")(MLP)
            
        values, indices = Lambda(lambda x: tf.math.top_k(tf.squeeze(x, axis = -1),k = k),name ="top_k")(ratings)
        
        return indices
    
    def get_DFEB_submodel(self):
        # submodel parameters
        # points - (B,N,G,F) batched N groups of size G of F point features
        #
        # returns - (B,N,F2) batched F2 features describing whole N groups
        
        mlp_shape = self.net_config['DFEB_mlp']
        inp_feature_count = self.net_config['DFEX_full_conn'] + 3
        
        DFE_input = Input(shape = (None,None,inp_feature_count))
        
        MLP = Dense(mlp_shape[0], name = f'DFEB_mlp_0')(DFE_input)
        for i in range(1, len(mlp_shape)):
            MLP = Dense(mlp_shape[i], name = f'DFEB_mlp_{i}')(MLP)
            
        max_pooled_features = Lambda(lambda x: tf.math.reduce_max(x,axis=[-2]))(MLP)
        
        return Model(DFE_input,max_pooled_features,name='DFEB_submodel')
    
    def apply_transform(self,points,transform):
        # points - (B,N,3) N batched point coordinates
        # transform - (B,4,4) batched transforms
        #
        # returns - (B,N,3) N batched transformed point coordinates
        
        def fun(p,t):
            s = tf.shape(p)
            one_pad_points = tf.concat([p,tf.ones([s[0],s[1],1])],axis = -1)
            t_points = tf.transpose(tf.matmul(t,one_pad_points,transpose_b=True),[0,2,1])[...,0:3]

            return t_points
        
        return Lambda(lambda x: fun(x[0],x[1]),name = "transform_multiply")([points,transform])
    
    def calculate_voxel_coordinates(self, points):
        # points - (B,N,3) batched point coordinates
        #
        # returns - (B,N,V,3) batched V voxel center coordinates placed in ball around each of N points
        
        sample_radius = self.net_config['voxel_sampling_radius']
        voxel_size = self.net_config['voxel_size']
        
        def fun(xyz, s_radius, v_size):
            
            n = s_radius // v_size
            steps = list(np.arange(-n* v_size, (n+1)*v_size, v_size))
            translations = []

            for i in steps:
                for j in steps:
                    for k in steps:
                        p = np.array([i*v_size,j*v_size,k*v_size])
                        dist = np.sqrt(np.sum(p**2))
                        if dist <= s_radius:
                            translations.append(p.tolist())
                            
            offsets = tf.constant(np.array(translations))
            
            voxel_count = tf.shape(offsets)[0]
            b_size = tf.shape(xyz)[0]
            xyz_count = tf.shape(xyz)[1]
            xyz_expanded = tf.expand_dims(xyz,axis = 2)
            xyz_repeated = tf.repeat(xyz_expanded,voxel_count,axis = 2)
            offsets_expanded = tf.expand_dims(tf.expand_dims(offsets,axis = 0),axis=0)
            offsets_repeated = tf.repeat(tf.repeat(offsets_expanded,b_size,axis =0 ),xyz_count,axis = 1)
           
            
            return tf.cast(offsets_repeated,tf.float32) + xyz_repeated
        
        
        voxels = Lambda(lambda x: fun(x,sample_radius,voxel_size),name = "calculate_voxel_centers")(points)
        self.voxel_dim_size = voxels.get_shape()[2]
        return voxels
    
    def group_to_voxels(self,target_xyz, target_features, voxel_xyz):
        # target_features - (B,N1,F) batched point features
        # target_xyz - (B,N1,3) batched point coordinates
        # voxel_xyz - (B,N2,V,3) batched V voxel center coordinates placed in ball around each of N2 points
        #
        # returns - (B,N2,V,G,3+F) batched G-sized groups of point features F and their xyz coordinates for each voxel V candidate for point N2
        #                          xyz in 3+F are zero translated to their voxel centers
        
        voxel_size =  self.net_config['voxel_size']
        group_size = self.net_config['group_size']
        
        def fun(tar_f, tar_xyz, v_xyz, v_size, group_size):
            shape = tf.shape(v_xyz)
            
            reshaped_voxels = tf.reshape(v_xyz,[shape[0],shape[1]*shape[2],3])

            #normal knn groupng for now, v_size is ignored
            #xyz are appended to features and zero translated to center
            #TODO make this into voxel limited grouping
            grouped_features = grouping(tar_f,group_size, tar_xyz, reshaped_voxels)[1]
        
            grouped_features_unreshaped = tf.reshape(grouped_features,[shape[0],shape[1], shape[2],group_size,-1])
            
            return grouped_features_unreshaped
            
            
        return Lambda(lambda x: fun(x[0], x[1], x[2], voxel_size, group_size),name = "voxels_grouping")([target_features,target_xyz,voxel_xyz])
    
    def radius_grouping(self,xyz,features,centers):
        # features - (B,N1,F) batched point features
        # xyz - (B,N1,3) batched point coordinates
        # centers - (B,N2,3) batched sampling coordinates of N2 points
        #
        # returns - (B,N2,G,3+F) batched G-sized groups of point features F and their xyz coordinates for point N2
        #                        xyz in 3+F are zero translated to their sampling centers
        
        ball_radius = voxel_size = self.net_config['ball_radius']
        group_size = self.net_config['group_size']
        
        def fun(_xyz, _features, _centers, radius, group_size):
            
            #normal knn groupng for now, radius is ignored
            #xyz are appended to features and zero translated to center
            #TODO make this into ball limited grouping
            grouped_features = grouping(_features, group_size, _xyz, _centers)[1]
            return grouped_features
        
        return Lambda(lambda x: fun(x[0], x[1], x[2], ball_radius, group_size),name = "ball_grouping")([xyz,features,centers])
    
    def normalize_grouped_voxels(self,xyzfeatures):
        voxel_size =  self.net_config['voxel_size']
        
        #grouping function already zero-translates
        def fun(xyzf,r):
            xyz_norm = xyzf[...,0:3]/(voxel_size/2)                        
            xyzf_norm = tf.concat([xyz_norm,xyzf[...,3:]],-1)
            return xyzf_norm

        return Lambda(lambda x: fun(x,voxel_size),name='voxel_normalize')(xyzfeatures)
    
    def normalize_grouped_balls(self,xyzfeatures):
        ball_radius =  self.net_config['ball_radius']
        
        #grouping function already zero-translates
        def fun(xyzf,r):
            xyz_norm = xyzf[...,0:3]/(ball_radius)                        
            xyzf_norm = tf.concat([xyz_norm,xyzf[...,3:]],-1)
            return xyzf_norm

        return Lambda(lambda x: fun(x,ball_radius),name='ball_normalize')(xyzfeatures)
    
    def dfeb_grouped_voxels(self,DFEB,grouped_voxels):
        n_pts = self.net_config['weighting_select_top_k']
        n_voxels = self.voxel_dim_size
        n_group_pts = self.net_config['group_size']
        n_features = self.net_config['DFEX_full_conn']
        n_dfeb_features = self.net_config['DFEB_mlp'][-1]
        
        reshaped_voxels = Lambda(lambda x: tf.reshape(x,[-1,n_pts*n_voxels,n_group_pts,n_features+3]),name='target_reshape_before_DFEB')(grouped_voxels)
        reshaped_voxel_dfeb = DFEB(reshaped_voxels)
        
        unreshaped_voxel_dfeb = Lambda(lambda x: tf.reshape(x,[-1,n_pts,n_voxels,n_dfeb_features]),name='target_reshape_after_DFEB')(reshaped_voxel_dfeb)
        
        return unreshaped_voxel_dfeb
    
    def CPG(self, target_candidates, target_features, source_features):
        
        def fun(t_f, s_f):
            
            C = tf.shape(t_f)[2]
            
            source_expanded_repeated = tf.repeat(tf.expand_dims(s_f,axis=2),C,axis = 2)
            
            f_diff = tf.sqrt((source_expanded_repeated - t_f)**2)
            
            return f_diff

        feature_diff = Lambda(lambda x: fun(x[0],x[1]),name ='CPG')([target_features,source_features])
        
        conv1 = Conv2D(16,3,padding='same',name='3Dconv_1')(feature_diff)
        conv2 = Conv2D(8,3,padding='same',name='3Dconv_2')(conv1)
        conv3 = Conv2D(1,3,padding='same',name='3Dconv_3')(conv2)
        
        mat = Lambda(lambda x: tf.nn.softmax(x,-1),name='softmax')(conv3)
        
        points = Lambda(lambda x: tf.reduce_sum(tf.repeat(x[0],3,axis=-1)*x[1],axis = -2),name = 'weights_multi')([mat,target_candidates])
        
        return points
            
        
        
        
        
        
        
    
        
        
            
            
            
        
           
        
            
            
        
        
        
        
        
        
        
        
        