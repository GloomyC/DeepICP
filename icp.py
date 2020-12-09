from realsense_utils import *
import open3d as o3d
from pathlib import Path
import numpy as np
import glob
import os    
import copy

class ICP:
    def __init__(self, bag_filepath, pointclouds_dir_path, max_frames = 10):
        self.bag_filepath = bag_filepath
        self.pointclouds_dir_path = pointclouds_dir_path
        self.max_frames = max_frames
        self.counter = 1
        self.remove_all_files()
        sourcePC = PointCloudSource(filepath = self.bag_filepath, max_frames = self.max_frames, ICP = True)
        for points, color in sourcePC:
            self.create_file_from_PC(points, color)
            self.counter += 1
            print("Getting frame...")
        
        self.transformation = self.run_ICP_for_all_files()
        self.print_transformation()
        self.validate()
            
    def get_ICP_files(self):
        result = []
        files = sorted(Path(self.pointclouds_dir_path).iterdir(), key=os.path.getmtime)
        for i in range(len(files)):
            if str(files[i]).endswith('.ply'):
                file_path = str(files[i]).replace('//','/')
                result.append(file_path)
        return result
    
    def remove_all_files(self):
        files = glob.glob(self.pointclouds_dir_path + '/*')
        for file in files:
            os.remove(file)
        
    def create_file_from_PC(self, points, color): 
        points.export_to_ply(f"./{self.pointclouds_dir_path}/{self.counter}.ply", color)
            
    def run_ICP(self, source, target, threshold = 1000):    
        source = o3d.io.read_point_cloud(source)
        target = o3d.io.read_point_cloud(target)
        threshold = threshold
    
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
        return reg_p2p.transformation
    
    def run_ICP_for_all_files(self):
        result = np.asarray([[1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0.0], 
                             [0.0, 0.0, 0.0, 1.0]])
        
        files = self.get_ICP_files()
        for i in range(len(files) - 1):
            print("calculating...")
            result = np.matmul(self.run_ICP(files[i], files[i+1]), result)
            print(result)
            
        return result
    
    def print_transformation(self):
        print("================= TRANSFORMATION =================")
        print("\n")    
        print(self.transformation)
        print("\n")    
        print("==================================================")
    
    def validate(self):
        source = o3d.io.read_point_cloud(f"./{self.pointclouds_dir_path}/1.ply")
        source.paint_uniform_color([1, 0, 0]) # red
        
        target = o3d.io.read_point_cloud(f"./{self.pointclouds_dir_path}/{self.max_frames}.ply")
        target.paint_uniform_color([0, 1, 0]) # green
        
        transformed = o3d.io.read_point_cloud(f"./{self.pointclouds_dir_path}/1.ply")
        transformed = transformed.transform(self.transformation)
        transformed.paint_uniform_color([0, 0, 1]) # blue

        o3d.visualization.draw_geometries([target, transformed], width = 640, height = 480)
        
        trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0], 
                                [0.0, 0.0, 0.0, 1.0]])
        evaluation = o3d.pipelines.registration.evaluate_registration(
        source, target, 100, trans_init)
        print(evaluation)
        
    




