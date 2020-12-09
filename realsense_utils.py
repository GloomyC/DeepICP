import pyrealsense2 as rs
import numpy as np
import numpy as np
import math
import open3d as o3d
import matplotlib
import os
import cv2

def batch_cmap(depth,bot=0,top=1):
    colors = np.ones((1,depth.shape[0],3))
    colors[:,:,0] = np.minimum((np.maximum(depth-bot,0))/(top-bot),1)/1.5
    colors = matplotlib.colors.hsv_to_rgb(colors)
    
    return colors[0]

def l2_batch_norm(array):
    assert len(array.shape) is 2
    return np.sum(array**2,axis = 1)**0.5

def drop_empty_vertices(vertices):
        return vertices[l2_batch_norm(vertices) > 1e-5]

class CloudMapVisualization:
    
#     h,w, d are render bounding box parameters, 
#     color_d is coolrscale from red to blue
    def __init__(self,h=4, w=4, d=6, color_d = 2, camera_look = False):
        self.bbx = (h, w, d)
        self.color_d = color_d
        self.geometry = geometry = o3d.geometry.PointCloud()
        self.camera_look = camera_look
        self.visualization =  o3d.visualization.Visualizer()
        
    def __enter__(self):
        self.window = self.visualization.create_window(width=800, height=600)
        self.set_bbx()
        return self
        
    def __exit__(self,exc_type, exc_value, exc_traceback):
        self.visualization.destroy_window()
        self.visualization = None
    
    def render_frame(self,vertices):
        if self.visualization is None:
            raise Exception("CloudMapVisualization has already beed closed") 
        
        self.geometry.colors = o3d.utility.Vector3dVector(batch_cmap(vertices[:,2],top = self.color_d))
        vertices[:,1] = -vertices[:,1]
        if(self.camera_look):
            vertices[:,2] = -vertices[:,2] + 4.1
        else:
            vertices[:,2] = -vertices[:,2]
        self.geometry.points = o3d.utility.Vector3dVector(vertices)
        self.visualization.update_geometry(self.geometry)
        self.visualization.poll_events()
        self.visualization.update_renderer()
        
    def set_bbx(self):
        self.geometry.points = o3d.utility.Vector3dVector(np.array([[-self.bbx[1]/2,-self.bbx[0]/2,-self.bbx[2]],[self.bbx[1]/2,self.bbx[0]/2,0]]))
#         self.geometry.points = o3d.utility.Vector3dVector(np.array([[-2,-2,-6],[2,2,0]]))
        self.visualization.add_geometry(self.geometry)
        
   
class PointCloudSource:
    
    def __init__(self,filepath = None, max_frames = 100, ignore_first_frames = 5,verbose = False, file_loop = False, ICP = False):
        self.filepath = filepath
        self.max_frames = max_frames
        self.ignore_first_frames = ignore_first_frames
        self.verbose = verbose
        self.frame_counter = 0
        self.ICP = ICP
        
        self.rs_config = rs.config()
        if filepath is not None:
            self.rs_config.enable_device_from_file(filepath)
            
    def get_frame(self):
        success, frames = self.pipe.try_wait_for_frames()

        if not success:
            return None
        
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        w = rs.video_frame(depth_frame).width
        h = rs.video_frame(depth_frame).height

#         if self.ICP:
#             self.save_frame_to_img(depth_frame, color_frame)

        pc = rs.pointcloud()
        points = pc.calculate(depth_frame)
        
        vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(h,w, 3)
        vertices = self.preprocess_point_cloud(vertices)

        self.frame_counter += 1
        
        if self.ICP:
            return points, color_frame
        
        return vertices
        
    def drop_frame(self):
        success, _ = self.pipe.try_wait_for_frames()
        if not success:
            raise Exception(f"Frames ended during initial dropping on frame {self.frame_counter}")
        self.frame_counter += 1
            
    def preprocess_point_cloud(self,vertices):
        #TODO put more here
        shape = vertices.shape
        vertices = drop_empty_vertices(vertices.reshape(shape[0]*shape[1],3))
        return vertices
        
    def __iter__(self):
        self.pipe = rs.pipeline()
        self.pipe.start(self.rs_config)
        self.frame_counter = 0
        for i in range(self.ignore_first_frames):
            self.drop_frame()
        self.log(f"dropped {self.ignore_first_frames} first frames")
        return self
            
    def __next__(self):
        if self.max_frames != None and self.frame_counter >= self.max_frames + self.ignore_first_frames:
            self.log(f"stopped at {self.frame_counter}")
            self.pipe.stop()
            raise StopIteration

        if self.ICP:
            return self.get_frame()
        
        vertices = self.get_frame()
        
        self.log(f"got frame {self.frame_counter}, {f'received {vertices.shape[0]} points' if vertices is not None else 'end of source, no points received'}")

        if vertices is None:
            self.pipe.stop()
            raise StopIteration
        else:
            return vertices
        
    def log(self,msg):
        if self.verbose is True:
            print(msg)

