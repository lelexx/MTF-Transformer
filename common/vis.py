from turtle import update
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2 as cv
import torch
import numpy as np
import os, sys
import time
from matplotlib.animation import FuncAnimation, writers


class Update():
    def __init__(self, parent):
        self.parent = parent

    def __call__(self, frame_id):
        pose_2d = self.parent.pose_2d
        pose_3d_list = self.parent.pose_3d_list
        title = self.parent.title
        B, J, C, N = pose_3d_list[0].shape
        
        bone_color = self.parent.cfg.VIS.BONE_COLOR
        bone_flag = self.parent.cfg.H36M_DATA.BONES_FLAG

        for i in range(frame_id, frame_id+1):
            radius = 1.7
            for view_id in range(self.parent.num_view):
                if self.parent.direction == 'h':
                    self.parent.ax_views[0][view_id].clear()
                    self.parent.ax_views[0][view_id].set_xticklabels([])
                    self.parent.ax_views[0][view_id].set_yticklabels([])
                    #self.parent.ax_views[0][view_id].set_title('camera {}'.format(view_id))
                else:
                    self.parent.ax_views[view_id][0].clear()
                    self.parent.ax_views[view_id][0].set_xticklabels([])
                    self.parent.ax_views[view_id][0].set_yticklabels([])
                    #self.parent.ax_views[view_id][0].set_title('camera {}'.format(view_id))      
                    self.parent.ax_views[view_id][0].tick_params(bottom=False,top=False,left=False,right=False)
                for k in range(self.parent.K-1):
                    if self.parent.direction == 'h':
                        self.parent.ax_views[k+1][view_id].clear()
                        self.parent.ax_views[k+1][view_id].set_xlim3d([-radius/2, radius/2])
                        self.parent.ax_views[k+1][view_id].set_ylim3d([-radius/2, radius/2])
                        self.parent.ax_views[k+1][view_id].set_zlim3d([-radius/2, radius/2])
                        self.parent.ax_views[k+1][view_id].set_xticklabels([])
                        self.parent.ax_views[k+1][view_id].set_yticklabels([])
                        self.parent.ax_views[k+1][view_id].set_zticklabels([])
                        if len(title) > 0:
                            self.parent.ax_views[k+1][view_id].set_title(title[k])
                        self.parent.ax_views[k+1][view_id].view_init(15, -90 + self.angle_y)
                    else:
                        self.parent.ax_views[view_id][k+1].clear()
                        
                        self.parent.ax_views[view_id][k+1].set_xlim3d([-radius/2, radius/2])
                        self.parent.ax_views[view_id][k+1].set_ylim3d([-radius/2, radius/2])
                        self.parent.ax_views[view_id][k+1].set_zlim3d([-radius/2, radius/2])
                        self.parent.ax_views[view_id][k+1].set_xticklabels('x', fontsize = 100)
                        self.parent.ax_views[view_id][k+1].set_xticklabels([])
                        self.parent.ax_views[view_id][k+1].set_yticklabels([])
                        self.parent.ax_views[view_id][k+1].set_zticklabels([])
                        if len(title) > 0 and view_id == 0:
                            self.parent.ax_views[view_id][k+1].set_title(title[k], fontsize = 15)

                        self.parent.ax_views[view_id][k+1].view_init(15, -90 + self.parent.angle_y)

                for n, pose_3d in enumerate(pose_3d_list):
                    for bone_id, l in enumerate(self.parent.link):
                        x = list(pose_3d[i, l, 0, view_id])
                        y = list(pose_3d[i, l, 2, view_id])
                        z = list(-1 * pose_3d[i, l, 1, view_id])
                        color = bone_color[bone_flag[bone_id]]
                        color = [i / 255 for i in color]
                        if self.parent.direction == 'h':
                            self.parent.ax_views[n+1][view_id].plot(x, y, z, zdir='z',color = color, linewidth=3)
                        else:
                            self.parent.ax_views[view_id][n+1].plot(x, y, z, zdir='z',color = color, linewidth=3)
                if self.parent.p2d_mode == 'pose':
                    for bone_id, l in enumerate(self.parent.link):
                        x = list(pose_2d[i, l, 0, view_id])
                        y = list(-pose_2d[i, l, 1, view_id])
                        color = bone_color[bone_flag[bone_id]][:3]
                        color = [i / 255 for i in color]
                        if self.parent.direction == 'h':
                            self.parent.ax_views[0][view_id].plot(x, y, color = color)
                        else:
                            self.parent.ax_views[view_id][0].plot(x, y, color = color)
                elif self.parent.p2d_mode == 'image':
                    if self.parent.direction == 'h':
                        self.parent.ax_views[0][view_id].imshow(pose_2d[view_id][i])
                    else:
                        self.parent.ax_views[view_id][0].imshow(pose_2d[view_id][i])
            #plt.pause(1)
class Vis():
    def __init__(self,cfg, num_3d, num_view = 4, p2d_mode = 'pose', direction = 'h', fig_name = 'compare_flex'):
        '''
        p2d_mode: ('pose', 'image')
        direction: ('h', 'v')
        '''
        self.cfg = cfg
        self.fig_name = fig_name
        self.direction = direction
        self.link = np.array(cfg.H36M_DATA.BONES)
        self.K = num_3d + 1
        self.num_view = num_view
        self.p2d_mode = p2d_mode
        self.scale = 1.6
        self.angle_y = 30
        if direction == 'h':
            self.fig = plt.figure(figsize=(14, 8))
        else:
            if cfg.VIS.DATASET == 'kth':
                self.fig = plt.figure(figsize=(self.K * 2, self.num_view * 2)) #(w, h)
            elif cfg.VIS.DATASET == 'h36m':
                self.fig = plt.figure(figsize=(self.K * 1.9, self.num_view * 1.81)) #(w, h)
        
        sin_y = np.sin(np.pi / 180 * self.angle_y)
        cos_y = np.cos(np.pi / 180 * self.angle_y)
        self.rot = torch.Tensor([
            [cos_y, 0, sin_y],
            [0, 1, 0],
            [-sin_y, 0, cos_y],
        ])
        self.ax_views = []
        if self.direction == 'h':
            for i in range(self.K):
                self.ax_views.append([])
        else:
            for i in range(self.num_view):
                self.ax_views.append([])
        
        for i in range(self.num_view):
            if self.direction == 'h':
                ax = self.fig.add_subplot(self.K, self.num_view, i+1+(self.K-1)*self.num_view)
                self.ax_views[0].append(ax)
            else:
                ax = self.fig.add_subplot(self.num_view, self.K, i*self.K + 1)
                self.ax_views[i].append(ax)
            
        for k in range(self.K-1):
            for i in range(self.num_view):
                if self.direction == 'h':
                    ax = self.fig.add_subplot(self.K, self.num_view,i+1 + k*self.num_view, projection='3d')
                    self.ax_views[k + 1].append(ax)
                else:
                    ax = self.fig.add_subplot(self.num_view,self.K, i*self.K + 1 + k+1, projection='3d')
                    plt.gca().set_box_aspect((4.5, 4.5, 5))
                    self.ax_views[i].append(ax)
        
        if cfg.VIS.DATASET == 'kth':
            plt.subplots_adjust(left=0.0, bottom=0.0, right=1,top = 0.955, wspace=0.0, hspace=0.)
        elif cfg.VIS.DATASET == 'h36m':
            plt.subplots_adjust(left=0.0, bottom=0.0, right=1,top = 0.955, wspace=0.0, hspace=0.)
        self.fps = 40
        self.pose_2d = None
        self.pose_3d_list = None
        self.title = None
        self.update_3d = Update(self)
        
    def show(self, pose_2d, *pose_3d_list, title = []):
        '''
        pose_2d:(B, J, C, N)
        '''
        # plt.pause(100)
        # return
        self.pose_2d = pose_2d.numpy() if self.p2d_mode == 'pose' else pose_2d
        self.pose_3d_list = list(pose_3d_list)
        for n, pose_3d in enumerate(self.pose_3d_list):
            self.pose_3d_list[n] = torch.einsum('tjcv, cq->tjqv', pose_3d, self.rot) * self.scale
            self.pose_3d_list[n] = self.pose_3d_list[n].cpu().numpy()
        self.title = title
        B, J, C, N = pose_3d_list[0].shape
        for i in range(B):
            self.update_3d(i)
            plt.pause(0.001)
            if not self.cfg.VIS.DEBUG:
                pass
            else:
                X = input()
                if X == 's':
                    dirs = './images/temp/{}'.format(self.fig_name)
                    if not os.path.exists(dirs):
                        os.makedirs(dirs)
                    path = os.path.join(dirs, 'image_dataset_{}_time_{}'.format(self.cfg.VIS.DATASET, time.strftime('%Y-%m-%d-%H-%M-%S')))
                    self.fig.savefig(path, dpi=500,bbox_inches='tight')
        # save gif
        #self.anim = FuncAnimation(self.fig, self.update_3d, frames=np.arange(0, B), interval=1000/self.fps, repeat=False)       
        #self.anim.save('./images/lele.gif', dpi=80, writer='imagemagick')
           

def crop_image(img, w, h, p2d, image_scale, scale_ratio):
    min_x, min_y = np.min(p2d, axis = 0)
    max_x, max_y = np.max(p2d, axis = 0)
    center_x = min_x + (max_x - min_x) / 2
    center_y = min_y + (max_y - min_y) / 2
    human_w = (max_x - min_x) * 1.1
    human_h = (max_y - min_y) * 1.1
    min_x = max(center_x - human_w / 2, 0)
    max_x = min(center_x + human_w / 2, w)
    min_y = max(center_y - human_h / 2, 0)
    max_y = min(center_y + human_h / 2, h)

    if max_x - min_x < (max_y - min_y) * scale_ratio:
        tmp_len = (max_y - min_y) * scale_ratio
        min_x = max(center_x - tmp_len / 2, 0)
        max_x = min(center_x + tmp_len / 2, w)
    elif max_x - min_x > (max_y - min_y) * scale_ratio:
        tmp_len = (max_x - min_x) / scale_ratio
        min_y = max(center_y - tmp_len / 2, 0)
        max_y = min(center_y + tmp_len / 2, h)
    min_x, max_x, min_y, max_y = int(min_x), int(max_x), int(min_y), int(max_y)

    img = img[min_y:max_y, min_x:max_x]
    img = cv.resize(img, image_scale, interpolation = cv.INTER_AREA)
    return img

def plot_pose2d(cfg, img, frame_pose2d):
    for bone_id, l in enumerate(cfg.H36M_DATA.BONES):
        bone_color = cfg.VIS.BONE_COLOR
        bone_flag = cfg.H36M_DATA.BONES_FLAG
        c = bone_color[bone_flag[bone_id]][:3]
        c.reverse()

        x = list(frame_pose2d[l, 0])
        y = list(frame_pose2d[l, 1])
        s = (int(x[0]), int(y[0]))
        e = (int(x[1]), int(y[1]))
        cv.line(img,s, e,c, 10, 1)
    