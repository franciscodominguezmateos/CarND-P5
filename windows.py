'''
Created on Feb 12, 2017

@author: Francisco Dominguez
'''
import cv2
import numpy as np
class WindowsFactory():
    def __init__(self):
        pass
    def process(self):
        pass
class WindowsSlider(WindowsFactory):
    def __init__(self):
        pass
    def process(self):
        pass
class WindowsProject(WindowsFactory):
    def __init__(self):
        self.K=np.array([[  1.15777818e+03,   0.00000000e+00,   6.67113986e+02],
                         [  0.00000000e+00 ,  1.15282212e+03 ,  3.86124636e+02],
                         [  0.00000000e+00 ,  0.00000000e+00 ,  1.00000000e+00]])
        self.dist=np.array([[-0.24688444, -0.02373395, -0.00109833,  0.00035111, -0.00259556]])
        self.rvec = np.array([0,0,0], np.float) # rotation vector
        self.tvec = np.array([0,0,0], np.float) # translation vector
        self.window3D=((-1.0,0.0,0.0),(1.0,2.0,0.0))
        self.Xstart=-8.0
        self.Xend  =8.0
        self.Xstep =0.75
        self.Zstart=10.0
        self.Zend  =30.0
        self.Zstep =2.5
    def getZWindows3D(self,Z):
        windowsList=[]
        starty=self.window3D[0][1]
        endy  =self.window3D[1][1]
        startz=self.window3D[0][2]+Z
        endz  =self.window3D[1][2]+Z
        for x in np.arange(self.Xstart,self.Xend+self.Xstep,self.Xstep):
            startx=x+self.window3D[0][0]
            endx  =x+self.window3D[1][0]
            windowsList.append(((startx,starty,startz),(endx,endy,endz)))
        return windowsList
    def getWindows3D(self):
        windowsList=[]
        for z in np.arange(self.Zstart,self.Zend+self.Zstep,self.Zstep):
            windowsList.extend(self.getZWindows3D(z))
        return windowsList
    def project(self,window3D):
        pts3D=np.array(window3D)
        window2D,_=cv2.projectPoints(pts3D,self.rvec,self.tvec, self.K, self.dist)
        window2D=window2D.astype(np.int)
        return (window2D[0][0][0],window2D[0][0][1]), \
               (window2D[1][0][0],window2D[1][0][1])
    def process(self):
        windows3D=self.getWindows3D()
        windows2D=[]
        for window3D in windows3D:
            window2D=self.project(window3D)
            windows2D.append(window2D)
        return windows2D