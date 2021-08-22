"""
Algorithm by nikorose87 to Eye Alignment
Test for Research Scientist position

"""
import cv2 as cv
import numpy as np
import requests
import os
import argparse

class face_alignment():
    def __init__(self, dir, image_name):
        self.dir = dir
        self.root_dir = os.getcwd()
        self.image_name = image_name
        self.image_dir = os.path.join(self.dir, self.image_name)
        parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
        parser.add_argument('--face_cascade', help='Path to face cascade.', default='/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')
        parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='/usr/share/opencv4/haarcascades/haarcascade_eye.xml')
        parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
        args = parser.parse_args()
        self.face_cascade_name = args.face_cascade
        self.eyes_cascade_name = args.eyes_cascade
        self.face_cascade = cv.CascadeClassifier()
        self.eyes_cascade = cv.CascadeClassifier()
        #Where to store the image object
        self.image = cv.imread(self.image_dir, cv.IMREAD_UNCHANGED)
        #Directory to load and save photos
        #-- 1. Load the cascades
        if not self.face_cascade.load(cv.samples.findFile(self.face_cascade_name)):
            print('--(!)Error loading face cascade')
            exit(0)
        if not self.eyes_cascade.load(cv.samples.findFile(self.eyes_cascade_name)):
            print('--(!)Error loading eyes cascade')
            exit(0)
    
    def scaling(self, percent=25):

        #calculate the 50 percent of original dimensions
        width = int(self.image.shape[1] * percent / 100)
        height = int(self.image.shape[0] * percent / 100)

        # dsize
        dsize = (width, height)

        # resize image
        self.image = cv.resize(self.image, dsize)
        return cv.imshow('', self.image)
    
    def gray_scale_plus_face(self, scale=1.03, neighbors= 4):
        # Converting the image into grayscale
        self.gray=cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        # Creating variable faces
        faces= self.face_cascade.detectMultiScale (self.gray, scale, neighbors)
        # Defining and drawing the rectangle around the face
        for(x , y,  w,  h) in faces:
            cv.rectangle(self.image, (x,y) ,(x+w, y+h), (0,255,0), 3)
        # Creating two regions of interest
        self.roi_gray=self.gray[y:(y+h), x:(x+w)]
        self.roi_color=self.image[y:(y+h), x:(x+w)]
        return cv.imshow('', self.image)
    
    def eye_detection(self, scale=1.1, neighbors= 4):
        # Creating variable eyes
        eyes = self.eyes_cascade.detectMultiScale(self.roi_gray, scale, neighbors)
        index=0
        # Creating for loop in order to divide one eye from another
        for (ex , ey,  ew,  eh) in eyes:
            if index == 0:
                self.eye_1 = (ex, ey, ew, eh)
            elif index == 1:
                self.eye_2 = (ex, ey, ew, eh)
            # Drawing rectangles around the eyes
            cv.rectangle(self.roi_color, (ex,ey) ,(ex+ew, ey+eh), (0,0,255), 3)
            index += 1
        return cv.imshow('', self.image)

    def which_eye(self):
        if self.eye_1[0] < self.eye_2[0]:
            self.left_eye = self.eye_1
            self.right_eye = self.eye_2
        else:
            self.left_eye = self.eye_2
            self.right_eye = self.eye_1
        return
    
    def calc_angle(self):
        # Calculating coordinates of a central points of the rectangles
        left_eye_center = (int(self.left_eye[0] + (self.left_eye[2] / 2)), int(self.left_eye[1] + (self.left_eye[3] / 2)))
        self.left_eye_x = left_eye_center[0] 
        self.left_eye_y = left_eye_center[1]
        
        right_eye_center = (int(self.right_eye[0] + (self.right_eye[2]/2)), int(self.right_eye[1] + (self.right_eye[3]/2)))
        self.right_eye_x = right_eye_center[0]
        self.right_eye_y = right_eye_center[1]
        
        cv.circle(self.roi_color, left_eye_center, 5, (255, 0, 0) , -1)
        cv.circle(self.roi_color, right_eye_center, 5, (255, 0, 0) , -1)
        cv.line(self.roi_color,right_eye_center, left_eye_center,(0,200,200),3)

        if self.left_eye_y > self.right_eye_y:
            A = (self.right_eye_x, self.left_eye_y)
            #    Integer -1 indicates that the image will rotate in the clockwise direction
            direction = -1 
        else:
            A = (self.left_eye_x, self.right_eye_y)
            # Integer 1 indicates that image will rotate in the counter clockwise  
            direction = 1 

        cv.circle(self.roi_color, A, 5, (255, 0, 0) , -1)
        
        cv.line(self.roi_color,right_eye_center, left_eye_center,(0,200,200),3)
        cv.line(self.roi_color,left_eye_center, A,(0,200,200),3)
        cv.line(self.roi_color,right_eye_center, A,(0,200,200),3)
        return cv.imshow('', self.image)
    
    def rotate_img(self):
        delta_x = self.right_eye_x - self.left_eye_x
        delta_y = self.right_eye_y - self.left_eye_y
        angle=np.arctan(delta_y/delta_x)
        angle = (angle * 180) / np.pi

        # Width and height of the image
        h, w = self.image.shape[:2]
        # Calculating a center point of the image
        # Integer division "//"" ensures that we receive whole numbers
        center = (w // 2, h // 2)
        # Defining a matrix M and calling
        # cv2.getRotationMatrix2D method
        M = cv.getRotationMatrix2D(center, (angle), 1.0)
        # Applying the rotation to our image using the
        # cv2.warpAffine method
        self.image = cv.warpAffine(self.image, M, (w, h))
        return cv.imshow('', self.image)
    
    def export_img(self, name):
        os.chdir(self.dir)
        cv.imwrite(name, self.image)
        os.chdir(self.root_dir)
        return cv.imshow('', self.image)

    def perform_all(self, scale=True):
        if scale:
            self.scaling()
        self.gray_scale_plus_face()
        self.eye_detection()
        self.which_eye()
        self.calc_angle()
        self.rotate_img()
        self.export_img(self.image_name[:-4]+'_m.jpg')


# dir = os.path.join('static','uploads')
# img = "IMG_20210724_191355.jpg"
# CV = face_alignment(dir, img)
# CV.perform_all()








