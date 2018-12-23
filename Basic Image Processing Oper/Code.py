# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 17:07:19 2018

@author: PC
"""

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QWidget, QFileDialog
from PyQt5.QtGui import QIcon, QPixmap
from shutil import copyfile
from tasarim import Ui_Form
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from scipy.ndimage import rotate
import cv2
import sys
import scipy.misc
import numpy as np

class MainWindow(QWidget,Ui_Form):
    
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setupUi(self)
        
        self.btn_close.clicked.connect(self.close)
        self.btn_image_filter_add.clicked.connect(self.Add_Image)
        self.radio_gray.clicked.connect(self.grayScale)
        self.radio_binary.clicked.connect(self.Binary)
        self.radio_zero.clicked.connect(self.Zero)
        self.radio_otzu.clicked.connect(self.OTSU)
        self.radio_histogram.clicked.connect(self.Histogram)
        self.radio_clahe.clicked.connect(self.clahe)
        self.btn_draw.clicked.connect(self.DrawCircle)   
        self.btn_image1_add.clicked.connect(self.Image1_Add)
        self.btn_image2_add.clicked.connect(self.Image2_Add)
        self.btn_apply.clicked.connect(self.Logical)
        self.btn_original_image_add.clicked.connect(self.lableling_image1_load)
        self.btn_searching_image_add.clicked.connect(self.lableling_image2_load)
        self.btn_lableling.clicked.connect(self.Lableling)
        self.btn_addImage_Noisy.clicked.connect(self.noisy_addImage)
        self.btn_noisy.clicked.connect(self.Noisy)
        self.btn_rotate_addPic.clicked.connect(self.Rotate_Image_Add)
        self.btn_rotateLeft.clicked.connect(self.RotateLeft)
        self.btn_rotateRight.clicked.connect(self.RotateRight)
        self.btn_resize.clicked.connect(self.Resize)
        self.btn_resizeAddPic.clicked.connect(self.ImageAddResize)
        
    
    def Add_Image(self):
        filename = QFileDialog.getOpenFileName()
        imagePath = filename[0]
        path = str(imagePath)
        pixmap = QPixmap(imagePath)
        self.lbl_original.setScaledContents(True)
        self.lbl_original.setPixmap(pixmap)
        self.show()
        copyfile(path, './images/original.jpg')
    
    def grayScale(self):
        resim = cv2.imread('./images/original.jpg', cv2.IMREAD_GRAYSCALE)
        cv2.imwrite('./images/Gray.jpg', resim)
        pixmap = QPixmap('./images/Gray.jpg')
        self.lbl_filtre_image.setScaledContents(True)
        self.lbl_filtre_image.setPixmap(pixmap)
        self.show()
    
    def Binary(self):
        img = cv2.imread('./images/original.jpg', 0)
        ret, resim = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite('./images/binary.jpg', resim)
        pixmap = QPixmap('./images/binary.jpg')
        self.lbl_filtre_image.setScaledContents(True)
        self.lbl_filtre_image.setPixmap(pixmap)
        self.show()
        
    def Zero(self):
        img = cv2.imread('./images/original.jpg', 0)
        ret, resim = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
        cv2.imwrite('./images/zero.jpg', resim)
        pixmap = QPixmap('./images/zero.jpg')
        self.lbl_filtre_image.setScaledContents(True)
        self.lbl_filtre_image.setPixmap(pixmap)
        self.show()
        
    def OTSU(self):
        img = cv2.imread('./images/original.jpg', 0)
        ret, resim = cv2.threshold(img, 127, 255, cv2.THRESH_OTSU)
        cv2.imwrite('./images/otsu.jpg', resim)
        pixmap = QPixmap('./images/otsu.jpg')
        self.lbl_filtre_image.setScaledContents(True)
        self.lbl_filtre_image.setPixmap(pixmap)
        self.show()
        
    def Histogram(self):
        img = cv2.imread('./images/original.jpg', 0)
        equ = cv2.equalizeHist(img)
        cv2.imwrite('./images/histogram.jpg',equ)
        pixmap = QPixmap('./images/histogram.jpg')
        self.lbl_filtre_image.setScaledContents(True)
        self.lbl_filtre_image.setPixmap(pixmap)
        self.show()
        
    def clahe(self):
        img = cv2.imread('./images/original.jpg', 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl1 = clahe.apply(img)
        cv2.imwrite('./images/clahe.jpg',cl1)
        pixmap = QPixmap('./images/clahe.jpg')
        self.lbl_filtre_image.setScaledContents(True)
        self.lbl_filtre_image.setPixmap(pixmap)
        self.show()
                
    def DrawCircle(self):
        width_ = int(self.line_width.text())
        height_ = int(self.line_height.text())
        img = np.zeros((512,512,3), np.uint8)
        image = cv2.circle(img,(width_,height_), 63, (255,255,0), -1)
        cv2.imwrite('./images/circle.png', image)
        pixmap = QPixmap('./images/circle.png')
        self.lbl_circle.setScaledContents(True)
        self.lbl_circle.setPixmap(pixmap)
        self.show()
        
        
    def Image1_Add(self):
        filename = QFileDialog.getOpenFileName()
        imagePath = filename[0]
        path = str(imagePath)
        pixmap = QPixmap(imagePath)
        self.lbl_image1.setScaledContents(True)
        self.lbl_image1.setPixmap(pixmap)
        self.show()
        copyfile(path, './images/image1.jpg')
        
    def Image2_Add(self):
        filename = QFileDialog.getOpenFileName()
        imagePath = filename[0]
        path = str(imagePath)
        pixmap = QPixmap(imagePath)
        self.lbl_image2.setScaledContents(True)
        self.lbl_image2.setPixmap(pixmap)
        self.show()
        copyfile(path, './images/image2.jpg')
        
    def Rotate_Image_Add(self):
        filename = QFileDialog.getOpenFileName()
        imagePath = filename[0]
        path = str(imagePath)
        pixmap = QPixmap(imagePath)
        self.lbl_rotate.setScaledContents(True)
        self.lbl_rotate.setPixmap(pixmap)
        self.show()
        copyfile(path, './images/RotateAdd.jpg')
        
    def RotateLeft(self):
        img = cv2.imread('./images/rotateAdd.jpg')
        img2 = rotate(img, 270)
        cv2.imwrite('./images/RotateLeft.jpg', img2)
        stri = './images/RotateLeft.jpg'
        pixmap = QPixmap(stri)
        self.lbl_rotate.setScaledContents(True)
        self.lbl_rotate.setPixmap(pixmap)
        self.show()
        
    def ImageAddResize(self):
        filename = QFileDialog.getOpenFileName()
        imagePath = filename[0]
        path = str(imagePath)
        pixmap = QPixmap(imagePath)
        self.lbl_resize.setScaledContents(True)
        self.lbl_resize.setPixmap(pixmap)
        self.show()
        copyfile(path, './images/resize.jpg')
        oriimg = cv2.imread('./images/resize.jpg')
        height, width, channel = oriimg.shape
        self.txt_height.setText(str(height))
        self.txt_width.setText(str(width))
        
    def Resize(self):
        height_value = int(self.txt_height.text())
        width_value = int(self.txt_width.text())
        filename = './images/resize.jpg'
        oriimg = cv2.imread(filename)
        newimg = cv2.resize(oriimg, (width_value, height_value))
        cv2.imwrite('./images/resizeResult.jpg', newimg)
        pixmap = QPixmap('./images/resizeResult.jpg')
        self.lbl_resize.setScaledContents(False)
        self.lbl_resize.setPixmap(pixmap)
        self.show()
        
    
    def RotateRight(self):
        img = cv2.imread('./images/rotateAdd.jpg')
        img2 = rotate(img, 90)
        cv2.imwrite('./images/RotateRight.jpg', img2)
        stri = './images/RotateRight.jpg'
        pixmap = QPixmap(stri)
        self.lbl_rotate.setScaledContents(True)
        self.lbl_rotate.setPixmap(pixmap)
        self.show()
        
    def Logical(self):
        image1 = cv2.imread('./images/image1.jpg', 0)
        image2 = cv2.imread('./images/image2.jpg', 0)
        
        resized1 = cv2.resize(image1, (600, 600))
        resized2 = cv2.resize(image2, (600, 600))
        
        text = str(self.combo_logic.currentText())
        
        if text == 'OR':
            bit_OR = cv2.bitwise_or(resized1, resized2)
            cv2.imwrite('./images/logical.jpg', bit_OR)
            pixmap = QPixmap('./images/logical.jpg')
            self.lbl_image_result.setScaledContents(True)
            self.lbl_image_result.setPixmap(pixmap)
            self.show()
        elif text == 'AND':
            bit_AND = cv2.bitwise_and(resized1, resized2)
            cv2.imwrite('./images/logical.jpg', bit_AND)
            pixmap = QPixmap('./images/logical.jpg')
            self.lbl_image_result.setScaledContents(True)
            self.lbl_image_result.setPixmap(pixmap)
            self.show()
        elif text == 'NOT':
            bit_NOT = cv2.bitwise_not(resized1, resized2)
            cv2.imwrite('./images/logical.jpg', bit_NOT)
            pixmap = QPixmap('./images/logical.jpg')
            self.lbl_image_result.setScaledContents(True)
            self.lbl_image_result.setPixmap(pixmap)
            self.show()
        elif text == 'XOR':
            bit_XOR = cv2.bitwise_xor(resized1, resized2)
            cv2.imwrite('./images/logical.jpg', bit_XOR)
            pixmap = QPixmap('./images/logical.jpg')
            self.lbl_image_result.setScaledContents(True)
            self.lbl_image_result.setPixmap(pixmap)
            self.show()
            
    def lableling_image1_load(self):
        filename = QFileDialog.getOpenFileName()
        imagePath = filename[0]
        path = str(imagePath)
        pixmap = QPixmap(imagePath)
        self.lbl_image_lable.setScaledContents(True)
        self.lbl_image_lable.setPixmap(pixmap)
        self.show()
        copyfile(path, './images/Lableling_Original.jpg')
        
    def lableling_image2_load(self):
        filename = QFileDialog.getOpenFileName()
        imagePath = filename[0]
        path = str(imagePath)
        pixmap = QPixmap(imagePath)
        self.lbl_image_lable_2.setScaledContents(True)
        self.lbl_image_lable_2.setPixmap(pixmap)
        self.show()
        copyfile(path, './images/Lableling_Detected.jpg')
        
    def Lableling(self):
        img = cv2.imread('./images/Lableling_Original.jpg', 0)
        img2 = img.copy()
        template = cv2.imread('./images/Lableling_Detected.jpg',0)
        w, h = template.shape[::-1]
        methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
        for meth in methods:
            img = img2.copy()   
            method = eval(meth)
            res = cv2.matchTemplate(img,template,method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(img,top_left, bottom_right, 255, 2)
        
        self.lbl_image_detect.setScaledContents(True)
        scipy.misc.toimage(img, cmin=0.0, cmax=...).save('./images/outfile.jpg')
        result = './images/outfile.jpg'
        pixmap = QPixmap(result)
        self.lbl_image_detect.setPixmap(pixmap)
        
    def noisy_addImage(self):
        filename = QFileDialog.getOpenFileName()
        imagePath = filename[0]
        path = str(imagePath)
        pixmap = QPixmap(imagePath)
        self.lbl_image_Orginal.setScaledContents(True)
        self.lbl_image_Orginal.setPixmap(pixmap)
        self.show()
        copyfile(path, './images/Original_Noisy.jpg')
        
    def Noisy(self):
        image = cv2.imread('./images/Original_Gurultu.jpg', 0)
        height, width = image.shape[:2]
        noise = np.zeros((height, width))
        cv2.randu(noise, 0, 256)
        noisy_gray = image + np.array(0.2*noise, dtype=np.int)
        cv2.imwrite('./images/noisy.jpg', noisy_gray)
        sonuc = './images/noisy.jpg'
        pixmap = QPixmap(sonuc)
        self.lbl_image_noisy.setScaledContents(True)
        self.lbl_image_noisy.setPixmap(pixmap)
    
    def close(self):
        sys.exit()