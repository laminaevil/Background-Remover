# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from tkinter import *
from tkinter import messagebox
import tkinter as tk
import numpy as np
import cv2 
from PIL import Image, ImageTk
import hashlib
import time

LOG_LINE_NUM = 0

class MY_GUI():
    def __init__(self,init_window_name):
        self.init_window_name = init_window_name

    #設置窗口
    def set_init_window(self):
        self.init_window_name.title("Bakground Remover")           #窗口名
        self.init_window_name.geometry('700x200')
        self.init_window_name.resizable(width=False,height=False)   # 固定长宽不可拉伸

        #標籤
        self.init_path_label = Label(self.init_window_name, text="圖片路徑 : ",  height= 2)
        self.init_path_label.place(x=20,  y=10,anchor='nw')

        self.save_path_label = Label(self.init_window_name, text="圖片存取路徑 : ",  height= 2)
        self.save_path_label.place(x=20,  y=50,anchor='nw')

        # #文本框
        self.init_path_Text = Text(self.init_window_name, width=80, height=1)  #原始數據錄入框
        self.init_path_Text.place(x=100,  y=18,anchor='nw')

        self.init_save_Text = Text(self.init_window_name, width=80, height=1)  #原始數據錄入框
        self.init_save_Text.place(x=110,  y=58,anchor='nw')

        # Display Button
        self.original_image_button = Button(self.init_window_name, text="顯示原圖", bg="lightblue", width=10, command=self.ori_img_display)  # 調用內部方法  加()為直接調用
        self.original_image_button.place(x=20,  y=100,anchor='nw')

        self.filter_button = Button(self.init_window_name, text="過濾", bg="lightblue", width=10, command=self.filter_img_display)  # 調用內部方法  加()為直接調用
        self.filter_button.place(x=110,  y=100,anchor='nw')

        self.remove_background_button = Button(self.init_window_name, text="去背", bg="lightblue", width=10, command=self.remove_background_display)  # 調用內部方法  加()為直接調用
        self.remove_background_button.place(x=200,  y=100,anchor='nw')
 
        # Save Button
        self.save_ori_button = Button(self.init_window_name, text="存取原圖", bg="lightblue", width=10, command=self.ori_img_save)  # 調用內部方法  加()為直接調用
        self.save_ori_button.place(x=20,  y=140,anchor='nw')

        self.save_fil_button = Button(self.init_window_name, text="存取過濾圖片", bg="lightblue", width=10, command=self.filter_img_save)  # 調用內部方法  加()為直接調用
        self.save_fil_button.place(x=110,  y=140,anchor='nw')

        self.save_remove_background_button = Button(self.init_window_name, text="存取去背圖片", bg="lightblue", width=10, command=self.remove_background_save)  # 調用內部方法  加()為直接調用
        self.save_remove_background_button.place(x=200,  y=140,anchor='nw')

    def ori_img_display(self):
        src = self.init_path_Text.get(1.0,END).replace("\n", "")
        print(src)
        try:
            original_image = cv2.imread(src)
            cv2.imshow("Original image", original_image)
        except:
            messagebox.showerror("Error", "Please, Enter the right path.")

    def filter_img_display(self):
        src = self.init_path_Text.get(1.0,END).replace("\n", "")
        try:
            original_image = cv2.imread(src)
            cv2.imshow("Original image", original_image)
        except:
            messagebox.showerror("Error", "Please, Enter the right path.")

    def remove_background_display(self):
        src = self.init_path_Text.get(1.0,END).replace("\n", "")
        try:
            original_image = cv2.imread(src)
            cv2.imshow("Original image", original_image)
        except:
            messagebox.showerror("Error", "Please, Enter the right path.")

    def ori_img_save(self):
        src = self.init_path_Text.get(1.0,END).replace("\n", "")
        try:
            original_image = cv2.imread(src)
            cv2.imwrite("Original image", original_image)
        except:
            messagebox.showerror("Error", "Please, Enter the right path.")

    def filter_img_save(self):
        src = self.init_path_Text.get(1.0,END).replace("\n", "")
        try:
            original_image = cv2.imread(src)
            cv2.imshow("Original image", original_image)
        except:
            messagebox.showerror("Error", "Please, Enter the right path.")

    def remove_background_save(self):
        src = self.init_path_Text.get(1.0,END).replace("\n", "")
        try:
            original_image = cv2.imread(src)
            cv2.imshow("Original image", original_image)
        except:
            messagebox.showerror("Error", "Please, Enter the right path.")


def gui_start():
    init_window = Tk()           
    ZMJ_PORTAL = MY_GUI(init_window)
    ZMJ_PORTAL.set_init_window()
    init_window.mainloop()         

if __name__ == '__main__':

    gui_start()

