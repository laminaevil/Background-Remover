# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from tkinter import *
from tkinter import messagebox

import numpy as np
import PIL
import hashlib
import time

LOG_LINE_NUM = 0

class MY_GUI():
    def __init__(self,init_window_name):
        self.init_window_name = init_window_name

    #設置窗口
    def set_init_window(self):
        self.init_window_name.title("Bakground Remover")           #窗口名
        self.init_window_name.geometry('800x500')
        self.init_window_name.resizable(width=False,height=False)   # 固定长宽不可拉伸

        #標籤
        self.init_path_label = Label(self.init_window_name, text="圖片路徑 : ",  height= 5)
        self.init_path_label.place(x=20,  y=10,anchor='nw')
        
        self.init_original_image_label = Label(self.init_window_name, text="原圖", width=3)
        self.init_original_image_label.place(x=100,  y=120,anchor='nw')

        # #文本框
        self.init_path_Text = Text(self.init_window_name, width=80, height=1)  #原始數據錄入框
        self.init_path_Text.place(x=100,  y=42,anchor='nw')

        # #按鈕
        self.original_image_button = Button(self.init_window_name, text="顯示原圖", bg="lightblue", width=10, command=self.str_trans_to_md5)  # 調用內部方法  加()為直接調用
        self.original_image_button.place(x=20,  y=80,anchor='nw')

    #功能函數
    def str_trans_to_md5(self):
        src = self.init_path_Text.get(1.0,END)

        try:
            im=PIL.Image.open("message.jpg")
        except:
            messagebox.showerror("Error", "Please Enter a certain path.")
                
        # #原圖顯示
        self.Image_Label = Label(self.init_window_name)
        self.Image_Label.place(x=100,  y=350,anchor='nw')


def gui_start():
    init_window = Tk()           
    ZMJ_PORTAL = MY_GUI(init_window)
    ZMJ_PORTAL.set_init_window()
    init_window.mainloop()         

if __name__ == '__main__':
    gui_start()

