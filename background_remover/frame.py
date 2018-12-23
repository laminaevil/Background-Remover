from keras.models import load_model
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk

from tkinter.filedialog import askdirectory
from Transparent import transparent
from extra import extra_validations
from Validation import validation

import tkinter as tk
import tkinter.filedialog

import numpy as np
import hashlib
import time

import cv2 

model = load_model('training_phase/models/test.h5', compile=False)

class MY_GUI():
    def __init__(self,init_window_name):
        self.init_window_name = init_window_name

    #Set Windows
    def set_init_window(self):
        self.init_window_name.title("Bakground Remover")           #Windows name
        self.init_window_name.geometry('800x450')
        self.init_window_name.resizable(width=False,height=False)   # Stable Width

        # Save Button
        self.init_path_label = Button(self.init_window_name, text="Choose Image Path",  height= 1, command=self.open_button)
        self.init_path_label.place(x=20,  y=10,anchor='nw')

        # # Content
        self.init_path_Text = Label(self.init_window_name, height=1)  # Input Content
        self.init_path_Text.place(x=160,  y=10,anchor='nw')

        # # Effective content
        self.init_precision_Text = Label(self.init_window_name, text="ave Precision:",  height=1)  
        self.init_precision_Text.place(x=30,  y=120,anchor='nw')

        self.init_recall_Text = Label(self.init_window_name, text="ave Recall:", height=1)  
        self.init_recall_Text.place(x=30,  y=170,anchor='nw')

        self.init_meassure_Text = Label(self.init_window_name, text="ave F_measure:", height=1)  
        self.init_meassure_Text.place(x=30,  y=220,anchor='nw')

        self.init_MAE_Text = Label(self.init_window_name, text="ave MAE:", height=1)  
        self.init_MAE_Text.place(x=30,  y=270,anchor='nw')

        self.init_image_Text = Label(self.init_window_name, text="image predict and make image time: ", height=1)  
        self.init_image_Text.place(x=30,  y=320,anchor='nw')

        self.init_average_Text = Label(self.init_window_name, text="average execute time:", height=1)  
        self.init_average_Text.place(x=30,  y=370,anchor='nw')

        # # Effective Results
        self.precision_Text = Label(self.init_window_name, text="0", height=1)  
        self.precision_Text.place(x=120,  y=120,anchor='nw')

        self.recall_Text = Label(self.init_window_name, text="0", height=1)  
        self.recall_Text.place(x=100,  y=170,anchor='nw')

        self.meassure_Text = Label(self.init_window_name, text="0", height=1)  
        self.meassure_Text.place(x=130,  y=220,anchor='nw')

        self.MAE_Text = Label(self.init_window_name, text="0", height=1)  
        self.MAE_Text.place(x=90,  y=270,anchor='nw')

        self.image_Text = Label(self.init_window_name, text="0", height=1)  
        self.image_Text.place(x=250,  y=320,anchor='nw')

        self.average_Text = Label(self.init_window_name, text="0", height=1)  
        self.average_Text.place(x=170,  y=370,anchor='nw')

        # Display Button
        self.filter_button = Button(self.init_window_name, text="Filter", bg="lightblue", width=10, command=self.filter_img_display)  # 調用內部方法  加()為直接調用
        self.filter_button.place(x=20,  y=50,anchor='nw')

        self.Ground_filter_button = Button(self.init_window_name, text="Effective Analysis", bg="lightblue", width=15, command=self.ground_filter_img_display)  # 調用內部方法  加()為直接調用
        self.Ground_filter_button.place(x=110,  y=50,anchor='nw')

        self.remove_background_button = Button(self.init_window_name, text="Background Remove", bg="lightblue", width=20, command=self.remove_background_display)  # 調用內部方法  加()為直接調用
        self.remove_background_button.place(x=240,  y=50,anchor='nw')

    def ground_filter_img_display(self):
        src = self.init_path_Text.cget("text")
        try:
            precision, recall, meassure, MAE, image_predict, average_exe_time=validation(model)
            self.precision_Text.config(text = str(precision))
            self.recall_Text.config(text = str(recall))
            self.meassure_Text.config(text = str(meassure))
            self.MAE_Text.config(text = str(MAE))
            self.image_Text.config(text = str(image_predict) + "s")
            self.average_Text.config(text = str(average_exe_time) + "s")
        except:
            messagebox.showerror("Error", "Please, Enter the right path.")

    def filter_img_display(self):
        src = self.init_path_Text.cget("text")
        try:
            image_predict, average_exe_time=extra_validations(model, src)
            self.image_Text.config(text = image_predict + "s")
            self.average_Text.config(text = average_exe_time + "s")
        except:
            messagebox.showerror("Error", "Please, Enter the right path.")
    
    def remove_background_display(self):
        try:
            src = self.init_path_Text.cget("text")
            transparent()
        except:
            messagebox.showerror("Error", "Please, Enter the right path.")
    
    def open_button(self):
        filename = askdirectory()
        if filename != '':
            self.init_path_Text.config(text = filename)
        else:
            self.init_path_Text.config(text ="You have to choose right path.")

def gui_start():
    init_window = Tk()           
    ZMJ_PORTAL = MY_GUI(init_window)
    ZMJ_PORTAL.set_init_window()
    init_window.mainloop()

if __name__ == '__main__':
    gui_start()

