# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as tp
import tkinter

    
if __name__ == '__main__':
    #windows set
    window = tkinter.Tk()
    window.title('BackGround Remover')
    window.geometry('500x500')
    
    # to generate the windows
    window.mainloop()
    l = window.Label(window, 
    text='OMG! this is TK!',    # 标签的文字
    bg='green',     # 背景颜色
    font=('Arial', 12),     # 字体和字体大小
    width=15, height=2  # 标签长宽
    )
    l.pack()    # 固定窗口位置


