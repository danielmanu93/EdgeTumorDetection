# -*- coding: utf-8 -*-
from calendar import c
from fileinput import filename
from tkinter import filedialog, ttk
from tkinter import *
import tkinter as tk
from PIL import ImageTk, Image
import os
import customtkinter
import sys
import numpy as np
import datetime
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch import nn
import torchvision
from torchvision import transforms
import numpy as np
import utils
import network
from dataset import FWIDataset
import transforms as T
import math
import tkinter.scrolledtext as tkscrolled
import matplotlib.pyplot as plt
from visualize import PlotUltrasound, TaskBasedReconstruction, TaskBasedTumor, Reconstruction, ReconstructionTumor
from matplotlib import colors
from torchvision.transforms import Compose
sys.path.append('../fwi_ultrasound')
import transforms as T
import classifier
import time

plt.rcParams.update({'figure.max_open_warning': 0})

root = Tk()
# root = tk.Toplevel()
root.attributes('-zoomed', True)
# w, h = root.winfo_screenwidth(), root.winfo_screenheight()
# root.geometry('%dx%d+0+0' % (w, h))
root.title("Ultrasound Computed Tomography GUI App")
canvas = tk.Canvas(root, height=1900, width=1900)
canvas.pack()

def restart():
    screen.delete('1.0', END)
    canvas1.delete("all")
    canvas2.delete("all")
    canvas3.delete(canv_img3)
    canvas4.delete(canv_img4)
    
  
menubar = Menu(root, font=("Times", "13"), selectcolor="gray80")
filemenu = Menu(menubar, tearoff=0)
newImg = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/exp/fcnvmb/images/new.png').resize((20, 20), Image.ANTIALIAS))
filemenu.add_command(label=" New", font = ("helvetica", 14), image=newImg, compound=LEFT, command=())
openImg = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/exp/fcnvmb/images/open.png').resize((20, 20), Image.ANTIALIAS))
filemenu.add_command(label=" Open...", font = ("helvetica", 14), image=openImg, compound=LEFT, command=())
saveImg = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/exp/fcnvmb/images/save.png').resize((20, 20), Image.ANTIALIAS))
filemenu.add_command(label=" Save", font = ("helvetica", 14), image=saveImg, compound=LEFT,  command=())
filemenu.add_separator()
resImg = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/exp/fcnvmb/images/restart.png').resize((20, 20), Image.ANTIALIAS))
filemenu.add_command(label=" Restart", font = ("helvetica", 14), image=resImg, compound=LEFT, command=restart)
exitImg = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/exp/fcnvmb/images/close.png').resize((20, 20), Image.ANTIALIAS))
filemenu.add_command(label=" Exit", font = ("helvetica", 14), image=exitImg, compound=LEFT, command=root.destroy)
menubar.add_cascade(label="File", menu=filemenu)


frame1 = customtkinter.CTkFrame(master=root, width=480, height=450, fg_color = "gray80", corner_radius=15, border_width=1, border_color="gray")
frame1.place(x=2, y=287)
frame1.pack_propagate(0)
acoustic_label = customtkinter.CTkLabel(master=frame1, text="Ultrasound Data", text_font=("helvetica", 13), fg_color="gray70", corner_radius=8)
acoustic_label.pack(pady=5)
canvas1 = tk.Canvas(frame1, bg="gray80", bd=1, height=400, width=470, highlightbackground="gray80")
canvas1.pack()

frame2 = customtkinter.CTkFrame(master=root, width=678, height=450, fg_color = "gray80", corner_radius=15, border_width=1, border_color="gray")
frame2.place(x=484, y=287)
frame2.pack_propagate(0)
encode_label = customtkinter.CTkLabel(master=frame2, text="Reconstruction", text_font=("helvetica", 13), fg_color="gray70", corner_radius=8)
encode_label.pack(pady=5)
canvas3 = tk.Canvas(frame2, bg="gray80", bd=1, height=400, width=600, highlightbackground="gray80")
canvas3.pack()

frame3 = customtkinter.CTkFrame(master=root, width=750, height=540, fg_color = "gray80", corner_radius=15, border_width=1, border_color="gray")
frame3.place(x=1167, y=197)
frame3.pack_propagate(0)
task_label = customtkinter.CTkLabel(master=frame3, text="Task-Based Reconstruction", text_font=("helvetica", 13), fg_color="gray70", corner_radius=8)
task_label.pack(pady=5)
canvas2 = tk.Canvas(frame3, bg="gray80", bd=1, height=520, width=700, highlightbackground="gray80")
canvas2.pack()

frame4 = customtkinter.CTkFrame(master=root, width=750, height=257, fg_color = "gray80", corner_radius=15, border_width=1, border_color="gray")
frame4.place(x=1165, y=739)
frame4.pack_propagate(0)
class_label = customtkinter.CTkLabel(master=frame4, text="Task-based Tumor", text_font=("helvetica", 13), fg_color="gray70", corner_radius=8)
class_label.pack(pady=5)
canvas4 = tk.Canvas(frame4, bg="gray80", bd=1, height=207, width=700, highlightbackground="gray80")
canvas4.pack()


frame5 = customtkinter.CTkFrame(master=root, width=480, height=252, fg_color = "gray80", corner_radius=15, border_width=1, border_color="gray")
frame5.place(x=2, y=739)
frame5.pack_propagate(0)
out_screen = customtkinter.CTkLabel(master=frame5, text="Output Display", text_font=("helvetica", 13), fg_color="gray70", corner_radius=8)
out_screen.pack(pady=3)

frame6 = customtkinter.CTkFrame(master=root, width=675, height=257, fg_color = "gray80", corner_radius=15, border_width=1, border_color="gray")
frame6.place(x=485, y=739)
frame6.pack_propagate(0)
recons_tumor_label = customtkinter.CTkLabel(master=frame6, text="Reconstruction Tumor", text_font=("helvetica", 13), fg_color="gray70", corner_radius=8)
recons_tumor_label.pack(pady=5)
canvas5 = tk.Canvas(frame6, bg="gray80", bd=1, height=207, width=600, highlightbackground="gray80")
canvas5.pack()

acoustic_labelFrame = LabelFrame(root, text=" Ultrasound Measurement ", font=("helvetica", 13), labelanchor="n", height=280, width=480, bg="gray80", bd=4, relief=GROOVE)
acoustic_labelFrame.place(x=3, y=7)

task_labelFrame = LabelFrame(root, text=" Task-Based/Tumor Prediction ", font=("helvetica", 13), labelanchor="n", height=190, width=750, bg="gray80", bd=4, relief=GROOVE)
task_labelFrame.place(x=1167, y=7)

encode_labelFrame = LabelFrame(root, text=" Reconstruction/Tumor Prediction ", font=("helvetica", 13), labelanchor="n", height=280, width=683, bg="gray80", bd=4, relief=GROOVE)
encode_labelFrame.place(x=483, y=7)

# vel_labelFrame = LabelFrame(root, text=" Velocity Locations ", font=("helvetica", 13), labelanchor="n", height=190, width=472, bg="gray80", bd=4, relief=GROOVE)
# vel_labelFrame.place(x=1448, y=7)

# data_labelFrame = LabelFrame(root, text=" ...... ", font=("helvetica", 13), labelanchor="n", height=258, width=480, bg="gray80", bd=4, relief=GROOVE)
# data_labelFrame.place(x=2, y=739)

unm = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/exp/fcnvmb/images/unm.PNG').resize((80, 70), Image.ANTIALIAS))
lanl = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/exp/fcnvmb/images/lanl.PNG').resize((80, 70), Image.ANTIALIAS))

def help_menu():   
    root_new = tk.Toplevel()
    canvas = tk.Canvas(root_new, height=260, width=500)
    infoscreen = Text(root_new, bd=2, wrap="word")
    infoscreen.tag_configure("center", justify='center')
    infoscreen.pack(side=LEFT, fill=BOTH)
    infoscreen.pack(side=RIGHT, fill=BOTH)
    infoscreen.configure(font=("helvetica", 12, "bold"))
    infoscreen.insert(INSERT, "Designed by:" + "\n")
    infoscreen.insert(INSERT, " " + "\n")
    infoscreen.insert(INSERT, "Daniel Manu, Youzuo Lin, Xiang Sun" + '\n')
    infoscreen.insert(INSERT, " " + "\n")
    infoscreen.insert(INSERT, "University of New Mexico, Los Alamos National Laboratory" + '\n')
    infoscreen.insert(INSERT, " " + "\n")
    infoscreen.tag_add("center", "1.0", "end")
    infoscreen.image_create(tk.END, image = unm)
    infoscreen.image_create(tk.END, image = lanl)
    infoscreen.insert(INSERT, " " + "\n")
    infoscreen.insert(INSERT, " " + "\n")
    infoscreen.insert(INSERT,"\u00A9 2022 Copyright \u00AE\u2122" + "\n")
    infoscreen.insert(INSERT, " " + "\n")
    infoscreen.insert(INSERT, "Daniel Manu (Ph.D. Candidate in University of New Mexico)" + '\n')
    infoscreen.insert(INSERT, " " + "\n")
    # infoscreen.insert(INSERT, "Zhirun Li (Ph.D. Student in University of New Mexico)" + '\n')
    # infoscreen.insert(INSERT, " " + "\n")
    infoscreen.insert(INSERT, "Dr. Youzuo Lin (Staff Scientist at Los Alamos National Laboratory)" + '\n')
    infoscreen.insert(INSERT, " " + "\n")
    infoscreen.insert(INSERT, "Dr. Xiang Sun (Assistant Professor in University of New Mexico)" + '\n')
    infoscreen.insert(INSERT, " " + "\n")
    infoscreen.config(state=DISABLED)


orig_acous = (400, 410)
orig_recons = (900, 410)
orig_recons_tumor = (1100, 200)
orig_tumor = (950, 200)
orig_task = (1200, 470)

def AcousticImage():
    global acous, Acous, receive_button, canv_img1, orig_acous
    Acous = Image.open('/home/pi/Desktop/USCT/results/Ultrasound.png')
    acous = ImageTk.PhotoImage(Acous.resize(orig_acous, Image.ANTIALIAS))
    canv_img1 = canvas1.create_image(235, 180, image=acous, anchor=CENTER)

def ReconsImage():
    global recons, Recons, reconst_button, canv_img3, orig_recons
    Recons = Image.open('/home/pi/Desktop/USCT/results/Recons.png')
    recons = ImageTk.PhotoImage(Recons.resize(orig_recons, Image.ANTIALIAS))
    canv_img3 = canvas3.create_image(290, 190, image=recons, anchor=CENTER)

def TumorImage():
    global tumor, Tumor, tumor_button, canv_img4, orig_tumor
    Tumor = Image.open('/home/pi/Desktop/USCT/results/TaskTumor.png')
    tumor = ImageTk.PhotoImage(Tumor.resize(orig_tumor, Image.ANTIALIAS))
    canv_img4 = canvas4.create_image(335, 100, image=tumor, anchor=CENTER)

def ReconsTumorImage():
    global rec_tumor, Rec_tumor, recons_tumor_button, canv_img5, orig_recons_tumor
    Rec_tumor = Image.open('/home/pi/Desktop/USCT/results/ReconsTumor.png')
    rec_tumor = ImageTk.PhotoImage(Rec_tumor.resize(orig_recons_tumor, Image.ANTIALIAS))
    canv_img5 = canvas5.create_image(285, 100, image=rec_tumor, anchor=CENTER)

def TaskImage():
    global task, Task, task_button, canv_img2, orig_task
    Task = Image.open('/home/pi/Desktop/USCT/results/TaskBased.png')
    task = ImageTk.PhotoImage(Task.resize(orig_task, Image.ANTIALIAS))
    canv_img2 = canvas2.create_image(330, 230, image=task, anchor=CENTER)

def zoom_acoust(zoom):
    global acousImg, orig_acous
    zoom = float(zoom)
    zoom = min(max(zoom, 0.1), 5.0)
    newsize = (int(orig_acous[0]* zoom), 
                int(orig_acous[1]*zoom))
    scaledacous = Acous.resize(newsize, Image.LINEAR)
    acousImg = ImageTk.PhotoImage(scaledacous)
    canvas1.itemconfig(canv_img1, image=acousImg)

def zoom_recons(zoom):
    global reconsImg, orig_recons
    zoom = float(zoom)
    zoom = min(max(zoom, 0.1), 5.0)
    newsize = (int(orig_recons[0]* zoom), 
                int(orig_recons[1]*zoom))
    scaledrecons = Recons.resize(newsize, Image.LINEAR)
    reconsImg = ImageTk.PhotoImage(scaledrecons)
    canvas3.itemconfig(canv_img3, image=reconsImg)

def zoom_task(zoom):
    global taskImg, orig_task
    zoom = float(zoom)
    zoom = min(max(zoom, 0.1), 5.0)
    newsize = (int(orig_task[0]* zoom), 
                int(orig_task[1]*zoom))
    scaledtask = Task.resize(newsize, Image.LINEAR)
    taskImg = ImageTk.PhotoImage(scaledtask)
    canvas2.itemconfig(canv_img2, image=taskImg)

def zoom_recons_tumor(zoom):
    global reconstumorImg, orig_recons_tumor
    zoom = float(zoom)
    zoom = min(max(zoom, 0.1), 5.0)
    newsize = (int(orig_recons_tumor[0]* zoom), 
                int(orig_recons_tumor[1]*zoom))
    scaledreconstumor = Rec_tumor.resize(newsize, Image.LINEAR)
    reconstumorImg = ImageTk.PhotoImage(scaledreconstumor)
    canvas5.itemconfig(canv_img5, image=reconstumorImg)

def zoom_tumor(zoom):
    global tumorImg, orig_tumor
    zoom = float(zoom)
    zoom = min(max(zoom, 0.1), 5.0)
    newsize = (int(orig_tumor[0]* zoom), 
                int(orig_tumor[1]*zoom))
    scaledtumor = Tumor.resize(newsize, Image.LINEAR)
    tumorImg = ImageTk.PhotoImage(scaledtumor)
    canvas4.itemconfig(canv_img4, image=tumorImg)

# Create a dictionary to map canvases to their move functions
canvas_move_functions = {}

# Create a dictionary to store the selected item for each canvas
selected_items = {}

# Function to handle button press event
def move_from(event, canvas):
    canvas.scan_mark(event.x, event.y)
    # Set the canvas as active
    canvas_move_functions[canvas] = canvas

# Function to handle mouse motion event
def move_to(event, canvas):
    active_canvas = canvas_move_functions[canvas]
    if active_canvas:
        active_canvas.scan_dragto(event.x, event.y, gain=1)

# Bind mouse events to the canvases
canvas1.bind('<ButtonPress-1>', lambda event, c=canvas1: move_from(event, c))
canvas1.bind('<B1-Motion>', lambda event, c=canvas1: move_to(event, c))

canvas2.bind('<ButtonPress-1>', lambda event, c=canvas2: move_from(event, c))
canvas2.bind('<B1-Motion>', lambda event, c=canvas2: move_to(event, c))

canvas3.bind('<ButtonPress-1>', lambda event, c=canvas3: move_from(event, c))
canvas3.bind('<B1-Motion>', lambda event, c=canvas3: move_to(event, c))

canvas4.bind('<ButtonPress-1>', lambda event, c=canvas4: move_from(event, c))
canvas4.bind('<B1-Motion>', lambda event, c=canvas4: move_to(event, c))

canvas5.bind('<ButtonPress-1>', lambda event, c=canvas5: move_from(event, c))
canvas5.bind('<B1-Motion>', lambda event, c=canvas5: move_to(event, c))

active_canvas = None

# def move_from_1(event):
#     canvas1.scan_mark(event.x, event.y)

# def move_to_1(event):
#     canvas1.scan_dragto(event.x, event.y, gain=1)

# def move_from_2(event):
#     canvas2.scan_mark(event.x, event.y)

# def move_to_2(event):
#     canvas2.scan_dragto(event.x, event.y, gain=1)

# canvas1.bind('<ButtonPress-1>', move_from)
# canvas1.bind('<B1-Motion>', move_to)

# canvas2.bind('<ButtonPress-1>', move_from)
# canvas2.bind('<B1-Motion>', move_to)

def TaskPred():
    TaskBased()
    TaskImage()

def ReceiveCall():
    orig_acoustic()
    AcousticImage()

def ReconsPred():
    Reconstruct()
    ReconsImage()

def ReconsTumorPred():
    ReconstructTumor()
    ReconsTumorImage()

def TumorPred():
    TaskTumor()
    TumorImage()

# def zoom_acoustic(zoom=0):
#     zoom_acoust(zoom)

# def zoom_encoder(zoom=0):
#     zoom_recons(zoom)

# def zoom_tumor_pred(zoom=0):
#     zoom_tumor(zoom)

# def zoom_classifier(zoom=0):
#     zoom_class(zoom)

# def zoom_task_based(zoom=0):
#     zoom_task(zoom)

img1 = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/USCT/images/zoom1.png').resize((25, 25), Image.ANTIALIAS))
img2 = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/USCT/images/zoom2.png').resize((25, 25), Image.ANTIALIAS))
img3 = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/USCT/images/zoom4.png').resize((30, 30), Image.ANTIALIAS))
img4 = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/USCT/images/zoom5.png').resize((30, 30), Image.ANTIALIAS))
img5 = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/USCT/images/data-receive.png').resize((40, 30), Image.ANTIALIAS))
img6 = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/USCT/images/task.jpg').resize((40, 30), Image.ANTIALIAS))
img7 = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/USCT/images/recons.png').resize((40, 30), Image.ANTIALIAS))
img8 = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/USCT/images/tumor.png').resize((40, 30), Image.ANTIALIAS))
img9 = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/USCT/images/noise.png').resize((100, 30), Image.ANTIALIAS))
img10 = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/USCT/images/zoom3.png').resize((25, 25), Image.ANTIALIAS))
img11 = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/USCT/images/class.png').resize((40, 30), Image.ANTIALIAS))

acous_scaler_label = customtkinter.CTkLabel(master=root, text="Zoom Ultrasound Data ", text_font=("helvetica", 13), width=220, image=img1, compound=RIGHT,  
                                        fg_color="gray80", corner_radius=1).place(x=135, y=110)

task_scaler_label = customtkinter.CTkLabel(master=root, text="Zoom Task-based Reconstruction ", text_font=("helvetica", 13), image=img2, compound=RIGHT,  
                                        fg_color="gray80", corner_radius=1).place(x=1230, y=110)

tumor_scaler_label = customtkinter.CTkLabel(master=root, text="Zoom Task-based Tumor ", text_font=("helvetica", 13), image=img10, compound=RIGHT,  
                                        fg_color="gray80", corner_radius=1).place(x=1605, y=110)

recons_scaler_label = customtkinter.CTkLabel(master=root, text="Zoom Reconstruction", text_font=("helvetica", 13), image=img3, compound=RIGHT,  
                                        fg_color="gray80", corner_radius=1).place(x=865, y=100)

recons_tumor_label = customtkinter.CTkLabel(master=root, text="Zoom Reconstruction Tumor", text_font=("helvetica", 13), image=img4, compound=RIGHT,  
                                        fg_color="gray80", corner_radius=1).place(x=840, y=190)

var1 = StringVar()
acous_scale = tk.Scale(root, variable=var1, font=("helvetica", 11), orient='horizontal', bg = "gray80", bd=1, repeatdelay=1000000000, 
                        from_=1, to=5, length=200, resolution=1, command=zoom_acoust)
acous_scale.place(x=141, y=140)

var2 = StringVar()
task_scale = tk.Scale(root, variable=var2, font=("helvetica", 11), orient='horizontal', bg = "gray80", bd=1, repeatdelay=1000000000, 
                        from_=1, to=5, length=200, resolution=1, command=zoom_task)
task_scale.place(x=1280, y=140) 

var3 = StringVar()
tumor_scale = tk.Scale(root, variable=var3, font=("helvetica", 11), orient='horizontal', bg = "gray80", bd=1, repeatdelay=1000000000, 
                        from_=1, to=5, length=200, resolution=1, command=zoom_tumor)
tumor_scale.place(x=1615, y=140) 

var4 = StringVar()
recons_scale = tk.Scale(root, variable=var4, font=("helvetica", 11), orient='horizontal', bg = "gray80", bd=1, repeatdelay=1000000000, 
                        from_=1, to=5, length=200, resolution=1, command=zoom_recons)
recons_scale.place(x=870, y=135) 

var5 = StringVar()
recons_tumor_scale = tk.Scale(root, variable=var5, font=("helvetica", 11), orient='horizontal', bg = "gray80", bd=1, repeatdelay=1000000000, 
                        from_=1, to=5, length=200, resolution=1, command=zoom_recons_tumor)
recons_tumor_scale.place(x=865, y=220) 

noise_scaler_label = customtkinter.CTkLabel(master=root, text=" Noise (dB) ", text_font=("helvetica", 13), width=100, image=img9, compound=RIGHT, 
                                    fg_color="gray80", corner_radius=1).place(x=548, y=100)

transform_data = None
label_min = 1.4
label_max = 1.6

# transform_valid_data = torchvision.transforms.Compose([
#     T.LogTransform(k=1),
#     T.MinMaxNormalize(T.log_transform(data_min, k=1), T.log_transform(data_max, k=1))
# ])

transform_valid_label = torchvision.transforms.Compose([
    T.MinMaxNormalize(label_min, label_max)
])

val_anno ='/home/pi/Desktop/USCT/test_data.txt'

if val_anno[-3:] == 'txt':
    dataset_valid = FWIDataset(
    val_anno,
    preload=True,
    sample_ratio=1,
    sources=2,
    file_size=1,
    transform_data=transform_data,
    transform_label=transform_valid_label
    )
else:
    dataset_valid = torch.load(val_anno)

dataloader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=False)


def open_phantom():

    global dataloader_valid

    filename = filedialog.askopenfilename(initialdir="/home/pi/Desktop/USCT/ultrasound", title="Select An Ultrasound Data", 
                               filetypes=(("numpy files", "*.npy"), ("all files", "*.*")))
    screen.insert(INSERT, " " + '\n')
    screen.insert(INSERT, " ***** Selected Ultrasound Data Loaded *****" + '\n')

    for data in filename:
        screen.insert(END, data, '\n')

    dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=1, shuffle=True)

# tumor_clf = classifier.TumorClassifier() #.to(dev)
# class_sd = torch.load('/home/pi/Desktop/USCT/models/Classifier/classifier.pth',map_location=torch.device('cpu') )['model']
# tumor_clf.load_state_dict(class_sd)
# tumor_clf.eval()

# if source == 1:
#     true_sos = np.load("/home/pi/Desktop/USCT/velocity_maps/data33.npy")
# elif source == 0:

# true_sos = np.load("/home/pi/Desktop/USCT/velocity_maps/data34.npy")

# true_sos = true_sos[0:1, 0:1, :, :]
# tumor = tumor_clf(torch.from_numpy(true_sos).float()).cpu().detach().numpy() > 0.5

def orig_acoustic():
    global noiseImg1, noiseImg2, noiseImg3, noiseImg4, noiseImg5, noiseImg6, noiseImg7, data, label_ten, source, label_noiseImg1, label_noiseImg2, label_noiseImg3, label_noiseImg4, label_noiseImg5, label_noiseImg6, label_noiseImg7

    import time 

    since = time.time()

    font2 = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 15,
        }
    font3 = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 18,
        }

    path = "/home/pi/Desktop/USCT/results"

    data, label_ten, source = iter(dataloader_valid).next()
    data_np = data.numpy() 
    label_np = np.load("/home/pi/Desktop/USCT/classic_test_results_rob.npy")
    label = label_np[0, 0, :, :]
    # tumor_np = np.load("/home/pi/Desktop/USCT/tumor_maps/dataset0.npy")
    # tumor = tumor_np[0, 0, :, :]

    if check_clean.get() == 1 and len(acous_source_num.get()) == 0: 
        
        clean_data = data_np.reshape(1, 64, 500, 256)

        img0 = clean_data[0, 0, :, :]

        data_min = 0.01 * img0.min()
        data_max = 0.01 * img0.max()

        PlotUltrasound(img0, data_min, data_max, f'{path}/Ultrasound.png')

    elif check_clean.get() == 1 and len(acous_source_num.get()) != 0:

        clean_data = data_np.reshape(1, 64, 500, 256)

        acous_chann = int(var10.get())

        img0 = clean_data[0, acous_chann, :, :]

        data_min = 0.01 * img0.min()
        data_max = 0.01 * img0.max()

        PlotUltrasound(img0, data_min, data_max, f'{path}/Ultrasound.png')

        
    if (scale.get()) == 0 and check_clean.get() == 0:

        # flatten images
        image = np.reshape(data_np, -1)
        label = np.reshape(label, -1)
        # tumor = np.reshape(tumor, -1)
        # print(tumor.min(), tumor.max())

        # set the target SNR
        target_SNR_dB = 0

        # compute power of image values
        image_power = image ** 2
        label_power = label ** 2
        # tumor_power = tumor ** 2
        # print("tumor power", tumor_power)

        # compute average image and convert to dB
        avg_image_power = np.mean(image_power)
        avg_image_dB = 10 * np.log10(avg_image_power)

        avg_label_power = np.mean(label_power)
        avg_label_dB = 10 * np.log10(avg_label_power)

        # avg_tumor_power = np.mean(tumor_power)
        # avg_tumor_dB = 10 * np.log10(avg_tumor_power)

        # compute noise and convert to  dB
        avg_noiseImg_dB = avg_image_dB - target_SNR_dB
        avg_noiseImg_power = 10 ** (avg_noiseImg_dB / 10)

        avg_labelnoiseImg_dB = avg_label_dB - target_SNR_dB
        avg_labelnoiseImg_power = 10 ** (avg_labelnoiseImg_dB / 10)

        # avg_tumornoiseImg_dB = avg_tumor_dB - target_SNR_dB
        # avg_tumornoiseImg_power = 10 ** (avg_tumornoiseImg_dB / 10)

        # generate sample of white noise
        mean_noise = 0
        noiseImg1 = np.random.normal(mean_noise, np.sqrt(avg_noiseImg_power), len(image_power))
        
        label_noiseImg1 = np.random.normal(mean_noise, np.sqrt(avg_labelnoiseImg_power), len(label_power))

        # tumor_noiseImg1 = np.random.normal(mean_noise, np.sqrt(avg_tumornoiseImg_power), len(label_power))
        # print(tumor_noiseImg1.min(), tumor_noiseImg1.max())

        image = image + noiseImg1

        image = image.reshape(1, 64, 500, 256)

        img1 = image[0, 0, :, :]

        data_min = 0.01 * img1.min()
        data_max = 0.01 * img1.max()

        PlotUltrasound(img1, data_min, data_max, f'{path}/Ultrasound.png')

    if (scale.get()) == 5 and check_clean.get() == 0:

        # flatten images
        image = np.reshape(data_np, -1)
        label =np.reshape(label, -1)

        # set the target SNR
        target_SNR_dB = 5

        # compute power of image values
        image_power = image ** 2
        label_power = label ** 2

        # compute average image and convert to dB
        avg_image_power = np.mean(image_power)
        avg_image_dB = 10 * np.log10(avg_image_power)

        avg_label_power = np.mean(label_power)
        avg_label_dB = 10 * np.log10(avg_label_power)

        # compute noise and convert to  dB
        avg_noiseImg_dB = avg_image_dB - target_SNR_dB
        avg_noiseImg_power = 10 ** (avg_noiseImg_dB / 10)

        avg_labelnoiseImg_dB = avg_label_dB - target_SNR_dB
        avg_labelnoiseImg_power = 10 ** (avg_labelnoiseImg_dB / 10)

        # generate sample of white noise
        mean_noise = 0
        noiseImg2 = np.random.normal(mean_noise, np.sqrt(avg_noiseImg_power), len(image_power))
        
        label_noiseImg2 = np.random.normal(mean_noise, np.sqrt(avg_labelnoiseImg_power), len(label_power))

        image = image + noiseImg2

        image = image.reshape(1, 64, 500, 256)

        img2 = image[0, 0, :, :]

        data_min = 0.01 * img2.min()
        data_max = 0.01 * img2.max()

        PlotUltrasound(img2, data_min, data_max, f'{path}/Ultrasound.png')

    if (scale.get()) == 10 and check_clean.get() == 0:

        # flatten images
        image = np.reshape(data_np, -1)
        label =np.reshape(label, -1)

        # set the target SNR
        target_SNR_dB = 10

        # compute power of image values
        image_power = image ** 2
        label_power = label ** 2

        # compute average image and convert to dB
        avg_image_power = np.mean(image_power)
        avg_image_dB = 10 * np.log10(avg_image_power)

        avg_label_power = np.mean(label_power)
        avg_label_dB = 10 * np.log10(avg_label_power)

        # compute noise and convert to  dB
        avg_noiseImg_dB = avg_image_dB - target_SNR_dB
        avg_noiseImg_power = 10 ** (avg_noiseImg_dB / 10)

        avg_labelnoiseImg_dB = avg_label_dB - target_SNR_dB
        avg_labelnoiseImg_power = 10 ** (avg_labelnoiseImg_dB / 10)

        # generate sample of white noise
        mean_noise = 0
        noiseImg3 = np.random.normal(mean_noise, np.sqrt(avg_noiseImg_power), len(image_power))
        
        label_noiseImg3 = np.random.normal(mean_noise, np.sqrt(avg_labelnoiseImg_power), len(label_power))

        image = image + noiseImg3

        image = image.reshape(1, 64, 500, 256)

        img3 = image[0, 0, :, :]

        data_min = 0.01 * img3.min()
        data_max = 0.01 * img3.max()

        PlotUltrasound(img3, data_min, data_max, f'{path}/Ultrasound.png')

    if (scale.get()) == 15 and check_clean.get() == 0:

        # flatten images
        image = np.reshape(data_np, -1)
        label =np.reshape(label, -1)

        # set the target SNR
        target_SNR_dB = 15

        # compute power of image values
        image_power = image ** 2
        label_power = label ** 2

        # compute average image and convert to dB
        avg_image_power = np.mean(image_power)
        avg_image_dB = 10 * np.log10(avg_image_power)

        avg_label_power = np.mean(label_power)
        avg_label_dB = 10 * np.log10(avg_label_power)

        # compute noise and convert to  dB
        avg_noiseImg_dB = avg_image_dB - target_SNR_dB
        avg_noiseImg_power = 10 ** (avg_noiseImg_dB / 10)

        avg_labelnoiseImg_dB = avg_label_dB - target_SNR_dB
        avg_labelnoiseImg_power = 10 ** (avg_labelnoiseImg_dB / 10)

        # generate sample of white noise
        mean_noise = 0
        noiseImg4 = np.random.normal(mean_noise, np.sqrt(avg_noiseImg_power), len(image_power))
        
        label_noiseImg4 = np.random.normal(mean_noise, np.sqrt(avg_labelnoiseImg_power), len(label_power))

        image = image + noiseImg4

        image = image.reshape(1, 64, 500, 256)

        img4 = image[0, 0, :, :]

        data_min = 0.01 * img4.min()
        data_max = 0.01 * img4.max()

        PlotUltrasound(img4, data_min, data_max, f'{path}/Ultrasound.png')

    if (scale.get()) == 20 and check_clean.get() == 0:

        # flatten images
        image = np.reshape(data_np, -1)
        label =np.reshape(label, -1)

        # set the target SNR
        target_SNR_dB = 20

        # compute power of image values
        image_power = image ** 2
        label_power = label ** 2

        # compute average image and convert to dB
        avg_image_power = np.mean(image_power)
        avg_image_dB = 10 * np.log10(avg_image_power)

        avg_label_power = np.mean(label_power)
        avg_label_dB = 10 * np.log10(avg_label_power)

        # compute noise and convert to  dB
        avg_noiseImg_dB = avg_image_dB - target_SNR_dB
        avg_noiseImg_power = 10 ** (avg_noiseImg_dB / 10)

        avg_labelnoiseImg_dB = avg_label_dB - target_SNR_dB
        avg_labelnoiseImg_power = 10 ** (avg_labelnoiseImg_dB / 10)

        # generate sample of white noise
        mean_noise = 0
        noiseImg5 = np.random.normal(mean_noise, np.sqrt(avg_noiseImg_power), len(image_power))
        
        label_noiseImg5 = np.random.normal(mean_noise, np.sqrt(avg_labelnoiseImg_power), len(label_power))

        image = image + noiseImg5

        image = image.reshape(1, 64, 500, 256)

        img5 = image[0, 0, :, :]

        data_min = 0.01 * img5.min()
        data_max = 0.01 * img5.max()

        PlotUltrasound(img5, data_min, data_max, f'{path}/Ultrasound.png')

    if (scale.get()) == 25 and check_clean.get() == 0:

        # flatten images
        image = np.reshape(data_np, -1)
        label =np.reshape(label, -1)

        # set the target SNR
        target_SNR_dB = 25

        # compute power of image values
        image_power = image ** 2
        label_power = label ** 2

        # compute average image and convert to dB
        avg_image_power = np.mean(image_power)
        avg_image_dB = 10 * np.log10(avg_image_power)

        avg_label_power = np.mean(label_power)
        avg_label_dB = 10 * np.log10(avg_label_power)

        # compute noise and convert to  dB
        avg_noiseImg_dB = avg_image_dB - target_SNR_dB
        avg_noiseImg_power = 10 ** (avg_noiseImg_dB / 10)

        avg_labelnoiseImg_dB = avg_label_dB - target_SNR_dB
        avg_labelnoiseImg_power = 10 ** (avg_labelnoiseImg_dB / 10)

        # generate sample of white noise
        mean_noise = 0
        noiseImg6 = np.random.normal(mean_noise, np.sqrt(avg_noiseImg_power), len(image_power))
        
        label_noiseImg6 = np.random.normal(mean_noise, np.sqrt(avg_labelnoiseImg_power), len(label_power))

        image = image + noiseImg6

        image = image.reshape(1, 64, 500, 256)

        img6 = image[0, 0, :, :]

        data_min = 0.01 * img6.min()
        data_max = 0.01 * img6.max()

        PlotUltrasound(img6, data_min, data_max, f'{path}/Ultrasound.png')

    if (scale.get()) == 30 and check_clean.get() == 0:

        # flatten images
        image = np.reshape(data_np, -1)
        label =np.reshape(label, -1)
        # tumor = np.reshape(tumor, -1)

        # set the target SNR
        target_SNR_dB = 30

        # compute power of image values
        image_power = image ** 2
        label_power = label ** 2
        # tumor_power = tumor ** 2

        # compute average image and convert to dB
        avg_image_power = np.mean(image_power)
        avg_image_dB = 10 * np.log10(avg_image_power)

        avg_label_power = np.mean(label_power)
        avg_label_dB = 10 * np.log10(avg_label_power)

        # avg_tumor_power = np.mean(tumor_power)
        # avg_tumor_dB = 10 * np.log10(avg_tumor_power)

        # compute noise and convert to  dB
        avg_noiseImg_dB = avg_image_dB - target_SNR_dB
        avg_noiseImg_power = 10 ** (avg_noiseImg_dB / 10)

        avg_labelnoiseImg_dB = avg_label_dB - target_SNR_dB
        avg_labelnoiseImg_power = 10 ** (avg_labelnoiseImg_dB / 10)

        # avg_tumornoiseImg_dB = avg_tumor_dB - target_SNR_dB
        # avg_tumornoiseImg_power = 10 ** (avg_tumornoiseImg_dB / 10)

        # generate sample of white noise
        mean_noise = 0
        noiseImg7 = np.random.normal(mean_noise, np.sqrt(avg_noiseImg_power), len(image_power))
        
        label_noiseImg7 = np.random.normal(mean_noise, np.sqrt(avg_labelnoiseImg_power), len(label_power))

        # tumor_noiseImg7 = np.random.normal(mean_noise, np.sqrt(avg_tumornoiseImg_power), len(label_power))
        # print(tumor_noiseImg7.min(), tumor_noiseImg7.max())

        image = image + noiseImg7

        image = image.reshape(1, 64, 500, 256)

        img7 = image[0, 0, :, :]

        data_min = 0.01 * img7.min()
        data_max = 0.01 * img7.max()

        PlotUltrasound(img7, data_min, data_max, f'{path}/Ultrasound.png')

    if len(acous_source_num.get()) != 0 and check_clean.get() == 0:

        if (scale.get()) == 0:

            # flatten images
            image = np.reshape(data_np, -1)

            image = image + noiseImg1

            image = image.reshape(1, 64, 500, 256)

            acous_chann = int(var10.get())

            img1 = image[0, acous_chann, :, :]

            data_min = 0.01 * img1.min()
            data_max = 0.01 * img1.max()

            PlotUltrasound(img1, data_min, data_max, f'{path}/Ultrasound.png')

        if (scale.get()) == 5:

            # flatten images
            image = np.reshape(data_np, -1)

            image = image + noiseImg2

            image = image.reshape(1, 64, 500, 256)

            acous_chann = int(var10.get())

            img2 = image[0, acous_chann, :, :]

            data_min = 0.01 * img2.min()
            data_max = 0.01 * img2.max()

            PlotUltrasound(img2, data_min, data_max, f'{path}/Ultrasound.png')

        if (scale.get()) == 10:

            # flatten images
            image = np.reshape(data_np, -1)

            image = image + noiseImg3

            image = image.reshape(1, 64, 500, 256)

            acous_chann = int(var10.get())

            img3 = image[0, acous_chann, :, :]

            data_min = 0.01 * img3.min()
            data_max = 0.01 * img3.max()

            PlotUltrasound(img3, data_min, data_max, f'{path}/Ultrasound.png')

        if (scale.get()) == 15:

            # flatten images
            image = np.reshape(data_np, -1)

            image = image + noiseImg4

            image = image.reshape(1, 64, 500, 256)

            acous_chann = int(var10.get())

            img4 = image[0, acous_chann, :, :]

            data_min = 0.01 * img4.min()
            data_max = 0.01 * img4.max()

            PlotUltrasound(img4, data_min, data_max, f'{path}/Ultrasound.png')

        if (scale.get()) == 20:

            # flatten images
            image = np.reshape(data_np, -1)

            image = image + noiseImg5

            image = image.reshape(1, 64, 500, 256)

            acous_chann = int(var10.get())

            img5 = image[0, acous_chann, :, :]

            data_min = 0.01 * img5.min()
            data_max = 0.01 * img5.max()

            PlotUltrasound(img5, data_min, data_max, f'{path}/Ultrasound.png')

        if (scale.get()) == 25:

            # flatten images
            image = np.reshape(data_np, -1)

            image = image + noiseImg6

            image = image.reshape(1, 64, 500, 256)

            acous_chann = int(var10.get())

            img6 = image[0, acous_chann, :, :]

            data_min = 0.01 * img6.min()
            data_max = 0.01 * img6.max()

            PlotUltrasound(img6, data_min, data_max, f'{path}/Ultrasound.png')

        if (scale.get()) == 30:

            # flatten images
            image = np.reshape(data_np, -1)

            image = image + noiseImg7

            image = image.reshape(1, 64, 500, 256)

            acous_chann = int(var10.get())

            img7 = image[0, acous_chann, :, :]

            data_min = 0.01 * img7.min()
            data_max = 0.01 * img7.max()

            PlotUltrasound(img7, data_min, data_max, f'{path}/Ultrasound.png')

    # if len(phantom_num.get()) != 0:

    #     if (scale.get()) == 0:

    #         # flatten images
    #         image = np.reshape(data_np, -1)

    #         image = image + noiseImg1

    #         image = image.reshape(5, 64, 500, 256)

    #         phant = int(var11.get())

    #         img1 = image[phant, 0, :, :]

    #         data_min = 0.01 * img1.min()
    #         data_max = 0.01 * img1.max()

    #         PlotUltrasound(img1, data_min, data_max, f'{path}/Ultrasound.png')

    #     if (scale.get()) == 5:

    #         # flatten images
    #         image = np.reshape(data_np, -1)

    #         image = image + noiseImg2

    #         image = image.reshape(5, 64, 500, 256)

    #         phant = int(var11.get())

    #         img2 = image[phant, 0, :, :]

    #         data_min = 0.01 * img2.min()
    #         data_max = 0.01 * img2.max()

    #         PlotUltrasound(img2, data_min, data_max, f'{path}/Ultrasound.png')

    #     if (scale.get()) == 10:

    #         # flatten images
    #         image = np.reshape(data_np, -1)

    #         image = image + noiseImg3

    #         image = image.reshape(5, 64, 500, 256)

    #         phant = int(var11.get())

    #         img3 = image[phant, 0, :, :]

    #         data_min = 0.01 * img3.min()
    #         data_max = 0.01 * img3.max()

    #         PlotUltrasound(img3, data_min, data_max, f'{path}/Ultrasound.png')

    #     if (scale.get()) == 15:

    #         # flatten images
    #         image = np.reshape(data_np, -1)

    #         image = image + noiseImg4

    #         image = image.reshape(5, 64, 500, 256)

    #         phant = int(var11.get())

    #         img4 = image[phant, 0, :, :]

    #         data_min = 0.01 * img4.min()
    #         data_max = 0.01 * img4.max()

    #         PlotUltrasound(img4, data_min, data_max, f'{path}/Ultrasound.png')

    #     if (scale.get()) == 20:

    #         # flatten images
    #         image = np.reshape(data_np, -1)

    #         image = image + noiseImg5

    #         image = image.reshape(5, 64, 500, 256)

    #         phant = int(var11.get())

    #         img5 = image[phant, 0, :, :]

    #         data_min = 0.01 * img5.min()
    #         data_max = 0.01 * img5.max()

    #         PlotUltrasound(img5, data_min, data_max, f'{path}/Ultrasound.png')

    #     if (scale.get()) == 25:

    #         # flatten images
    #         image = np.reshape(data_np, -1)

    #         image = image + noiseImg6

    #         image = image.reshape(5, 64, 500, 256)

    #         phant = int(var11.get())

    #         img6 = image[phant, 0, :, :]

    #         data_min = 0.01 * img6.min()
    #         data_max = 0.01 * img6.max()

    #         PlotUltrasound(img6, data_min, data_max, f'{path}/Ultrasound.png')

    #     if (scale.get()) == 30:

    #         # flatten images
    #         image = np.reshape(data_np, -1)

    #         image = image + noiseImg7

    #         image = image.reshape(5, 64, 500, 256)

    #         phant = int(var11.get())

    #         img7 = image[phant, 0, :, :]

    #         data_min = 0.01 * img7.min()
    #         data_max = 0.01 * img7.max()

    #         PlotUltrasound(img7, data_min, data_max, f'{path}/Ultrasound.png')


    # if len(acous_source_num.get()) != 0 and len(phantom_num.get()) != 0:

    #     if (scale.get()) == 0:

    #         # flatten images
    #         image = np.reshape(data_np, -1)

    #         image = image + noiseImg1

    #         image = image.reshape(5, 64, 500, 256)

    #         acous_chann = int(var10.get())
    #         phant = int(var11.get())

    #         img1 = image[phant, acous_chann, :, :]

    #         data_min = 0.01 * img1.min()
    #         data_max = 0.01 * img1.max()

    #         PlotUltrasound(img1, data_min, data_max, f'{path}/Ultrasound.png')

    #     if (scale.get()) == 5:

    #         # flatten images
    #         image = np.reshape(data_np, -1)

    #         image = image + noiseImg2

    #         image = image.reshape(5, 64, 500, 256)

    #         acous_chann = int(var10.get())
    #         phant = int(var11.get())

    #         img2 = image[phant, acous_chann, :, :]

    #         data_min = 0.01 * img2.min()
    #         data_max = 0.01 * img2.max()

    #         PlotUltrasound(img2, data_min, data_max, f'{path}/Ultrasound.png')

    #     if (scale.get()) == 10:

    #         # flatten images
    #         image = np.reshape(data_np, -1)

    #         image = image + noiseImg3

    #         image = image.reshape(5, 64, 500, 256)

    #         acous_chann = int(var10.get())
    #         phant = int(var11.get())

    #         img3 = image[phant, acous_chann, :, :]

    #         data_min = 0.01 * img3.min()
    #         data_max = 0.01 * img3.max()

    #         PlotUltrasound(img3, data_min, data_max, f'{path}/Ultrasound.png')

    #     if (scale.get()) == 15:

    #         # flatten images
    #         image = np.reshape(data_np, -1)

    #         image = image + noiseImg4

    #         image = image.reshape(5, 64, 500, 256)

    #         acous_chann = int(var10.get())
    #         phant = int(var11.get())

    #         img4 = image[phant, acous_chann, :, :]

    #         data_min = 0.01 * img4.min()
    #         data_max = 0.01 * img4.max()

    #         PlotUltrasound(img4, data_min, data_max, f'{path}/Ultrasound.png')

    #     if (scale.get()) == 20:

    #         # flatten images
    #         image = np.reshape(data_np, -1)

    #         image = image + noiseImg5

    #         image = image.reshape(5, 64, 500, 256)

    #         acous_chann = int(var10.get())
    #         phant = int(var11.get())

    #         img5 = image[phant, acous_chann, :, :]

    #         data_min = 0.01 * img5.min()
    #         data_max = 0.01 * img5.max()

    #         PlotUltrasound(img5, data_min, data_max, f'{path}/Ultrasound.png')

    #     if (scale.get()) == 25:

    #         # flatten images
    #         image = np.reshape(data_np, -1)

    #         image = image + noiseImg6

    #         image = image.reshape(5, 64, 500, 256)

    #         acous_chann = int(var10.get())
    #         phant = int(var11.get())

    #         img6 = image[phant, acous_chann, :, :]

    #         data_min = 0.01 * img6.min()
    #         data_max = 0.01 * img6.max()

    #         PlotUltrasound(img6, data_min, data_max, f'{path}/Ultrasound.png')

    #     if (scale.get()) == 30:

    #         # flatten images
    #         image = np.reshape(data_np, -1)

    #         image = image + noiseImg7

    #         image = image.reshape(5, 64, 500, 256)

    #         acous_chann = int(var10.get())
    #         phant = int(var11.get())

    #         img7 = image[phant, acous_chann, :, :]

    #         data_min = 0.01 * img7.min()
    #         data_max = 0.01 * img7.max()

    #         PlotUltrasound(img7, data_min, data_max, f'{path}/Ultrasound.png')

    # Record the consuming time
    time_elapsed = time.time() - since
    time_elapsed = round(time_elapsed)
    time = f"The ultrasound data is recieved in {time_elapsed} second(s)"
    screen.insert(INSERT, " " + "\n")
    screen.insert(END, time + '\n')

def TaskBased(scale_value=0):
    global pred0, pred1, pred2, pred3, pred4, pred5, pred6, pred7

    import time

    since = time.time()

    #set arguments
    path = '/home/pi/Desktop/USCT/results'
    os.makedirs(path, exist_ok=True)
    data_np = data.numpy()
    
    with torch.no_grad():

        model = network.model_dict['FCN4_Deep_Resize_Enc'](upsample_mode='nearest')
        sd = torch.load('/home/pi/Desktop/USCT/models/TaskBased/quant_model.pth', map_location=torch.device('cpu'))['model']
        model.encoder.copy_(torch.load('/home/pi/Desktop/USCT/models/TaskBased/quant_model.pth',map_location=torch.device('cpu') )['encoder'])
        model.load_state_dict(sd)
        model.eval()

        classf = classifier.TumorClassifier() #.to(dev)
        sdc = torch.load('/home/pi/Desktop/USCT/models/Classifier/classifier.pth',map_location=torch.device('cpu') )['model']
        classf.load_state_dict(sdc)
        classf.eval()

        if check_recons.get() == 0 and check_recons_tumor.get() == 0 and check_task.get() == 1 and check_tumor.get() == 0 or check_recons.get() == 1 and check_recons_tumor.get() == 1 and check_task.get() == 0 and check_tumor.get() == 0 or check_recons.get() == 1 and check_recons_tumor.get() == 0 and check_task.get() == 1 and check_tumor.get() == 0 or check_recons.get() == 1 and check_recons_tumor.get() == 0 and check_task.get() == 0 and check_tumor.get() == 1 or check_recons.get() == 0 and check_recons_tumor.get() == 1 and check_task.get() == 1 and check_tumor.get() == 0 or check_recons.get() == 0 and check_recons_tumor.get() == 1 and check_task.get() == 0 and check_tumor.get() == 1 or check_recons.get() == 0 and check_recons_tumor.get() == 0 and check_task.get() == 1 and check_tumor.get() == 1 or check_recons.get() == 1 and check_recons_tumor.get() == 1 and check_task.get() == 1 and check_tumor.get() == 0 or check_recons.get() == 1 and check_recons_tumor.get() == 0 and check_task.get() == 1 and check_tumor.get() == 1 or check_recons.get() == 0 and check_recons_tumor.get() == 1 and check_task.get() == 1 and check_tumor.get() == 1 or check_recons.get() == 1 and check_recons_tumor.get() == 1 and check_task.get() == 0 and check_tumor.get() == 1 or check_recons.get() == 1 and check_recons_tumor.get() == 1 and check_task.get() == 1 and check_tumor.get() == 1:

            if check_clean.get() == 1 and len(acous_source_num.get()) == 0: 
        
                clean_data = data_np.reshape(1, 64, 500, 256)

                image_small = clean_data[0:1, :, :, :]

                image = torch.from_numpy(image_small).float()

                pred0 = model(image)
                pred0 += 1
                pred0 *= (1.6 - 1.4) / 2
                pred0 += 1.4

                pred_np = pred0.cpu().detach().numpy()

                img0 = pred_np[0, 0, :, :]

                TaskBasedReconstruction(img0, f'{path}/TaskBased.png')

            elif check_clean.get() == 1 and len(acous_source_num.get()) != 0:

                clean_data = data_np.reshape(1, 64, 500, 256)

                image_small = clean_data[0:1, :, :, :]

                image = torch.from_numpy(image_small).float()

                pred0 = model(image)
                pred0 += 1
                pred0 *= (1.6 - 1.4) / 2
                pred0 += 1.4

                pred_np = pred0.cpu().detach().numpy()

                img0 = pred_np[0, 0, :, :]

                TaskBasedReconstruction(img0, f'{path}/TaskBased.png')
                
            if (scale.get()) == 0 and check_clean.get() == 0:

                # flatten images
                image = np.reshape(data_np, -1)

                image = image + noiseImg1

                image = image.reshape(1, 64, 500, 256)

                image_small = image[0:1, :, :, :]
                image = torch.from_numpy(image_small).float()

                pred1 = model(image)
                pred1 += 1
                pred1 *= (1.6 - 1.4) / 2
                pred1 += 1.4

                pred_np = pred1.cpu().detach().numpy()

                img1 = pred_np[0, 0, :, :]

                TaskBasedReconstruction(img1, f'{path}/TaskBased.png')

            if (scale.get()) == 5 and check_clean.get() == 0:

                # flatten images
                image = np.reshape(data_np, -1)

                image = image + noiseImg2

                image = image.reshape(1, 64, 500, 256)

                image_small = image[0:1, :, :, :]
                image = torch.from_numpy(image_small).float()

                pred2 = model(image)
                pred2 += 1
                pred2 *= (1.6 - 1.4) / 2
                pred2 += 1.4

                pred_np = pred2.cpu().detach().numpy()

                img2 = pred_np[0, 0, :, :]

                TaskBasedReconstruction(img2, f'{path}/TaskBased.png')

            if (scale.get()) == 10 and check_clean.get() == 0:

                # flatten images
                image = np.reshape(data_np, -1)

                image = image + noiseImg3

                image = image.reshape(1, 64, 500, 256)

                image_small = image[0:1, :, :, :]
                image = torch.from_numpy(image_small).float()

                pred3 = model(image)
                pred3 += 1
                pred3 *= (1.6 - 1.4) / 2
                pred3 += 1.4

                pred_np = pred3.cpu().detach().numpy()

                img3 = pred_np[0, 0, :, :]

                TaskBasedReconstruction(img3, f'{path}/TaskBased.png')

            if (scale.get()) == 15 and check_clean.get() == 0:

                # flatten images
                image = np.reshape(data_np, -1)

                image = image + noiseImg4

                image = image.reshape(1, 64, 500, 256)

                image_small = image[0:1, :, :, :]
                image = torch.from_numpy(image_small).float()

                pred4 = model(image)
                pred4 += 1
                pred4 *= (1.6 - 1.4) / 2
                pred4 += 1.4

                pred_np = pred4.cpu().detach().numpy()

                img4 = pred_np[0, 0, :, :]

                TaskBasedReconstruction(img4, f'{path}/TaskBased.png')

            if (scale.get()) == 20 and check_clean.get() == 0:

                # flatten images
                image = np.reshape(data_np, -1)

                image = image + noiseImg5

                image = image.reshape(1, 64, 500, 256)

                image_small = image[0:1, :, :, :]
                image = torch.from_numpy(image_small).float()

                pred5 = model(image)
                pred5 += 1
                pred5 *= (1.6 - 1.4) / 2
                pred5 += 1.4

                pred_np = pred5.cpu().detach().numpy()

                img5 = pred_np[0, 0, :, :]

                TaskBasedReconstruction(img5, f'{path}/TaskBased.png')

            if (scale.get()) == 25 and check_clean.get() == 0:

                # flatten images
                image = np.reshape(data_np, -1)

                image = image + noiseImg6

                image = image.reshape(1, 64, 500, 256)

                image_small = image[0:1, :, :, :]
                image = torch.from_numpy(image_small).float()

                pred6 = model(image)
                pred6 += 1
                pred6 *= (1.6 - 1.4) / 2
                pred6 += 1.4

                pred_np = pred6.cpu().detach().numpy()

                img6 = pred_np[0, 0, :, :]

                TaskBasedReconstruction(img6, f'{path}/TaskBased.png')

            if (scale.get()) == 30 and check_clean.get() == 0:

                # flatten images
                image = np.reshape(data_np, -1)

                image = image + noiseImg7

                image = image.reshape(1, 64, 500, 256)

                image_small = image[0:1, :, :, :]
                image = torch.from_numpy(image_small).float()

                pred7 = model(image)
                pred7 += 1
                pred7 *= (1.6 - 1.4) / 2
                pred7+= 1.4

                pred_np = pred7.cpu().detach().numpy()

                img7 = pred_np[0, 0, :, :]

                TaskBasedReconstruction(img7, f'{path}/TaskBased.png')

            if len(acous_source_num.get()) != 0 and check_clean.get() == 0:

                if (scale.get()) == 0:

                    # flatten images
                    image = np.reshape(data_np, -1)

                    image = image + noiseImg1

                    image = image.reshape(1, 64, 500, 256)

                    image_small = image[0:1, :, :, :]
                    image = torch.from_numpy(image_small).float()

                    pred1 = model(image)
                    pred1 += 1
                    pred1 *= (1.6 - 1.4) / 2
                    pred1 += 1.4

                    pred_np = pred1.cpu().detach().numpy()
                    acous_chann = int(var10.get())

                    img1 = pred_np[0, 0, :, :]

                    TaskBasedReconstruction(img1, f'{path}/TaskBased.png')

                if (scale.get()) == 5:

                    # flatten images
                    image = np.reshape(data_np, -1)

                    image = image + noiseImg2

                    image = image.reshape(1, 64, 500, 256)

                    image_small = image[0:1, :, :, :]
                    image = torch.from_numpy(image_small).float()

                    pred2 = model(image)
                    pred2 += 1
                    pred2 *= (1.6 - 1.4) / 2
                    pred2 += 1.4

                    pred_np = pred2.cpu().detach().numpy()
                    acous_chann = int(var10.get())

                    img2 = pred_np[0, 0, :, :]

                    TaskBasedReconstruction(img2, f'{path}/TaskBased.png')

                if (scale.get()) == 10:

                    # flatten images
                    image = np.reshape(data_np, -1)

                    image = image + noiseImg3

                    image = image.reshape(1, 64, 500, 256)

                    image_small = image[0:1, :, :, :]
                    image = torch.from_numpy(image_small).float()

                    pred3 = model(image)
                    pred3 += 1
                    pred3 *= (1.6 - 1.4) / 2
                    pred3 += 1.4

                    pred_np = pred3.cpu().detach().numpy()
                    acous_chann = int(var10.get())

                    img3 = pred_np[0, 0, :, :]

                    TaskBasedReconstruction(img3, f'{path}/TaskBased.png')

                if (scale.get()) == 15:

                    # flatten images
                    image = np.reshape(data_np, -1)

                    image = image + noiseImg4

                    image = image.reshape(1, 64, 500, 256)

                    image_small = image[0:1, :, :, :]
                    image = torch.from_numpy(image_small).float()

                    pred4 = model(image)
                    pred4 += 1
                    pred4 *= (1.6 - 1.4) / 2
                    pred4 += 1.4

                    pred_np = pred4.cpu().detach().numpy()
                    acous_chann = int(var10.get())

                    img4 = pred_np[0, 0, :, :]

                    TaskBasedReconstruction(img4, f'{path}/TaskBased.png')

                if (scale.get()) == 20:

                    # flatten images
                    image = np.reshape(data_np, -1)

                    image = image + noiseImg5

                    image = image.reshape(1, 64, 500, 256)

                    image_small = image[0:1, :, :, :]
                    image = torch.from_numpy(image_small).float()

                    pred5 = model(image)
                    pred5 += 1
                    pred5 *= (1.6 - 1.4) / 2
                    pred5 += 1.4

                    pred_np = pred5.cpu().detach().numpy()
                    acous_chann = int(var10.get())

                    img5 = pred_np[0, 0, :, :]

                    TaskBasedReconstruction(img5, f'{path}/TaskBased.png')

                if (scale.get()) == 25:

                    # flatten images
                    image = np.reshape(data_np, -1)

                    image = image + noiseImg6

                    image = image.reshape(1, 64, 500, 256)

                    image_small = image[0:1, :, :, :]
                    image = torch.from_numpy(image_small).float()

                    pred6 = model(image)
                    pred6 += 1
                    pred6 *= (1.6 - 1.4) / 2
                    pred6 += 1.4

                    pred_np = pred6.cpu().detach().numpy()
                    acous_chann = int(var10.get())

                    img6 = pred_np[0, 0, :, :]

                    TaskBasedReconstruction(img6, f'{path}/TaskBased.png')

                if (scale.get()) == 30:

                    # flatten images
                    image = np.reshape(data_np, -1)

                    image = image + noiseImg7

                    image = image.reshape(1, 64, 500, 256)

                    image_small = image[0:1, :, :, :]
                    image = torch.from_numpy(image_small).float()

                    pred7 = model(image)
                    pred7 += 1
                    pred7 *= (1.6 - 1.4) / 2
                    pred7 += 1.4

                    pred_np = pred7.cpu().detach().numpy()
                    acous_chann = int(var10.get())

                    img7 = pred_np[0, 0, :, :]

                    TaskBasedReconstruction(img7, f'{path}/TaskBased.png')

            time_elapsed = time.time() - since
            time_elapsed = round(time_elapsed)
            time = f"The time for task-based prediction is {time_elapsed} seconds"
            screen.insert(INSERT, " " + "\n")
            screen.insert(END, time + '\n')


def TaskTumor(scale_value=0):

    import time
        
    since = time.time()

    #set arguments
    path = '/home/pi/Desktop/USCT/results'
    os.makedirs(path, exist_ok=True)
    data_np = data.numpy()

    with torch.no_grad():

        model = network.model_dict['FCN4_Deep_Resize_Enc'](upsample_mode='nearest')
        sd = torch.load('/home/pi/Desktop/USCT/models/TaskBased/quant_model.pth', map_location=torch.device('cpu'))['model']
        model.encoder.copy_(torch.load('/home/pi/Desktop/USCT/models/TaskBased/quant_model.pth',map_location=torch.device('cpu') )['encoder'])
        model.load_state_dict(sd)
        model.eval()

        classf = classifier.TumorClassifier() #.to(dev)
        sdc = torch.load('/home/pi/Desktop/USCT/models/Classifier/classifier.pth',map_location=torch.device('cpu') )['model']
        classf.load_state_dict(sdc)
        classf.eval()

        if check_recons.get() == 0 and check_recons_tumor.get() == 0 and check_task.get() == 0 and check_tumor.get() == 1 or check_recons.get() == 1 and check_recons_tumor.get() == 1 and check_task.get() == 0 and check_tumor.get() == 0 or check_recons.get() == 1 and check_recons_tumor.get() == 0 and check_task.get() == 1 and check_tumor.get() == 0 or check_recons.get() == 1 and check_recons_tumor.get() == 0 and check_task.get() == 0 and check_tumor.get() == 1 or check_recons.get() == 0 and check_recons_tumor.get() == 1 and check_task.get() == 1 and check_tumor.get() == 0 or check_recons.get() == 0 and check_recons_tumor.get() == 1 and check_task.get() == 0 and check_tumor.get() == 1 or check_recons.get() == 0 and check_recons_tumor.get() == 0 and check_task.get() == 1 and check_tumor.get() == 1 or check_recons.get() == 1 and check_recons_tumor.get() == 1 and check_task.get() == 1 and check_tumor.get() == 0 or check_recons.get() == 1 and check_recons_tumor.get() == 0 and check_task.get() == 1 and check_tumor.get() == 1 or check_recons.get() == 0 and check_recons_tumor.get() == 1 and check_task.get() == 1 and check_tumor.get() == 1 or check_recons.get() == 1 and check_recons_tumor.get() == 1 and check_task.get() == 0 and check_tumor.get() == 1 or check_recons.get() == 1 and check_recons_tumor.get() == 1 and check_task.get() == 1 and check_tumor.get() == 1:
            
            tmaps =  classf(label_ten).cpu().detach().numpy() > 0.5

            tm_small = tmaps[0:1, :, :, :]

            if check_clean.get() == 1 and len(acous_source_num.get()) == 0:

                if check_task.get() == 0:

                    clean_data = data_np.reshape(1, 64, 500, 256)

                    image_small = clean_data[0:1, :, :, :]
                    image = torch.from_numpy(image_small).float()

                    pred = model(image)
                    pred += 1
                    pred *= (1.6 - 1.4) / 2
                    pred += 1.4

                    learned_tumort = classf(pred) > 0.02

                    learned_tumor = learned_tumort.cpu().detach()

                    img0 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                else:
                    learned_tumort = classf(pred0) > 0.02

                    learned_tumor = learned_tumort.cpu().detach()

                    img0 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                TaskBasedTumor(img0, f'{path}/TaskTumor.png')

            elif check_clean.get() == 1 and len(acous_source_num.get()) != 0:

                if check_task.get() == 0:

                    clean_data = data_np.reshape(1, 64, 500, 256)

                    image_small = clean_data[0:1, :, :, :]
                    image = torch.from_numpy(image_small).float()

                    pred = model(image)
                    pred += 1
                    pred *= (1.6 - 1.4) / 2
                    pred += 1.4

                    learned_tumort = classf(pred) > 0.02

                    learned_tumor = learned_tumort.cpu().detach()

                    img0 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                else:

                    learned_tumort = classf(pred0) > 0.02

                    learned_tumor = learned_tumort.cpu().detach()
                    
                    img0 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                TaskBasedTumor(img0, f'{path}/TaskTumor.png')

            if (scale.get()) == 0 and check_clean.get() == 0:

                if check_task.get() == 0:

                    # flatten images
                    image = np.reshape(data_np, -1)

                    image = image + noiseImg1

                    image = image.reshape(1, 64, 500, 256)

                    image_small = image[0:1, :, :, :]
                    image = torch.from_numpy(image_small).float()

                    pred = model(image)
                    pred += 1
                    pred *= (1.6 - 1.4) / 2
                    pred += 1.4

                    learned_tumort = classf(pred) > 0.02

                    learned_tumor = learned_tumort.cpu().detach()
                    

                    img1 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                else:

                    learned_tumort = classf(pred1) > 0.02

                    learned_tumor = learned_tumort.cpu().detach()
                    
                    img1 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                TaskBasedTumor(img1, f'{path}/TaskTumor.png')

            if (scale.get()) == 5 and check_clean.get() == 0:

                if check_task.get() == 0:

                    # flatten images
                    image = np.reshape(data_np, -1)

                    image = image + noiseImg2

                    image = image.reshape(1, 64, 500, 256)

                    image_small = image[0:1, :, :, :]
                    image = torch.from_numpy(image_small).float()

                    pred = model(image)
                    pred += 1
                    pred *= (1.6 - 1.4) / 2
                    pred += 1.4

                    learned_tumort = classf(pred) > 0.02

                    learned_tumor = learned_tumort.cpu().detach()

                    img2 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                else:

                    learned_tumort = classf(pred2) > 0.02

                    learned_tumor = learned_tumort.cpu().detach()

                    img2 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                TaskBasedTumor(img2, f'{path}/TaskTumor.png')

            if (scale.get()) == 10 and check_clean.get() == 0:

                if check_task.get() == 0:

                    # flatten images
                    image = np.reshape(data_np, -1)

                    image = image + noiseImg3

                    image = image.reshape(1, 64, 500, 256)

                    image_small = image[0:1, :, :, :]
                    image = torch.from_numpy(image_small).float()

                    pred = model(image)
                    pred += 1
                    pred *= (1.6 - 1.4) / 2
                    pred += 1.4

                    learned_tumort = classf(pred) > 0.02

                    learned_tumor = learned_tumort.cpu().detach()

                    img3 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                else:

                    learned_tumort = classf(pred3) > 0.02

                    learned_tumor = learned_tumort.cpu().detach()

                    img3 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                TaskBasedTumor(img3, f'{path}/TaskTumor.png')

            if (scale.get()) == 15 and check_clean.get() == 0:

                if check_task.get():

                    # flatten images
                    image = np.reshape(data_np, -1)

                    image = image + noiseImg4

                    image = image.reshape(1, 64, 500, 256)

                    image_small = image[0:1, :, :, :]
                    image = torch.from_numpy(image_small).float()

                    pred = model(image)
                    pred += 1
                    pred *= (1.6 - 1.4) / 2
                    pred += 1.4

                    learned_tumort = classf(pred) > 0.02

                    learned_tumor = learned_tumort.cpu().detach()

                    img4 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                else:

                    learned_tumort = classf(pred4) > 0.02

                    learned_tumor = learned_tumort.cpu().detach()

                    img4 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                TaskBasedTumor(img4, f'{path}/TaskTumor.png')


            if (scale.get()) == 20 and check_clean.get() == 0:

                if check_task.get() == 0:

                    # flatten images
                    image = np.reshape(data_np, -1)

                    image = image + noiseImg5

                    image = image.reshape(1, 64, 500, 256)

                    image_small = image[0:1, :, :, :]
                    image = torch.from_numpy(image_small).float()

                    pred = model(image)
                    pred += 1
                    pred *= (1.6 - 1.4) / 2
                    pred += 1.4

                    learned_tumort = classf(pred) > 0.02

                    learned_tumor = learned_tumort.cpu().detach()

                    img5 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                else:

                    learned_tumort = classf(pred5) > 0.02

                    learned_tumor = learned_tumort.cpu().detach()

                    img5 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                TaskBasedTumor(img5, f'{path}/TaskTumor.png')

            if (scale.get()) == 25 and check_clean.get() == 0:

                if check_task.get() == 0:

                    # flatten images
                    image = np.reshape(data_np, -1)

                    image = image + noiseImg6

                    image = image.reshape(1, 64, 500, 256)

                    image_small = image[0:1, :, :, :]
                    image = torch.from_numpy(image_small).float()

                    pred = model(image)
                    pred += 1
                    pred *= (1.6 - 1.4) / 2
                    pred += 1.4

                    learned_tumort = classf(pred) > 0.02

                    learned_tumor = learned_tumort.cpu().detach()

                    img6 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                else:

                    learned_tumort = classf(pred6) > 0.02

                    learned_tumor = learned_tumort.cpu().detach()

                    img6 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                TaskBasedTumor(img6, f'{path}/TaskTumor.png')

            if (scale.get()) == 30 and check_clean.get() == 0:

                if check_task.get() == 0:

                    # flatten images
                    image = np.reshape(data_np, -1)

                    image = image + noiseImg7

                    image = image.reshape(1, 64, 500, 256)

                    image_small = image[0:1, :, :, :]
                    image = torch.from_numpy(image_small).float()

                    pred = model(image)
                    pred += 1
                    pred *= (1.6 - 1.4) / 2
                    pred += 1.4

                    learned_tumort = classf(pred) > 0.02

                    learned_tumor = learned_tumort.cpu().detach()

                    img7 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                else:

                    learned_tumort = classf(pred7) > 0.02

                    learned_tumor = learned_tumort.cpu().detach()

                    img7 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                TaskBasedTumor(img7, f'{path}/TaskTumor.png')

            if len(acous_source_num.get()) != 0 and check_clean.get() == 0:

                if (scale.get()) == 0:
                    
                    if check_task == 0:

                        # flatten images
                        image = np.reshape(data_np, -1)

                        image = image + noiseImg1

                        image = image.reshape(1, 64, 500, 256)

                        image_small = image[0:1, :, :, :]
                        image = torch.from_numpy(image_small).float()

                        pred = model(image)
                        pred += 1
                        pred *= (1.6 - 1.4) / 2
                        pred += 1.4

                        learned_tumort = classf(pred) > 0.02

                        learned_tumor = learned_tumort.cpu().detach()

                        img1 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                    else:

                        learned_tumort = classf(pred1) > 0.02

                        learned_tumor = learned_tumort.cpu().detach()

                        img1 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                    TaskBasedTumor(img1, f'{path}/TaskTumor.png')

                if (scale.get()) == 5:
                    
                    if check_task == 0:

                        # flatten images
                        image = np.reshape(data_np, -1)

                        image = image + noiseImg2

                        image = image.reshape(1, 64, 500, 256)

                        image_small = image[0:1, :, :, :]
                        image = torch.from_numpy(image_small).float()

                        pred = model(image)
                        pred += 1
                        pred *= (1.6 - 1.4) / 2
                        pred += 1.4

                        learned_tumort = classf(pred) > 0.02

                        learned_tumor = learned_tumort.cpu().detach()

                        img2 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                    else:

                        learned_tumort = classf(pred2) > 0.02

                        learned_tumor = learned_tumort.cpu().detach()

                        img2 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                    TaskBasedTumor(img2, f'{path}/TaskTumor.png')

                if (scale.get()) == 10:

                    if check_task == 0:

                        # flatten images
                        image = np.reshape(data_np, -1)

                        image = image + noiseImg3

                        image = image.reshape(1, 64, 500, 256)

                        image_small = image[0:1, :, :, :]
                        image = torch.from_numpy(image_small).float()

                        pred = model(image)
                        pred += 1
                        pred *= (1.6 - 1.4) / 2
                        pred += 1.4

                        learned_tumort = classf(pred) > 0.02

                        learned_tumor = learned_tumort.cpu().detach()

                        img3 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                    else:

                        learned_tumort = classf(pred3) > 0.02

                        learned_tumor = learned_tumort.cpu().detach()

                        img3 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                    TaskBasedTumor(img3, f'{path}/TaskTumor.png')

                if (scale.get()) == 15:

                    if check_task == 0:

                        # flatten images
                        image = np.reshape(data_np, -1)

                        image = image + noiseImg4

                        image = image.reshape(1, 64, 500, 256)

                        image_small = image[0:1, :, :, :]
                        image = torch.from_numpy(image_small).float()

                        pred = model(image)
                        pred += 1
                        pred *= (1.6 - 1.4) / 2
                        pred += 1.4

                        learned_tumort = classf(pred) > 0.02

                        learned_tumor = learned_tumort.cpu().detach()

                        img4 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                    else:

                        learned_tumort = classf(pred4) > 0.02

                        learned_tumor = learned_tumort.cpu().detach()

                        img4 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                    TaskBasedTumor(img4, f'{path}/TaskTumor.png')

                if (scale.get()) == 20:

                    if check_task == 0:

                        # flatten images
                        image = np.reshape(data_np, -1)

                        image = image + noiseImg5

                        image = image.reshape(1, 64, 500, 256)

                        image_small = image[0:1, :, :, :]
                        image = torch.from_numpy(image_small).float()

                        pred = model(image)
                        pred += 1
                        pred *= (1.6 - 1.4) / 2
                        pred += 1.4

                        learned_tumort = classf(pred) > 0.02

                        learned_tumor = learned_tumort.cpu().detach()

                        img5 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                    else:

                        learned_tumort = classf(pred5) > 0.02

                        learned_tumor = learned_tumort.cpu().detach()

                        img5 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                    TaskBasedTumor(img5, f'{path}/TaskTumor.png')

                if (scale.get()) == 25:

                    if check_task == 0:

                        # flatten images
                        image = np.reshape(data_np, -1)

                        image = image + noiseImg6

                        image = image.reshape(1, 64, 500, 256)

                        image_small = image[0:1, :, :, :]
                        image = torch.from_numpy(image_small).float()

                        pred = model(image)
                        pred += 1
                        pred *= (1.6 - 1.4) / 2
                        pred += 1.4

                        learned_tumort = classf(pred) > 0.02

                        learned_tumor = learned_tumort.cpu().detach()

                        img6 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                    else:

                        learned_tumort = classf(pred6) > 0.02

                        learned_tumor = learned_tumort.cpu().detach()

                        img6 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                    TaskBasedTumor(img6, f'{path}/TaskTumor.png')

                if (scale.get()) == 30:

                    if check_task == 0:

                        # flatten images
                        image = np.reshape(data_np, -1)

                        image = image + noiseImg7

                        image = image.reshape(1, 64, 500, 256)

                        image_small = image[0:1, :, :, :]
                        image = torch.from_numpy(image_small).float()

                        pred = model(image)
                        pred += 1
                        pred *= (1.6 - 1.4) / 2
                        pred += 1.4

                        learned_tumort = classf(pred) > 0.02

                        learned_tumor = learned_tumort.cpu().detach()

                        img7 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                    else:

                        learned_tumort = classf(pred7) > 0.02

                        learned_tumor = learned_tumort.cpu().detach()

                        img7 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                    TaskBasedTumor(img7, f'{path}/TaskTumor.png')

            time_elapsed = time.time() - since
            time_elapsed = round(time_elapsed)
            time = f"The time for task-based tumor prediction is {time_elapsed} seconds"
            screen.insert(INSERT, " " + "\n")
            screen.insert(END, time + '\n')


def Reconstruct(scale_value=0):
    global pred10, pred11, pred12, pred13, pred14, pred15, pred16, pred17

    import time

    since = time.time()

    #set arguments
    path = '/home/pi/Desktop/USCT/results'
    os.makedirs(path, exist_ok=True)
    data_np = data.numpy()
    
    with torch.no_grad():

        model = network.model_dict['FCN4_Deep_Resize_Enc'](upsample_mode='nearest')
        sd = torch.load('/home/pi/Desktop/USCT/models/TaskBased/quant_gamma001.pth', map_location=torch.device('cpu'))['model']
        model.encoder.copy_(torch.load('/home/pi/Desktop/USCT/models/TaskBased/quant_gamma001.pth',map_location=torch.device('cpu') )['encoder'])
        model.load_state_dict(sd)
        model.eval()

        classf = classifier.TumorClassifier() #.to(dev)
        sdc = torch.load('/home/pi/Desktop/USCT/models/Classifier/classifier.pth',map_location=torch.device('cpu') )['model']
        classf.load_state_dict(sdc)
        classf.eval()

        if check_recons.get() == 1 and check_recons_tumor.get() == 0 and check_task.get() == 0 and check_tumor.get() == 0 or check_recons.get() == 1 and check_recons_tumor.get() == 1 and check_task.get() == 0 and check_tumor.get() == 0 or check_recons.get() == 1 and check_recons_tumor.get() == 0 and check_task.get() == 1 and check_tumor.get() == 0 or check_recons.get() == 1 and check_recons_tumor.get() == 0 and check_task.get() == 0 and check_tumor.get() == 1 or check_recons.get() == 0 and check_recons_tumor.get() == 1 and check_task.get() == 1 and check_tumor.get() == 0 or check_recons.get() == 0 and check_recons_tumor.get() == 1 and check_task.get() == 0 and check_tumor.get() == 1 or check_recons.get() == 0 and check_recons_tumor.get() == 0 and check_task.get() == 1 and check_tumor.get() == 1 or check_recons.get() == 1 and check_recons_tumor.get() == 1 and check_task.get() == 1 and check_tumor.get() == 0 or check_recons.get() == 1 and check_recons_tumor.get() == 0 and check_task.get() == 1 and check_tumor.get() == 1 or check_recons.get() == 0 and check_recons_tumor.get() == 1 and check_task.get() == 1 and check_tumor.get() == 1 or check_recons.get() == 1 and check_recons_tumor.get() == 1 and check_task.get() == 0 and check_tumor.get() == 1 or check_recons.get() == 1 and check_recons_tumor.get() == 1 and check_task.get() == 1 and check_tumor.get() == 1:

            if check_clean.get() == 1 and len(acous_source_num.get()) == 0: 
        
                clean_data = data_np.reshape(1, 64, 500, 256)

                image_small = clean_data[0:1, :, :, :]

                image = torch.from_numpy(image_small).float()

                pred10 = model(image)
                pred10 += 1
                pred10 *= (1.6 - 1.4) / 2
                pred10 += 1.4

                pred_np = pred10.cpu().detach().numpy()

                img10 = pred_np[0, 0, :, :]

                Reconstruction(img10, f'{path}/Recons.png')

            elif check_clean.get() == 1 and len(acous_source_num.get()) != 0:

                clean_data = data_np.reshape(1, 64, 500, 256)

                image_small = clean_data[0:1, :, :, :]

                image = torch.from_numpy(image_small).float()

                pred10 = model(image)
                pred10 += 1
                pred10 *= (1.6 - 1.4) / 2
                pred10 += 1.4

                pred_np = pred10.cpu().detach().numpy()

                img10 = pred_np[0, 0, :, :]

                Reconstruction(img10, f'{path}/Recons.png')
                
            if (scale.get()) == 0 and check_clean.get() == 0:

                # flatten images
                image = np.reshape(data_np, -1)

                image = image + noiseImg1

                image = image.reshape(1, 64, 500, 256)

                image_small = image[0:1, :, :, :]
                image = torch.from_numpy(image_small).float()

                pred11 = model(image)
                pred11 += 1
                pred11 *= (1.6 - 1.4) / 2
                pred11 += 1.4

                pred_np = pred11.cpu().detach().numpy()

                img11 = pred_np[0, 0, :, :]

                Reconstruction(img11, f'{path}/Recons.png')

            if (scale.get()) == 5 and check_clean.get() == 0:

                # flatten images
                image = np.reshape(data_np, -1)

                image = image + noiseImg2

                image = image.reshape(1, 64, 500, 256)

                image_small = image[0:1, :, :, :]
                image = torch.from_numpy(image_small).float()

                pred12 = model(image)
                pred12 += 1
                pred12 *= (1.6 - 1.4) / 2
                pred12 += 1.4

                pred_np = pred12.cpu().detach().numpy()

                img12 = pred_np[0, 0, :, :]

                Reconstruction(img12, f'{path}/Recons.png')

            if (scale.get()) == 10 and check_clean.get() == 0:

                # flatten images
                image = np.reshape(data_np, -1)

                image = image + noiseImg3

                image = image.reshape(1, 64, 500, 256)

                image_small = image[0:1, :, :, :]
                image = torch.from_numpy(image_small).float()

                pred13 = model(image)
                pred13 += 1
                pred13 *= (1.6 - 1.4) / 2
                pred13 += 1.4

                pred_np = pred13.cpu().detach().numpy()

                img13 = pred_np[0, 0, :, :]

                Reconstruction(img13, f'{path}/Recons.png')

            if (scale.get()) == 15 and check_clean.get() == 0:

                # flatten images
                image = np.reshape(data_np, -1)

                image = image + noiseImg4

                image = image.reshape(1, 64, 500, 256)

                image_small = image[0:1, :, :, :]
                image = torch.from_numpy(image_small).float()

                pred14 = model(image)
                pred14 += 1
                pred14 *= (1.6 - 1.4) / 2
                pred14 += 1.4

                pred_np = pred14.cpu().detach().numpy()

                img14 = pred_np[0, 0, :, :]

                Reconstruction(img14, f'{path}/Recons.png')

            if (scale.get()) == 20 and check_clean.get() == 0:

                # flatten images
                image = np.reshape(data_np, -1)

                image = image + noiseImg5

                image = image.reshape(1, 64, 500, 256)

                image_small = image[0:1, :, :, :]
                image = torch.from_numpy(image_small).float()

                pred15 = model(image)
                pred15 += 1
                pred15 *= (1.6 - 1.4) / 2
                pred15 += 1.4

                pred_np = pred15.cpu().detach().numpy()

                img15 = pred_np[0, 0, :, :]

                Reconstruction(img15, f'{path}/Recons.png')

            if (scale.get()) == 25 and check_clean.get() == 0:

                # flatten images
                image = np.reshape(data_np, -1)

                image = image + noiseImg6

                image = image.reshape(1, 64, 500, 256)

                image_small = image[0:1, :, :, :]
                image = torch.from_numpy(image_small).float()

                pred16 = model(image)
                pred16 += 1
                pred16 *= (1.6 - 1.4) / 2
                pred16 += 1.4

                pred_np = pred16.cpu().detach().numpy()

                img16 = pred_np[0, 0, :, :]

                Reconstruction(img16, f'{path}/Recons.png')

            if (scale.get()) == 30 and check_clean.get() == 0:

                # flatten images
                image = np.reshape(data_np, -1)

                image = image + noiseImg7

                image = image.reshape(1, 64, 500, 256)

                image_small = image[0:1, :, :, :]
                image = torch.from_numpy(image_small).float()

                pred17 = model(image)
                pred17 += 1
                pred17 *= (1.6 - 1.4) / 2
                pred17+= 1.4

                pred_np = pred17.cpu().detach().numpy()

                img17 = pred_np[0, 0, :, :]

                Reconstruction(img17, f'{path}/Recons.png')

            if len(acous_source_num.get()) != 0 and check_clean.get() == 0:

                if (scale.get()) == 0:

                    # flatten images
                    image = np.reshape(data_np, -1)

                    image = image + noiseImg1

                    image = image.reshape(1, 64, 500, 256)

                    image_small = image[0:1, :, :, :]
                    image = torch.from_numpy(image_small).float()

                    pred11 = model(image)
                    pred11 += 1
                    pred11 *= (1.6 - 1.4) / 2
                    pred11 += 1.4

                    pred_np = pred11.cpu().detach().numpy()
                    acous_chann = int(var10.get())

                    img11 = pred_np[0, 0, :, :]

                    Reconstruction(img11, f'{path}/Recons.png')

                if (scale.get()) == 5:

                    # flatten images
                    image = np.reshape(data_np, -1)

                    image = image + noiseImg2

                    image = image.reshape(1, 64, 500, 256)

                    image_small = image[0:1, :, :, :]
                    image = torch.from_numpy(image_small).float()

                    pred12 = model(image)
                    pred12 += 1
                    pred12 *= (1.6 - 1.4) / 2
                    pred12 += 1.4

                    pred_np = pred12.cpu().detach().numpy()
                    acous_chann = int(var10.get())

                    img12 = pred_np[0, 0, :, :]

                    Reconstruction(img12, f'{path}/Recons.png')

                if (scale.get()) == 10:

                    # flatten images
                    image = np.reshape(data_np, -1)

                    image = image + noiseImg3

                    image = image.reshape(1, 64, 500, 256)

                    image_small = image[0:1, :, :, :]
                    image = torch.from_numpy(image_small).float()

                    pred13 = model(image)
                    pred13 += 1
                    pred13 *= (1.6 - 1.4) / 2
                    pred13 += 1.4

                    pred_np = pred13.cpu().detach().numpy()
                    acous_chann = int(var10.get())

                    img13 = pred_np[0, 0, :, :]

                    Reconstruction(img13, f'{path}/Recons.png')

                if (scale.get()) == 15:

                    # flatten images
                    image = np.reshape(data_np, -1)

                    image = image + noiseImg4

                    image = image.reshape(1, 64, 500, 256)

                    image_small = image[0:1, :, :, :]
                    image = torch.from_numpy(image_small).float()

                    pred14 = model(image)
                    pred14 += 1
                    pred14 *= (1.6 - 1.4) / 2
                    pred14 += 1.4

                    pred_np = pred14.cpu().detach().numpy()
                    acous_chann = int(var10.get())

                    img14 = pred_np[0, 0, :, :]

                    Reconstruction(img14, f'{path}/Recons.png')

                if (scale.get()) == 20:

                    # flatten images
                    image = np.reshape(data_np, -1)

                    image = image + noiseImg5

                    image = image.reshape(1, 64, 500, 256)

                    image_small = image[0:1, :, :, :]
                    image = torch.from_numpy(image_small).float()

                    pred15 = model(image)
                    pred15 += 1
                    pred15 *= (1.6 - 1.4) / 2
                    pred15 += 1.4

                    pred_np = pred15.cpu().detach().numpy()
                    acous_chann = int(var10.get())

                    img15 = pred_np[0, 0, :, :]

                    Reconstruction(img15, f'{path}/Recons.png')

                if (scale.get()) == 25:

                    # flatten images
                    image = np.reshape(data_np, -1)

                    image = image + noiseImg6

                    image = image.reshape(1, 64, 500, 256)

                    image_small = image[0:1, :, :, :]
                    image = torch.from_numpy(image_small).float()

                    pred16 = model(image)
                    pred16 += 1
                    pred16 *= (1.6 - 1.4) / 2
                    pred16 += 1.4

                    pred_np = pred16.cpu().detach().numpy()
                    acous_chann = int(var10.get())

                    img16 = pred_np[0, 0, :, :]

                    Reconstruction(img16, f'{path}/Recons.png')

                if (scale.get()) == 30:

                    # flatten images
                    image = np.reshape(data_np, -1)

                    image = image + noiseImg7

                    image = image.reshape(1, 64, 500, 256)

                    image_small = image[0:1, :, :, :]
                    image = torch.from_numpy(image_small).float()

                    pred17 = model(image)
                    pred17 += 1
                    pred17 *= (1.6 - 1.4) / 2
                    pred17 += 1.4

                    pred_np = pred17.cpu().detach().numpy()
                    acous_chann = int(var10.get())

                    img17 = pred_np[0, 0, :, :]

                    Reconstruction(img17, f'{path}/Recons.png')

            time_elapsed = time.time() - since
            time_elapsed = round(time_elapsed)
            time = f"The time for original prediction is {time_elapsed} seconds"
            screen.insert(INSERT, " " + "\n")
            screen.insert(END, time + '\n')


def ReconstructTumor(scale_value=0):

    import time
        
    since = time.time()

    #set arguments
    path = '/home/pi/Desktop/USCT/results'
    os.makedirs(path, exist_ok=True)
    data_np = data.numpy()

    with torch.no_grad():

        model = network.model_dict['FCN4_Deep_Resize_Enc'](upsample_mode='nearest')
        sd = torch.load('/home/pi/Desktop/USCT/models/TaskBased/quant_gamma001.pth', map_location=torch.device('cpu'))['model']
        model.encoder.copy_(torch.load('/home/pi/Desktop/USCT/models/TaskBased/quant_gamma001.pth',map_location=torch.device('cpu') )['encoder'])
        model.load_state_dict(sd)
        model.eval()

        classf = classifier.TumorClassifier() #.to(dev)
        sdc = torch.load('/home/pi/Desktop/USCT/models/Classifier/classifier.pth',map_location=torch.device('cpu') )['model']
        classf.load_state_dict(sdc)
        classf.eval()

        if check_recons.get() == 0 and check_recons_tumor.get() == 1 and check_task.get() == 0 and check_tumor.get() == 0 or check_recons.get() == 1 and check_recons_tumor.get() == 1 and check_task.get() == 0 and check_tumor.get() == 0 or check_recons.get() == 1 and check_recons_tumor.get() == 0 and check_task.get() == 1 and check_tumor.get() == 0 or check_recons.get() == 1 and check_recons_tumor.get() == 0 and check_task.get() == 0 and check_tumor.get() == 1 or check_recons.get() == 0 and check_recons_tumor.get() == 1 and check_task.get() == 1 and check_tumor.get() == 0 or check_recons.get() == 0 and check_recons_tumor.get() == 1 and check_task.get() == 0 and check_tumor.get() == 1 or check_recons.get() == 0 and check_recons_tumor.get() == 0 and check_task.get() == 1 and check_tumor.get() == 1 or check_recons.get() == 1 and check_recons_tumor.get() == 1 and check_task.get() == 1 and check_tumor.get() == 0 or check_recons.get() == 1 and check_recons_tumor.get() == 0 and check_task.get() == 1 and check_tumor.get() == 1 or check_recons.get() == 0 and check_recons_tumor.get() == 1 and check_task.get() == 1 and check_tumor.get() == 1 or check_recons.get() == 1 and check_recons_tumor.get() == 1 and check_task.get() == 0 and check_tumor.get() == 1 or check_recons.get() == 1 and check_recons_tumor.get() == 1 and check_task.get() == 1 and check_tumor.get() == 1:
            
            tmaps =  classf(label_ten).cpu().detach().numpy() > 0.5

            tm_small = tmaps[0:1, :, :, :]

            if check_clean.get() == 1 and len(acous_source_num.get()) == 0:

                if check_task.get() == 0:

                    clean_data = data_np.reshape(1, 64, 500, 256)

                    image_small = clean_data[0:1, :, :, :]
                    image = torch.from_numpy(image_small).float()

                    pred = model(image)
                    pred += 1
                    pred *= (1.6 - 1.4) / 2
                    pred += 1.4

                    learned_tumort = classf(pred) > 0.02

                    learned_tumor = learned_tumort.cpu().detach()

                    img10 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                else:
                    learned_tumort = classf(pred10) > 0.02

                    learned_tumor = learned_tumort.cpu().detach()

                    img10 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                ReconstructionTumor(img10, f'{path}/ReconsTumor.png')

            elif check_clean.get() == 1 and len(acous_source_num.get()) != 0:

                if check_task.get() == 0:

                    clean_data = data_np.reshape(1, 64, 500, 256)

                    image_small = clean_data[0:1, :, :, :]
                    image = torch.from_numpy(image_small).float()

                    pred = model(image)
                    pred += 1
                    pred *= (1.6 - 1.4) / 2
                    pred += 1.4

                    learned_tumort = classf(pred) > 0.02

                    learned_tumor = learned_tumort.cpu().detach()

                    img10 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                else:

                    learned_tumort = classf(pred10) > 0.02

                    learned_tumor = learned_tumort.cpu().detach()
                    
                    img10 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                ReconstructionTumor(img10, f'{path}/ReconsTumor.png')

            if (scale.get()) == 0 and check_clean.get() == 0:

                if check_task.get() == 0:

                    # flatten images
                    image = np.reshape(data_np, -1)

                    image = image + noiseImg1

                    image = image.reshape(1, 64, 500, 256)

                    image_small = image[0:1, :, :, :]
                    image = torch.from_numpy(image_small).float()

                    pred = model(image)
                    pred += 1
                    pred *= (1.6 - 1.4) / 2
                    pred += 1.4

                    learned_tumort = classf(pred) > 0.02

                    learned_tumor = learned_tumort.cpu().detach()
                    

                    img11 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                else:

                    learned_tumort = classf(pred11) > 0.02

                    learned_tumor = learned_tumort.cpu().detach()
                    
                    img11 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                ReconstructionTumor(img11, f'{path}/ReconsTumor.png')

            if (scale.get()) == 5 and check_clean.get() == 0:

                if check_task.get() == 0:

                    # flatten images
                    image = np.reshape(data_np, -1)

                    image = image + noiseImg2

                    image = image.reshape(1, 64, 500, 256)

                    image_small = image[0:1, :, :, :]
                    image = torch.from_numpy(image_small).float()

                    pred = model(image)
                    pred += 1
                    pred *= (1.6 - 1.4) / 2
                    pred += 1.4

                    learned_tumort = classf(pred) > 0.02

                    learned_tumor = learned_tumort.cpu().detach()

                    img12 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                else:

                    learned_tumort = classf(pred12) > 0.02

                    learned_tumor = learned_tumort.cpu().detach()

                    img12 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                ReconstructionTumor(img12, f'{path}/ReconsTumor.png')

            if (scale.get()) == 10 and check_clean.get() == 0:

                if check_task.get() == 0:

                    # flatten images
                    image = np.reshape(data_np, -1)

                    image = image + noiseImg3

                    image = image.reshape(1, 64, 500, 256)

                    image_small = image[0:1, :, :, :]
                    image = torch.from_numpy(image_small).float()

                    pred = model(image)
                    pred += 1
                    pred *= (1.6 - 1.4) / 2
                    pred += 1.4

                    learned_tumort = classf(pred) > 0.02

                    learned_tumor = learned_tumort.cpu().detach()

                    img13 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                else:

                    learned_tumort = classf(pred13) > 0.02

                    learned_tumor = learned_tumort.cpu().detach()

                    img13 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                ReconstructionTumor(img13, f'{path}/ReconsTumor.png')

            if (scale.get()) == 15 and check_clean.get() == 0:

                if check_task.get():

                    # flatten images
                    image = np.reshape(data_np, -1)

                    image = image + noiseImg4

                    image = image.reshape(1, 64, 500, 256)

                    image_small = image[0:1, :, :, :]
                    image = torch.from_numpy(image_small).float()

                    pred = model(image)
                    pred += 1
                    pred *= (1.6 - 1.4) / 2
                    pred += 1.4

                    learned_tumort = classf(pred) > 0.02

                    learned_tumor = learned_tumort.cpu().detach()

                    img14 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                else:

                    learned_tumort = classf(pred14) > 0.02

                    learned_tumor = learned_tumort.cpu().detach()

                    img14 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                ReconstructionTumor(img14, f'{path}/ReconsTumor.png')


            if (scale.get()) == 20 and check_clean.get() == 0:

                if check_task.get() == 0:

                    # flatten images
                    image = np.reshape(data_np, -1)

                    image = image + noiseImg5

                    image = image.reshape(1, 64, 500, 256)

                    image_small = image[0:1, :, :, :]
                    image = torch.from_numpy(image_small).float()

                    pred = model(image)
                    pred += 1
                    pred *= (1.6 - 1.4) / 2
                    pred += 1.4

                    learned_tumort = classf(pred) > 0.02

                    learned_tumor = learned_tumort.cpu().detach()

                    img15 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                else:

                    learned_tumort = classf(pred15) > 0.02

                    learned_tumor = learned_tumort.cpu().detach()

                    img15 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                ReconstructionTumor(img15, f'{path}/ReconsTumor.png')

            if (scale.get()) == 25 and check_clean.get() == 0:

                if check_task.get() == 0:

                    # flatten images
                    image = np.reshape(data_np, -1)

                    image = image + noiseImg6

                    image = image.reshape(1, 64, 500, 256)

                    image_small = image[0:1, :, :, :]
                    image = torch.from_numpy(image_small).float()

                    pred = model(image)
                    pred += 1
                    pred *= (1.6 - 1.4) / 2
                    pred += 1.4

                    learned_tumort = classf(pred) > 0.02

                    learned_tumor = learned_tumort.cpu().detach()

                    img16 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                else:

                    learned_tumort = classf(pred16) > 0.02

                    learned_tumor = learned_tumort.cpu().detach()

                    img16 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                ReconstructionTumor(img16, f'{path}/ReconsTumor.png')

            if (scale.get()) == 30 and check_clean.get() == 0:

                if check_task.get() == 0:

                    # flatten images
                    image = np.reshape(data_np, -1)

                    image = image + noiseImg7

                    image = image.reshape(1, 64, 500, 256)

                    image_small = image[0:1, :, :, :]
                    image = torch.from_numpy(image_small).float()

                    pred = model(image)
                    pred += 1
                    pred *= (1.6 - 1.4) / 2
                    pred += 1.4

                    learned_tumort = classf(pred) > 0.02

                    learned_tumor = learned_tumort.cpu().detach()

                    img17 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                else:

                    learned_tumort = classf(pred17) > 0.02

                    learned_tumor = learned_tumort.cpu().detach()

                    img17 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                ReconstructionTumor(img17, f'{path}/ReconsTumor.png')

            if len(acous_source_num.get()) != 0 and check_clean.get() == 0:

                if (scale.get()) == 0:
                    
                    if check_task == 0:

                        # flatten images
                        image = np.reshape(data_np, -1)

                        image = image + noiseImg1

                        image = image.reshape(1, 64, 500, 256)

                        image_small = image[0:1, :, :, :]
                        image = torch.from_numpy(image_small).float()

                        pred = model(image)
                        pred += 1
                        pred *= (1.6 - 1.4) / 2
                        pred += 1.4

                        learned_tumort = classf(pred) > 0.02

                        learned_tumor = learned_tumort.cpu().detach()

                        img11 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                    else:

                        learned_tumort = classf(pred11) > 0.02

                        learned_tumor = learned_tumort.cpu().detach()

                        img11 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                    ReconstructionTumor(img11, f'{path}/ReconsTumor.png')

                if (scale.get()) == 5:
                    
                    if check_task == 0:

                        # flatten images
                        image = np.reshape(data_np, -1)

                        image = image + noiseImg2

                        image = image.reshape(1, 64, 500, 256)

                        image_small = image[0:1, :, :, :]
                        image = torch.from_numpy(image_small).float()

                        pred = model(image)
                        pred += 1
                        pred *= (1.6 - 1.4) / 2
                        pred += 1.4

                        learned_tumort = classf(pred) > 0.02

                        learned_tumor = learned_tumort.cpu().detach()

                        img12 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                    else:

                        learned_tumort = classf(pred12) > 0.02

                        learned_tumor = learned_tumort.cpu().detach()

                        img12 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                    ReconstructionTumor(img12, f'{path}/ReconsTumor.png')

                if (scale.get()) == 10:

                    if check_task == 0:

                        # flatten images
                        image = np.reshape(data_np, -1)

                        image = image + noiseImg3

                        image = image.reshape(1, 64, 500, 256)

                        image_small = image[0:1, :, :, :]
                        image = torch.from_numpy(image_small).float()

                        pred = model(image)
                        pred += 1
                        pred *= (1.6 - 1.4) / 2
                        pred += 1.4

                        learned_tumort = classf(pred) > 0.02

                        learned_tumor = learned_tumort.cpu().detach()

                        img13 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                    else:

                        learned_tumort = classf(pred13) > 0.02

                        learned_tumor = learned_tumort.cpu().detach()

                        img13 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                    ReconstructionTumor(img13, f'{path}/ReconsTumor.png')

                if (scale.get()) == 15:

                    if check_task == 0:

                        # flatten images
                        image = np.reshape(data_np, -1)

                        image = image + noiseImg4

                        image = image.reshape(1, 64, 500, 256)

                        image_small = image[0:1, :, :, :]
                        image = torch.from_numpy(image_small).float()

                        pred = model(image)
                        pred += 1
                        pred *= (1.6 - 1.4) / 2
                        pred += 1.4

                        learned_tumort = classf(pred) > 0.02

                        learned_tumor = learned_tumort.cpu().detach()

                        img14 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                    else:

                        learned_tumort = classf(pred14) > 0.02

                        learned_tumor = learned_tumort.cpu().detach()

                        img14 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                    ReconstructionTumor(img14, f'{path}/ReconsTumor.png')

                if (scale.get()) == 20:

                    if check_task == 0:

                        # flatten images
                        image = np.reshape(data_np, -1)

                        image = image + noiseImg5

                        image = image.reshape(1, 64, 500, 256)

                        image_small = image[0:1, :, :, :]
                        image = torch.from_numpy(image_small).float()

                        pred = model(image)
                        pred += 1
                        pred *= (1.6 - 1.4) / 2
                        pred += 1.4

                        learned_tumort = classf(pred) > 0.02

                        learned_tumor = learned_tumort.cpu().detach()

                        img15 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                    else:

                        learned_tumort = classf(pred15) > 0.02

                        learned_tumor = learned_tumort.cpu().detach()

                        img15 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                    ReconstructionTumor(img15, f'{path}/ReconsTumor.png')

                if (scale.get()) == 25:

                    if check_task == 0:

                        # flatten images
                        image = np.reshape(data_np, -1)

                        image = image + noiseImg6

                        image = image.reshape(1, 64, 500, 256)

                        image_small = image[0:1, :, :, :]
                        image = torch.from_numpy(image_small).float()

                        pred = model(image)
                        pred += 1
                        pred *= (1.6 - 1.4) / 2
                        pred += 1.4

                        learned_tumort = classf(pred) > 0.02

                        learned_tumor = learned_tumort.cpu().detach()

                        img16 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                    else:

                        learned_tumort = classf(pred16) > 0.02

                        learned_tumor = learned_tumort.cpu().detach()

                        img16 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                    ReconstructionTumor(img16, f'{path}/ReconsTumor.png')

                if (scale.get()) == 30:

                    if check_task == 0:

                        # flatten images
                        image = np.reshape(data_np, -1)

                        image = image + noiseImg7

                        image = image.reshape(1, 64, 500, 256)

                        image_small = image[0:1, :, :, :]
                        image = torch.from_numpy(image_small).float()

                        pred = model(image)
                        pred += 1
                        pred *= (1.6 - 1.4) / 2
                        pred += 1.4

                        learned_tumort = classf(pred) > 0.02

                        learned_tumor = learned_tumort.cpu().detach()

                        img17 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                    else:

                        learned_tumort = classf(pred17) > 0.02

                        learned_tumor = learned_tumort.cpu().detach()

                        img17 = learned_tumor[0, 0, :, :] * (2-tm_small[0, 0, :, :])

                    ReconstructionTumor(img17, f'{path}/ReconsTumor.png')

            time_elapsed = time.time() - since
            time_elapsed = round(time_elapsed)
            time = f"The time for original tumor prediction is {time_elapsed} seconds"
            screen.insert(INSERT, " " + "\n")
            screen.insert(END, time + '\n')

# def Reconstruct(scale_value=0):
#     global pred10, pred11, pred12, pred13, pred14, pred15, pred16, pred17

#     from skimage.metrics import structural_similarity as ssim
#     import time

#     since = time.time()
    
#     #set arguments
#     path = '/home/pi/Desktop/USCT/results'
#     os.makedirs(path, exist_ok=True)
#     data_np = data.numpy()

#     tumor_classifier = classifier.TumorClassifier() #.to(dev)
#     class_sd = torch.load('/home/pi/Desktop/USCT/models/Classifier/classifier.pth',map_location=torch.device('cpu') )['model']
#     tumor_classifier.load_state_dict(class_sd)
#     tumor_classifier.eval()

#     if source == 1:
#         true_sos = np.load("/home/pi/Desktop/USCT/velocity_maps/data33.npy")
#     elif source == 0:
#         true_sos = np.load("/home/pi/Desktop/USCT/velocity_maps/data34.npy")
	
#     recons = np.load("/home/pi/Desktop/USCT/classic_test_results_rob.npy")

#     SSIMS = []
#     RMSES = []

#     struct = np.ones((3,3), dtype=np.int)

#     if check_recons.get() == 1 and check_recons_tumor.get() == 0 and check_task.get() == 0 and check_tumor.get() == 0 or check_recons.get() == 1 and check_recons_tumor.get() == 1 and check_task.get() == 0 and check_tumor.get() == 0 or check_recons.get() == 1 and check_recons_tumor.get() == 0 and check_task.get() == 1 and check_tumor.get() == 0 or check_recons.get() == 1 and check_recons_tumor.get() == 0 and check_task.get() == 0 and check_tumor.get() == 1 or check_recons.get() == 0 and check_recons_tumor.get() == 1 and check_task.get() == 1 and check_tumor.get() == 0 or check_recons.get() == 0 and check_recons_tumor.get() == 1 and check_task.get() == 0 and check_tumor.get() == 1 or check_recons.get() == 0 and check_recons_tumor.get() == 0 and check_task.get() == 1 and check_tumor.get() == 1 or check_recons.get() == 1 and check_recons_tumor.get() == 1 and check_task.get() == 1 and check_tumor.get() == 0 or check_recons.get() == 1 and check_recons_tumor.get() == 0 and check_task.get() == 1 and check_tumor.get() == 1 or check_recons.get() == 0 and check_recons_tumor.get() == 1 and check_task.get() == 1 and check_tumor.get() == 1 or check_recons.get() == 1 and check_recons_tumor.get() == 1 and check_task.get() == 0 and check_tumor.get() == 1 or check_recons.get() == 1 and check_recons_tumor.get() == 1 and check_task.get() == 1 and check_tumor.get() == 1:

#         if check_clean.get() == 1 and len(acous_source_num.get()) == 0: 
    
#             clean_label = true_sos

#             SSIMS.append(ssim(clean_label[0, 0, :, :], true_sos[0, 0, :, :], data_range=0.2))
#             RMSES.append(np.linalg.norm(clean_label[0, 0, :, :] - true_sos[0, 0, :, :]))

#             rmse_error = np.array(RMSES).mean()
#             rmse_std = np.array(RMSES).std()
#             test_ssim = np.array(SSIMS).mean()
#             ssim_std = np.array(SSIMS).std()

#             screen.insert(INSERT, " " + '\n')
#             screen.insert(INSERT, "***** Performing FWI Reconstruction *****" + '\n') 
#             Reconstruction(clean_label[0, 0, :, :], f'{path}/Recons.png')
#             error = f"Average testing error = {rmse_error}"
#             error_std = f"Testing error standard deviation = {rmse_std}"
#             ssim_test = f"Average Testing SSIM = {test_ssim}"
#             std_ssim = f"Testing error SSIM STD = {ssim_std}"

#             screen.insert(END, error + '\n')
#             screen.insert(END, error_std + '\n')
#             screen.insert(END, ssim_test + '\n')
#             screen.insert(END, std_ssim + '\n')

#         elif check_clean.get() == 1 and len(acous_source_num.get()) != 0:

#             clean_label = true_sos

#             SSIMS.append(ssim(clean_label[0, 0, :, :], true_sos[0, 0, :, :], data_range=0.2))
#             RMSES.append(np.linalg.norm(clean_label[0, 0, :, :] - true_sos[0, 0, :, :]))

#             rmse_error = np.array(RMSES).mean()
#             rmse_std = np.array(RMSES).std()
#             test_ssim = np.array(SSIMS).mean()
#             ssim_std = np.array(SSIMS).std()

#             screen.insert(INSERT, " " + '\n')
#             screen.insert(INSERT, "***** Performing FWI Reconstruction *****" + '\n') 
#             Reconstruction(clean_label[0, 0, :, :], f'{path}/Recons.png')
#             error = f"Average testing error = {rmse_error}"
#             error_std = f"Testing error standard deviation = {rmse_std}"
#             ssim = f"Average Testing SSIM = {test_ssim}"
#             std_ssim = f"Testing error SSIM STD = {ssim_std}"
#             screen.insert(END, error + '\n')
#             screen.insert(END, error_std + '\n')
#             screen.insert(END, ssim + '\n')
#             screen.insert(END, std_ssim + '\n')
            
#         if (scale.get()) == 0 and check_clean.get() == 0:

#             label1 = true_sos[0:1, 0:1, :, :]
#             recons1 = recons[0:1, 0:1, :, :]

#             # flatten images
#             label1 = np.reshape(label1, -1)
#             recons1 = np.reshape(recons1, -1)

#             label1 = label1 + label_noiseImg1
#             recons1 = recons1 + label_noiseImg1
            
#             noise_label1 = label1.reshape(1, 1, 214, 214)
#             noise_recons1 = recons1.reshape(1, 1, 214, 214)

#             SSIMS.append(ssim(noise_label1[0, 0, :, :], noise_recons1[0, 0, :, :], data_range=0.2))
#             RMSES.append(np.linalg.norm(noise_label1[0, 0, :, :] - noise_recons1[0, 0, :, :]))

#             rmse_error = np.array(RMSES).mean()
#             rmse_std = np.array(RMSES).std()
#             test_ssim = np.array(SSIMS).mean()
#             ssim_std = np.array(SSIMS).std()

#             screen.insert(INSERT, " " + '\n')
#             screen.insert(INSERT, "***** Performing FWI Reconstruction *****" + '\n') 
#             Reconstruction(noise_recons1[0, 0, :, :], f'{path}/Recons.png')
#             error = f"Average testing error = {rmse_error}"
#             error_std = f"Testing error standard deviation = {rmse_std}"
#             ssim = f"Average Testing SSIM = {test_ssim}"
#             std_ssim = f"Testing error SSIM STD = {ssim_std}"
#             screen.insert(END, error + '\n')
#             screen.insert(END, error_std + '\n')
#             screen.insert(END, ssim + '\n')
#             screen.insert(END, std_ssim + '\n')

#         if (scale.get()) == 5 and check_clean.get() == 0:

#             label2 = true_sos[0:1, 0:1, :, :]
#             recons2 = recons[0:1, 0:1, :, :]

#             # flatten images
#             label2 = np.reshape(label2, -1)
#             recons2 = np.reshape(recons2, -1)

#             label2 = label2 + label_noiseImg2
#             recons2 = recons2 + label_noiseImg2
            
#             noise_label2 = label2.reshape(1, 1, 214, 214)
#             noise_recons2 = recons2.reshape(1, 1, 214, 214)

#             SSIMS.append(ssim(noise_label2[0, 0, :, :], noise_recons2[0, 0, :, :], data_range=0.2))
#             RMSES.append(np.linalg.norm(noise_label2[0, 0, :, :] - noise_recons2[0, 0, :, :]))

#             rmse_error = np.array(RMSES).mean()
#             rmse_std = np.array(RMSES).std()
#             test_ssim = np.array(SSIMS).mean()
#             ssim_std = np.array(SSIMS).std()

#             screen.insert(INSERT, " " + '\n')
#             screen.insert(INSERT, "***** Performing FWI Reconstruction *****" + '\n') 
#             Reconstruction(noise_recons2[0, 0, :, :], f'{path}/Recons.png')
#             error = f"Average testing error = {rmse_error}"
#             error_std = f"Testing error standard deviation = {rmse_std}"
#             ssim = f"Average Testing SSIM = {test_ssim}"
#             std_ssim = f"Testing error SSIM STD = {ssim_std}"
#             screen.insert(END, error + '\n')
#             screen.insert(END, error_std + '\n')
#             screen.insert(END, ssim + '\n')
#             screen.insert(END, std_ssim + '\n')

#         if (scale.get()) == 10 and check_clean.get() == 0:

#             label3 = true_sos[0:1, 0:1, :, :]
#             recons3 = recons[0:1, 0:1, :, :]

#             # flatten images
#             label3 = np.reshape(label3, -1)
#             recons3 = np.reshape(recons3, -1)

#             label3 = label3 + label_noiseImg3
#             recons3 = recons3 + label_noiseImg3
            
#             noise_label3 = label3.reshape(1, 1, 214, 214)
#             noise_recons3 = recons3.reshape(1, 1, 214, 214)

#             SSIMS.append(ssim(noise_label3[0, 0, :, :], noise_recons3[0, 0, :, :], data_range=0.2))
#             RMSES.append(np.linalg.norm(noise_label3[0, 0, :, :] - noise_recons3[0, 0, :, :]))

#             rmse_error = np.array(RMSES).mean()
#             rmse_std = np.array(RMSES).std()
#             test_ssim = np.array(SSIMS).mean()
#             ssim_std = np.array(SSIMS).std()

#             screen.insert(INSERT, " " + '\n')
#             screen.insert(INSERT, "***** Performing FWI Reconstruction *****" + '\n') 
#             Reconstruction(noise_recons3[0, 0, :, :], f'{path}/Recons.png')
#             error = f"Average testing error = {rmse_error}"
#             error_std = f"Testing error standard deviation = {rmse_std}"
#             ssim = f"Average Testing SSIM = {test_ssim}"
#             std_ssim = f"Testing error SSIM STD = {ssim_std}"
#             screen.insert(END, error + '\n')
#             screen.insert(END, error_std + '\n')
#             screen.insert(END, ssim + '\n')
#             screen.insert(END, std_ssim + '\n')

#         if (scale.get()) == 15 and check_clean.get() == 0:

#             label4 = true_sos[0:1, 0:1, :, :]
#             recons4 = recons[0:1, 0:1, :, :]

#             # flatten images
#             label4 = np.reshape(label4, -1)
#             recons4 = np.reshape(recons4, -1)

#             label4 = label4 + label_noiseImg4
#             recons4 = recons4 + label_noiseImg4
            
#             noise_label4 = label4.reshape(1, 1, 214, 214)
#             noise_recons4 = recons4.reshape(1, 1, 214, 214)

#             SSIMS.append(ssim(noise_label4[0, 0, :, :], noise_recons4[0, 0, :, :], data_range=0.2))
#             RMSES.append(np.linalg.norm(noise_label4[0, 0, :, :] - noise_recons4[0, 0, :, :]))
#             rmse_error = np.array(RMSES).mean()
#             rmse_std = np.array(RMSES).std()
#             test_ssim = np.array(SSIMS).mean()
#             ssim_std = np.array(SSIMS).std()

#             screen.insert(INSERT, " " + '\n')
#             screen.insert(INSERT, "***** Performing FWI Reconstruction *****" + '\n') 
#             Reconstruction(noise_recons4[0, 0, :, :], f'{path}/Recons.png')
#             error = f"Average testing error = {rmse_error}"
#             error_std = f"Testing error standard deviation = {rmse_std}"
#             ssim = f"Average Testing SSIM = {test_ssim}"
#             std_ssim = f"Testing error SSIM STD = {ssim_std}"
#             screen.insert(END, error + '\n')
#             screen.insert(END, error_std + '\n')
#             screen.insert(END, ssim + '\n')
#             screen.insert(END, std_ssim + '\n')

#         if (scale.get()) == 20 and check_clean.get() == 0:

#             label5 = true_sos[0:1, 0:1, :, :]
#             recons5 = recons[0:1, 0:1, :, :]

#             # flatten images
#             label5 = np.reshape(label5, -1)
#             recons5 = np.reshape(recons5, -1)

#             label5 = label5 + label_noiseImg5
#             recons5 = recons5 + label_noiseImg5
            
#             noise_label5 = label5.reshape(1, 1, 214, 214)
#             noise_recons5 = recons5.reshape(1, 1, 214, 214)

#             SSIMS.append(ssim(noise_label5[0, 0, :, :], noise_recons5[0, 0, :, :], data_range=0.2))
#             RMSES.append(np.linalg.norm(noise_label5[0, 0, :, :] - noise_recons5[0, 0, :, :]))

#             rmse_error = np.array(RMSES).mean()
#             rmse_std = np.array(RMSES).std()
#             test_ssim = np.array(SSIMS).mean()
#             ssim_std = np.array(SSIMS).std()

#             screen.insert(INSERT, " " + '\n')
#             screen.insert(INSERT, "***** Performing FWI Reconstruction *****" + '\n') 
#             Reconstruction(noise_recons5[0, 0, :, :], f'{path}/Recons.png')
#             error = f"Average testing error = {rmse_error}"
#             error_std = f"Testing error standard deviation = {rmse_std}"
#             ssim = f"Average Testing SSIM = {test_ssim}"
#             std_ssim = f"Testing error SSIM STD = {ssim_std}"
#             screen.insert(END, error + '\n')
#             screen.insert(END, error_std + '\n')
#             screen.insert(END, ssim + '\n')
#             screen.insert(END, std_ssim + '\n')

#         if (scale.get()) == 25 and check_clean.get() == 0:

#             label6 = true_sos[0:1, 0:1, :, :]
#             recons6 = recons[0:1, 0:1, :, :]

#             # flatten images
#             label6 = np.reshape(label6, -1)
#             recons6 = np.reshape(recons6, -1)

#             label6 = label6 + label_noiseImg6
#             recons6 = recons6 + label_noiseImg6
            
#             noise_label6 = label6.reshape(1, 1, 214, 214)
#             noise_recons6 = recons6.reshape(1, 1, 214, 214)

#             SSIMS.append(ssim(noise_label6[0, 0, :, :], noise_recons6[0, 0, :, :], data_range=0.2))
#             RMSES.append(np.linalg.norm(noise_label6[0, 0, :, :] - noise_recons6[0, 0, :, :]))

#             rmse_error = np.array(RMSES).mean()
#             rmse_std = np.array(RMSES).std()
#             test_ssim = np.array(SSIMS).mean()
#             ssim_std = np.array(SSIMS).std()

#             screen.insert(INSERT, " " + '\n')
#             screen.insert(INSERT, "***** Performing FWI Reconstruction *****" + '\n') 
#             Reconstruction(noise_recons6[0, 0, :, :], f'{path}/Recons.png')
#             error = f"Average testing error = {rmse_error}"
#             error_std = f"Testing error standard deviation = {rmse_std}"
#             ssim = f"Average Testing SSIM = {test_ssim}"
#             std_ssim = f"Testing error SSIM STD = {ssim_std}"
#             screen.insert(END, error + '\n')
#             screen.insert(END, error_std + '\n')
#             screen.insert(END, ssim + '\n')
#             screen.insert(END, std_ssim + '\n')

#         if (scale.get()) == 30 and check_clean.get() == 0:

#             label7 = true_sos[0:1, 0:1, :, :]
#             recons7 = recons[0:1, 0:1, :, :]

#             # flatten images
#             label7 = np.reshape(label7, -1)
#             recons7 = np.reshape(recons7, -1)

#             label7 = label7 + label_noiseImg7
#             recons7 = recons7 + label_noiseImg7
 
#             noise_label7 = label7.reshape(1, 1, 214, 214)
#             noise_recons7 = recons7.reshape(1, 1, 214, 214)

#             SSIMS.append(ssim(noise_label7[0, 0, :, :], noise_recons7[0, 0, :, :], data_range=0.2))
#             RMSES.append(np.linalg.norm(noise_label7[0, 0, :, :] - noise_recons7[0, 0, :, :]))

#             rmse_error = np.array(RMSES).mean()
#             rmse_std = np.array(RMSES).std()
#             test_ssim = np.array(SSIMS).mean()
#             ssim_std = np.array(SSIMS).std()

#             screen.insert(INSERT, " " + '\n')
#             screen.insert(INSERT, "***** Performing FWI Reconstruction *****" + '\n') 
#             Reconstruction(noise_recons7[0, 0, :, :], f'{path}/Recons.png')
#             error = f"Average testing error = {rmse_error}"
#             error_std = f"Testing error standard deviation = {rmse_std}"
#             ssim = f"Average Testing SSIM = {test_ssim}"
#             std_ssim = f"Testing error SSIM STD = {ssim_std}"
#             screen.insert(END, error + '\n')
#             screen.insert(END, error_std + '\n')
#             screen.insert(END, ssim + '\n')
#             screen.insert(END, std_ssim + '\n')

#         elif len(acous_source_num.get()) != 0 and check_clean.get() == 0:

#             from skimage.metrics import structural_similarity as ssim

#             if (scale.get()) == 0:

#                 label1 = true_sos[0:1, 0:1, :, :]
#                 recons1 = recons[0:1, 0:1, :, :]

#                 # flatten images
#                 label1 = np.reshape(label1, -1)
#                 recons1 = np.reshape(recons1, -1)

#                 label1 = label1 + label_noiseImg1
#                 recons1 = recons1 + label_noiseImg1
                
#                 noise_label1 = label1.reshape(1, 1, 214, 214)
#                 noise_recons1 = recons1.reshape(1, 1, 214, 214)

#                 SSIMS.append(ssim(noise_label1[0, 0, :, :], noise_recons1[0, 0, :, :], data_range=0.2))
#                 RMSES.append(np.linalg.norm(noise_label1[0, 0, :, :] - noise_recons1[0, 0, :, :]))

#                 rmse_error = np.array(RMSES).mean()
#                 rmse_std = np.array(RMSES).std()
#                 test_ssim = np.array(SSIMS).mean()
#                 ssim_std = np.array(SSIMS).std()

#                 screen.insert(INSERT, " " + '\n')
#                 screen.insert(INSERT, "***** Performing FWI Reconstruction *****" + '\n') 
#                 Reconstruction(noise_recons1[0, 0, :, :], f'{path}/Recons.png')
#                 error = f"Average testing error = {rmse_error}"
#                 error_std = f"Testing error standard deviation = {rmse_std}"
#                 ssim = f"Average Testing SSIM = {test_ssim}"
#                 std_ssim = f"Testing error SSIM STD = {ssim_std}"
#                 screen.insert(END, error + '\n')
#                 screen.insert(END, error_std + '\n')
#                 screen.insert(END, ssim + '\n')
#                 screen.insert(END, std_ssim + '\n')

#             if (scale.get()) == 5:

#                 label2 = true_sos[0:1, 0:1, :, :]
#                 recons2 = recons[0:1, 0:1, :, :]

#                 # flatten images
#                 label2 = np.reshape(label2, -1)
#                 recons2 = np.reshape(recons2, -1)

#                 label2 = label2 + label_noiseImg2
#                 recons2 = recons2 + label_noiseImg2
                
#                 noise_label2 = label2.reshape(1, 1, 214, 214)
#                 noise_recons2 = recons2.reshape(1, 1, 214, 214)

#                 SSIMS.append(ssim(noise_label2[0, 0, :, :], noise_recons2[0, 0, :, :], data_range=0.2))
#                 RMSES.append(np.linalg.norm(noise_label2[0, 0, :, :] - noise_recons2[0, 0, :, :]))

#                 rmse_error = np.array(RMSES).mean()
#                 rmse_std = np.array(RMSES).std()
#                 test_ssim = np.array(SSIMS).mean()
#                 ssim_std = np.array(SSIMS).std()

#                 screen.insert(INSERT, " " + '\n')
#                 screen.insert(INSERT, "***** Performing FWI Reconstruction *****" + '\n') 
#                 Reconstruction(noise_recons2[0, 0, :, :], f'{path}/Recons.png')
#                 error = f"Average testing error = {rmse_error}"
#                 error_std = f"Testing error standard deviation = {rmse_std}"
#                 ssim = f"Average Testing SSIM = {test_ssim}"
#                 std_ssim = f"Testing error SSIM STD = {ssim_std}"
#                 screen.insert(END, error + '\n')
#                 screen.insert(END, error_std + '\n')
#                 screen.insert(END, ssim + '\n')
#                 screen.insert(END, std_ssim + '\n')

#             if (scale.get()) == 10:

#                 label3 = true_sos[0:1, 0:1, :, :]
#                 recons3 = recons[0:1, 0:1, :, :]

#                 # flatten images
#                 label3 = np.reshape(label3, -1)
#                 recons3 = np.reshape(recons3, -1)

#                 label3 = label3 + label_noiseImg3
#                 recons3 = recons3 + label_noiseImg3
                
#                 noise_label3 = label3.reshape(1, 1, 214, 214)
#                 noise_recons3 = recons3.reshape(1, 1, 214, 214)

#                 SSIMS.append(ssim(noise_label3[0, 0, :, :], noise_recons3[0, 0, :, :], data_range=0.2))
#                 RMSES.append(np.linalg.norm(noise_label3[0, 0, :, :] - noise_recons3[0, 0, :, :]))

#                 rmse_error = np.array(RMSES).mean()
#                 rmse_std = np.array(RMSES).std()
#                 test_ssim = np.array(SSIMS).mean()
#                 ssim_std = np.array(SSIMS).std()

#                 screen.insert(INSERT, " " + '\n')
#                 screen.insert(INSERT, "***** Performing FWI Reconstruction *****" + '\n') 
#                 Reconstruction(noise_recons3[0, 0, :, :], f'{path}/Recons.png')
#                 error = f"Average testing error = {rmse_error}"
#                 error_std = f"Testing error standard deviation = {rmse_std}"
#                 ssim = f"Average Testing SSIM = {test_ssim}"
#                 std_ssim = f"Testing error SSIM STD = {ssim_std}"
#                 screen.insert(END, error + '\n')
#                 screen.insert(END, error_std + '\n')
#                 screen.insert(END, ssim + '\n')
#                 screen.insert(END, std_ssim + '\n')

#             if (scale.get()) == 15:

#                 label4 = true_sos[0:1, 0:1, :, :]
#                 recons4 = recons[0:1, 0:1, :, :]

#                 # flatten images
#                 label4 = np.reshape(label4, -1)
#                 recons4 = np.reshape(recons4, -1)

#                 label4 = label4 + label_noiseImg4
#                 recons4 = recons4 + label_noiseImg4
                
#                 noise_label4 = label4.reshape(1, 1, 214, 214)
#                 noise_recons4 = recons4.reshape(1, 1, 214, 214)

#                 SSIMS.append(ssim(noise_label4[0, 0, :, :], noise_recons4[0, 0, :, :], data_range=0.2))
#                 RMSES.append(np.linalg.norm(noise_label4[0, 0, :, :] - noise_recons4[0, 0, :, :]))
#                 rmse_error = np.array(RMSES).mean()
#                 rmse_std = np.array(RMSES).std()
#                 test_ssim = np.array(SSIMS).mean()
#                 ssim_std = np.array(SSIMS).std()

#                 screen.insert(INSERT, " " + '\n')
#                 screen.insert(INSERT, "***** Performing FWI Reconstruction *****" + '\n') 
#                 Reconstruction(noise_recons4[0, 0, :, :], f'{path}/Recons.png')
#                 error = f"Average testing error = {rmse_error}"
#                 error_std = f"Testing error standard deviation = {rmse_std}"
#                 ssim = f"Average Testing SSIM = {test_ssim}"
#                 std_ssim = f"Testing error SSIM STD = {ssim_std}"
#                 screen.insert(END, error + '\n')
#                 screen.insert(END, error_std + '\n')
#                 screen.insert(END, ssim + '\n')
#                 screen.insert(END, std_ssim + '\n')

#             if (scale.get()) == 20:

#                 label5 = true_sos[0:1, 0:1, :, :]
#                 recons5 = recons[0:1, 0:1, :, :]

#                 # flatten images
#                 label5 = np.reshape(label5, -1)
#                 recons5 = np.reshape(recons5, -1)

#                 label5 = label5 + label_noiseImg5
#                 recons5 = recons5 + label_noiseImg5
                
#                 noise_label5 = label5.reshape(1, 1, 214, 214)
#                 noise_recons5 = recons5.reshape(1, 1, 214, 214)

#                 SSIMS.append(ssim(noise_label5[0, 0, :, :], noise_recons5[0, 0, :, :], data_range=0.2))
#                 RMSES.append(np.linalg.norm(noise_label5[0, 0, :, :] - noise_recons5[0, 0, :, :]))

#                 rmse_error = np.array(RMSES).mean()
#                 rmse_std = np.array(RMSES).std()
#                 test_ssim = np.array(SSIMS).mean()
#                 ssim_std = np.array(SSIMS).std()

#                 screen.insert(INSERT, " " + '\n')
#                 screen.insert(INSERT, "***** Performing FWI Reconstruction *****" + '\n') 
#                 Reconstruction(noise_recons5[0, 0, :, :], f'{path}/Recons.png')
#                 error = f"Average testing error = {rmse_error}"
#                 error_std = f"Testing error standard deviation = {rmse_std}"
#                 ssim = f"Average Testing SSIM = {test_ssim}"
#                 std_ssim = f"Testing error SSIM STD = {ssim_std}"
#                 screen.insert(END, error + '\n')
#                 screen.insert(END, error_std + '\n')
#                 screen.insert(END, ssim + '\n')
#                 screen.insert(END, std_ssim + '\n')

#             if (scale.get()) == 25:

#                 label6 = true_sos[0:1, 0:1, :, :]
#                 recons6 = recons[0:1, 0:1, :, :]

#                 # flatten images
#                 label6 = np.reshape(label6, -1)
#                 recons6 = np.reshape(recons6, -1)

#                 label6 = label6 + label_noiseImg6
#                 recons6 = recons6 + label_noiseImg6
                
#                 noise_label6 = label6.reshape(1, 1, 214, 214)
#                 noise_recons6 = recons6.reshape(1, 1, 214, 214)

#                 SSIMS.append(ssim(noise_label6[0, 0, :, :], noise_recons6[0, 0, :, :], data_range=0.2))
#                 RMSES.append(np.linalg.norm(noise_label6[0, 0, :, :] - noise_recons6[0, 0, :, :]))

#                 rmse_error = np.array(RMSES).mean()
#                 rmse_std = np.array(RMSES).std()
#                 test_ssim = np.array(SSIMS).mean()
#                 ssim_std = np.array(SSIMS).std()

#                 screen.insert(INSERT, " " + '\n')
#                 screen.insert(INSERT, "***** Performing FWI Reconstruction *****" + '\n') 
#                 Reconstruction(noise_recons6[0, 0, :, :], f'{path}/Recons.png')
#                 error = f"Average testing error = {rmse_error}"
#                 error_std = f"Testing error standard deviation = {rmse_std}"
#                 ssim = f"Average Testing SSIM = {test_ssim}"
#                 std_ssim = f"Testing error SSIM STD = {ssim_std}"
#                 screen.insert(END, error + '\n')
#                 screen.insert(END, error_std + '\n')
#                 screen.insert(END, ssim + '\n')
#                 screen.insert(END, std_ssim + '\n')

#             if (scale.get()) == 30:

#                 label7 = true_sos[0:1, 0:1, :, :]
#                 recons7 = recons[0:1, 0:1, :, :]

#                 # flatten images
#                 label7 = np.reshape(label7, -1)
#                 recons7 = np.reshape(recons7, -1)

#                 label7 = label7 + label_noiseImg7
#                 recons7 = recons7 + label_noiseImg7
                
#                 noise_label7 = label7.reshape(1, 1, 214, 214)
#                 noise_recons7 = recons7.reshape(1, 1, 214, 214)

#                 SSIMS.append(ssim(noise_label7[0, 0, :, :], noise_recons7[0, 0, :, :], data_range=0.2))
#                 RMSES.append(np.linalg.norm(noise_label7[0, 0, :, :] - noise_recons7[0, 0, :, :]))

#                 rmse_error = np.array(RMSES).mean()
#                 rmse_std = np.array(RMSES).std()
#                 test_ssim = np.array(SSIMS).mean()
#                 ssim_std = np.array(SSIMS).std()

#                 screen.insert(INSERT, " " + '\n')
#                 screen.insert(INSERT, "***** Performing FWI Reconstruction *****" + '\n') 
#                 Reconstruction(noise_recons7[0, 0, :, :], f'{path}/Recons.png')
#                 error = f"Average testing error = {rmse_error}"
#                 error_std = f"Testing error standard deviation = {rmse_std}"
#                 ssim = f"Average Testing SSIM = {test_ssim}"
#                 std_ssim = f"Testing error SSIM STD = {ssim_std}"
#                 screen.insert(END, error + '\n')
#                 screen.insert(END, error_std + '\n')
#                 screen.insert(END, ssim + '\n')
#                 screen.insert(END, std_ssim + '\n')

#         time_elapsed = time.time() - since
#         time_elapsed = round(time_elapsed)
#         time = f"The time for FWI reconstruction prediction is {time_elapsed} seconds"
#         screen.insert(INSERT, " " + "\n")
#         screen.insert(END, time + '\n')


# def ReconstructTumor(scale_value=0):

#     from scipy.ndimage.measurements import label
#     import time

#     since = time.time()
    
#     #set arguments
#     path = '/home/pi/Desktop/USCT/results'
#     os.makedirs(path, exist_ok=True)
#     data_np = data.numpy()

#     tumor_classifier = classifier.TumorClassifier() #.to(dev)
#     class_sd = torch.load('/home/pi/Desktop/USCT/models/Classifier/classifier.pth',map_location=torch.device('cpu') )['model']
#     tumor_classifier.load_state_dict(class_sd)
#     tumor_classifier.eval()

#     if source == 1:
#         true_sos = np.load("/home/pi/Desktop/USCT/velocity_maps/data33.npy")
#     elif source == 0:
#         true_sos = np.load("/home/pi/Desktop/USCT/velocity_maps/data34.npy")

#     true_sos = true_sos[0:1, 0:1, :, :]
#     true_tumor = tumor_classifier(torch.from_numpy(true_sos).float()).cpu().detach().numpy() > 0.5	
#     recons = np.load("/home/pi/Desktop/USCT/classic_test_results_rob.npy")
#     recons = recons[0:1, 0:1, :, :]
#     tumor_recon = tumor_classifier(torch.from_numpy(recons).float()).cpu().detach().numpy() > 0.02

#     DICE = []
#     struct = np.ones((3,3), dtype=np.int)

#     if check_recons.get() == 0 and check_recons_tumor.get() == 1 and check_task.get() == 0 and check_tumor.get() == 0 or check_recons.get() == 1 and check_recons_tumor.get() == 1 and check_task.get() == 0 and check_tumor.get() == 0 or check_recons.get() == 1 and check_recons_tumor.get() == 0 and check_task.get() == 1 and check_tumor.get() == 0 or check_recons.get() == 1 and check_recons_tumor.get() == 0 and check_task.get() == 0 and check_tumor.get() == 1 or check_recons.get() == 0 and check_recons_tumor.get() == 1 and check_task.get() == 1 and check_tumor.get() == 0 or check_recons.get() == 0 and check_recons_tumor.get() == 1 and check_task.get() == 0 and check_tumor.get() == 1 or check_recons.get() == 0 and check_recons_tumor.get() == 0 and check_task.get() == 1 and check_tumor.get() == 1 or check_recons.get() == 1 and check_recons_tumor.get() == 1 and check_task.get() == 1 and check_tumor.get() == 0 or check_recons.get() == 1 and check_recons_tumor.get() == 0 and check_task.get() == 1 and check_tumor.get() == 1 or check_recons.get() == 0 and check_recons_tumor.get() == 1 and check_task.get() == 1 and check_tumor.get() == 1 or check_recons.get() == 1 and check_recons_tumor.get() == 1 and check_task.get() == 0 and check_tumor.get() == 1 or check_recons.get() == 1 and check_recons_tumor.get() == 1 and check_task.get() == 1 and check_tumor.get() == 1:

#         if check_clean.get() == 1 and len(acous_source_num.get()) == 0: 

#             tumor = tumor_recon[0, 0, :, :] * (2-true_tumor[0, 0, :, : ])

#             l, n0 = label(tumor_recon[0, 0, :, :], struct)
#             l, n1 = label(true_tumor[0, 0, :, :], struct)
#             dice_den = n0 + n1
#             if dice_den > 0:
#                 l, n2 = label(tumor_recon[0, 0, :, :]*true_tumor[0, 0, :, :], struct)
#                 DICE.append(2*n2/dice_den)
#             else:
#                 DICE.append(1.)
            
#             test_dice = np.array(DICE).mean()
#             dice_std = np.array(DICE).std()

#             screen.insert(INSERT, " " + '\n')
#             screen.insert(INSERT, "***** Predicting FWI Tumor *****" + '\n') 
#             ReconstructionTumor(tumor[:, :], f'{path}/ReconsTumor.png')

#             dice = f"Average Testing Dice = {test_dice}"
#             std_dice = f"Testing Dice STD = {dice_std}"
            
#             screen.insert(END, dice + '\n')
#             screen.insert(END, std_dice + '\n')

#         elif check_clean.get() == 1 and len(acous_source_num.get()) != 0:

#             tumor = tumor_recon[0, 0, :, :] * (2-true_tumor[0, 0, :, : ])

#             l, n0 = label(tumor_recon[0, 0, :, :], struct)
#             l, n1 = label(true_tumor[0, 0, :, :], struct)
#             dice_den = n0 + n1
#             if dice_den > 0:
#                 l, n2 = label(tumor_recon[0, 0, :, :]*true_tumor[0, 0, :, :], struct)
#                 DICE.append(2*n2/dice_den)
#             else:
#                 DICE.append(1.)
            
#             test_dice = np.array(DICE).mean()
#             dice_std = np.array(DICE).std()

#             screen.insert(INSERT, " " + '\n')
#             screen.insert(INSERT, "***** Performing FWI Tumor *****" + '\n') 
#             ReconstructionTumor(tumor[0, 0, :, :], f'{path}/ReconsTumor.png')

#             dice = f"Average Testing Dice = {test_dice}"
#             std_dice = f"Testing Dice STD = {dice_std}"
            
#             screen.insert(END, dice + '\n')
#             screen.insert(END, std_dice + '\n')
            
#         if (scale.get()) == 0 and check_clean.get() == 0:

#             # flatten reconstruct array
#             tumor_recon1 = np.reshape(tumor_recon, -1)
#             true_tumor1 = np.reshape(true_tumor, -1)

#             tumor_recon1 = tumor_recon1 + label_noiseImg1
#             true_tumor1 = true_tumor1 + label_noiseImg1

#             tumor_recon1 = tumor_recon1.reshape(1, 1, 214, 214)
#             true_tumor1 = true_tumor1.reshape(1, 1, 214, 214)
            
#             data_min = tumor_recon1.min()
#             data_max = tumor_recon1.max()

#             tumor1 = tumor_recon1[0, 0, :, :] * (2-true_tumor1[0, 0, :, : ])
            
#             data_min = tumor1.min()
#             data_max = tumor1.max()

#             l, n0 = label(tumor_recon1[0, 0, :, :], struct)
#             l, n1 = label(true_tumor1[0, 0, :, :], struct)
#             dice_den = n0 + n1
#             if dice_den > 0:
#                 l, n2 = label(tumor_recon1[0, 0, :, :]*true_tumor1[0, 0, :, :], struct)
#                 DICE.append(2*n2/dice_den)
#             else:
#                 DICE.append(1.)
            
#             test_dice = np.array(DICE).mean()
#             dice_std = np.array(DICE).std()

#             screen.insert(INSERT, " " + '\n')
#             screen.insert(INSERT, "***** Performing FWI Tumor *****" + '\n') 
#             ReconstructionTumor(tumor1[:, :], f'{path}/ReconsTumor.png')

#             dice = f"Average Testing Dice = {test_dice}"
#             std_dice = f"Testing Dice STD = {dice_std}"
            
#             screen.insert(END, dice + '\n')
#             screen.insert(END, std_dice + '\n')

#         if (scale.get()) == 5 and check_clean.get() == 0:
            
#              # flatten reconstruct array
#             tumor_recon2 = np.reshape(tumor_recon, -1)
#             true_tumor2 = np.reshape(true_tumor, -1)

#             tumor_recon2 = tumor_recon2 + label_noiseImg2
#             true_tumor2 = true_tumor2 + label_noiseImg2

#             tumor_recon2 = tumor_recon2.reshape(1, 1, 214, 214)
#             true_tumor2 = true_tumor2.reshape(1, 1, 214, 214)
            
#             data_min = tumor_recon2.min()
#             data_max = tumor_recon2.max()

#             tumor2 = tumor_recon2[0, 0, :, :] * (2-true_tumor2[0, 0, :, : ])
            
#             l, n0 = label(tumor_recon2[0, 0, :, :], struct)
#             l, n1 = label(true_tumor2[0, 0, :, :], struct)
#             dice_den = n0 + n1
#             if dice_den > 0:
#                 l, n2 = label(tumor_recon2[0, 0, :, :]*true_tumor2[0, 0, :, :], struct)
#                 DICE.append(2*n2/dice_den)
#             else:
#                 DICE.append(1.)
            
#             test_dice = np.array(DICE).mean()
#             dice_std = np.array(DICE).std()

#             screen.insert(INSERT, " " + '\n')
#             screen.insert(INSERT, "***** Performing FWI Tumor *****" + '\n') 
#             ReconstructionTumor(tumor2[:, :], f'{path}/ReconsTumor.png')

#             dice = f"Average Testing Dice = {test_dice}"
#             std_dice = f"Testing Dice STD = {dice_std}"
            
#             screen.insert(END, dice + '\n')
#             screen.insert(END, std_dice + '\n')

#         if (scale.get()) == 10 and check_clean.get() == 0:

#             # flatten reconstruct array
#             tumor_recon3 = np.reshape(tumor_recon, -1)
#             true_tumor3 = np.reshape(true_tumor, -1)

#             tumor_recon3 = tumor_recon3 + label_noiseImg3
#             true_tumor3 = true_tumor3 + label_noiseImg3

#             tumor_recon3 = tumor_recon3.reshape(1, 1, 214, 214)
#             true_tumor3 = true_tumor3.reshape(1, 1, 214, 214)

#             data_min = tumor_recon3.min()
#             data_max = tumor_recon3.max()

#             tumor3 = tumor_recon3[0, 0, :, :] * (2-true_tumor3[0, 0, :, : ])
            
#             l, n0 = label(tumor_recon3[0, 0, :, :], struct)
#             l, n1 = label(true_tumor3[0, 0, :, :], struct)
#             dice_den = n0 + n1
#             if dice_den > 0:
#                 l, n2 = label(tumor_recon3[0, 0, :, :]*true_tumor3[0, 0, :, :], struct)
#                 DICE.append(2*n2/dice_den)
#             else:
#                 DICE.append(1.)
            
#             test_dice = np.array(DICE).mean()
#             dice_std = np.array(DICE).std()

#             screen.insert(INSERT, " " + '\n')
#             screen.insert(INSERT, "***** Performing FWI Tumor *****" + '\n') 
#             ReconstructionTumor(tumor3[:, :], f'{path}/ReconsTumor.png')

#             dice = f"Average Testing Dice = {test_dice}"
#             std_dice = f"Testing Dice STD = {dice_std}"
            
#             screen.insert(END, dice + '\n')
#             screen.insert(END, std_dice + '\n')

#         if (scale.get()) == 15 and check_clean.get() == 0:

#             # flatten reconstruct array
#             tumor_recon4 = np.reshape(tumor_recon, -1)
#             true_tumor4 = np.reshape(true_tumor, -1)

#             tumor_recon4 = tumor_recon4 + label_noiseImg4
#             true_tumor4 = true_tumor4 + label_noiseImg4

#             tumor_recon4 = tumor_recon4.reshape(1, 1, 214, 214)
#             true_tumor4 = true_tumor4.reshape(1, 1, 214, 214)

#             data_min = tumor_recon4.min()
#             data_max = tumor_recon4.max()

#             tumor4 = tumor_recon4[0, 0, :, :] * (2-true_tumor4[0, 0, :, : ])
            
#             l, n0 = label(tumor_recon4[0, 0, :, :], struct)
#             l, n1 = label(true_tumor4[0, 0, :, :], struct)
#             dice_den = n0 + n1
#             if dice_den > 0:
#                 l, n2 = label(tumor_recon4[0, 0, :, :]*true_tumor4[0, 0, :, :], struct)
#                 DICE.append(2*n2/dice_den)
#             else:
#                 DICE.append(1.)
            
#             test_dice = np.array(DICE).mean()
#             dice_std = np.array(DICE).std()

#             screen.insert(INSERT, " " + '\n')
#             screen.insert(INSERT, "***** Performing FWI Tumor *****" + '\n') 
#             ReconstructionTumor(tumor4[:, :], f'{path}/ReconsTumor.png')

#             dice = f"Average Testing Dice = {test_dice}"
#             std_dice = f"Testing Dice STD = {dice_std}"
            
#             screen.insert(END, dice + '\n')
#             screen.insert(END, std_dice + '\n')

#         if (scale.get()) == 20 and check_clean.get() == 0:

#             # flatten reconstruct array
#             tumor_recon5 = np.reshape(tumor_recon, -1)
#             true_tumor5 = np.reshape(true_tumor, -1)

#             tumor_recon5 = tumor_recon5 + label_noiseImg5
#             true_tumor5 = true_tumor5 + label_noiseImg5

#             tumor_recon5 = tumor_recon5.reshape(1, 1, 214, 214)
#             true_tumor5 = true_tumor5.reshape(1, 1, 214, 214)

#             data_min = tumor_recon5.min()
#             data_max = tumor_recon5.max()

#             tumor5 = tumor_recon5[0, 0, :, :] * (2-true_tumor5[0, 0, :, : ])
            
#             l, n0 = label(tumor_recon5[0, 0, :, :], struct)
#             l, n1 = label(true_tumor5[0, 0, :, :], struct)
#             dice_den = n0 + n1
#             if dice_den > 0:
#                 l, n2 = label(tumor_recon5[0, 0, :, :]*true_tumor5[0, 0, :, :], struct)
#                 DICE.append(2*n2/dice_den)
#             else:
#                 DICE.append(1.)
            
#             test_dice = np.array(DICE).mean()
#             dice_std = np.array(DICE).std()

#             screen.insert(INSERT, " " + '\n')
#             screen.insert(INSERT, "***** Performing FWI Tumor *****" + '\n') 
#             ReconstructionTumor(tumor5[:, :], f'{path}/ReconsTumor.png')

#             dice = f"Average Testing Dice = {test_dice}"
#             std_dice = f"Testing Dice STD = {dice_std}"
            
#             screen.insert(END, dice + '\n')
#             screen.insert(END, std_dice + '\n')

#         if (scale.get()) == 25 and check_clean.get() == 0:

#             # flatten reconstruct array
#             tumor_recon6 = np.reshape(tumor_recon, -1)
#             true_tumor6 = np.reshape(true_tumor, -1)

#             tumor_recon6 = tumor_recon6 + label_noiseImg6
#             true_tumor6 = true_tumor6 + label_noiseImg6

#             tumor_recon6 = tumor_recon6.reshape(1, 1, 214, 214)
#             true_tumor6 = true_tumor6.reshape(1, 1, 214, 214)

#             data_min = tumor_recon6.min()
#             data_max = tumor_recon6.max()

#             tumor6 = tumor_recon6[0, 0, :, :] * (2-true_tumor6[0, 0, :, : ])
            
#             l, n0 = label(tumor_recon6[0, 0, :, :], struct)
#             l, n1 = label(true_tumor6[0, 0, :, :], struct)
#             dice_den = n0 + n1
#             if dice_den > 0:
#                 l, n2 = label(tumor_recon6[0, 0, :, :]*true_tumor6[0, 0, :, :], struct)
#                 DICE.append(2*n2/dice_den)
#             else:
#                 DICE.append(1.)
            
#             test_dice = np.array(DICE).mean()
#             dice_std = np.array(DICE).std()

#             screen.insert(INSERT, " " + '\n')
#             screen.insert(INSERT, "***** Performing FWI Tumor *****" + '\n') 
#             ReconstructionTumor(tumor6[:, :], f'{path}/ReconsTumor.png')

#             dice = f"Average Testing Dice = {test_dice}"
#             std_dice = f"Testing Dice STD = {dice_std}"
            
#             screen.insert(END, dice + '\n')
#             screen.insert(END, std_dice + '\n')

#         if (scale.get()) == 30 and check_clean.get() == 0:

#             # flatten reconstruct array
#             tumor_recon7 = np.reshape(tumor_recon, -1)
#             true_tumor7 = np.reshape(true_tumor, -1)

#             tumor_recon7 = tumor_recon7 + label_noiseImg7
#             true_tumor7 = true_tumor7 + label_noiseImg7

#             tumor_recon7 = tumor_recon7.reshape(1, 1, 214, 214)
#             true_tumor7 = true_tumor7.reshape(1, 1, 214, 214)

#             data_min = tumor_recon7.min()
#             data_max = tumor_recon7.max()

#             tumor7 = tumor_recon7[0, 0, :, :] * (2-true_tumor7[0, 0, :, : ])
            
#             l, n0 = label(tumor_recon7[0, 0, :, :], struct)
#             l, n1 = label(true_tumor7[0, 0, :, :], struct)
#             dice_den = n0 + n1
#             if dice_den > 0:
#                 l, n2 = label(tumor_recon7[0, 0, :, :]*true_tumor7[0, 0, :, :], struct)
#                 DICE.append(2*n2/dice_den)
#             else:
#                 DICE.append(1.)
            
#             test_dice = np.array(DICE).mean()
#             dice_std = np.array(DICE).std()

#             screen.insert(INSERT, " " + '\n')
#             screen.insert(INSERT, "***** Performing FWI Tumor *****" + '\n') 
#             ReconstructionTumor(tumor7[:, :], f'{path}/ReconsTumor.png')

#             dice = f"Average Testing Dice = {test_dice}"
#             std_dice = f"Testing Dice STD = {dice_std}"
            
#             screen.insert(END, dice + '\n')
#             screen.insert(END, std_dice + '\n')

#         if len(acous_source_num.get()) != 0 and check_clean.get() == 0:

#             if (scale.get()) == 0:

#                 # flatten reconstruct array
#                 tumor_recon1 = np.reshape(tumor_recon, -1)
#                 true_tumor1 = np.reshape(true_tumor, -1)

#                 tumor_recon1 = tumor_recon1 + label_noiseImg1
#                 true_tumor1 = true_tumor1 + label_noiseImg1

#                 tumor_recon1 = tumor_recon1.reshape(1, 1, 214, 214)
#                 true_tumor1 = true_tumor1.reshape(1, 1, 214, 214)

#                 data_min = tumor_recon1.min()
#                 data_max = tumor_recon1.max()

#                 tumor1 = tumor_recon1[0, 0, :, :] * (2-true_tumor1[0, 0, :, : ])

#                 l, n0 = label(tumor_recon1[0, 0, :, :], struct)
#                 l, n1 = label(true_tumor1[0, 0, :, :], struct)
#                 dice_den = n0 + n1
#                 if dice_den > 0:
#                     l, n2 = label(tumor_recon1[0, 0, :, :]*true_tumor1[0, 0, :, :], struct)
#                     DICE.append(2*n2/dice_den)
#                 else:
#                     DICE.append(1.)
                
#                 test_dice = np.array(DICE).mean()
#                 dice_std = np.array(DICE).std()

#                 screen.insert(INSERT, " " + '\n')
#                 screen.insert(INSERT, "***** Performing FWI Tumor *****" + '\n') 
#                 ReconstructionTumor(tumor1[:, :], f'{path}/ReconsTumor.png')

#                 dice = f"Average Testing Dice = {test_dice}"
#                 std_dice = f"Testing Dice STD = {dice_std}"
                
#                 screen.insert(END, dice + '\n')
#                 screen.insert(END, std_dice + '\n')

#             if (scale.get()) == 5:

#                 # flatten reconstruct array
#                 tumor_recon2 = np.reshape(tumor_recon, -1)
#                 true_tumor2 = np.reshape(true_tumor, -1)

#                 tumor_recon2 = tumor_recon2 + label_noiseImg2
#                 true_tumor2 = true_tumor2 + label_noiseImg2

#                 tumor_recon2 = tumor_recon2.reshape(1, 1, 214, 214)
#                 true_tumor2 = true_tumor2.reshape(1, 1, 214, 214)

#                 data_min = tumor_recon2.min()
#                 data_max = tumor_recon2.max()

#                 tumor2 = tumor_recon2[0, 0, :, :] * (2-true_tumor2[0, 0, :, : ])

#                 l, n0 = label(tumor_recon2[0, 0, :, :], struct)
#                 l, n1 = label(true_tumor2[0, 0, :, :], struct)
#                 dice_den = n0 + n1
#                 if dice_den > 0:
#                     l, n2 = label(tumor_recon2[0, 0, :, :]*true_tumor2[0, 0, :, :], struct)
#                     DICE.append(2*n2/dice_den)
#                 else:
#                     DICE.append(1.)
                
#                 test_dice = np.array(DICE).mean()
#                 dice_std = np.array(DICE).std()

#                 screen.insert(INSERT, " " + '\n')
#                 screen.insert(INSERT, "***** Performing FWI Tumor *****" + '\n') 
#                 ReconstructionTumor(tumor2[:, :], f'{path}/ReconsTumor.png')

#                 dice = f"Average Testing Dice = {test_dice}"
#                 std_dice = f"Testing Dice STD = {dice_std}"
                
#                 screen.insert(END, dice + '\n')
#                 screen.insert(END, std_dice + '\n')

#             if (scale.get()) == 10:

#                 # flatten reconstruct array
#                 tumor_recon3 = np.reshape(tumor_recon, -1)
#                 true_tumor3 = np.reshape(true_tumor, -1)

#                 tumor_recon3 = tumor_recon3 + label_noiseImg3
#                 true_tumor3 = true_tumor3 + label_noiseImg3

#                 tumor_recon3 = tumor_recon3.reshape(1, 1, 214, 214)
#                 true_tumor3 = true_tumor3.reshape(1, 1, 214, 214)

#                 data_min = tumor_recon3.min()
#                 data_max = tumor_recon3.max()

#                 tumor3 = tumor_recon3[0, 0, :, :] * (2-true_tumor3[0, 0, :, : ])

#                 l, n0 = label(tumor_recon3[0, 0, :, :], struct)
#                 l, n1 = label(true_tumor3[0, 0, :, :], struct)
#                 dice_den = n0 + n1
#                 if dice_den > 0:
#                     l, n2 = label(tumor_recon3[0, 0, :, :]*true_tumor3[0, 0, :, :], struct)
#                     DICE.append(2*n2/dice_den)
#                 else:
#                     DICE.append(1.)
                
#                 test_dice = np.array(DICE).mean()
#                 dice_std = np.array(DICE).std()

#                 screen.insert(INSERT, " " + '\n')
#                 screen.insert(INSERT, "***** Performing FWI Tumor *****" + '\n') 
#                 ReconstructionTumor(tumor3[:, :], f'{path}/ReconsTumor.png')

#                 dice = f"Average Testing Dice = {test_dice}"
#                 std_dice = f"Testing Dice STD = {dice_std}"
                
#                 screen.insert(END, dice + '\n')
#                 screen.insert(END, std_dice + '\n')

#             if (scale.get()) == 15:

#                 # flatten reconstruct array
#                 tumor_recon4 = np.reshape(tumor_recon, -1)
#                 true_tumor4 = np.reshape(true_tumor, -1)

#                 tumor_recon4 = tumor_recon4 + label_noiseImg4
#                 true_tumor4 = true_tumor4 + label_noiseImg4

#                 tumor_recon4 = tumor_recon4.reshape(1, 1, 214, 214)
#                 true_tumor4 = true_tumor4.reshape(1, 1, 214, 214)

#                 data_min = tumor_recon4.min()
#                 data_max = tumor_recon4.max()

#                 tumor4 = tumor_recon4[0, 0, :, :] * (2-true_tumor4[0, 0, :, : ])

#                 l, n0 = label(tumor_recon4[0, 0, :, :], struct)
#                 l, n1 = label(true_tumor4[0, 0, :, :], struct)
#                 dice_den = n0 + n1
#                 if dice_den > 0:
#                     l, n2 = label(tumor_recon4[0, 0, :, :]*true_tumor4[0, 0, :, :], struct)
#                     DICE.append(2*n2/dice_den)
#                 else:
#                     DICE.append(1.)
                
#                 test_dice = np.array(DICE).mean()
#                 dice_std = np.array(DICE).std()

#                 screen.insert(INSERT, " " + '\n')
#                 screen.insert(INSERT, "***** Performing FWI Tumor *****" + '\n') 
#                 ReconstructionTumor(tumor4[:, :], f'{path}/ReconsTumor.png')

#                 dice = f"Average Testing Dice = {test_dice}"
#                 std_dice = f"Testing Dice STD = {dice_std}"
                
#                 screen.insert(END, dice + '\n')
#                 screen.insert(END, std_dice + '\n')

#             if (scale.get()) == 20:

#                 # flatten reconstruct array
#                 tumor_recon5 = np.reshape(tumor_recon, -1)
#                 true_tumor5 = np.reshape(true_tumor, -1)

#                 tumor_recon5 = tumor_recon5 + label_noiseImg5
#                 true_tumor5 = true_tumor5 + label_noiseImg5

#                 tumor_recon5 = tumor_recon5.reshape(1, 1, 214, 214)
#                 true_tumor5 = true_tumor5.reshape(1, 1, 214, 214)

#                 data_min = tumor_recon5.min()
#                 data_max = tumor_recon5.max()

#                 tumor5 = tumor_recon5[0, 0, :, :] * (2-true_tumor5[0, 0, :, : ])

#                 l, n0 = label(tumor_recon5[0, 0, :, :], struct)
#                 l, n1 = label(true_tumor5[0, 0, :, :], struct)
#                 dice_den = n0 + n1
#                 if dice_den > 0:
#                     l, n2 = label(tumor_recon5[0, 0, :, :]*true_tumor5[0, 0, :, :], struct)
#                     DICE.append(2*n2/dice_den)
#                 else:
#                     DICE.append(1.)
                
#                 test_dice = np.array(DICE).mean()
#                 dice_std = np.array(DICE).std()

#                 screen.insert(INSERT, " " + '\n')
#                 screen.insert(INSERT, "***** Performing FWI Tumor *****" + '\n') 
#                 ReconstructionTumor(tumor5[:, :], f'{path}/ReconsTumor.png')

#                 dice = f"Average Testing Dice = {test_dice}"
#                 std_dice = f"Testing Dice STD = {dice_std}"
                
#                 screen.insert(END, dice + '\n')
#                 screen.insert(END, std_dice + '\n')

#             if (scale.get()) == 25:

#                 # flatten reconstruct array
#                 tumor_recon6 = np.reshape(tumor_recon, -1)
#                 true_tumor6 = np.reshape(true_tumor, -1)

#                 tumor_recon6 = tumor_recon6 + label_noiseImg6
#                 true_tumor6 = true_tumor6 + label_noiseImg6

#                 tumor_recon6 = tumor_recon6.reshape(1, 1, 214, 214)
#                 true_tumor6 = true_tumor6.reshape(1, 1, 214, 214)

#                 data_min = tumor_recon6.min()
#                 data_max = tumor_recon6.max()

#                 tumor6 = tumor_recon6[0, 0, :, :] * (2-true_tumor6[0, 0, :, : ])

#                 l, n0 = label(tumor_recon6[0, 0, :, :], struct)
#                 l, n1 = label(true_tumor6[0, 0, :, :], struct)
#                 dice_den = n0 + n1
#                 if dice_den > 0:
#                     l, n2 = label(tumor_recon6[0, 0, :, :]*true_tumor6[0, 0, :, :], struct)
#                     DICE.append(2*n2/dice_den)
#                 else:
#                     DICE.append(1.)
                
#                 test_dice = np.array(DICE).mean()
#                 dice_std = np.array(DICE).std()

#                 screen.insert(INSERT, " " + '\n')
#                 screen.insert(INSERT, "***** Performing FWI Tumor *****" + '\n') 
#                 ReconstructionTumor(tumor6[:, :], f'{path}/ReconsTumor.png')

#                 dice = f"Average Testing Dice = {test_dice}"
#                 std_dice = f"Testing Dice STD = {dice_std}"
                
#                 screen.insert(END, dice + '\n')
#                 screen.insert(END, std_dice + '\n')

#             if (scale.get()) == 30:

#                 # flatten reconstruct array
#                 tumor_recon7 = np.reshape(tumor_recon, -1)
#                 true_tumor7 = np.reshape(true_tumor, -1)

#                 tumor_recon7 = tumor_recon7 + label_noiseImg7
#                 true_tumor7 = true_tumor7 + label_noiseImg7

#                 tumor_recon7 = tumor_recon7.reshape(1, 1, 214, 214)
#                 true_tumor7 = true_tumor7.reshape(1, 1, 214, 214)

#                 data_min = tumor_recon7.min()
#                 data_max = tumor_recon7.max()

#                 tumor7 = tumor_recon7[0, 0, :, :] * (2-true_tumor7[0, 0, :, : ])

#                 l, n0 = label(tumor_recon7[0, 0, :, :], struct)
#                 l, n1 = label(true_tumor7[0, 0, :, :], struct)
#                 dice_den = n0 + n1
#                 if dice_den > 0:
#                     l, n2 = label(tumor_recon7[0, 0, :, :]*true_tumor7[0, 0, :, :], struct)
#                     DICE.append(2*n2/dice_den)
#                 else:
#                     DICE.append(1.)
                
#                 test_dice = np.array(DICE).mean()
#                 dice_std = np.array(DICE).std()

#                 screen.insert(INSERT, " " + '\n')
#                 screen.insert(INSERT, "***** Performing FWI Tumor *****" + '\n') 
#                 ReconstructionTumor(tumor7[:, :], f'{path}/ReconsTumor.png')

#                 dice = f"Average Testing Dice = {test_dice}"
#                 std_dice = f"Testing Dice STD = {dice_std}"
                
#                 screen.insert(END, dice + '\n')
#                 screen.insert(END, std_dice + '\n')

#         time_elapsed = time.time() - since
#         time_elapsed = round(time_elapsed)
#         time = f"The time for FWI reconstruction tumor is {time_elapsed} seconds"
#         screen.insert(INSERT, " " + "\n")
#         screen.insert(END, time + '\n')


var3 = StringVar()
scale = tk.Scale(root, variable=var3, font=("helvetica", 11), orient='horizontal', bg = "gray80", bd=1, from_=0, to=30, 
                            tickinterval=5, resolution=5, length=200, command=())
scale.place(x=548, y=135)

screen = tkscrolled.ScrolledText(root, bd=2, wrap="word")
screen.place(x=5, y=770, width=472, height=215)
screen.configure(font=("courier", 12))

var10 = StringVar()
acous_source_num = Entry(root, textvariable=var10, font = ("Times", 12, "bold"), bg="gray80", justify=CENTER, highlightthickness=2)
acous_source_num.config(highlightbackground="black", highlightcolor="black")
acous_source_num.place(x=215, y=240, width=60, height=25)
acous_source_label = customtkinter.CTkLabel(master=root, text="Enter Source No. (Range: 0-63)", text_font=("helvetica", 13), width=15, fg_color="gray80", corner_radius=1)
acous_source_label.place(x=110, y=210)

def clean_check():
    if check_clean.get() == 1:
        screen.insert(INSERT, "***** Clean Ultrasound Data Loaded *****" + '\n') 


check_recons = IntVar()
check_recons_tumor = IntVar()
check_task = IntVar()
check_tumor = IntVar()
check_clean = IntVar()

receive_button = customtkinter.CTkButton(master=root, image=img5, text="Receive Ultrasound Data", text_font=("helvetica", 13), width=220, height=25, 
                                    corner_radius=10, compound="right", fg_color="skyblue2", bg_color="skyblue2", command=ReceiveCall)
receive_button.place(x=113, y=50)

task_button = customtkinter.CTkButton(master=root, image=img6, text="Predict Task-based Reconstruction", text_font=("helvetica", 13), width=180, height=25, compound="right", 
                                   corner_radius=10, fg_color="skyblue2", bg_color="skyblue2", command=TaskPred)
task_button.place(x=1210, y=50)

reconst_button = customtkinter.CTkButton(master=root, image=img7, text="Predict Reconstruction", text_font=("helvetica", 13), width=220, height=25,
                                    corner_radius=10, compound="right", fg_color="skyblue2", bg_color="skyblue2", command=ReconsPred)
reconst_button.place(x=525, y=50)

tumor_button = customtkinter.CTkButton(master=root, image=img8, text="Predict Task-based Tumor ", text_font=("helvetica", 13), width=220, height=25,
                                corner_radius=10, compound="right", fg_color="skyblue2", bg_color="skyblue2", command=TumorPred)
tumor_button.place(x=1585, y=50)

recons_tumor_button = customtkinter.CTkButton(master=root, image=img11, text="Predict Reconstruction Tumor", text_font=("helvetica", 13), width=220, height=25,
                                corner_radius=10, compound="right", fg_color="skyblue2", bg_color="skyblue2", command=ReconsTumorPred)
recons_tumor_button.place(x=820, y=50)

uncheck = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/USCT/images/uncheck.PNG'))
check = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/USCT/images/check.PNG'))
recons_uncheck = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/USCT/images/recons_uncheck.PNG').resize((35, 35), Image.ANTIALIAS))
recons_check = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/USCT/images/recons_check.PNG').resize((35, 35), Image.ANTIALIAS))
recons_tumor_uncheck = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/USCT/images/recons_tumor_uncheck.PNG').resize((35, 35), Image.ANTIALIAS))
recons_tumor_check = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/USCT/images/recons_tumor_check.PNG').resize((35, 35), Image.ANTIALIAS))
task_uncheck = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/USCT/images/task_uncheck.PNG').resize((35, 35), Image.ANTIALIAS))
task_check = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/USCT/images/task_check.PNG').resize((35, 35), Image.ANTIALIAS))
tumor_uncheck = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/USCT/images/tumor_uncheck.PNG').resize((35, 35), Image.ANTIALIAS))
tumor_check = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/USCT/images/tumor_check.PNG').resize((35, 35), Image.ANTIALIAS))

clean_data_check = Checkbutton(root, text = ' Clean Ultrasound Data ', font = ("helvetica", 13), bd=0, variable = check_clean, compound="left", image=uncheck,
                selectimage=check, indicatoron=False, selectcolor="gray80", bg="gray80", activebackground="gray80", command=clean_check)
clean_data_check.place(x=535, y=230)
# clean_data_label = customtkinter.CTkLabel(master=root, text="Enter Source No. (Range: 0-63)", text_font=("helvetica", 13), width=15, fg_color="gray80", corner_radius=1)
# clean_data_label.place(x=1000, y=210)

datamenu = Menu(menubar, tearoff=0, selectcolor="gray80")
ultraImg = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/USCT/images/ultra.png').resize((120,35), Image.ANTIALIAS))
datamenu.add_command(label="  ", image=ultraImg, compound=RIGHT, command=open_phantom)
menubar.add_cascade(label="Local Data", menu=datamenu)

modelmenu = Menu(menubar, tearoff=0, selectcolor="gray80")
modelmenu.add_checkbutton(label="Reconstruction Model                ", font = ("helvetica", 14), compound="right", image=recons_uncheck, selectimage=check, 
                                variable=check_recons, indicatoron=False, selectcolor="gray80", activebackground="gray80", command=())
modelmenu.add_checkbutton(label="Reconstruction Tumor Model    ", font = ("helvetica", 14), compound="right", image=recons_tumor_uncheck, selectimage=check, 
                               variable=check_recons_tumor, indicatoron=False, selectcolor="gray80", activebackground="gray80", command=())
modelmenu.add_checkbutton(label="Task-based Model                     ", font = ("helvetica", 14), compound="right", image=task_uncheck, selectimage=check, 
                               variable=check_task, indicatoron=False, selectcolor="gray80", activebackground="gray80", command=())
modelmenu.add_checkbutton(label="Task-based Tumor Model          ", font = ("helvetica", 14), compound="right", image=tumor_uncheck, selectimage=check, 
                               variable=check_tumor, indicatoron=False, selectcolor="gray80", activebackground="gray80", command=())
menubar.add_cascade(label="Models", menu=modelmenu)

helpmenu = Menu(menubar, tearoff=0, selectcolor="gray80")
helpmenu.add_command(label="About", font = ("helvetica", 14), command=help_menu)
menubar.add_cascade(label="Help", menu=helpmenu)

root.config(menu=menubar)
root.mainloop()