from tkinter import *
from tkinter import filedialog as fd
from PIL import Image, ImageShow, ImageTk

import os
import pathlib

import numpy as np
import cv2 as cv

import argparse
import sys

modes = (cv.Stitcher_PANORAMA, cv.Stitcher_SCANS)

class Application(Frame):
    def __init__ (self):
        self.main_window = Tk()
        #self.main_window.geometry("1200x500")
        self.main_window.title("Panorama Stitcher")
        self.main_window.configure(bg='grey23')
        self.mainFrame = Frame(self.main_window, bg='grey23')
        self.mainFrame.grid(padx=20, pady=20, sticky='nsew')
        
        self.imagesChosen = StringVar()
        self.imageSaved = StringVar()

        self.chooseImagesButton = Button(self.mainFrame, padx=5, pady=5, 
            bg='grey45', fg='white', activebackground='grey55', activeforeground='white', 
            text = "Select 2+ images", font = ("Roboto", 14), command = self.chooseImages)
        self.chooseImagesButton.grid(row=1, column=1)

        self.imagesChosenLabel = Label(self.mainFrame, textvariable = self.imagesChosen, font = ("Roboto", 10), bg='grey23', fg='white')
        self.imagesChosenLabel.grid(row=2, column=1)
        
        self.saveImageButton = Button(self.mainFrame, padx=5, pady=5,
            bg='grey45', fg='white', activebackground='grey55', activeforeground='white', 
            text = "Save to", font = ("Roboto", 14), command = self.saveImage)
        self.saveImageButton.grid(row=1, column=2)

        self.saveImageLabel = Label(self.mainFrame, textvariable = self.imageSaved, font = ("Roboto", 10), bg='grey23', fg='white')
        self.saveImageLabel.grid(row=2, column=2)

        self.panoramaButton = Button(self.mainFrame, padx=5, pady=5,
            bg='grey45', fg='white', activebackground='grey55', activeforeground='white', 
            text = "Compile Panorama", font = ("Roboto", 14), command = self.runPanorama)
        self.panoramaButton.grid(row=1, column=3)

        """
        self.img = PhotoImage(self.imageSaved.get())
        self.img1 = self.img.subsample(20, 20)
        Label(self.mainFrame, image = self.img1).grid(row = 0, column = 4, columnspan = 2, rowspan = 2, padx = 5, pady = 5)
        """

        
        self.saturation_slider = Scale(self.main_window, length=300, from_=-100, to=100, 
            tickinterval=10, orient=HORIZONTAL, bg="grey20", fg="white")
        self.saturation_slider.grid(padx=10, pady=10, row=4, column=2)
        self.saturation_slider.set(0)
        self.satLabel = Label(self.mainFrame, text="Saturation slider", font = ("Roboto", 12), bg='grey23', fg='white')
        self.satLabel.grid(row=3, column=2)
        

        self.sizeOptBox = Listbox(self.main_window, bg="grey20", fg="white", selectbackground="grey40", 
            selectmode=SINGLE, font=(14), height=5) #setting listbox style to show up to 5 lines at once
        #listbox items in order they appear
        self.sizeOptBox.insert(1, "Small")
        self.sizeOptBox.insert(2, "Medium")
        self.sizeOptBox.insert(3, "Large")
        self.sizeOptBox.grid(padx=10, pady=10, row=4, column=1, sticky="w")

        mainloop()

    def chooseImages(self):
        names = fd.askopenfilenames(filetypes = fileTypes)
        temp = ""
        loadNames.clear()
        for i in names:
            temp += pathlib.PurePath(i).name + " "
            print(pathlib.PurePath(i).name)
            loadNames.append(pathlib.PurePath(i))
        self.imagesChosen.set(temp)
        #print(name)

    def saveImage(self):
        saveName.clear()
        names = ""
        names = fd.asksaveasfilename(filetypes = fileTypes, defaultextension = '.jpg', initialfile = "result.jpg")
        temp = pathlib.Path(names).name
        saveName.append(pathlib.Path(names))
        self.imageSaved.set(temp)

    def runPanorama(self):
        if (self.imagesChosen.get() == "" or self.imageSaved.get() == ""):
            print("Select an image to save or load first")
        else:
            stitchFunc()
            #print(saveName[0])
            result = Image.open(self.imageSaved.get())
            # result = Image.show(saveName[0])
            # result.thumbnail((400, 400))
            result.show()
            self.img = PhotoImage(file=r"result.jpg")
            #self.img1 = self.img.subsample(90, 50)
            Label(self.mainFrame, image = self.img, width=90, height=50).grid(row = 3, column = 4, columnspan = 2, rowspan = 2, padx = 5, pady = 5)


loadNames = []
saveName = []
fileTypes = [('Images', '.jpg'), ('Images', '.png'), ('All files', '*')]

def stitchFunc():
    imgs = []
    p = pathlib.PurePath(os.getcwd())
    pIn = p / 'march22Demo'
    searchPath = str(pIn)
    for imgName in loadNames:
        
        cv.samples.addSamplesDataSearchPath(searchPath)
        cv.samples.addSamplesDataSearchPath(str(imgName.parent))
        img_name = str(imgName.name)
        print(img_name)

        img = cv.imread(cv.samples.findFile(img_name))
        if img is None:
            print("can't read image ", img_name)
            sys.exit(-1)
        imgs.append(img)

    stitcher = cv.Stitcher.create(cv.Stitcher_PANORAMA)
    status, pano = stitcher.stitch(imgs)

    if status != cv.Stitcher_OK:
        print("can't stitch images error code %d" % status)
        sys.exit(-1)

    cv.imwrite(str(saveName[0]), pano)
    print("stitching completed successfully. %s saved!" % saveName[0].name)

    print('Done')
    

def main():
    app = Application()

if __name__ == '__main__':
    main()