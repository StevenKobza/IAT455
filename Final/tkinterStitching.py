from tkinter import *
from tkinter import filedialog as fd
from PIL import Image, ImageShow, ImageTk

import os
import pathlib

import numpy as np
import cv2 as cv

import argparse
import sys

import imutils

modes = (cv.Stitcher_PANORAMA, cv.Stitcher_SCANS)


class Application(Frame):
    def __init__ (self):
        self.main_window = Tk()
        self.main_window.geometry("1080x720")
        self.main_window.title("Panorama Stitcher")

        self.mainFrame = Frame(self.main_window, width = 1080, height = 720)
        self.mainFrame.place(x = 0, y = 0)
        
        self.chooseImagesButton = Button(self.mainFrame, text = "Select at least 2 images", font = ("Arial", 14), command = self.chooseImages)
        self.chooseImagesButton.place(x = 50, y = 100)

        self.imagesChosen = StringVar()
        self.imageSaved = StringVar()
        self.cropping = IntVar()

        self.imagesChosenLabel = Label(self.mainFrame, textvariable = self.imagesChosen, font = ("Arial", 12))
        self.imagesChosenLabel.place(x = 50, y = 150)
        
        self.saveImageButton = Button(self.mainFrame, text = "Choose your save location", font = ("Arial", 14), command = self.saveImage)
        self.saveImageButton.place(x = 50, y = 220)

        self.saveImageLabel = Label(self.mainFrame, textvariable = self.imageSaved, font = ("Arial", 12))
        self.saveImageLabel.place(x = 50, y = 270)

        self.panoramaButton = Button(self.mainFrame, text = "Compile Panorama", font = ("Arial", 14), command = self.runPanorama)
        self.panoramaButton.place(x = 50, y = 400)

        self.cropBox = Checkbutton(self.mainFrame, text = "Crop", font = ("Arial", 14), variable = self.cropping, onvalue = 1, offvalue = 0)
        self.cropBox.place(x = 50, y = 600)

        # saturation_slider = Scale(self.main_window, from_=0, to=200, interval=10 orient=HORIZONTAL)
        # saturation_slider.set(100)
        # saturation_slider.pack()

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
        if (self.cropping.get() == 1):
            crop = True
        if (self.imagesChosen.get() == "" or self.imageSaved.get() == ""):
            print("Select an image to save or load first")
        else:
            stitchFunc(crop)
            #print(saveName[0])
            result = Image.open(saveName[0])
            # result.thumbnail((400, 400))
            #result.show()
            self.main_window.quit()

loadNames = []
saveName = []
fileTypes = [('Images', '.jpg'), ('Images', '.png'), ('All files', '*')]

def stitchFunc(crop):
    imgs = []
    p = pathlib.PurePath(os.getcwd())
    pIn = p / 'march22Demo'
    searchPath = str(pIn)
    for imgName in loadNames:
        
        cv.samples.addSamplesDataSearchPath(searchPath)
        cv.samples.addSamplesDataSearchPath(str(imgName.parent))
        img_name = str(imgName.name)
        #print(img_name)

        img = cv.imread(cv.samples.findFile(img_name))
        if img is None:
            print("can't read image ", img_name)
            sys.exit(-1)
        imgs.append(img)

    stitcher = cv.createStitcher() if imutils.is_cv3() else cv.Stitcher_create()
    #stitcher = cv.Stitcher.create(cv.Stitcher_PANORAMA)
    status, pano = stitcher.stitch(imgs)

    if status != cv.Stitcher_OK:
        print("can't stitch images error code %d" % status)
        sys.exit(-1)

    print(crop)
    if (crop == True):
        print("cropping")
        stitched = cv.copyMakeBorder(pano, 10, 10, 10, 10, cv.BORDER_CONSTANT, (0, 0, 0))

        gray = cv.cvtColor(stitched, cv.COLOR_BGR2GRAY)
        thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)[1]
        
        cv.namedWindow("thresh", cv.WINDOW_NORMAL)
        cv.imshow("thresh", thresh)
        cv.waitKey(0)
    #cv.imwrite(str(saveName[0]), pano)
    #cv.imshow("Stitched", pano)
    print("stitching completed successfully. %s saved!" % saveName[0].name)

    print('Done')
    cv.destroyAllWindows()
    

def main():
    app = Application()

if __name__ == '__main__':
    main()