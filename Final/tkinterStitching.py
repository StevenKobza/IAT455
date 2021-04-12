from tkinter import *
from tkinter import filedialog as fd
from tkinter import messagebox
from PIL import Image, ImageShow, ImageTk
from collections import OrderedDict

import os
import pathlib

import numpy as np
import cv2 as cv

import argparse
import sys

import imutils

#setting up the options for Stitcher Detailed from OpenCV
EXPOS_COMP_CHOICES = OrderedDict()
EXPOS_COMP_CHOICES['gain_blocks'] = cv.detail.ExposureCompensator_GAIN_BLOCKS
EXPOS_COMP_CHOICES['gain'] = cv.detail.ExposureCompensator_GAIN
EXPOS_COMP_CHOICES['no'] = cv.detail.ExposureCompensator_NO

BA_COST_CHOICES = OrderedDict()
BA_COST_CHOICES['ray'] = cv.detail_BundleAdjusterRay
BA_COST_CHOICES['reproj'] = cv.detail_BundleAdjusterReproj
BA_COST_CHOICES['affine'] = cv.detail_BundleAdjusterAffinePartial
BA_COST_CHOICES['no'] = cv.detail_NoBundleAdjuster

FEATURES_FIND_CHOICES = OrderedDict()
FEATURES_FIND_CHOICES['orb'] = cv.ORB.create
try:
    FEATURES_FIND_CHOICES['sift'] = cv.xfeatures2d_SIFT.create
except AttributeError:
    print("SIFT not available")
try:
    FEATURES_FIND_CHOICES['brisk'] = cv.BRISK_create
except AttributeError:
    print("BRISK not available")
try:
    FEATURES_FIND_CHOICES['akaze'] = cv.AKAZE_create
except AttributeError:
    print("AKAZE not available")

SEAM_FIND_CHOICES = OrderedDict()
SEAM_FIND_CHOICES['gc_color'] = cv.detail_GraphCutSeamFinder('COST_COLOR')
SEAM_FIND_CHOICES['gc_colorgrad'] = cv.detail_GraphCutSeamFinder('COST_COLOR_GRAD')
SEAM_FIND_CHOICES['dp_color'] = cv.detail_DpSeamFinder('COLOR')
SEAM_FIND_CHOICES['dp_colorgrad'] = cv.detail_DpSeamFinder('COLOR_GRAD')
SEAM_FIND_CHOICES['no'] = cv.detail.SeamFinder_createDefault(cv.detail.SeamFinder_NO)

ESTIMATOR_CHOICES = OrderedDict()
ESTIMATOR_CHOICES['homography'] = cv.detail_HomographyBasedEstimator

WARP_CHOICES = (
    'spherical',
    'cylindrical',
    'fisheye',
    'stereographic',
)

WAVE_CORRECT_CHOICES = ('horiz', 'no', 'vert',)

BLEND_CHOICES = ('multiband', 'feather', 'no',)


class Application(Frame):
    def __init__ (self):
        self.main_window = Tk()
        self.main_window.geometry("1400x950")
        self.main_window.title("Panorama Stitcher")
        self.main_window.configure(bg='grey23')

        self.mainFrame = Frame(self.main_window, bg='grey23')
        self.mainFrame.grid(padx=20, pady=20, sticky='nsew')
        self.mainFrame.configure(bg='grey23')

        #Variables
        self.imagesChosen = StringVar()
        self.imageSaved = StringVar()
        self.cropping = IntVar()
        self.matchConf = DoubleVar()
        self.confThresh = DoubleVar()
        self.workMegapix = DoubleVar()
        self.seamMegapix = DoubleVar()

        #Buttons + Labls. Setting them along a grid for good viewing
        self.chooseImagesButton = Button(self.mainFrame, padx=5, pady=5, 
            bg='grey45', fg='white', activebackground='grey55', activeforeground='white', 
            text = "Select 2+ images", font = ("Roboto", 14), command = self.chooseImages)
        self.chooseImagesButton.grid(row=1, column=1)

        #Images that have been chosen
        self.imagesChosenLabel = Label(self.mainFrame, textvariable = self.imagesChosen, font = ("Roboto", 10), bg='grey23', fg='white')
        self.imagesChosenLabel.grid(row=2, column=1)
        
        #Selecting where to save
        self.saveImageButton = Button(self.mainFrame, padx=5, pady=5,
            bg='grey45', fg='white', activebackground='grey55', activeforeground='white', 
            text = "Save to", font = ("Roboto", 14), command = self.saveImage)
        self.saveImageButton.grid(row=1, column=2)

        #Name of that image
        self.saveImageLabel = Label(self.mainFrame, textvariable = self.imageSaved, font = ("Roboto", 10), bg='grey23', fg='white')
        self.saveImageLabel.grid(row=2, column=2)

        #Run the panorama
        self.panoramaButton = Button(self.mainFrame, padx=5, pady=5,
            bg='grey45', fg='white', activebackground='grey55', activeforeground='white', 
            text = "Compile Panorama", font = ("Roboto", 14), command = self.runPanorama)
        self.panoramaButton.grid(row=1, column=3)

        #Match confidence label
        self.matchConfLabel = Label(self.mainFrame, text="Match Confidence Slider", font = ("Roboto", 12), bg='grey23', fg='white')
        self.matchConfLabel.grid(row=3, column=2)
        self.matchConfSlider = Scale(self.mainFrame, length=300, from_=0.0, to=1.0, tickinterval=0.1, resolution = 0.01, variable = self.matchConf, orient=HORIZONTAL, bg="grey23", fg="white")
        self.matchConfSlider.grid(padx=10, pady=10, row=3, column=0)
        self.matchConfSlider.set(0.3)

        #Confidence Threshold label and slider
        self.confThreshLabel = Label(self.mainFrame, text="Confidence Threshold Slider", font = ("Roboto", 12), bg='grey23', fg='white')
        self.confThreshLabel.grid(row=4, column=2)
        self.confThreshSlider = Scale(self.mainFrame, length=300, from_=0.0, to=1.0, tickinterval=0.1, resolution = 0.01, variable = self.confThresh, orient=HORIZONTAL, bg="grey23", fg="white")
        self.confThreshSlider.grid(padx=10, pady=10, row=4, column=0)
        self.confThreshSlider.set(0.3)

        self.workMegapixLabel = Label(self.mainFrame, text="Megapixels to work with", font = ("Roboto", 12), bg='grey23', fg='white')
        self.workMegapixLabel.grid(row=5, column=2)
        self.workMegapixSlider = Scale(self.mainFrame, length=300, from_=0.0, to=5, tickinterval=0.5, resolution = 0.1, variable = self.workMegapix, orient=HORIZONTAL, bg="grey23", fg="white")
        self.workMegapixSlider.grid(padx=10, pady=10, row=5, column=0)
        self.workMegapixSlider.set(0.6)

        self.seamMegapixLabel = Label(self.mainFrame, text="Megapixels to detect seams with", font = ("Roboto", 12), bg='grey23', fg='white')
        self.seamMegapixLabel.grid(row=6, column=2)
        self.seamMegapixSlider = Scale(self.mainFrame, length=300, from_=0.0, to=5, tickinterval=0.5, resolution = 0.1, variable = self.seamMegapix, orient=HORIZONTAL, bg="grey23", fg="white")
        self.seamMegapixSlider.grid(padx=10, pady=10, row=6, column=0)
        self.seamMegapixSlider.set(0.1)

        self.seamLabel = Label(self.mainFrame, text="Seam Estimation Algorithm", font = ("Roboto", 12), bg='grey23', fg='white')
        self.seamLabel.grid(row=7, column=2)
        self.seamListBox = Listbox(self.mainFrame, height = 5, width = 35, font = ("Roboto", 12), bg="grey20", fg="white", selectbackground="grey40", selectmode=SINGLE, exportselection = 0)
        self.seamListBox.insert(1, "gc_color")
        self.seamListBox.insert(2, "gc_colorgrad")
        self.seamListBox.insert(3, "dp_color")
        self.seamListBox.insert(4, "dp_colorgrad")
        self.seamListBox.insert(6, "no")
        self.seamListBox.grid(row = 7, column = 0)
        self.seamListBox.activate(1)
        

        self.warpLabel = Label(self.mainFrame, text="Warp Type", font = ("Roboto", 12), bg='grey23', fg='white')
        self.warpLabel.grid(row=8, column=2)
        self.warpListBox = Listbox(self.mainFrame, height = 4, width = 35, font = ("Roboto", 12), bg="grey20", fg="white", selectbackground="grey40", selectmode=SINGLE, exportselection = 0)
        self.warpListBox.insert(1, "spherical")
        self.warpListBox.insert(2, "cylindrical")
        self.warpListBox.insert(3, "fisheye")
        self.warpListBox.insert(4, "stereographic")
        self.warpListBox.grid(row=8, column = 0)
        self.warpListBox.activate(1)

        self.waveCorrLabel = Label(self.mainFrame, text="Wave Correction", font = ("Roboto", 12), bg='grey23', fg='white')
        self.waveCorrLabel.grid(row=9, column=2)
        self.waveCorrListBox = Listbox(self.mainFrame, height = 3, width = 35, font = ("Roboto", 12), bg="grey20", fg="white", selectbackground="grey40", selectmode=SINGLE, exportselection = 0)
        self.waveCorrListBox.insert(1, "horiz")
        self.waveCorrListBox.insert(2, "vert")
        self.waveCorrListBox.insert(3, "no")
        self.waveCorrListBox.grid(row=9, column = 0)
        self.waveCorrListBox.activate(1)

        self.blendTypeLabel = Label(self.mainFrame, text="Blend Type", font = ("Roboto", 12), bg='grey23', fg='white')
        self.blendTypeLabel.grid(row=10, column=2)
        self.blendListBox = Listbox(self.mainFrame, height = 3, width = 35, font = ("Roboto", 12), bg="grey20", fg="white", selectbackground="grey40", selectmode=SINGLE, exportselection = 0)
        self.blendListBox.insert(1, "multiband")
        self.blendListBox.insert(2, "feather")
        self.blendListBox.insert(3, "no")
        self.blendListBox.grid(row=10, column = 0)
        self.blendListBox.activate(1)

        self.exposureCompLabel = Label(self.mainFrame, text="Exposure Compensation", font = ("Roboto", 12), bg='grey23', fg='white')
        self.exposureCompLabel.grid(row=11, column=2)
        self.exposCompListBox = Listbox(self.mainFrame, height = 3, width = 35, font = ("Roboto", 12), bg="grey20", fg="white", selectbackground="grey40", selectmode=SINGLE, exportselection = 0)
        self.exposCompListBox.insert(1, "gain_blocks")
        self.exposCompListBox.insert(2, "gain")
        self.exposCompListBox.insert(3, "no")
        self.exposCompListBox.grid(row=11, column = 0)
        self.exposCompListBox.activate(1)

        self.baCostLabel = Label(self.mainFrame, text="BA Cost", font = ("Roboto", 12), bg='grey23', fg='white')
        self.baCostLabel.grid(row=12, column=2)
        self.baCostListBox = Listbox(self.mainFrame, height = 3, width = 35, font = ("Roboto", 12), bg="grey20", fg="white", selectbackground="grey40", selectmode=SINGLE, exportselection = 0)
        self.baCostListBox.insert(1, "ray")
        self.baCostListBox.insert(2, "reproj")
        self.baCostListBox.insert(3, "no")
        self.baCostListBox.grid(row=12, column = 0)
        self.baCostListBox.activate(1)

        self.featureFindLabel = Label(self.mainFrame, text="Feature Finder", font = ("Roboto", 12), bg='grey23', fg='white')
        self.featureFindLabel.grid(row=13, column=2)
        self.featureListBox = Listbox(self.mainFrame, height = 4, width = 35, font = ("Roboto", 12), bg="grey20", fg="white", selectbackground="grey40", selectmode=SINGLE, exportselection = 0)
        self.featureListBox.insert(1, "orb")
        self.featureListBox.insert(2, "sift")
        self.featureListBox.insert(3, "brisk")
        self.featureListBox.insert(4, "akaze")
        self.featureListBox.grid(row=13, column = 0)
        self.featureListBox.activate(1)


        mainloop()

    def chooseImages(self):
        names = fd.askopenfilenames(filetypes = fileTypes)
        temp = ""
        loadNames.clear()
        for i in names:
            #Adding the names that are being loaded in, in an array for future use. Using PurePath to swap between different OSes easily
            temp += pathlib.PurePath(i).name + " "
            loadNames.append(pathlib.PurePath(i))
            loadNamesNames.append(str(pathlib.PurePath(i).name))
        #setting the string variable
        self.imagesChosen.set(temp)

    def saveImage(self):
        saveName.clear()
        names = ""
        #Opening a dialogue like the previous one.
        names = fd.asksaveasfilename(filetypes = fileTypes, defaultextension = '.jpg', initialfile = "result.jpg")
        #Using pathlib again
        temp = pathlib.Path(names).name
        saveName.append(pathlib.Path(names))
        self.imageSaved.set(temp)

    def runPanorama(self):
        if (self.cropping.get() == 1):
            crop = True
        else:
            crop = False

        #Getting the current selection for the following listboxes and if there isn't anything selected, just selecting the default option.
        if (self.seamListBox.curselection() == ()):
            options.append("gc_color")
        else:
            options.append(self.seamListBox.get(self.seamListBox.curselection()))

        if (self.warpListBox.curselection() == ()):
            options.append("spherical")
        else:
            options.append(self.warpListBox.get(self.warpListBox.curselection()))

        if (self.waveCorrListBox.curselection() == ()):
            options.append("horiz")
        else:
            options.append(self.waveCorrListBox.get(self.waveCorrListBox.curselection()))

        if (self.blendListBox.curselection() == ()):
            options.append("multiband")
        else:
            options.append(self.blendListBox.get(self.blendListBox.curselection()))

        if (self.exposCompListBox.curselection() == ()):
            options.append("gain_blocks")
        else:
            options.append(self.exposCompListBox.get(self.exposCompListBox.curselection()))

        if (self.baCostListBox.curselection() == ()):
            options.append("ray")
        else:
            options.append(self.baCostListBox.get(self.baCostListBox.curselection()))

        if (self.featureListBox.curselection() == ()):
            options.append("orb")
        else:
            options.append(self.featureListBox.get(self.featureListBox.curselection()))

        if (self.imagesChosen.get() == "" or self.imageSaved.get() == ""):
            messagebox.showerror("Select Images", "Select an image to save or load first")
        else:
            stitchFuncNew(crop, self)
            result = Image.open(saveName[0])
            self.main_window.quit()

loadNames = []
loadNamesNames = []
saveName = []
fileTypes = [('Images', '.jpg'), ('Images', '.png'), ('All files', '*')]
options = []    
    

def getMatcher(self):
    try_cuda = False
    match_conf = self.matchConf.get()
    matcher = cv.detail.BestOf2NearestMatcher_create(try_cuda, match_conf)
    return matcher

def getCompensator(self):
    
    expos_comp_type = EXPOS_COMP_CHOICES[options[4]]
    compensator = cv.detail.ExposureCompensator_createDefault(expos_comp_type)
    return compensator

def stitchFuncNew(crop, self):
    #set up search
    path = pathlib.PurePath(os.getcwd())
    pIn = path / 'Final'
    searchPath = str(pIn)
    cv.samples.addSamplesDataSearchPath(searchPath)
    cv.samples.addSamplesDataSearchPath(str(loadNames[0].parent))


    img_names = loadNamesNames
    print(img_names)
    work_megapix = self.workMegapix.get()
    seam_megapix = self.seamMegapix.get()
    compose_megapix = -1
    conf_thresh = self.confThresh.get()
    ba_refine_mask = 'xxxxx'
    wave_correct = options[2]
    save_graph = 'Final/text2.txt'

    if wave_correct == 'no':
        do_wave_correct = False
    else:
        do_wave_correct = True

    if save_graph is None:
        save_graph = False
    else:
        save_graph = True

    warp_type = options[1]
    blend_type = options[3]
    blend_strength = 5
    result_name = str(saveName[0])

    timelapse = None
    if timelapse is not None:
        timelapse = True
        if timelapse == "as_is":
            timelapse_type = cv.detail.Timelapser_AS_IS
        elif timelapse == "crop":
            timelapse_type = cv.detail.Timelapser_CROP
        else:
            print("Bad timelapse method")
            cv.waitKey(0)
            exit()
    else:
        timelapse = False

    finder = FEATURES_FIND_CHOICES[options[6]]()
    seam_work_aspect = 1
    full_img_sizes = []
    features = []
    images = []
    is_work_scale_set = False
    is_seam_scale_set = False
    is_compose_scale_set = False
    for name in img_names:
        full_img = cv.imread(cv.samples.findFile(name))
        if full_img is None:
            print("Cannot read image ", name)
            cv.waitKey(0)
            exit()
        full_img_sizes.append((full_img.shape[1], full_img.shape[0]))
        if work_megapix < 0:
            img = full_img
            work_scale = 1
            is_work_scale_set = True
        else:
            if is_work_scale_set is False:
                work_scale = min(1.0, np.sqrt(work_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
                is_work_scale_set = True
            img = cv.resize(src=full_img, dsize=None, fx=work_scale, fy=work_scale, interpolation=cv.INTER_LINEAR_EXACT)
        if is_seam_scale_set is False:
            seam_scale = min(1.0, np.sqrt(seam_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
            seam_work_aspect = seam_scale / work_scale
            is_seam_scale_set = True
        img_feat = cv.detail.computeImageFeatures2(finder, img)
        features.append(img_feat)
        img = cv.resize(src=full_img, dsize=None, fx=seam_scale, fy=seam_scale, interpolation=cv.INTER_LINEAR_EXACT)
        images.append(img)

    matcher = getMatcher(self)
    p = matcher.apply2(features)
    matcher.collectGarbage()
    if save_graph:
        with open(save_graph, 'w') as fh:
            fh.write(cv.detail.matchesGraphAsString(img_names, p, conf_thresh))

    indices = cv.detail.leaveBiggestComponent(features, p, 0.3)
    img_subset = []
    img_names_subset = []
    full_img_sizes_subset = []
    for i in range(len(indices)):
        img_names_subset.append(img_names[indices[i, 0]])
        img_subset.append(images[indices[i, 0]])
        full_img_sizes_subset.append(full_img_sizes[indices[i, 0]])
    images = img_subset
    img_names = img_names_subset
    full_img_sizes = full_img_sizes_subset
    num_images = len(img_names)
    if num_images < 2:
        print("Need more images")
        cv.waitKey(0)
        exit()

    estimator = ESTIMATOR_CHOICES['homography']()
    b, cameras = estimator.apply(features, p, None)
    if not b:
        print("Homography estimation failed.")
        cv.waitKey(0)
        exit()
    for cam in cameras:
        cam.R = cam.R.astype(np.float32)

    adjuster = BA_COST_CHOICES[options[5]]()
    adjuster.setConfThresh(1)
    refine_mask = np.zeros((3, 3), np.uint8)
    if ba_refine_mask[0] == 'x':
        refine_mask[0, 0] = 1
    if ba_refine_mask[1] == 'x':
        refine_mask[0, 1] = 1
    if ba_refine_mask[2] == 'x':
        refine_mask[0, 2] = 1
    if ba_refine_mask[3] == 'x':
        refine_mask[1, 1] = 1
    if ba_refine_mask[4] == 'x':
        refine_mask[1, 2] = 1
    adjuster.setRefinementMask(refine_mask)
    b, cameras = adjuster.apply(features, p, cameras)
    if not b:
        print("Camera parameters adjusting failed.")
        cv.waitKey()
        exit()
    focals = []
    for cam in cameras:
        focals.append(cam.focal)
    focals.sort()
    if len(focals) % 2 == 1:
        warped_image_scale = focals[len(focals) // 2]
    else:
        warped_image_scale = (focals[len(focals) // 2] + focals[len(focals) // 2 - 1]) / 2
    if do_wave_correct:
        rmats = []
        for cam in cameras:
            rmats.append(np.copy(cam.R))
        rmats = cv.detail.waveCorrect(rmats, cv.detail.WAVE_CORRECT_HORIZ)
        for idx, cam in enumerate(cameras):
            cam.R = rmats[idx]
    corners = []
    masks_warped = []
    images_warped = []
    sizes = []
    masks = []
    for i in range(0, num_images):
        um = cv.UMat(255 * np.ones((images[i].shape[0], images[i].shape[1]), np.uint8))
        masks.append(um)

    warper = cv.PyRotationWarper(warp_type, warped_image_scale * seam_work_aspect)  # warper could be nullptr?
    for idx in range(0, num_images):
        K = cameras[idx].K().astype(np.float32)
        swa = seam_work_aspect
        K[0, 0] *= swa
        K[0, 2] *= swa
        K[1, 1] *= swa
        K[1, 2] *= swa
        corner, image_wp = warper.warp(images[idx], K, cameras[idx].R, cv.INTER_LINEAR, cv.BORDER_REFLECT)
        corners.append(corner)
        sizes.append((image_wp.shape[1], image_wp.shape[0]))
        images_warped.append(image_wp)
        p, mask_wp = warper.warp(masks[idx], K, cameras[idx].R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)
        masks_warped.append(mask_wp.get())

    images_warped_f = []
    for img in images_warped:
        imgf = img.astype(np.float32)
        images_warped_f.append(imgf)

    compensator = getCompensator(self)
    compensator.feed(corners=corners, images=images_warped, masks=masks_warped)

    seam_finder = SEAM_FIND_CHOICES[options[0]]
    seam_finder.find(images_warped_f, corners, masks_warped)
    compose_scale = 1
    corners = []
    sizes = []
    blender = None
    timelapser = None
    # https://github.com/opencv/opencv/blob/master/samples/cpp/stitching_detailed.cpp#L725 ?
    for idx, name in enumerate(img_names):
        print(name)
        full_img = cv.imread(cv.samples.findFile(name))
        if not is_compose_scale_set:
            if compose_megapix > 0:
                compose_scale = min(1.0, np.sqrt(compose_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
            is_compose_scale_set = True
            compose_work_aspect = compose_scale / work_scale
            warped_image_scale *= compose_work_aspect
            warper = cv.PyRotationWarper(warp_type, warped_image_scale)
            for i in range(0, len(img_names)):
                cameras[i].focal *= compose_work_aspect
                cameras[i].ppx *= compose_work_aspect
                cameras[i].ppy *= compose_work_aspect
                sz = (full_img_sizes[i][0] * compose_scale, full_img_sizes[i][1] * compose_scale)
                K = cameras[i].K().astype(np.float32)
                roi = warper.warpRoi(sz, K, cameras[i].R)
                corners.append(roi[0:2])
                sizes.append(roi[2:4])
        if abs(compose_scale - 1) > 1e-1:
            img = cv.resize(src=full_img, dsize=None, fx=compose_scale, fy=compose_scale,
                            interpolation=cv.INTER_LINEAR_EXACT)
        else:
            img = full_img
        _img_size = (img.shape[1], img.shape[0])
        K = cameras[idx].K().astype(np.float32)
        corner, image_warped = warper.warp(img, K, cameras[idx].R, cv.INTER_LINEAR, cv.BORDER_REFLECT)
        mask = 255 * np.ones((img.shape[0], img.shape[1]), np.uint8)
        p, mask_warped = warper.warp(mask, K, cameras[idx].R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)
        compensator.apply(idx, corners[idx], image_warped, mask_warped)
        image_warped_s = image_warped.astype(np.int16)
        dilated_mask = cv.dilate(masks_warped[idx], None)
        seam_mask = cv.resize(dilated_mask, (mask_warped.shape[1], mask_warped.shape[0]), 0, 0, cv.INTER_LINEAR_EXACT)
        mask_warped = cv.bitwise_and(seam_mask, mask_warped)
        if blender is None and not timelapse:
            blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)
            dst_sz = cv.detail.resultRoi(corners=corners, sizes=sizes)
            blend_width = np.sqrt(dst_sz[2] * dst_sz[3]) * blend_strength / 100
            if blend_width < 1:
                blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)
            elif blend_type == "multiband":
                blender = cv.detail_MultiBandBlender()
                blender.setNumBands((np.log(blend_width) / np.log(2.) - 1.).astype(np.int))
            elif blend_type == "feather":
                blender = cv.detail_FeatherBlender()
                blender.setSharpness(1. / blend_width)
            blender.prepare(dst_sz)
        elif timelapser is None and timelapse:
            timelapser = cv.detail.Timelapser_createDefault(timelapse_type)
            timelapser.initialize(corners, sizes)
        if timelapse:
            ma_tones = np.ones((image_warped_s.shape[0], image_warped_s.shape[1]), np.uint8)
            timelapser.process(image_warped_s, ma_tones, corners[idx])
            pos_s = img_names[idx].rfind("/")
            if pos_s == -1:
                fixed_file_name = "fixed_" + img_names[idx]
            else:
                fixed_file_name = img_names[idx][:pos_s + 1] + "fixed_" + img_names[idx][pos_s + 1:]
            cv.imwrite(fixed_file_name, timelapser.getDst())
        else:
            blender.feed(cv.UMat(image_warped_s), mask_warped, corners[idx])
    if not timelapse:
        result = None
        result_mask = None
        result, result_mask = blender.blend(result, result_mask)
        cv.imwrite(result_name, result)
        zoom_x = 600.0 / result.shape[1]
        dst = cv.normalize(src=result, dst=None, alpha=255., norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        dst = cv.resize(dst, dsize=None, fx=zoom_x, fy=zoom_x)
        cv.imshow(result_name, dst)
        cv.waitKey()
    cv.waitKey(0)
    print("Done")

def main():
    app = Application()

if __name__ == '__main__':
    main()