import numpy as np
import cv2 as cv
import os
from AOA import AOA
from FHO import FHO
from FSA import FSA
from GLCM import Image_GLCM
from numpy import matlib
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from skimage.feature import hog
from skimage.color import rgb2gray
from GTO import GTO
from Global_vars import Global_vars
from LBP import LBP
from Model_Ada_F_ANN import Model_Ada_F_ANN
from Model_OACF import Model_OACF
from Model_Trans_Unet import Model_Trans_Unet
from Model_UNET import Model_Unet
from Model_unet_plus_plus import Model_unet_plus_plus
from Obj_fun import objfun_Segmentation, objfun_Obj_Detection
from PROPOSED import PROPOSED
from Plot_results import *

no_of_dataset = 5

# Read the Dataset
an = 0
if an == 1:
    Dataset = './Dataset/Multi Cancer/'
    Data_path = os.listdir(Dataset)
    for i in range(len(Data_path)):
        Images = []
        Targ = []
        Cancer_types = Dataset + Data_path[i]
        Cancer_folder = os.listdir(Cancer_types)
        for j in range(len(Cancer_folder)):
            Cancer_classes = Cancer_types + '/' + Cancer_folder[j]
            Cancer_class_file = os.listdir(Cancer_classes)
            for k in range(1000):  # 1000 images from every class
                print(i, len(Data_path), j, len(Cancer_folder), k, len(Cancer_class_file))
                img_dir = Cancer_classes + '/' + Cancer_class_file[k]
                image = cv.imread(img_dir)
                image = cv.resize(image, (256, 256))

                name = img_dir.split('/')[-2]
                Images.append(image)
                Targ.append(name)

        label_encoder = LabelEncoder()
        Tar_encoded = label_encoder.fit_transform(Targ)
        class_tar = to_categorical(Tar_encoded, dtype="uint8")

        index = np.arange(len(Images))
        np.random.shuffle(index)
        Images = np.asarray(Images)
        Shuffled_Images = Images[index]
        Shuffled_Target = class_tar[index]

        np.save('Images_' + str(i + 1) + '.npy', Shuffled_Images)
        np.save('Targets_' + str(i + 1) + '.npy', Shuffled_Target)

# optimization for object detection
an = 0
if an == 1:
    BESTSOL = []
    for n in range(no_of_dataset):
        Feat = np.load('Images_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Images
        Target = np.load('Targets_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Target
        Global_vars.Feat = Feat
        Global_vars.Target = Target
        Npop = 10
        Chlen = 3  # Scale Factor, Minimum No of Neighbors, Minimum Size in OACF
        xmin = matlib.repmat(np.asarray([0.5, 3, 12]), Npop, 1)
        xmax = matlib.repmat(np.asarray([1.5, 15, 48]), Npop, 1)
        fname = objfun_Obj_Detection
        initsol = np.zeros((Npop, Chlen))
        for p1 in range(initsol.shape[0]):
            for p2 in range(initsol.shape[1]):
                initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
        Max_iter = 50
        print("GTO...")
        [bestfit1, fitness1, bestsol1, time1] = GTO(initsol, fname, xmin, xmax, Max_iter)  # LBO

        print("AOA...")
        [bestfit2, fitness2, bestsol2, time2] = AOA(initsol, fname, xmin, xmax, Max_iter)  # GEO

        print("FSA...")
        [bestfit3, fitness3, bestsol3, time3] = FSA(initsol, fname, xmin, xmax, Max_iter)  # TSA

        print("FHO...")
        [bestfit4, fitness4, bestsol4, time4] = FHO(initsol, fname, xmin, xmax, Max_iter)  # FHO

        print("PROPOSED...")
        [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)  # PROPOSED

        BestSol = [bestsol1.squeeze(), bestsol2.squeeze(), bestsol3.squeeze(), bestsol4.squeeze(), bestsol5.squeeze()]
        BESTSOL.append(BestSol)
    np.save('BestSol_obj.npy', np.asarray(BESTSOL))  # Bestsol classification

# object detection
an = 0
if an == 1:
    for n in range(no_of_dataset):
        Feat = np.load('Images_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Images
        BestSol = np.load('BestSol_obj.npy', allow_pickle=True)[n]  # Load the Target
        Proposed = Model_OACF(Feat, BestSol[4, :])
        np.save('Object_detect_' + str(n + 1) + '.npy', Proposed)

# Feature Extraction
an = 0
if an == 1:
    for n in range(no_of_dataset):
        Images = np.load('Object_detect_' + str(n + 1) + '.npy', allow_pickle=True)
        GLCM_Feat = []
        HOG_Feat = []
        CIF_Feat = []
        for i in range(len(Images)):
            print(i, len(Images))
            image = Images[i]

            # Bounding_box
            img = image.copy()
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            thresh = cv.threshold(gray, 128, 255, cv.THRESH_BINARY)[1]
            Bounding_box = img.copy()
            contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]
            for cntr in contours:
                x, y, w, h = cv.boundingRect(cntr)
                cv.rectangle(Bounding_box, (x, y), (x + w, y + h), (0, 0, 255), 2)

            Image = Bounding_box
            if len(Image.shape) == 3:
                Image = cv.cvtColor(Image, cv.COLOR_BGR2GRAY)
            # Texture feature(GLCM + LBP)
            glcm = Image_GLCM(Image)
            lbp = LBP(Image)  # LBP
            Falt_lbp = lbp.flatten()
            Glcm_flat = np.zeros(len(Falt_lbp))
            glCM = glcm[0].flatten()
            for i in range(len(glCM)):
                print(i, len(glCM))
                Glcm_flat[i] = glCM[i]
            GLCM_Feat.append(Glcm_flat)

            # HOG Feature
            gray_image = rgb2gray(Bounding_box)
            hog_features, hog_image = hog(gray_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
            HOG_Feat.append(hog_features)

            # Channel intensity features
            b_channel, g_channel, r_channel = cv.split(Bounding_box)
            b_mean = np.mean(b_channel)
            b_std = np.std(b_channel)
            g_mean = np.mean(g_channel)
            g_std = np.std(g_channel)
            r_mean = np.mean(r_channel)
            r_std = np.std(r_channel)
            CIF = [b_mean, b_std, g_mean, g_std, r_mean, r_std]

            CIF_Feat.append(CIF)
        GLCM_Feat = np.asarray(GLCM_Feat)
        HOG_Feat = np.asarray(HOG_Feat)
        CIF_Feat = np.asarray(CIF_Feat)
        Feature = np.concatenate((GLCM_Feat, HOG_Feat, CIF_Feat), axis=1)
        np.save('Feature_' + str(n + 1) + '.npy', np.asarray(Feature))


# Optimization for Segmentation
an = 0
if an == 1:
    BESTSOL = []
    FITNESS = []
    for n in range(no_of_dataset):
        Feat = np.load('Object_detect_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Images
        Target = np.load('Ground_Truth_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Target
        Global_vars.Feat = Feat
        Global_vars.Target = Target
        Npop = 10
        Chlen = 3  # Hidden Neuron Count, Epoch, Activation Function in Ada-F-ANN
        xmin = matlib.repmat(np.asarray([5, 5, 1]), Npop, 1)
        xmax = matlib.repmat(np.asarray([255, 50, 5]), Npop, 1)
        fname = objfun_Segmentation
        initsol = np.zeros((Npop, Chlen))
        for p1 in range(initsol.shape[0]):
            for p2 in range(initsol.shape[1]):
                initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
        Max_iter = 50

        print("GTO...")
        [bestfit1, fitness1, bestsol1, time1] = GTO(initsol, fname, xmin, xmax, Max_iter)  # GTO

        print("AOA...")
        [bestfit2, fitness2, bestsol2, time2] = AOA(initsol, fname, xmin, xmax, Max_iter)  # AOA

        print("FSA...")
        [bestfit3, fitness3, bestsol3, time3] = FSA(initsol, fname, xmin, xmax, Max_iter)  # FSA

        print("FHO...")
        [bestfit4, fitness4, bestsol4, time4] = FHO(initsol, fname, xmin, xmax, Max_iter)  # FHO

        print("PROPOSED...")
        [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)  # Enchanced FHO

        BestSol = [bestsol1.squeeze(), bestsol2.squeeze(), bestsol3.squeeze(), bestsol4.squeeze(), bestsol5.squeeze()]
        fitness = [fitness1.squeeze(), fitness2.squeeze(), fitness3.squeeze(), fitness4.squeeze(), fitness5.squeeze()]

        FITNESS.append(fitness)
        BESTSOL.append(BestSol)
    np.save('Fitness.npy', np.asarray(FITNESS))
    np.save('BestSol_Seg.npy', np.asarray(BESTSOL))  # Bestsol classification


# Segmentation
an = 0
if an == 1:
    for n in range(no_of_dataset):
        Data_path = './Images/Original_images/Dataset_'+str(n+1)
        Data = np.load('Object_detect_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Data
        BestSol = np.load('BestSol_Seg.npy', allow_pickle=True)[n]  # Load the Data
        Target = np.load('Ground_Truth_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the ground truth
        Unet = Model_Unet(Data_path)
        Res_Unet = Model_unet_plus_plus(Data, Target)
        Trans_Unet = Model_Trans_Unet(Data, Target)
        Ada_F_ANN = Model_Ada_F_ANN(Data, Target)
        Proposed = Model_Ada_F_ANN(Data, Target, BestSol[4, :])
        Seg = [Unet, Res_Unet, Trans_Unet, Ada_F_ANN, Proposed]
        np.save('Segmented_image_' + str(n + 1) + '.npy', Proposed)
        np.save('Seg_img_' + str(n + 1) + '.npy', Seg)

plot_Con_results()
plot_results_Seg()
Image_Boundary_comparision()
Image_segment()
