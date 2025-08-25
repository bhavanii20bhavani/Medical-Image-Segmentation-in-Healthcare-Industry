import numpy as np
import cv2 as cv
import warnings
from matplotlib import pylab
from prettytable import PrettyTable
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


no_of_dataset = 5


def plot_Con_results():
    Dataset = ['Acute Lymphoblastic Leukemia ', 'Kidney Cancer', 'Lung and Colon Cancer', 'Lymphoma', 'Oral Cancer']
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'GTO-OFyANN', 'AOA-OFyANN', 'FSA-OFyANN', 'FHO-OFyANN', 'OFHA-OFyANN']
    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    for n in range(Fitness.shape[0]):
        Conv_Graph = np.zeros((5, 5))
        for j in range(5):
            Conv_Graph[j, :] = Statistical(Fitness[n, j, :])
        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('------------------------------ Statistical Report Dataset', n + 1,
              '------------------------------')
        print(Table)

        length = np.arange(Fitness.shape[2])
        Conv_Graph = Fitness[n]

        Algorithm = ['TERMS', 'GTO-OFyANN', 'AOA-OFyANN', 'FSA-OFyANN', 'FHO-OFyANN', 'OFHA-OFyANN']
        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=2, label='GTO-OFyANN')
        plt.plot(length, Conv_Graph[1, :], color='g', linewidth=2, label='AOA-OFyANN')
        plt.plot(length, Conv_Graph[2, :], color='b', linewidth=2, label='FSA-OFyANN')
        plt.plot(length, Conv_Graph[3, :], color='m', linewidth=2, label='FHO-OFyANN')
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=2, label='OFHA-OFyANN')
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('Convergence Curve of ' + str(Dataset[n]))
        plt.savefig("./Results/Convergence_%s.png" % (Dataset[n]))
        plt.show()


def plot_results_Seg():
    Eval_all = np.load('Eval_all_seg.npy', allow_pickle=True)
    Terms = ['Dice Coefficient', 'Jaccard', 'Accuracy', 'PSNR', 'MSE', 'Sensitivity', 'Specificity', 'Precision', 'FPR',
             'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']

    stats = np.zeros((Eval_all.shape[0], Eval_all.shape[3] - 4, Eval_all.shape[1] + 4, 5))
    for n in range(Eval_all.shape[0]):
        value_all = Eval_all[n, :]
        for i in range(Eval_all.shape[-1] - 4):
            for j in range(value_all.shape[0] + 4):
                if j < value_all.shape[0]:
                    stats[n, i, j, 0] = np.max(value_all[j][:, i + 4]) * 100
                    stats[n, i, j, 1] = np.min(value_all[j][:, i + 4]) * 100
                    stats[n, i, j, 2] = np.mean(value_all[j][:, i + 4]) * 100
                    stats[n, i, j, 3] = np.median(value_all[j][:, i + 4]) * 100
                    stats[n, i, j, 4] = np.std(value_all[j][:, i + 4]) * 100

    for i in range(Eval_all.shape[-1] - 4):
        if Terms[i] == 'PSNR':
            stats[:, i, :, :] = stats[:, i, :, :] / 100
        else:
            stats[:, i, :, :] = stats[:, i, :, :]
        X = np.arange(stats.shape[0])
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

        ax.bar(X + 0.00, stats[:, i, 0, 2], color='#f075e6', edgecolor='w', width=0.10, label="GTO-OFyANN")  # r
        ax.bar(X + 0.10, stats[:, i, 1, 2], color='#0cff0c', edgecolor='w', width=0.10, label="AOA-OFyANN")  # g
        ax.bar(X + 0.20, stats[:, i, 2, 2], color='#0165fc', edgecolor='w', width=0.10, label="FSA-OFyANN")  # b
        ax.bar(X + 0.30, stats[:, i, 3, 2], color='#fd411e', edgecolor='w', width=0.10, label="FHO-OFyANN")  # m
        ax.bar(X + 0.40, stats[:, i, 4, 2], color='k', edgecolor='w', width=0.10, label="OFHA-OFyANN")  # k
        plt.xticks(X + 0.20, (
        'Acute \nLymphoblastic Leukemia ', 'Kidney Cancer', 'Lung and \nColon Cancer', 'Lymphoma', 'Oral Cancer'))
        plt.ylabel(Terms[i])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('Statisticsal Analysis vs ' + Terms[i])
        path = "./Results/Mean_Seg_%s_alg.png" % (Terms[i])
        plt.savefig(path)
        plt.show()

        X = np.arange(stats.shape[0])
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.bar(X + 0.00, stats[:, i, 5, 2], color='#ff028d', edgecolor='k', width=0.10, label="Unet")
        ax.bar(X + 0.10, stats[:, i, 6, 2], color='#0cff0c', edgecolor='k', width=0.10, label="Unet++")
        ax.bar(X + 0.20, stats[:, i, 7, 2], color='#0165fc', edgecolor='k', width=0.10, label="TransUNet")
        ax.bar(X + 0.30, stats[:, i, 8, 2], color='#fd411e', edgecolor='k', width=0.10, label="Ada-F-ANN")
        ax.bar(X + 0.40, stats[:, i, 4, 2], color='k', edgecolor='k', width=0.10, label="OFHA-OFyANN")
        plt.xticks(X + 0.20, (
        'Acute \nLymphoblastic Leukemia ', 'Kidney Cancer', 'Lung and \nColon Cancer', 'Lymphoma', 'Oral Cancer'))
        plt.ylabel(Terms[i])
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('Statisticsal Analysis vs ' + Terms[i])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
        path = "./Results/Mean_Seg_%s_mtd.png" % (Terms[i])
        plt.savefig(path)
        plt.show()


def Image_Boundary_comparision():
    for n in range(no_of_dataset):
        Original = np.load('Images_' + str(n + 1) + '.npy', allow_pickle=True)
        segmented = np.load('Object_detect_' + str(n + 1) + '.npy', allow_pickle=True)
        Images = [[25, 26, 28, 30, 34], [0, 3, 7, 9, 15], [13, 21, 44, 46, 53], [1, 2, 7, 12, 15],
                  [19, 95, 1041, 1223, 1315]]
        Images = np.asarray(Images)
        for i in range(Images.shape[1]):
            Orig = Original[Images[n][i]]
            Seg = segmented[Images[n][i]]
            plt.suptitle('Segmented Images from Dataset ' + str(n + 1), fontsize=20)

            plt.subplot(1, 2, 1).axis('off')
            plt.imshow(Orig)
            plt.title('Original', fontsize=10)

            plt.subplot(1, 2, 2).axis('off')
            plt.imshow(Seg)
            plt.title('Bounding box', fontsize=10)

            path = "./Results/Image_Results/Dataset_%s_image_%s.png" % (n + 1, i + 1)
            plt.savefig(path)
            plt.show()
            cv.imwrite('./Results/Image_Results/Dataset_' + str(n + 1) + 'Orig_image_' + str(i + 1) + '.png', Orig)
            cv.imwrite('./Results/Image_Results/Dataset_' + str(n + 1) + 'OACF_' + str(i + 1) + '.png', Seg)


def Image_segment():
    for n in range(no_of_dataset):
        Original = np.load('Images_' + str(n + 1) + '.npy', allow_pickle=True)
        segmented = np.load('Seg_img_' + str(n + 1) + '.npy', allow_pickle=True)
        Ground_truth = np.load('Ground_Truth_' + str(n + 1) + '.npy', allow_pickle=True)
        Images = [[25, 26, 28, 30, 34], [0, 3, 7, 9, 15], [13, 21, 44, 46, 53], [1, 2, 7, 12, 15],
                  [19, 95, 1041, 1223, 1315]]
        Images = np.asarray(Images)
        for i in range(Images.shape[1]):
            print(n, no_of_dataset, i, Images.shape[1], Images[n][i])
            Orig = Original[Images[n][i]]
            Seg = segmented[Images[n][i]]
            GT = Ground_truth[Images[n][i]]
            for j in range(1):
                Orig_2 = Seg[j + 1]
                Orig_3 = Seg[j + 2]
                Orig_4 = Seg[j + 3]
                Orig_5 = Seg[j + 4]
                plt.suptitle('Segmented Images from Dataset', fontsize=20)

                plt.subplot(2, 3, 1).axis('off')
                plt.imshow(GT)
                plt.title('Ground Truth', fontsize=10)

                plt.subplot(2, 3, 2).axis('off')
                plt.imshow(Orig_2)
                plt.title('RESUnet', fontsize=10)

                plt.subplot(2, 3, 3).axis('off')
                plt.imshow(Orig)
                plt.title('Original', fontsize=10)

                plt.subplot(2, 3, 4).axis('off')
                plt.imshow(Orig_3)
                plt.title('Trans_ResUnet ', fontsize=10)

                plt.subplot(2, 3, 5).axis('off')
                plt.imshow(Orig_4)
                plt.title('TransUnet++', fontsize=10)

                plt.subplot(2, 3, 6).axis('off')
                plt.imshow(Orig_5)
                plt.title('MDTUNet++', fontsize=10)

                path = "./Results/Image_Results/Dataset_%s_image_%s.png" % (n + 1, i + 1)
                plt.savefig(path)
                plt.show()

                cv.imwrite('./Results/Image_Results/seg_Dataset_' + str(n + 1) + 'Orig_image_' + str(i + 1) + '.png',
                           Orig)
                cv.imwrite('./Results/Image_Results/seg_Dataset_' + str(n + 1) + 'Ground_Truth_' + str(i + 1) + '.png',
                           GT)
                cv.imwrite('./Results/Image_Results/seg_Dataset_' + str(n + 1) + 'segm_Unet++_' + str(i + 1) + '.png',
                           Orig_2)
                cv.imwrite(
                    './Results/Image_Results/seg_Dataset_' + str(n + 1) + 'segm_TransUnet_' + str(i + 1) + '.png',
                    Orig_3)
                cv.imwrite(
                    './Results/Image_Results/seg_Dataset_' + str(n + 1) + 'segm_Ada-F-ANN_' + str(i + 1) + '.png',
                    Orig_4)
                cv.imwrite(
                    './Results/Image_Results/seg_Dataset_' + str(n + 1) + 'segm_Proposed_' + str(i + 1) + '.png',
                    Orig_5)


if __name__ == '__main__':
    plot_Con_results()
    plot_results_Seg()
    Image_Boundary_comparision()
    Image_segment()
