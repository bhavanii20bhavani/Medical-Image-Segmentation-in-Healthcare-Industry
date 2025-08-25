import numpy as np
from Evaluation import net_evaluation
from Global_vars import Global_vars
from Model_Ada_F_ANN import Model_Ada_F_ANN
from Model_OACF import Model_OACF


def objfun_Obj_Detection(Soln):
    Feat = Global_vars.Feat
    Tar = Global_vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            predict = Model_OACF(Feat, sol)
            EVAl = []
            for img in range(len(predict)):
                Eval = net_evaluation(predict[img], Tar[img])
                EVAl.append(Eval)
            mean_EVAl = np.mean(EVAl, axis=0, keepdims=True)
            Fitn[i] = 1 / (mean_EVAl[0, 11])  # Precision
        return Fitn
    else:
        sol = np.round(Soln).astype(np.int16)
        predict = Model_OACF(Feat, sol)
        EVAl = []
        for img in range(len(predict)):
            Eval = net_evaluation(predict[img], Tar[img])
            EVAl.append(Eval)
        mean_EVAl = np.mean(EVAl, axis=0, keepdims=True)
        Fitn = 1 / (mean_EVAl[0, 11])  # Precision
        return Fitn


def objfun_Segmentation(Soln):
    Feat = Global_vars.Feat
    Tar = Global_vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            predict = Model_Ada_F_ANN(Feat, Tar, sol)
            EVAl = []
            for img in range(len(predict)):
                Eval = net_evaluation(predict[img], Tar[img])
                EVAl.append(Eval)
            mean_EVAl = np.mean(EVAl, axis=0, keepdims=True)
            Fitn[i] = 1 / (mean_EVAl[0, 4])  # Dice coefficient
        return Fitn
    else:
        sol = np.round(Soln).astype(np.int16)
        predict = Model_Ada_F_ANN(Feat, Tar, sol)
        EVAl = []
        for img in range(len(predict)):
            Eval = net_evaluation(predict[img], Tar[img])
            EVAl.append(Eval)
        mean_EVAl = np.mean(EVAl, axis=0, keepdims=True)
        Fitn = 1 / (mean_EVAl[0, 4])  # Dice coefficient
        return Fitn
