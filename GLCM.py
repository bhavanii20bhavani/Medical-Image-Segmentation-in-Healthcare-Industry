from skimage.feature import graycomatrix, graycoprops
import numpy as np

def Image_GLCM(image):
    bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255])  # 16-bit
    inds = np.digitize(image, bins)

    max_value = np.max(inds) + 1
    matrix_coocurrence = graycomatrix(inds, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=max_value,
                                      normed=False, symmetric=False)
    contrast = contrast_feature(matrix_coocurrence)
    dissimilarity = dissimilarity_feature(matrix_coocurrence)
    homogeneity = homogeneity_feature(matrix_coocurrence)
    energy = energy_feature(matrix_coocurrence)
    correlation = correlation_feature(matrix_coocurrence)
    asm = asm_feature(matrix_coocurrence)
    glcm = np.append(contrast, np.append(dissimilarity, np.append(homogeneity,
                                                                  np.append(energy, np.append(correlation, asm, axis=1),
                                                                            axis=1), axis=1), axis=1), axis=1)
    return glcm



# GLCM properties
def contrast_feature(matrix_coocurrence):
    contrast = graycoprops(matrix_coocurrence, 'contrast')
    return contrast

def dissimilarity_feature(matrix_coocurrence):
    dissimilarity = graycoprops(matrix_coocurrence, 'dissimilarity')
    return dissimilarity

def homogeneity_feature(matrix_coocurrence):
    homogeneity = graycoprops(matrix_coocurrence, 'homogeneity')
    return homogeneity

def energy_feature(matrix_coocurrence):
    energy = graycoprops(matrix_coocurrence, 'energy')
    return energy

def correlation_feature(matrix_coocurrence):
    correlation = graycoprops(matrix_coocurrence, 'correlation')
    return correlation


def entropy_feature(matrix_coocurrence):
    entropy = graycoprops(matrix_coocurrence, 'entropy')
    return entropy

def asm_feature(matrix_coocurrence):
    asm = graycoprops(matrix_coocurrence, 'ASM')
    return asm