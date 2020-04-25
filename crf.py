
import os
import numpy as np
import cv2
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral

def apply_crf(ori_image, mask):
    """ Conditional Random Field
    ori_image: np.array with value between 0-255
    mask: np.array with value between 0-1
    """

    ## Grayscale to RGB
    # if len(mask.shape) < 3:
    #     mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    ## Converting the anotations RGB to single 32  bit color
    annotated_label = mask.astype(np.int32)
    # annotated_label = mask[:,:,0] + (mask[:,:,1]<<8) + (mask[:,:,2]<<16)

    ## Convert the 32bit integer color to 0,1, 2, ... labels.
    colors, labels = np.unique(annotated_label, return_inverse=True)
    n_labels = 2

    ## Setting up the CRF model
    d = dcrf.DenseCRF2D(ori_image.shape[1], ori_image.shape[0], n_labels)

    ## Get unary potentials (neg log probability)
    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(U)

    ## This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    ## Run Inference for 10 steps
    Q = d.inference(10)

    ## Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    return MAP.reshape((ori_image.shape[0], ori_image.shape[1]))


