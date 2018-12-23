
import numpy as np
import cv2

def Precision(rs, gt):
    """
    Compute precision between result and ground true image.

    Arguments:
    rs -- result mask
    gt -- ground true mask
    Return:
    precision -- 
    """    
    assert rs.shape == gt.shape  
    precision = np.count_nonzero(cv2.bitwise_and(rs, gt))/np.count_nonzero(rs)   
    return precision

def Recall(rs, gt):
    """
    Compute Recall between result and ground true image.

    Arguments:
    rs -- result mask
    gt -- ground true mask
    Return:
    recall -- 
    """    
    assert rs.shape == gt.shape  
    recall = np.count_nonzero(cv2.bitwise_and(rs, gt))/np.count_nonzero(gt)   
    return recall

def F_measure(rs, gt):
    """
    Compute 
    
    Arguments:
    rs -- result mask
    gt -- ground true mask
    Return:
    F1 -- 
    """    
    assert rs.shape == gt.shape
    F1 = 2 * Precision(rs, gt) * Recall(rs, gt) / (Precision(rs, gt) + Recall(rs, gt))
    return F1

def MAE(rs, gt):
    """
    Compute 
    
    Arguments:
    rs -- result image
    gt -- ground true image
    Return:
    MAE -- 
    """    
    assert rs.shape == gt.shape
    cv2.normalize(rs,  rs, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(gt,  gt, 0, 1, cv2.NORM_MINMAX)    
    MAE = np.sum(np.abs(rs[:, :, 0] - gt[:, :, 0])) / (rs.shape[0] * rs.shape[1])
    cv2.normalize(rs,  rs, 0, 255, cv2.NORM_MINMAX)
    cv2.normalize(gt,  gt, 0, 255, cv2.NORM_MINMAX)  
    return MAE

def evaluate(rs, gt):
    p = Precision(rs, gt)
    r = Recall(rs, gt)
    f = F_measure(rs, gt)
    m = MAE(rs, gt)
    print ('Precision: ', p)
    print ('Recall:', r)
    print ('F_measure:',f)
    print ('MAE:', m)
    return p, r, f, m
    