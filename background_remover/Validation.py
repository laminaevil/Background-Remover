from keras.models import load_model
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # don't show image
from time import time
np.set_printoptions(suppress=True)
from evaluate import *


def loadDatas(direction):
    """
    Loading image datas from the the same folder.

    Arguments:
    direction -- A direction of image data from the same folder.

    Return:
    np.array(imageList) -- Data in the form of numpy.
    """
    imageList=[]
    list = os.listdir(direction)
    for item in list:
        path = os.path.join(direction, item) # get the path's data
        if(os.path.isfile(path)):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256)) # resize to network's input shape
            
            imageList.append(img)
    return np.array(imageList)

def validation(model):
        # load data
        X_test = loadDatas('test1/test data').astype(np.float32)
        Y_test = loadDatas('test1/gt').astype(np.float32)

        # predict mask Y_hat
        cv2.normalize(X_test,  X_test, 0, 1, cv2.NORM_MINMAX)
        Y_hat =  (model.predict(X_test))
        cv2.normalize(X_test,  X_test, 0, 255, cv2.NORM_MINMAX)

        tic = time()
        # change Y_hat dimension
        Y_hat1 = (Y_hat > 0.5).astype(np.float32)
        Y_hat = np.append(Y_hat1, Y_hat1, axis = 3)
        Y_hat = np.append(Y_hat, Y_hat1, axis = 3)
        cv2.normalize(Y_hat, Y_hat, 0, 255, cv2.NORM_MINMAX)
        Y_hat.astype(np.uint8)

        # Make dst
        dst = (cv2.bitwise_and(X_test.astype(np.uint8), Y_hat.astype(np.uint8)))

        def ShowImageWithGT(X_test, Y_hat, dst, Y_test):
                assert X_test.shape == Y_hat.shape
                assert Y_hat.shape == Y_test.shape
                assert dst.shape == Y_test.shape

                for i in range(X_test.shape[0]):
                        
                        fig, axarr = plt.subplots(1, 4, figsize=(80, 20))
                        [idx.axis('off') for idx in axarr.flatten()]      
                        axarr[0].imshow(X_test[i].astype(np.uint8))
                        axarr[1].imshow(Y_hat[i].astype(np.uint8))
                        axarr[2].imshow(dst[i].astype(np.uint8))
                        axarr[3].imshow(Y_test[i].astype(np.uint8))
                        plt.tight_layout(h_pad=0.1, w_pad=0.1)
                        plt.show()
                        
                        fig.savefig("test1/filter_data/test_result_NO "+str(i + 1)+" .png")
                
        # show result
        ShowImageWithGT(X_test, Y_hat, dst, Y_test)
        toc = time()
        # evalution
        Precision, Recall, F_measure, MAE = 0, 0, 0, 0
        for i in range(Y_test.shape[0]):
                print ('Image No '+str(i+1)+': ')
                p, r, f, m = evaluate(Y_hat[i], Y_test[i])
                Precision += p
                Recall += r
                F_measure += f
                MAE += m
                print ()
        
        print ('ave Precision: ', Precision/Y_test.shape[0])
        print ('ave Recall: ', Recall/Y_test.shape[0])
        print ('ave F_measure: ', F_measure/Y_test.shape[0])
        print ('ave MAE: ', MAE/Y_test.shape[0])

        print (str(X_test.shape[0]) + ' image predict and make image time: ' + str((toc - tic)) + ' sec')
        print ('average execute time: ' + str((toc - tic)/ X_test.shape[0]) + ' sec')

        return Precision/Y_test.shape[0], Recall/Y_test.shape[0], F_measure/Y_test.shape[0], MAE/Y_test.shape[0], str((toc - tic)), ((toc - tic)/ X_test.shape[0])
