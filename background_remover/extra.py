from keras.models import load_model
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # don't show image
from time import time
np.set_printoptions(suppress=True)


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
        path = os.path.join(direction, item) # Get path's data
        if(os.path.isfile(path)):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256)) # resize to network's input shape
            
            imageList.append(img)
    return np.array(imageList)

def extra_validations(model, exdirection):
        # load data
        X_test = loadDatas(str(exdirection)).astype(np.float32)

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

        def ShowImage(X_test, Y_hat, dst):
                assert X_test.shape == Y_hat.shape
                assert Y_hat.shape == dst.shape
                
                for i in range(X_test.shape[0]):
                        
                        fig, axarr = plt.subplots(1, 3, figsize=(60, 20))
                        [idx.axis('off') for idx in axarr.flatten()]      
                        axarr[0].imshow(X_test[i].astype(np.uint8))
                        axarr[1].imshow(Y_hat[i].astype(np.uint8))
                        axarr[2].imshow(dst[i].astype(np.uint8))
                        plt.tight_layout(h_pad=0.1, w_pad=0.1)
                        plt.show()        
                        fig.savefig("test1/filter_data/test_extra_result_NO "+str(i + 1)+" .png")
                
        # show result
        ShowImage(X_test, Y_hat, dst)
        toc = time()


        print (str(X_test.shape[0]) + ' image predict and make image time: ' + str((toc - tic)) + ' sec')
        print ('average execute time: ' + str((toc - tic)/ X_test.shape[0]) + ' sec')
        return str((toc - tic)), str((toc - tic)/ X_test.shape[0])