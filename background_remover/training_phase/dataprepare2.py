import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from time import time 
get_ipython().run_line_magic('matplotlib', 'inline')

def loadDatas(direction, color = 'rgb'):
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
        path = os.path.join(direction, item) # 把list裡面的檔名加到路徑
        if(os.path.isfile(path)):
            img = cv2.imread(path)
            if color == 'rgb':                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif color == 'gray':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (256, 256)) # resize到網絡input的shape
            
            imageList.append(img)
    return np.array(imageList)

def datasets():
    # First file
    mask = loadDatas('train6/train_label', color = 'gray')

    # sample num , row, col, rgb 3 channels
    mask = mask[ :,:,:,np.newaxis]
    
    # to check the dimension num
    print (mask.shape)

    x = loadDatas('train6/train_image')
    print (x.shape)

    mask = (mask>0.5).astype(np.float32)
    
    tic = time()
    # save the data
    file = h5py.File('all.h5', 'w')
    file.create_dataset('X_pre', data = x)
    file.create_dataset('Y_pre', data = mask)
    file.close()
    toc = time()
    print ('execute time: ' + str((toc - tic)) + ' sec')

    mask_show = np.append(mask, mask, axis = 3)
    mask_show = np.append(mask_show, mask, axis = 3)
    cv2.normalize(mask_show, mask_show, 0, 255, cv2.NORM_MINMAX)
    dst = (cv2.bitwise_and(x.astype(np.uint8), mask_show.astype(np.uint8)))
    fig, axarr = plt.subplots(30, 3, figsize=(15, 150))
    [idx.axis('off') for idx in axarr.flatten()]

    for i in range(axarr.shape[0]):    
        axarr[i,0].imshow(x[i])
        axarr[i,1].imshow((mask_show[i]).astype(np.uint8))
        axarr[i,2].imshow((dst[i]).astype(np.uint8))

    plt.tight_layout(h_pad=0.1, w_pad=0.1)
    plt.show()
    fig.savefig('training_dataset_example_gray_6.png')

    # Second file
    # # 3 - Making development dataset
    x_dev = loadDatas('dev2/devimage')
    mask_dev = loadDatas('dev2/devlabel', 'gray')
    print (x_dev.shape, mask_dev.shape)
    mask_dev = mask_dev[ :,:,:,np.newaxis]
    print (mask_dev.shape)
    mask_dev = (mask_dev>0.5).astype(np.float32)
    tic = time()
    file = h5py.File('dev_dataset_gray_2.h5', 'w')
    file.create_dataset('X_pre', data = x_dev)
    file.create_dataset('Y_pre', data = mask_dev)
    file.close()
    toc = time()
    print ('execute time: ' + str((toc - tic)) + ' sec')

    mask_dev_show = np.append(mask_dev, mask_dev, axis = 3)
    mask_dev_show = np.append(mask_dev_show, mask_dev, axis = 3)
    cv2.normalize(mask_dev_show, mask_dev_show, 0, 255, cv2.NORM_MINMAX)
    dst_dev = (cv2.bitwise_and(x_dev.astype(np.uint8), mask_dev_show.astype(np.uint8)))
    fig, axarr = plt.subplots(30, 3, figsize=(15, 150))
    [idx.axis('off') for idx in axarr.flatten()]
    for i in range(axarr.shape[0]):    
        axarr[i,0].imshow(x_dev[i])
        axarr[i,1].imshow(mask_dev_show[i].astype(np.uint8))
        axarr[i,2].imshow(dst_dev[i].astype(np.uint8))
    plt.tight_layout(h_pad=0.1, w_pad=0.1)
    plt.show()
    fig.savefig('dev_dataset_example_gray_2.png')