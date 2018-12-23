import cv2
import os 

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
        path = os.path.join(direction, item) # Get files from paths.
        if(os.path.isfile(path)):
            img = cv2.imread(path)
            imageList.append(img)
    return imageList

def transparent():
    num = 1
    imageList = loadDatas("test1/filter_data")
    for image in imageList:
        num += 1 
        tmp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
        b, g, r = cv2.split(image)
        rgba = [b,g,r, alpha]
        dst = cv2.merge(rgba,4)
        final_image = cv2.resize(dst, (256,256))
        cv2.imwrite("test1/rm_background/"+"Filter" + str(num+1) + ".png", dst)