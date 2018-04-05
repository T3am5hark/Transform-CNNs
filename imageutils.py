import cv2
import numpy as np
import matplotlib.pyplot as plt

def bgr2rgb(image):
    out = image.copy()
    out[:,:,0] = image[:,:,2]
    out[:,:,1] = image[:,:,1]
    out[:,:,2] = image[:,:,0]
    return out

def aspect_ratio(image):
    return image.shape[1]/float(image.shape[0])

def cv2imshow(image, xlen=10):
    plt.figure(figsize=(xlen, aspect_ratio(image)*xlen))
    plt.imshow(bgr2rgb(image))
    plt.show()

def get_block_dct_dims(image, dct_size, stride):
    output_height = np.floor(dct_size*(image.shape[0]/float(stride)-1))
    output_width  = np.floor(dct_size*(image.shape[1]/float(stride)-1))
    return (int(output_height), int(output_width))

def block_dct(image, dct_size=8, stride=4, debug=False):
    if len(image.shape) != 3:
        raise('Input must be 3d with dims [H,W,Channels]')
    height,width = get_block_dct_dims(image, dct_size, stride)
    if debug is True:
        print((height,width))
    block_dct = np.ndarray((height, width, image.shape[2]))
    if debug is True:
        print(block_dct.shape)
    for ch in np.arange(start=0, stop=image.shape[2]):
        block_y = 0
        stop_y = image.shape[0]-dct_size+1
        stop_x = image.shape[1]-dct_size+1
        for y in np.arange(start=0, stop=stop_y, step=stride):
            block_x = 0
            for x in np.arange(start=0, stop=stop_x, step=stride):
                block = cv2.dct(image[y:(y+dct_size),x:(x+dct_size),ch])
                block_dct[block_y:(block_y+dct_size), block_x:(block_x+dct_size), ch] = block
                block_x = block_x + dct_size
            block_y = block_y + dct_size
    return block_dct

def dataset_transform_block_dct(dataset, dct_size=8, stride=4, debug=False):
    if len(dataset.shape) != 4:
        raise('Input must be 4d with dimensions [nData, H, W, Channels]')
    height,width = get_block_dct_dims(dataset[0,:,:,:], dct_size, stride)
    xfrm_shape = (dataset.shape[0], height, width, dataset.shape[3])
    if debug is True:
        print(xfrm_shape)
    xfrm_set = np.ndarray(xfrm_shape)
    if debug is True:
        print((height,width))
    for i in range(0, dataset.shape[0]):
        xfrm_set[i,:,:,:] = block_dct(dataset[i,:,:,:], dct_size, stride, debug)
    return xfrm_set
