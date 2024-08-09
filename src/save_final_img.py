import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

IMG_DIRECTORY = "C:\\Users\\Prade\\Desktop\\heartData\\splitZip1\\img\\extracted_gzs"
LABEL_DIRECTORY = "C:\\Users\\Prade\\Desktop\\heartData\\splitZip1\\labels\\extracted_gzs"
SAVE_IMG = "C:\\Users\\Prade\\Desktop\\heartData\\final_images"
SAVE_LABELS = "C:\\Users\\Prade\\Desktop\\heartData\\final_labels"

def save(path, save_path):
    i = 0
    for file in os.listdir(path):
        f = os.path.join(path, file)
        if os.path.isfile(f):
            i += 1
            test_load = nib.load(f).get_fdata()
            #test_load.shape
            test = test_load[:,:,59]
            plt.imshow(test)
            #plt.show()
            plt.savefig("{}\\Figure_{}.png".format(save_path, i))

try:
    save(IMG_DIRECTORY, SAVE_IMG)
except:
    save(LABEL_DIRECTORY, SAVE_LABELS)