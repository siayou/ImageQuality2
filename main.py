import numpy as np
import cv2
import os
import glob
import PIL
import scipy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as img
import sewar
from sewar.full_ref import uqi
from PIL import Image

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

path = '/home/siavash/Downloads/ImageDatasets'
dirs = os.listdir(path)

im_names_list = [""]
names_list = [""]
QualityMeasure = []
ImageIndex = []
print(type(im_names_list))

# Use the best image as your reference
img1 = cv2.imread("/home/siavash/Downloads/ImageDatasets/Position 9.tiff", 0)

# Define an ROI insite the image to compare, smaller regions are more efficient to process
img11 = img1[3700:3800, 2200:2300]

#Running over all images to calculate quality score with respect to the good image
for file in dirs:
  print(file)
  indexofTiff = file.rfind('.tiff')
  image_number = int(file[8:indexofTiff])
  #print(image_number)
  im_names_list.append(file)
  names_list.insert(image_number, file)

  #img = cv2.imread(path + "/" + file, 0)
  #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
  #cv2.imshow('image', img)
  #cv2.waitKey(0)
  #cv2.destroyAllWindows()

  img2 = cv2.imread(path + "/" + file, 0)
  img22 = img2[3700:3800, 2200:2300]

  # So far, liked ergas and psnr
  #Measure = (sewar.full_ref.ergas(img11, img22, r=4, ws=8))
  #Measure = (sewar.full_ref.mse(img11, img22))
  #Measure = (100 * sewar.full_ref.msssim(img11, img22, weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333], ws=11, K1=0.01, K2=0.03, MAX=None))
  #Measure = (sewar.full_ref.psnr(img11, img22, MAX=None))
  Measure = sewar.full_ref.rase(img11, img22, ws=8)
  print(Measure)

  QualityMeasure.insert(image_number, Measure)
  ImageIndex.insert(image_number, image_number)

#plt.figure()
#plt.interactive(False)
#plt.plot(ImageIndex, QualityMeasure)
#plt.show()


#print(QualityMeasure)
#print(ImageIndex)

list1, list2 = zip(*sorted(zip(ImageIndex, QualityMeasure)))

np1 = np.asarray(list1)
np2 = np.asarray(list2)

print(np1)
print(np2)

markerline, stemlines, baseline = plt.stem(np1, np2, markerfmt='D',use_line_collection = True)
plt.setp(baseline, color='r', linewidth=2)
plt.savefig('ScatterPlot.png')
plt.clf()


