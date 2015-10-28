#Import libraries for doing image analysis
from skimage.io import imread
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier as RF
import glob
import os
import sys
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from matplotlib import colors
from pylab import cm
from skimage import segmentation
from skimage.morphology import watershed
from skimage import measure
from skimage import morphology,filter
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.feature import peak_local_max
import cv2
import math
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.transform import warp, AffineTransform
# make graphics inline
#%matplotlib inline
n_estim = 100

import warnings
warnings.filterwarnings("ignore")

# get the classnames from the directory structure
directory_names = list(set(glob.glob(os.path.join("competition_data","train", "*"))\
 ).difference(set(glob.glob(os.path.join("competition_data","train","*.*")))))
 
# Example image
# This example was chosen for because it has two 
# noncontinguous pieces that will make the segmentation 
# example more illustrative
example_file = glob.glob(os.path.join(directory_names[79],"*.jpg"))[0]
print example_file
im = imread(example_file, as_grey=True)
plt.imshow(im, cmap=cm.gray)
#plt.show()

# First we threshold the image by only taking values 
# greater than the mean to reduce noise in the image
# to use later as a mask
f = plt.figure(figsize=(12,3))
imthr = im.copy()
imthr = np.where(im > np.mean(im),0.,1.0)
sub1 = plt.subplot(1,4,1)
plt.imshow(im, cmap=cm.gray)
sub1.set_title("Original Image")

sub2 = plt.subplot(1,4,2)
plt.imshow(imthr, cmap=cm.gray_r)
sub2.set_title("Thresholded Image")

imdilated = morphology.dilation(imthr, np.ones((4,4)))
sub3 = plt.subplot(1, 4, 3)
plt.imshow(imdilated, cmap=cm.gray_r)
sub3.set_title("Dilated Image")

labels = measure.label(imdilated)
labels = imthr*labels
labels = labels.astype(int)
sub4 = plt.subplot(1, 4, 4)
sub4.set_title("Labeled Image")
plt.imshow(labels)

# calculate common region properties for each region within the segmentation
regions = measure.regionprops(labels)
# find the largest nonzero region
def getLargestRegions(props=regions, labelmap=labels, imagethres=imthr):
	regionmaxprop = None
	region2ndmaxprop = None
	for regionprop in props:
		# check to see if the region is at least 50% nonzero
		if sum(imagethres[labelmap == regionprop.label])*1.0/regionprop.area < 0.50:
			continue
		if regionmaxprop is None:
			regionmaxprop = regionprop
		elif region2ndmaxprop is None:
			region2ndmaxprop = regionprop
		if regionmaxprop.filled_area < regionprop.filled_area:
			region2ndmaxprop = regionmaxprop
			regionmaxprop = regionprop
		elif ((not region2ndmaxprop is None) and (region2ndmaxprop.filled_area < regionprop.filled_area)):
			region2ndmaxprop = regionprop
	return regionmaxprop,region2ndmaxprop
	
regionmax,region2ndmax = getLargestRegions()
plt.imshow(np.where(labels == regionmax.label,1.0,0.0))
#plt.show()

def getMinorMajorRatio(image):
	image = image.copy()
	# Create the thresholded image to eliminate some of the background
	imagethr = np.where(image > np.mean(image),0.,1.0)
	imagethr2 = np.where(image > np.mean(image) - 2*np.std(image),0.,1.0)
	coords = corner_peaks(corner_harris(imagethr), min_distance=5)
	coords_subpix = corner_subpix(imagethr, coords, window_size=13)
	cornercentercoords = np.nanmean(coords_subpix, axis=0)
	cornerstdcoords = np.nanstd(coords_subpix, axis=0)
	
	#Dilate the image
	imdilated = morphology.dilation(imagethr, np.ones((4,4)))

	# Create the label list
	label_list = measure.label(imdilated)
	label_list2 = imagethr2*label_list
	label_list = imagethr*label_list
	label_list2 = label_list2.astype(int)
	label_list = label_list.astype(int)
	   
	region_list = measure.regionprops(label_list)
	region_list2 = measure.regionprops(label_list2)
	maxregion,max2ndregion = getLargestRegions(region_list, label_list, imagethr)
	maxregion2,max2ndregion2 = getLargestRegions(region_list2, label_list2, imagethr2)

	# guard against cases where the segmentation fails by providing zeros
	ratio = 0.0
	fillratio = 0.0
	largeeigen = 0.0
	smalleigen = 0.0
	eigenratio = 0.0
	solidity = 0.0
	perimratio = 0.0
	arearatio = 0.0
	orientation = 0.0
	centroid = (0.0,0.0)
	cornercenter = 0.0
	cornerstd = 0.0
	hu1 = hu2 = hu3 = hu12 = hu13 = hu23 = 0.0
	if ((not maxregion is None) and  (maxregion.major_axis_length != 0.0)):
		ratio = 0.0 if maxregion is None else  maxregion.minor_axis_length*1.0 / maxregion.major_axis_length
		largeeigen = 0.0 if maxregion is None else maxregion.inertia_tensor_eigvals[0]
		smalleigen = 0.0 if maxregion is None else maxregion.inertia_tensor_eigvals[1]
		fillratio = 0.0 if (maxregion2 is None or maxregion2.minor_axis_length == 0.0) else maxregion2.filled_area/(maxregion2.minor_axis_length*maxregion2.major_axis_length)
		solidity = 0.0 if maxregion2 is None else maxregion2.solidity
		hu1 = 0.0 if maxregion is None else maxregion.moments_hu[1]
		hu2 = 0.0 if maxregion is None else maxregion.moments_hu[2]
		hu3 = 0.0 if maxregion is None else maxregion.moments_hu[3]
		hu12 = 0.0 if (maxregion is None or hu1==0.0) else hu2/hu1
		hu13 = 0.0 if (maxregion is None or hu1==0.0) else hu3/hu1
		hu23 = 0.0 if (maxregion is None or hu2==0.0) else hu3/hu2
		perimratio = 0.0 if (maxregion is None or maxregion.minor_axis_length==0.0) else maxregion.perimeter/(maxregion.minor_axis_length*4.0+maxregion.major_axis_length*4.0)
		eigenratio = 0.0 if largeeigen == 0.0 else smalleigen/largeeigen
		orientation = 0.0 if maxregion is None else maxregion.orientation
		centroid = (0.0,0.0) if maxregion is None else maxregion.centroid
		cornercentercoords = np.absolute(cornercentercoords - centroid) if maxregion.major_axis_length==0.0 else np.absolute(cornercentercoords - centroid)/maxregion.major_axis_length
		cornercenter = np.linalg.norm(cornercentercoords)
		if maxregion.major_axis_length!=0.0: cornerstdcoords = np.absolute(cornerstdcoords)/maxregion.major_axis_length
		cornerstd = np.linalg.norm(cornerstdcoords)
	if ((not maxregion is None) and (not max2ndregion is None)):
		arearatio = max2ndregion.area/maxregion.area
	#print perimratio
	if np.isnan(cornercenter):
		cornercenter = 0.0
	if sum(np.isnan(cornercentercoords)) > 0.0:
		cornercentercoords = np.array([0.0,0.0])
	if math.isnan(cornerstd):
		cornerstd = 0.0
	if sum(np.isnan(cornerstdcoords)) > 0.0:
		cornerstdcoords = np.array([0.0,0.0])
	return cornercenter,cornercentercoords,cornerstd,cornerstdcoords,ratio,fillratio,eigenratio,solidity,hu1,hu2,hu3,hu12,hu13,hu23,perimratio,arearatio,orientation,centroid

def getCorners(image, orientation):
	max_corners = 25
	image2 = image.copy()
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	gray = np.float32(gray)
	rows,cols = gray.shape
	M = cv2.getRotationMatrix2D(centroid,(orientation+math.pi/2.0)*180.0/math.pi,1)
	grayrot = cv2.warpAffine(gray,M,(cols,rows))
	dst = cv2.cornerHarris(gray,2,3,0.04)
	image2[dst>0.01*dst.max()]=[0,0,255]
	#cv2.imshow('image',image),cv2.imshow('image2',image2)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	corners = cv2.goodFeaturesToTrack(gray,max_corners,0.01,10)
	corners = np.int0(corners)
	num_corners = corners.size/2
	corner_ratio = float(num_corners)/max_corners
	#print corner_ratio
	return corner_ratio
# Rescale the images and create the combined metrics and training labels

#get the total training images
numberofImages = 0
for folder in directory_names:
	for fileNameDir in os.walk(folder):   
		for fileName in fileNameDir[2]:
			# Only read in the images
			if fileName[-4:] != ".jpg":
				continue
			numberofImages += 1

# We'll rescale the images to be 25x25
maxPixel = 25
imageSize = maxPixel * maxPixel
num_rows = numberofImages # one row for each image in the training dataset
#Zak's note: I think I need to add to num_features to add new metrics
num_features = imageSize + 23 # for our ratio


# X is the feature vector with one row of features per image
# consisting of the pixel values and our metric
X = np.zeros((num_rows, num_features), dtype=float)
# y is the numeric class label 
y = np.zeros((num_rows))

files = []
# Generate training data
i = 0    
label = 0
# List of string of class names
namesClasses = list()

print "Reading images"
# Navigate through the list of directories
for folder in directory_names:
	# Append the string class name for each class
	currentClass = folder.split(os.pathsep)[-1]
	namesClasses.append(currentClass)
	for fileNameDir in os.walk(folder):   
		for fileName in fileNameDir[2]:
			# Only read in the images
			if fileName[-4:] != ".jpg":
				continue
            
			# Read in the images and create the features
			nameFileImage = "{0}{1}{2}".format(fileNameDir[0], os.sep, fileName)            
			image = imread(nameFileImage, as_grey=True)
			# Added from https://github.com/Newmu/Stroke-Prediction/blob/master/startPredictingGenCode.py
			thresh = 0.9*255
			if image.min < 0.75*255:
				img = image < thresh
			else:
				img = image
			if img.sum() != 0:
				imgX,imgY = np.nonzero(img)
				imgW = imgX.max()-imgX.min()
				imgH = imgY.max()-imgY.min()
				if (imgW>1 and imgH>1):
					image = image[imgX.min():imgX.max(),imgY.min():imgY.max()]
			#----------------------------------
			cvimage = cv2.imread(nameFileImage)
			files.append(nameFileImage)
			(cornercenter,cornercentercoords,cornerstd,cornerstdcoords,axisratio,fillratio,eigenratio,solidity,hu1,hu2,hu3,hu12,hu13,hu23,
			perimratio,arearatio,orientation,centroid) = getMinorMajorRatio(image)
			corners = getCorners(cvimage,orientation)
			image = resize(image, (maxPixel, maxPixel))
    
			peaklocalmax = np.nanmean(peak_local_max(image))
			felzen = np.nanmean(segmentation.felzenszwalb(image))
			if np.isnan(peaklocalmax):
				peaklocalmax = 0.0
			if np.isnan(felzen):
				felzen = 0.0
			# Store the rescaled image pixels and the axis ratio
			X[i, 0:imageSize] = np.reshape(image, (1, imageSize))
			#Zak's note: I think I need to add new features into this array after axisratio
			X[i, imageSize] = axisratio
			X[i, imageSize+1] = fillratio
			X[i, imageSize+2] = eigenratio
			X[i, imageSize+3] = solidity
			X[i, imageSize+4] = corners
			X[i, imageSize+5] = hu1
			X[i, imageSize+6] = hu2
			X[i, imageSize+7] = hu3
			X[i, imageSize+8] = hu12
			X[i, imageSize+9] = hu13
			X[i, imageSize+10] = hu23
			X[i, imageSize+11] = perimratio
			X[i, imageSize+12] = arearatio
			X[i, imageSize+13] = cornercenter
			X[i, imageSize+14] = cornercentercoords[0]
			X[i, imageSize+15] = cornercentercoords[1]
			X[i, imageSize+16] = cornerstd
			X[i, imageSize+17] = cornerstdcoords[0]
			X[i, imageSize+18] = cornerstdcoords[1]
			X[i, imageSize+19] = np.nanmean(filter.vsobel(image))
			X[i, imageSize+20] = np.nanmean(filter.hsobel(image))
			X[i, imageSize+21] = felzen
			X[i, imageSize+22] = peaklocalmax
			# Store the classlabel
			y[i] = label
			i += 1
			# report progress for each 5% done  
			report = [int((j+1)*num_rows/20.) for j in range(20)]
			if i in report: print np.ceil(i *100.0 / num_rows), "% done"
	label += 1
	
# Loop through the classes two at a time and compare their distributions of the Width/Length Ratio

#Create a DataFrame object to make subsetting the data on the class 
df = pd.DataFrame({"class": y[:], "ratio": X[:, num_features-1]})

f = plt.figure(figsize=(30, 20))
#we suppress zeros and choose a few large classes to better highlight the distributions.
df = df.loc[df["ratio"] > 0]
minimumSize = 20 
counts = df["class"].value_counts()
largeclasses = [int(x) for x in list(counts.loc[counts > minimumSize].index)]
# Loop through 40 of the classes 
for j in range(0,40,2):
	subfig = plt.subplot(4, 5, j/2 +1)
	# Plot the normalized histograms for two classes
	classind1 = largeclasses[j]
	classind2 = largeclasses[j+1]
	n, bins,p = plt.hist(df.loc[df["class"] == classind1]["ratio"].values,\
						alpha=0.5, bins=[x*0.01 for x in range(100)], \
						label=namesClasses[classind1].split(os.sep)[-1], normed=1)

	n2, bins,p = plt.hist(df.loc[df["class"] == (classind2)]["ratio"].values,\
							alpha=0.5, bins=bins, label=namesClasses[classind2].split(os.sep)[-1],normed=1)
	subfig.set_ylim([0.,10.])
	plt.legend(loc='upper right')
	plt.xlabel("Fill Ratio")
	
print "Training"
# n_estimators is the number of decision trees
# max_features also known as m_try is set to the default value of the square root of the number of features
clf = RF(n_estimators=n_estim, n_jobs=3);
scores = cross_validation.cross_val_score(clf, X, y, cv=5, n_jobs=1);
print "Accuracy of all classes"
print np.mean(scores)

kf = KFold(y, n_folds=5)
y_pred = y * 0
for train, test in kf:
	X_train, X_test, y_train, y_test = X[train,:], X[test,:], y[train], y[test]
	clf = RF(n_estimators=n_estim, n_jobs=3)
	clf.fit(X_train, y_train)
	y_pred[test] = clf.predict(X_test)
print classification_report(y, y_pred, target_names=namesClasses)

def multiclass_log_loss(y_true, y_pred, eps=1e-15):
	"""Multi class version of Logarithmic Loss metric.
	https://www.kaggle.com/wiki/MultiClassLogLoss

	Parameters
	----------
	y_true : array, shape = [n_samples]
			true class, intergers in [0, n_classes - 1)
	y_pred : array, shape = [n_samples, n_classes]

	Returns
	-------
	loss : float
	"""
	predictions = np.clip(y_pred, eps, 1 - eps)

	# normalize row sums to 1
	predictions /= predictions.sum(axis=1)[:, np.newaxis]

	actual = np.zeros(y_pred.shape)
	n_samples = actual.shape[0]
	actual[np.arange(n_samples), y_true.astype(int)] = 1
	vectsum = np.sum(actual * np.log(predictions))
	loss = -1.0 / n_samples * vectsum
	return loss
	
# Get the probability predictions for computing the log-loss function
kf = KFold(y, n_folds=5)
# prediction probabilities number of samples, by number of classes
y_pred = np.zeros((len(y),len(set(y))))
for train, test in kf:
	X_train, X_test, y_train, y_test = X[train,:], X[test,:], y[train], y[test]
	clf = RF(n_estimators=n_estim, n_jobs=3)
	clf.fit(X_train, y_train)
	y_pred[test] = clf.predict_proba(X_test)
	
multiclass_log_loss(y, y_pred)
