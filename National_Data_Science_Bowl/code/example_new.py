#Import libraries for doing image analysis
import cPickle
from skimage.io import imread
from sklearn import decomposition
from skimage.feature import hog
from skimage.feature import local_binary_pattern
from skimage.transform import resize
from skimage.exposure import equalize_hist
from skimage.filter import gaussian_filter
from sklearn.ensemble import RandomForestClassifier as RF
from memory_profiler import profile
import glob
import os
import sys
from random import randint
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
from scipy import ndimage,signal
from skimage.feature import peak_local_max
import cv2
import math
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.transform import warp, AffineTransform,rotate
# make graphics inline
#%matplotlib inline
n_estim = 100
radius = 3
n_points = 8 * radius
METHOD = 'uniform'
n_folds = 5
import warnings
warnings.filterwarnings("ignore")
from classes import *
n_jobs = 1

# get the classnames from the directory structure

# For full set
directory_names = list(set(glob.glob(os.path.join("competition_data","train", "*"))\
 ).difference(set(glob.glob(os.path.join("competition_data","train","*.*")))))
 
# For smaller set
#directory_names = list(set(glob.glob(os.path.join("smaller_comp","train", "*"))\
# ).difference(set(glob.glob(os.path.join("smaller_comp","train","*.*")))))
 
# Example image
# This example was chosen for because it has two 
# noncontinguous pieces that will make the segmentation 
# example more illustrative
#example_file = glob.glob(os.path.join(directory_names[79],"*.jpg"))[0]
example_file = glob.glob(os.path.join(directory_names[1],"*.jpg"))[0]
#print example_file
im = imread(example_file, as_grey=True)
#plt.imshow(im, cmap=cm.gray)
#plt.show()

# First we threshold the image by only taking values 
# greater than the mean to reduce noise in the image
# to use later as a mask
#f = plt.figure(figsize=(12,3))
imthr = im.copy()
imthr = np.where(im > np.mean(im),0.,1.0)
#sub1 = plt.subplot(1,4,1)
#plt.imshow(im, cmap=cm.gray)
#sub1.set_title("Original Image")

#sub2 = plt.subplot(1,4,2)
#plt.imshow(imthr, cmap=cm.gray_r)
#sub2.set_title("Thresholded Image")

imdilated = morphology.dilation(imthr, np.ones((4,4)))
#sub3 = plt.subplot(1, 4, 3)
#plt.imshow(imdilated, cmap=cm.gray_r)
#sub3.set_title("Dilated Image")

labels = measure.label(imdilated)
#labels = imthr*labels
#labels = labels.astype(int)
#sub4 = plt.subplot(1, 4, 4)
#sub4.set_title("Labeled Image")
#plt.imshow(labels)

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
	
#regionmax,region2ndmax = getLargestRegions()
#plt.imshow(np.where(labels == regionmax.label,1.0,0.0))
#plt.show()

def rotImage(image):
	# Create the thresholded image to eliminate some of the background
	imagethr = np.where(image > np.mean(image),0.,1.0)
	
	#Dilate the image
	imdilated = morphology.dilation(imagethr, np.ones((4,4)))

	# Create the label list
	label_list = measure.label(imdilated)
	label_list = imagethr*label_list
	label_list = label_list.astype(int)
	   
	region_list = measure.regionprops(label_list)
	maxregion,max2ndregion = getLargestRegions(region_list, label_list, imagethr)
	if not maxregion is None: 
		pivot = maxregion.centroid
		padX = [image.shape[0] - pivot[0], pivot[0]]
		padY = [image.shape[1] - pivot[1], pivot[1]]
		padded = np.pad(image, [padX, padY], 'constant', constant_values = 255)
		#rotator = AffineTransform(rotation=maxregion.orientation)
		#image2 = warp(padded, rotator)
		image2 = rotate(padded,-maxregion.orientation*180.0/math.pi+90.0,mode='nearest')
	else:
		image2 = image.copy()
	# f = plt.figure(figsize=(9,3))
	# x0, y0 = maxregion.centroid	
	# x0_pad = x0 + padX[0]
	# y0_pad = y0 + padY[0]
	# orientation = maxregion.orientation
	# y1 = y0 + math.cos(orientation) * 0.5 * maxregion.major_axis_length
	# x1 = x0 - math.sin(orientation) * 0.5 * maxregion.major_axis_length
	# y2 = y0_pad
	# x2 = x0_pad - 0.5 * maxregion.major_axis_length
	# sub1 = plt.subplot(1,3,1)
	# sub1.plot((y0, y1), (x0, x1), '-r', linewidth=2.5)
	# plt.imshow(image, cmap=cm.gray)
	# sub1.set_title("Original Image")
	# sub2 = plt.subplot(1,3,2)
	# plt.imshow(padded, cmap=cm.gray)
	# sub2.set_title("Padded Image")
	# sub3 = plt.subplot(1,3,3)
	# sub3.plot((y0_pad, y2), (x0_pad, x2), '-r', linewidth=2.5)
	# plt.imshow(image2, cmap=cm.gray)
	# sub3.set_title("Rotated Image")
	# plt.show()
	#sys.exit()
	return image2

features = {'axisratio': 0.0, 'fillratio': 0.0, 'eigenvals': np.array([0.0,0.0]),
			'eigenratio': 0.0, 'solidity': 0.0, 'perimratio': 0.0, 'arearatio': 0.0,
			'orientation': 0.0, 'centroid': np.array([0.0,0.0]), 'wcentroiddiff': np.array([0.0,0.0]),
			'cornerctr': 0.0, 'cornerstd': 0.0, 'cornerctrcoords': np.array([0.0,0.0]), 
			'cornerstdcoords': np.array([0.0,0.0]), 'lrdiff': 0.0, 'tbdiff': 0.0, 'hu': np.zeros(7,float),
			'huratios': np.zeros(3,float), 'whu': np.zeros(7,float), 'whuratios': np.zeros(3,float), 
			'extent': 0.0, 'minintensity': 0.0, 'meanintensity': 0.0, 'maxintensity': 0.0,
			'intensityratios': np.zeros(3,float), 'vsobel': 0.0, 'hsobel': 0.0, 'felzen': 0.0,
			'peaklocalmax': 0.0, 'hormean': 0.0, 'horstd': 0.0, 'vertmean': 0.0, 'vertstd': 0.0,
			'horcount': 0.0, 'vertcount': 0.0, 'corners': 0.0,
			'area': 0.0, 'bbox': np.zeros(4,float), 'convex_area': 0.0, 'eccentricity': 0.0, 'equivalent_diameter': 0.0,
			'euler_number': 0.0, 'filled_area': 0.0, 'major_axis': 0.0, 'minor_axis': 0.0, 'moments': np.zeros(16,float),
			'moments_central': np.zeros(16,float), 'moments_normalized': np.zeros(16,float), 'perimeter': 0.0,
			'wcentroid': np.array([0.0,0.0]), 'weighted_moments': np.zeros(16,float), 'weighted_moments_central': np.zeros(16,float),
			'weighted_moments_normalized': np.zeros(13,float), 'original_size': 0			
			}
	
def getMinorMajorRatio(image, features = features):
	features = features.copy()
	image = image.copy()
	# Create the thresholded image to eliminate some of the background
	imagethr = np.where(image > np.mean(image),0.,1.0)
	imagethr2 = np.where(image > np.mean(image) - 2*np.std(image),0.,1.0)

	#Dilate the image
	imdilated = morphology.dilation(imagethr, np.ones((4,4)))

	# Create the label list
	label_list = measure.label(imdilated)
	label_list2 = imagethr2*label_list
	label_list = imagethr*label_list
	label_list2 = label_list2.astype(int)
	label_list = label_list.astype(int)
	   
	region_list = measure.regionprops(label_list, intensity_image=image)
	region_list2 = measure.regionprops(label_list2, intensity_image=image)
	maxregion,max2ndregion = getLargestRegions(region_list, label_list, imagethr)
	maxregion2,max2ndregion2 = getLargestRegions(region_list2, label_list2, imagethr2)

	# guard against cases where the segmentation fails by providing zeros
	if not maxregion is None:
		features['area'] = maxregion.area
		features['bbox'] = maxregion.bbox
		features['convex_area'] = maxregion.convex_area
		features['eccentricity'] = maxregion.eccentricity
		features['equivalent_diameter'] = maxregion.equivalent_diameter
		features['euler_number'] = maxregion.euler_number
		features['filled_area'] = maxregion.filled_area
		features['major_axis'] = maxregion.major_axis_length
		features['minor_axis'] = maxregion.minor_axis_length
		features['moments'] = maxregion.moments.flatten()
		features['moments_central'] = maxregion.moments_central.flatten()
		features['moments_normalized'] = maxregion.moments_normalized.flatten()[np.array([2,3,5,6,7,8,9,10,11,12,13,14,15])]
		features['perimeter'] = maxregion.perimeter
		features['wcentroid'] = maxregion.weighted_centroid
		features['weighted_moments'] = maxregion.weighted_moments.flatten()
		features['weighted_moments_central'] = maxregion.weighted_moments_central.flatten()
		features['weighted_moments_normalized'] = maxregion.weighted_moments_normalized.flatten()[np.array([2,3,5,6,7,8,9,10,11,12,13,14,15])]

		corners = corner_peaks(corner_harris(maxregion.image), min_distance=5)
		corners_subpix = corner_subpix(maxregion.image, corners, window_size=13)
		features['cornerctrcoords'] = np.nanmean(corners_subpix, axis=0)
		features['cornerstdcoords'] = np.nanstd(corners_subpix, axis=0)
		features['eigenvals'] = maxregion.inertia_tensor_eigvals
		features['hu'] = maxregion.moments_hu
		if not features['hu'][0] == 0.0:
			features['huratios'][0] = features['hu'][1]/features['hu'][0]
			features['huratios'][1] = features['hu'][2]/features['hu'][0]
		if not features['hu'][1] == 0.0:
			features['huratios'][2] = features['hu'][2]/features['hu'][1]
		features['whu'] = maxregion.weighted_moments_hu
		if not features['whu'][0] == 0.0:
			features['whuratios'][0] = features['whu'][1]/features['whu'][0]
			features['whuratios'][1] = features['whu'][2]/features['whu'][0]
		if not features['whu'][1] == 0.0:
			features['whuratios'][2] = features['whu'][2]/features['whu'][1]
		features['extent'] = maxregion.extent
		features['minintensity'] = maxregion.min_intensity
		features['meanintensity'] = maxregion.mean_intensity
		features['maxintensity'] = maxregion.max_intensity
		if not features['maxintensity'] == 0.0:
			features['intensityratios'][0] = features['meanintensity']/features['maxintensity']
			features['intensityratios'][1] = features['minintensity']/features['maxintensity']
		if not features['meanintensity'] == 0.0:
			features['intensityratios'][2] = features['minintensity']/features['meanintensity']
		if not maxregion.minor_axis_length == 0.0:
			features['perimratio'] = maxregion.perimeter/(maxregion.minor_axis_length*4.0+maxregion.major_axis_length*4.0)
		if not features['eigenvals'][0] == 0.0:
			features['eigenratio'] = features['eigenvals'][1]/features['eigenvals'][0]
		features['orientation'] = maxregion.orientation
		features['centroid'] = maxregion.centroid
		features['wcentroiddiff'] = np.absolute(features['centroid']-np.asarray(maxregion.weighted_centroid))/maxregion.major_axis_length
		features['cornerctrcoords'] = np.absolute(features['cornerctrcoords'] - features['centroid']) if maxregion.major_axis_length==0.0 else np.absolute(features['cornerctrcoords'] - features['centroid'])/maxregion.major_axis_length
		features['cornerctr'] = np.linalg.norm(features['cornerctrcoords'])
		if not maxregion.major_axis_length == 0.0:
			features['axisratio'] = maxregion.minor_axis_length / maxregion.major_axis_length
			features['cornerstdcoords'] = np.absolute(features['cornerstdcoords'])/maxregion.major_axis_length
		features['cornerstd'] = np.linalg.norm(features['cornerstdcoords'])
		left = np.sum(maxregion.image[:,maxregion.image.shape[1]/2:])
		if maxregion.image.shape[1] % 2 == 0:
			right = np.sum(maxregion.image[:,:maxregion.image.shape[1]/2])
		else:
			right = np.sum(maxregion.image[:,:maxregion.image.shape[1]/2+1])
		features['lrdiff'] = np.abs((right-left)/(right+left)) 
		top = np.sum(maxregion.image[maxregion.image.shape[0]/2:,:])
		if maxregion.image.shape[0] % 2 == 0:
			bottom = np.sum(maxregion.image[:maxregion.image.shape[0]/2,:])
		else:
			bottom = np.sum(maxregion.image[:maxregion.image.shape[0]/2+1,:])
		features['tbdiff'] = np.abs((top-bottom)/(top+bottom)) 
		if not max2ndregion is None:
			features['arearatio'] = max2ndregion.area/maxregion.area
	if not maxregion2 is None:
		if not maxregion2.minor_axis_length == 0.0:
			features['fillratio'] = maxregion2.filled_area/(maxregion2.minor_axis_length*maxregion2.major_axis_length)
		features['solidity'] = maxregion2.solidity
	if np.isnan(features['cornerctr']):
		features['cornerctr'] = 0.0
	if sum(np.isnan(features['cornerctrcoords'])) > 0.0:
		features['cornerctrcoords'] = np.array([0.0,0.0])
	if math.isnan(features['cornerstd']):
		features['cornerstd'] = 0.0
	if sum(np.isnan(features['cornerstdcoords'])) > 0.0:
		features['cornerstdcoords'] = np.array([0.0,0.0])
	return features
	
def getCorners(image, orientation, features = features):
	features = features.copy()
	max_corners = 25
	image2 = image.copy()
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	gray = np.float32(gray)
	rows,cols = gray.shape
	M = cv2.getRotationMatrix2D(tuple(features['centroid']),(orientation+math.pi/2.0)*180.0/math.pi,1)
	grayrot = cv2.warpAffine(gray,M,(cols,rows))
	dst = cv2.cornerHarris(gray,2,3,0.04)
	image2[dst>0.01*dst.max()]=[0,0,255]
	#cv2.imshow('image',image),cv2.imshow('image2',image2)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	corners = cv2.goodFeaturesToTrack(gray,max_corners,0.01,10)
	if corners is None:
		corner_ratio = 0.0
	else:
		corners = np.int0(corners)
		num_corners = corners.size/2
		corner_ratio = float(num_corners)/max_corners
	#print corner_ratio
	return corner_ratio
# Rescale the images and create the combined metrics and training labels

imagesperfile = 4
#get the total training images
numberofImages = 0
for folder in directory_names:
	for fileNameDir in os.walk(folder):   
		for fileName in fileNameDir[2]:
			# Only read in the images
			if fileName[-4:] != ".jpg":
				continue
			numberofImages += imagesperfile

# We'll rescale the images to be 25x25
maxPixel = 25
imageSize = maxPixel * maxPixel
num_rows = numberofImages # one row for each image in the training dataset
#Zak's note: I think I need to add to num_features to add new metrics
corr2dsize = 0 #2401
corrsize = 0 # 49
extrafeatures = 0
lbpsize = maxPixel+1
PCAsize = 25
for k,v in features.items():
    try:
		#print imageSize+extrafeatures,k,len(v),v
		extrafeatures = extrafeatures + len(v)
    except TypeError, te:
		#print
		extrafeatures = extrafeatures + 1
num_features = imageSize + corr2dsize + 3*corrsize + extrafeatures + lbpsize + PCAsize+1  # for our ratio


# X is the feature vector with one row of features per image
# consisting of the pixel values and our metric
X = np.zeros((num_rows, num_features), dtype=float)
# y is the numeric class label 
y = np.zeros((num_rows))

# report progress for each 5% done  
report = [int((j+1)*num_rows/20.) for j in range(20)]


# Loop through the classes two at a time and compare their distributions of the Width/Length Ratio
def comparison(featurechoice, features = features):
	#Create a DataFrame object to make subsetting the data on the class 
	df = pd.DataFrame({"class": y[:], "feature": X[:, featurechoice]})

	f = plt.figure(figsize=(30, 20))
	#we suppress zeros and choose a few large classes to better highlight the distributions.
	df = df.loc[df["feature"] > 0]
	counts = df["class"].value_counts()
	# Loop through 40 of the classes 
	for j in range(0,20):
		class1 = randint(0,121)
		class2 = randint(0,121)
		while class2==class1:
			class2 = randint(0,121)
		subfig = plt.subplot(4, 5, j+1)
		# Plot the normalized histograms for two classes
		n, bins,p = plt.hist(df.loc[df["class"] == class1]["feature"].values,\
							alpha=0.5, bins=[x*df["feature"].max()*0.01 for x in range(100)], \
							label=namesClasses[class1], normed=1)

		n2, bins,p = plt.hist(df.loc[df["class"] == (class2)]["feature"].values,\
								alpha=0.5, bins=bins, label=namesClasses[class2],normed=1)
		subfig.set_ylim([0.,max(n.max(),n2.max())])
		plt.legend(loc='upper right')
		plt.xlabel("Feature")
	plt.show()

# #Create a DataFrame object to make subsetting the data on the class 
# df = pd.DataFrame({"class": y[:], "ratio": X[:, num_features-1]})

# f = plt.figure(figsize=(30, 20))
# #we suppress zeros and choose a few large classes to better highlight the distributions.
# df = df.loc[df["ratio"] > 0]
# minimumSize = 20 
# counts = df["class"].value_counts()
# largeclasses = [int(x) for x in list(counts.loc[counts > minimumSize].index)]
# # Loop through 40 of the classes 
# for j in range(0,40,2):
	# subfig = plt.subplot(4, 5, j/2 +1)
	# # Plot the normalized histograms for two classes
	# classind1 = largeclasses[j]
	# classind2 = largeclasses[j+1]
	# n, bins,p = plt.hist(df.loc[df["class"] == classind1]["ratio"].values,\
						# alpha=0.5, bins=[x*0.01 for x in range(100)], \
						# label=namesClasses[classind1], normed=1)

	# n2, bins,p = plt.hist(df.loc[df["class"] == (classind2)]["ratio"].values,\
							# alpha=0.5, bins=bins, label=namesClasses[classind2],normed=1)
	# subfig.set_ylim([0.,10.])
	# plt.legend(loc='upper right')
	# plt.xlabel("Fill Ratio")


def multiclass_log_loss(y_true, y_pred, eps=1e-15):
	"""Multi class version of Logarithmic Loss metric.
	https://www.kaggle.com/wiki/MultiClassLogLoss

	Parameters
	----------
	y_true : array, shape = [n_samples]
			true class, integers in [0, n_classes - 1)
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

def prep():
	header = "acantharia_protist_big_center,acantharia_protist_halo,acantharia_protist,amphipods,appendicularian_fritillaridae,appendicularian_s_shape,appendicularian_slight_curve,appendicularian_straight,artifacts_edge,artifacts,chaetognath_non_sagitta,chaetognath_other,chaetognath_sagitta,chordate_type1,copepod_calanoid_eggs,copepod_calanoid_eucalanus,copepod_calanoid_flatheads,copepod_calanoid_frillyAntennae,copepod_calanoid_large_side_antennatucked,copepod_calanoid_large,copepod_calanoid_octomoms,copepod_calanoid_small_longantennae,copepod_calanoid,copepod_cyclopoid_copilia,copepod_cyclopoid_oithona_eggs,copepod_cyclopoid_oithona,copepod_other,crustacean_other,ctenophore_cestid,ctenophore_cydippid_no_tentacles,ctenophore_cydippid_tentacles,ctenophore_lobate,decapods,detritus_blob,detritus_filamentous,detritus_other,diatom_chain_string,diatom_chain_tube,echinoderm_larva_pluteus_brittlestar,echinoderm_larva_pluteus_early,echinoderm_larva_pluteus_typeC,echinoderm_larva_pluteus_urchin,echinoderm_larva_seastar_bipinnaria,echinoderm_larva_seastar_brachiolaria,echinoderm_seacucumber_auricularia_larva,echinopluteus,ephyra,euphausiids_young,euphausiids,fecal_pellet,fish_larvae_deep_body,fish_larvae_leptocephali,fish_larvae_medium_body,fish_larvae_myctophids,fish_larvae_thin_body,fish_larvae_very_thin_body,heteropod,hydromedusae_aglaura,hydromedusae_bell_and_tentacles,hydromedusae_h15,hydromedusae_haliscera_small_sideview,hydromedusae_haliscera,hydromedusae_liriope,hydromedusae_narco_dark,hydromedusae_narco_young,hydromedusae_narcomedusae,hydromedusae_other,hydromedusae_partial_dark,hydromedusae_shapeA_sideview_small,hydromedusae_shapeA,hydromedusae_shapeB,hydromedusae_sideview_big,hydromedusae_solmaris,hydromedusae_solmundella,hydromedusae_typeD_bell_and_tentacles,hydromedusae_typeD,hydromedusae_typeE,hydromedusae_typeF,invertebrate_larvae_other_A,invertebrate_larvae_other_B,jellies_tentacles,polychaete,protist_dark_center,protist_fuzzy_olive,protist_noctiluca,protist_other,protist_star,pteropod_butterfly,pteropod_theco_dev_seq,pteropod_triangle,radiolarian_chain,radiolarian_colony,shrimp_caridean,shrimp_sergestidae,shrimp_zoea,shrimp-like_other,siphonophore_calycophoran_abylidae,siphonophore_calycophoran_rocketship_adult,siphonophore_calycophoran_rocketship_young,siphonophore_calycophoran_sphaeronectes_stem,siphonophore_calycophoran_sphaeronectes_young,siphonophore_calycophoran_sphaeronectes,siphonophore_other_parts,siphonophore_partial,siphonophore_physonect_young,siphonophore_physonect,stomatopod,tornaria_acorn_worm_larvae,trichodesmium_bowtie,trichodesmium_multiple,trichodesmium_puff,trichodesmium_tuft,trochophore_larvae,tunicate_doliolid_nurse,tunicate_doliolid,tunicate_partial,tunicate_salp_chains,tunicate_salp,unknown_blobs_and_smudges,unknown_sticks,unknown_unclassified".split(',')
	with open('namesClasses.dat','rb') as f:
		namesClasses = cPickle.load(f)
	labels = map(lambda s: s.split('\\')[-1], namesClasses)
	for i in range(len(namesClasses)):
		currentClass = namesClasses[i]
		root_class.data_search(currentClass).id_list.append(i)
	#get the total test images
	#Full set
	fnames = glob.glob(os.path.join("competition_data", "test", "*.jpg"))
	#Smaller set
	#fnames = glob.glob(os.path.join("smaller_comp", "test", "*.jpg"))
	
	numberofTestImages = len(fnames)

	X = np.zeros((numberofTestImages, num_features), dtype=float)

	#Get filename separate from prefix path
	images = map(lambda fileName: fileName.split('\\')[-1], fnames)

	i = 0
	# report progress for each 1% done  
	report = [int((j+1)*numberofTestImages/100.) for j in range(100)]
	for fileName in fnames:
		# Read in the images and create the features
		image = imread(fileName, as_grey=True)
		image = rotImage(image)
		# Added from https://github.com/Newmu/Stroke-Prediction/blob/master/startPredictingGenCode.py
		thresh = 0.9*255
		# if image.min < 0.75*255:
			# img = image < thresh
		# else:
			# img = image
		# if img.sum() != 0:
			# imgX,imgY = np.nonzero(img)
			# imgW = imgX.max()-imgX.min()
			# imgH = imgY.max()-imgY.min()
			# if (imgW>1 and imgH>1):
				# image = image[imgX.min():imgX.max(),imgY.min():imgY.max()]
		# #----------------------------------
		cvimage = cv2.imread(fileName)
		features = getMinorMajorRatio(image)
		#__Begin moved region
		#From http://nbviewer.ipython.org/github/kqdtran/AY250-F13/blob/master/hw4/hw4.ipynb
		
		pca = decomposition.PCA(n_components=25)
		PCAFeatures = pca.fit_transform(image)
		PCAevr = pca.explained_variance_ratio_
		PCAevr_resized = PCAevr.copy()
		PCAevr_resized.resize(PCAsize)
		for evr in range(len(PCAevr)):
			if np.isnan(PCAevr[evr]):
				PCAevr[evr] = 0.0
		#_____________________________________________________________
		corners = getCorners(cvimage,features['orientation'])
		horslice = image[:,maxPixel/2]
		vertslice = image[maxPixel/2]
		#correlation = signal.correlate(image,image)
		#horcorrelation = signal.correlate(horslice,horslice)
		#vertcorrelation = signal.correlate(vertslice,vertslice)
		#crosscorrelation = signal.correlate(horslice,vertslice)
		#correlation = correlation/correlation[correlation.shape[0]/2,correlation.shape[0]/2]
		#horcorrelation = horcorrelation/horcorrelation[horcorrelation.shape[0]/2]
		#vertcorrelation = vertcorrelation/vertcorrelation[vertcorrelation.shape[0]/2]
		#crosscorrelation = crosscorrelation/crosscorrelation[horcorrelation.shape[0]/2]
		hormean = np.mean(horslice)
		horstd = np.std(horslice)
		vertmean = np.mean(vertslice)
		vertstd = np.std(vertslice)
		horcount = vertcount = 0
		for pix in horslice:
			graycheck = False
			if pix<thresh:
				if not graycheck:
					graycheck = True
					horcount = horcount + 1
			else:
				graycheck = False
		for pix in vertslice:
			graycheck = False
			if pix<thresh:
				if not graycheck:
					graycheck = True
					vertcount = vertcount + 1
			else:
				graycheck = False
		features['hsobel'] = np.nanmean(filter.hsobel(image))
		features['vsobel'] = np.nanmean(filter.vsobel(image))
		features['peaklocalmax'] = np.nanmean(peak_local_max(image))
		features['felzen'] = np.nanmean(segmentation.felzenszwalb(image))
		if np.isnan(features['peaklocalmax']):
			features['peaklocalmax'] = 0.0
		if np.isnan(features['felzen']):
			features['felzen'] = 0.0
		#hormirror = image[:maxPixel/2]-image[maxPixel-1:maxPixel/2:-1]
		#vertmirror = image[:,:maxPixel/2]-image[:,maxPixel-1:maxPixel/2:-1]		
		image = resize(image, (maxPixel, maxPixel))
					
		#From http://scikit-image.org/docs/dev/auto_examples/plot_local_binary_pattern.html
		lbp = local_binary_pattern(image, n_points, radius, METHOD)
		n_bins = lbp.max()+1
		lbpcounts = np.histogram(lbp,n_bins,normed=True,range=(0, n_bins))[0]
		#_____________________________________________________________
		#__Moved region was here
		#dist_trans = ndimage.distance_transform_edt(image[0])
		# Store the rescaled image pixels and the axis ratio
		#fd, hog_image = hog(image[0], orientations=8, pixels_per_cell=(2, 2),
		#    cells_per_block=(1, 1), visualise=True)
		#X[i, 0:imageSize] = np.reshape(dist_trans, (1, imageSize))
		#X[i, 0:imageSize] = np.reshape(hog_image, (1, imageSize))
					
		# Store the rescaled image pixels and the axis ratio
		X[i, 0:imageSize] = np.reshape(image, (1, imageSize))
		#X[i, imageSize:imageSize+corr2dsize] = np.reshape(correlation, (1,corr2dsize))
		#X[i, imageSize+corr2dsize:imageSize+corr2dsize+corrsize] = np.reshape(horcorrelation, (1,corrsize))
		#X[i, imageSize+corr2dsize+corrsize:imageSize+corr2dsize+2*corrsize] = np.reshape(vertcorrelation, (1,corrsize))
		#X[i, imageSize+corr2dsize+2*corrsize:imageSize+corr2dsize+3*corrsize] = np.reshape(crosscorrelation, (1,corrsize))
		featcount = imageSize+3*corrsize+corr2dsize
		for k,v in features.items():
			try:
				X[i, featcount:featcount+len(v)] = v
				featcount = featcount + len(v)
			except TypeError, te:
				X[i, featcount] = v
				featcount = featcount + 1
		X[i, featcount:featcount+lbpcounts.size] = lbpcounts
		X[i, featcount+lbpcounts.size:featcount+lbpcounts.size+PCAsize] = PCAevr_resized
		X[i, featcount+lbpcounts.size+PCAsize] = np.mean(PCAFeatures)
		i += 1
		if i in report: print np.ceil(i *100.0 / numberofTestImages), "% done"
	
	max_depth = root_class.max_depth()
	root_class.update_ids()
	root_class.trim()
	layers = [[] for i in range(max_depth)]
	layers = root_class.find_layers(layers,0,max_depth)
	y_all = [np.zeros(len(namesClasses)) for i in range(max_depth)]
	y_all[max_depth-1] = range(len(namesClasses))
	for i in range(len(namesClasses)):
		for layer in range(max_depth-1):
			for j in range(len(layers[layer])):
				if y[i] in layers[layer][j].id_list:
					y_all[layer][i] = j
					
	y_pred = [0]*max_depth
	clf = [0]*max_depth		
	for i in range(max_depth):
		clffile = 'clf' + str(i) + '.dat'
		with open(clffile,'rb') as f:
			clf[i] = cPickle.load(f)		
		y_pred[i] = clf[i].predict_proba(X)

	y_final = []
	y_final = y_pred[max_depth-1]
	for i in range(max_depth-1):
		for j in range(y_pred[max_depth-1].shape[0]):
			for k in range(len(namesClasses)):
				y_final[j][k] = y_final[j][k] * y_pred[i][j][y_all[i][j]]
				
	df = pd.DataFrame(y_final, columns=labels, index=images)
	
	df.index.name = 'image'
	
	df = df[header]
	
	df.to_csv('competition_data/submission.csv')
	
	#!gzip competition_data/submission.csv
	
	#!ls -l competition_data/submission.csv.gz	
#@profile		
def main():
	files = []
	# Generate training data
	i = 0    
	label = 0
	# List of string of class names
	namesClasses = list()
	features = {}
	features = features.copy()
	print "Reading images"
	# Navigate through the list of directories
	for folder in directory_names:
		#Get name of class directory separate from prefix path
		currentClass = folder.split(os.sep)[-1] 
		namesClasses.append(currentClass)
		root_class.data_search(currentClass).id_list.append(len(namesClasses)-1)
		for fileNameDir in os.walk(folder):   
			for fileName in fileNameDir[2]:
				# Only read in the images
				if fileName[-4:] != ".jpg":
					continue
				
				# Read in the images and create the features
				nameFileImage = "{0}{1}{2}".format(fileNameDir[0], os.sep, fileName)
				image = []
				image.append(imread(nameFileImage, as_grey=True))
				features['original_size'] = image[0].size
				#image[0] = equalize_hist(image[0])
				image[0] = rotImage(image[0])
				# Added from https://github.com/Newmu/Stroke-Prediction/blob/master/startPredictingGenCode.py
				thresh = 0.9*255
				# if image.min < 0.75*255:
					# img = image < thresh
				# else:
					# img = image
				# if img.sum() != 0:
					# imgX,imgY = np.nonzero(img)
					# imgW = imgX.max()-imgX.min()
					# imgH = imgY.max()-imgY.min()
					# if (imgW>1 and imgH>1):
						# image = image[imgX.min():imgX.max(),imgY.min():imgY.max()]
				#----------------------------------
				cvimage = cv2.imread(nameFileImage)
				#image[0] = gaussian_filter(image[0],sigma=2)
				files.append(nameFileImage)
				image.append(np.fliplr(image[0]))
				image.append(np.flipud(image[0]))
				image.append(np.fliplr(image[2]))
				# image.append(np.rot90(image[0]))
				# image.append(np.fliplr(image[4]))
				# image.append(np.flipud(image[4]))
				# image.append(np.fliplr(image[6]))
				for j in range(len(image)):
					features = getMinorMajorRatio(image[j])
					#__Begin moved region
					#From http://nbviewer.ipython.org/github/kqdtran/AY250-F13/blob/master/hw4/hw4.ipynb
					pca = decomposition.PCA(n_components=25)
					PCAFeatures = pca.fit_transform(image[0])
					#_____________________________________________________________
					corners = getCorners(cvimage,features['orientation'])
					horslice = image[j][:,maxPixel/2]
					vertslice = image[j][maxPixel/2]
					#correlation = signal.correlate(image,image)
					#horcorrelation = signal.correlate(horslice,horslice)
					#vertcorrelation = signal.correlate(vertslice,vertslice)
					#crosscorrelation = signal.correlate(horslice,vertslice)
					#correlation = correlation/correlation[correlation.shape[0]/2,correlation.shape[0]/2]
					#horcorrelation = horcorrelation/horcorrelation[horcorrelation.shape[0]/2]
					#vertcorrelation = vertcorrelation/vertcorrelation[vertcorrelation.shape[0]/2]
					#crosscorrelation = crosscorrelation/crosscorrelation[horcorrelation.shape[0]/2]
					hormean = np.mean(horslice)
					horstd = np.std(horslice)
					vertmean = np.mean(vertslice)
					vertstd = np.std(vertslice)
					horcount = vertcount = 0
					for pix in horslice:
						graycheck = False
						if pix<thresh:
							if not graycheck:
								graycheck = True
								horcount = horcount + 1
						else:
							graycheck = False
					for pix in vertslice:
						graycheck = False
						if pix<thresh:
							if not graycheck:
								graycheck = True
								vertcount = vertcount + 1
						else:
							graycheck = False
					features['hsobel'] = np.nanmean(filter.hsobel(image[j]))
					features['vsobel'] = np.nanmean(filter.vsobel(image[j]))
					features['peaklocalmax'] = np.nanmean(peak_local_max(image[j]))
					features['felzen'] = np.nanmean(segmentation.felzenszwalb(image[j]))
					if np.isnan(features['peaklocalmax']):
						features['peaklocalmax'] = 0.0
					if np.isnan(features['felzen']):
						features['felzen'] = 0.0
					#hormirror = image[j][:maxPixel/2]-image[j][maxPixel-1:maxPixel/2:-1]
					#vertmirror = image[j][:,:maxPixel/2]-image[j][:,maxPixel-1:maxPixel/2:-1]
					#__End moved region
					image[j] = resize(image[j], (maxPixel, maxPixel))
					
					#From http://scikit-image.org/docs/dev/auto_examples/plot_local_binary_pattern.html
					lbp = local_binary_pattern(image[j], n_points, radius, METHOD)
					n_bins = lbp.max()+1
					lbpcounts = np.histogram(lbp,n_bins,normed=True,range=(0, n_bins))[0]
					#_____________________________________________________________
					#__Moved region was here
					#dist_trans = ndimage.distance_transform_edt(image[0])
					# Store the rescaled image pixels and the axis ratio
					#fd, hog_image = hog(image[0], orientations=8, pixels_per_cell=(2, 2),
					#    cells_per_block=(1, 1), visualise=True)
					#X[i*imagesperfile+j, 0:imageSize] = np.reshape(dist_trans, (1, imageSize))
					#X[i*imagesperfile+j, 0:imageSize] = np.reshape(hog_image, (1, imageSize))
					X[i*imagesperfile+j, 0:imageSize] = np.reshape(image[j], (1, imageSize))
					#X[i*imagesperfile+j, imageSize:imageSize+corr2dsize] = np.reshape(correlation, (1,corr2dsize))
					#X[i*imagesperfile+j, imageSize+corr2dsize:imageSize+corr2dsize+corrsize] = np.reshape(horcorrelation, (1,corrsize))
					#X[i*imagesperfile+j, imageSize+corr2dsize+corrsize:imageSize+corr2dsize+2*corrsize] = np.reshape(vertcorrelation, (1,corrsize))
					#X[i*imagesperfile+j, imageSize+corr2dsize+2*corrsize:imageSize+corr2dsize+3*corrsize] = np.reshape(crosscorrelation, (1,corrsize))
					featcount = imageSize+3*corrsize+corr2dsize
					for k,v in features.items():
						try:
							X[i*imagesperfile+j, featcount:featcount+len(v)] = v
							featcount = featcount + len(v)
						except TypeError, te:
							X[i*imagesperfile+j, featcount] = v
							featcount = featcount + 1
					X[i*imagesperfile+j, featcount:featcount+lbpcounts.size] = lbpcounts
					X[i*imagesperfile+j, featcount+lbpcounts.size:featcount+lbpcounts.size+PCAsize] = pca.explained_variance_ratio_
					X[i*imagesperfile+j, featcount+lbpcounts.size+PCAsize] = np.mean(PCAFeatures)
					# Store the classlabel
					y[i*imagesperfile+j] = label
					if i*imagesperfile+j in report: print np.ceil((i*imagesperfile+j) *100.0 / (num_rows)), "% done"
				i += 1
		label += 1

	max_depth = root_class.max_depth()
	root_class.update_ids()
	root_class.trim()
	layers = [[] for i in range(max_depth)]
	layers = root_class.find_layers(layers,0,max_depth)
	y_all = [np.zeros(len(y)) for i in range(max_depth)]
	y_all[max_depth-1] = y
	for i in range(len(y)):
		for layer in range(max_depth-1):
			for j in range(len(layers[layer])):
				if y[i] in layers[layer][j].id_list:
					y_all[layer][i] = j
	end_list = []
	end_list = root_class.list_ends()

	print "Training"
	# n_estimators is the number of decision trees
	# max_features also known as m_try is set to the default value of the square root of the number of features
	clf = RF(n_estimators=n_estim, n_jobs=n_jobs,compute_importances=True);
	scores = cross_validation.cross_val_score(clf, X, y, cv=n_folds, n_jobs=n_jobs);
	print "Accuracy of all classes"
	print np.mean(scores)


	kf = KFold(y, n_folds, shuffle=True)
	y_pred = []
	for depth in range(max_depth):
		y_pred.append(y_all[depth] * 0)
	for train, test in kf:
		#Just main categories here, not sure I need to loop over categorization levels
		#for i in range(max_depth):
		i = max_depth-1
		X_train, X_test, y_train, y_test = X[train,:], X[test,:], y_all[i][train], y_all[i][test] 
		clf = RF(n_estimators=n_estim, n_jobs=n_jobs,compute_importances=True)
		clf.fit(X_train, y_train)
		y_pred[i][test] = clf.predict(X_test)

	print classification_report(y, y_pred[max_depth-1], target_names=namesClasses)
	# Get the probability predictions for computing the log-loss function
	kf = KFold(y, n_folds, shuffle=True)
	# prediction probabilities number of samples, by number of classes
	y_pred = []
	clf = [0]*max_depth
	for depth in range(max_depth):
		y_pred.append(np.zeros((len(y_all[depth]),len(set(y_all[depth])))))
	for train, test in kf:
		for i in range(max_depth):
			X_train, X_test, y_train, y_test = X[train,:], X[test,:], y_all[i][train], y_all[i][test]
			clf[i] = RF(n_estimators=n_estim, n_jobs=n_jobs,compute_importances=True)
			clf[i].fit(X_train, y_train)
			y_pred[i][test] = clf[i].predict_proba(X_test)
	# for i in range(len(y)):
		# for layer in range(max_depth-1):
			# for j in range(len(layers[layer])):
				# if y[i] in layers[layer][j].id_list:
					# y_all[layer][i] = j
			
	y_final = []
	y_final = y_pred[max_depth-1]
	for i in range(max_depth-1):
		for j in range(y.size):
			for k in range(len(set(y))):
				y_final[j][k] = y_final[j][k] * y_pred[i][j][y_all[i][j]]

	print multiclass_log_loss(y, y_final)
	for i in range(max_depth):
		clffile = 'clf' + str(i) + '.dat'
		with open(clffile,'wb') as f:
			cPickle.dump(clf[i],f)
	with open('namesClasses.dat','wb') as f:
		cPickle.dump(namesClasses,f)
	#
	# print
	# print "y_pred " + "y"
	# for guess,answer in zip(y_pred,y):
		# if guess.argmax() == int(answer):
			# print namesClasses[guess.argmax()],
				# guess.max(),namesClasses[int(answer)]
#main()
#prep()