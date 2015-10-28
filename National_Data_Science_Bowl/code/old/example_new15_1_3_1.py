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
from scipy import ndimage,signal
from skimage.feature import peak_local_max
import cv2
import math
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.transform import warp, AffineTransform,rotate
# make graphics inline
#%matplotlib inline
n_estim = 100

import warnings
warnings.filterwarnings("ignore")

importances = np.array([  7.32101347e-05,   9.80462978e-05,   1.71850420e-05,
         2.88009030e-05,   1.29364557e-05,   4.20253879e-05,
         4.59842740e-05,   1.14609212e-05,   2.06899500e-05,
         7.57051258e-05,   1.59308317e-04,   4.80436550e-04,
         1.73317745e-03,   6.71108517e-04,   1.62695054e-04,
         7.96962969e-05,   7.73359571e-05,   1.27345973e-05,
         1.24172525e-05,   1.63766117e-05,   1.36142393e-05,
         9.26991700e-06,   7.45792191e-06,   1.02203721e-05,
         7.62053606e-06,   2.97991338e-05,   1.96733500e-05,
         1.46210498e-05,   1.60907694e-05,   2.55943615e-05,
         3.05044449e-05,   4.16555098e-05,   4.42391579e-05,
         3.88187483e-05,   9.75539421e-05,   2.14713661e-04,
         6.38687989e-04,   2.29813221e-03,   8.60881835e-04,
         2.40350510e-04,   1.33365531e-04,   1.11874719e-04,
         1.23038452e-05,   1.29980397e-05,   2.41400671e-05,
         1.09024370e-05,   1.29211816e-05,   9.64091362e-06,
         6.94680206e-06,   9.16301728e-06,   6.23262219e-05,
         5.45294404e-05,   8.71835786e-06,   2.27159905e-05,
         4.12880840e-05,   1.85540546e-05,   2.31022152e-05,
         1.56277226e-05,   5.34176297e-05,   1.19457102e-04,
         3.04389186e-04,   8.31696936e-04,   2.29221205e-03,
         1.01854179e-03,   3.54542648e-04,   1.51808847e-04,
         6.94993937e-05,   1.96491018e-05,   4.24577057e-05,
         1.86662550e-05,   8.42885416e-06,   1.23504818e-05,
         9.07470919e-06,   6.05108148e-06,   1.34801444e-05,
         1.95517495e-05,   5.92190628e-05,   1.37161629e-05,
         1.17103316e-05,   4.55293630e-05,   7.37401304e-05,
         7.84037518e-05,   1.97287360e-05,   6.93432640e-05,
         1.94112656e-04,   4.76214673e-04,   1.13069094e-03,
         2.87087419e-03,   1.34634310e-03,   5.13916316e-04,
         2.58017665e-04,   1.23764076e-04,   6.19877882e-05,
         2.05659806e-05,   9.02069305e-06,   8.98877640e-06,
         1.90453586e-05,   1.45527296e-05,   1.47814585e-05,
         8.33446918e-06,   7.76836097e-05,   1.23003779e-05,
         4.97670262e-05,   1.51509746e-05,   2.80707002e-05,
         3.72288966e-05,   2.69004393e-05,   6.35755363e-05,
         1.46022411e-04,   2.96487014e-04,   7.58951907e-04,
         1.78810438e-03,   3.87186587e-03,   1.96224325e-03,
         9.44046830e-04,   4.24639011e-04,   2.25315126e-04,
         6.81257049e-05,   5.86470021e-05,   2.00764724e-05,
         1.07303664e-05,   1.00909803e-05,   7.40858667e-06,
         6.59160951e-06,   7.14060326e-06,   7.66982793e-06,
         4.00870948e-05,   2.34296899e-05,   2.49952549e-05,
         2.13911138e-05,   3.35667969e-05,   6.46110248e-05,
         1.27636909e-04,   3.22265266e-04,   6.19348546e-04,
         1.37841044e-03,   2.69772001e-03,   5.01539395e-03,
         2.91508956e-03,   1.73234981e-03,   8.49586892e-04,
         4.43398535e-04,   2.39335338e-04,   6.98714201e-05,
         3.55459073e-05,   2.98496995e-05,   1.60175080e-05,
         1.09474653e-05,   1.95718669e-05,   2.10929425e-05,
         1.46521512e-05,   1.59138744e-05,   2.39300561e-05,
         3.27761740e-05,   5.47786537e-05,   8.40284470e-05,
         1.98433808e-04,   3.84597739e-04,   6.22087493e-04,
         1.24401450e-03,   2.40543970e-03,   3.85389060e-03,
         6.24652111e-03,   4.21506268e-03,   2.89928412e-03,
         1.64692876e-03,   8.96968886e-04,   4.02982688e-04,
         2.21537811e-04,   8.62908225e-05,   1.38873341e-04,
         2.62374428e-05,   1.34849702e-05,   1.23715591e-05,
         1.49222123e-05,   5.66813248e-05,   1.30034562e-05,
         2.58998502e-05,   7.24512840e-05,   1.29746411e-04,
         2.51929407e-04,   4.62722698e-04,   6.93852048e-04,
         1.21110516e-03,   1.96471433e-03,   3.37952981e-03,
         5.00161147e-03,   7.42836552e-03,   5.44584894e-03,
         4.15769849e-03,   2.75372692e-03,   1.48576561e-03,
         7.81722620e-04,   3.69487467e-04,   2.24180375e-04,
         1.25096796e-04,   4.06767129e-05,   1.57846265e-05,
         1.63477326e-05,   9.40578376e-06,   1.12670801e-05,
         3.83510218e-05,   4.27606102e-05,   1.21263530e-04,
         2.27693194e-04,   4.00016224e-04,   5.96489418e-04,
         1.04055015e-03,   1.61678945e-03,   2.89336062e-03,
         4.53813563e-03,   6.00282914e-03,   8.47813653e-03,
         6.69821640e-03,   4.99020746e-03,   3.46290482e-03,
         2.20389318e-03,   1.19902945e-03,   6.18102141e-04,
         3.23366589e-04,   1.72163598e-04,   7.90927260e-05,
         2.33956157e-05,   1.62290369e-05,   6.54619458e-06,
         1.86035612e-05,   2.26336390e-05,   8.72357538e-05,
         1.14566966e-04,   2.66579660e-04,   4.87118045e-04,
         7.95804076e-04,   1.30124690e-03,   2.25867876e-03,
         3.65537115e-03,   4.94174052e-03,   6.66719056e-03,
         9.12200859e-03,   8.28518965e-03,   5.40746135e-03,
         4.18662437e-03,   2.81386035e-03,   1.64583373e-03,
         7.96340518e-04,   4.60217041e-04,   2.20556657e-04,
         9.36096173e-05,   3.18002467e-05,   1.47920657e-05,
         1.58535631e-05,   1.48731206e-05,   1.57032027e-05,
         3.04092834e-05,   1.05089029e-04,   2.95159569e-04,
         5.99201703e-04,   9.19030434e-04,   1.52255592e-03,
         2.65475274e-03,   4.01592470e-03,   5.16155396e-03,
         7.85600465e-03,   1.05883067e-02,   9.11011955e-03,
         6.78365203e-03,   4.27075230e-03,   3.10148201e-03,
         1.92460735e-03,   1.02824873e-03,   5.31909640e-04,
         2.28832179e-04,   8.83635926e-05,   3.41527845e-05,
         1.07425551e-05,   1.52270429e-05,   4.58176940e-05,
         1.69803732e-05,   5.59637192e-05,   1.13945681e-04,
         3.29132461e-04,   6.14347482e-04,   1.05101336e-03,
         1.72255158e-03,   2.78795263e-03,   4.01467587e-03,
         6.31897801e-03,   9.95674545e-03,   1.24321936e-02,
         1.17994344e-02,   7.94306258e-03,   5.02339429e-03,
         3.45988433e-03,   2.13160055e-03,   1.17478541e-03,
         6.56565012e-04,   3.16687358e-04,   1.11516581e-04,
         3.48890840e-05,   1.63580139e-05,   2.59325669e-05,
         1.99826364e-05,   1.91723405e-05,   3.03890856e-05,
         7.96812829e-05,   2.71031316e-04,   6.50329976e-04,
         1.06420051e-03,   1.76784755e-03,   2.94462662e-03,
         4.04257884e-03,   7.36016206e-03,   1.07022992e-02,
         1.54574885e-02,   1.39185143e-02,   8.49301761e-03,
         5.55911698e-03,   3.47399158e-03,   2.25147059e-03,
         1.32978820e-03,   7.82789255e-04,   3.77842867e-04,
         1.42943410e-04,   2.92976429e-05,   1.52379032e-05,
         7.78361681e-06,   1.86191842e-05,   4.24276187e-05,
         5.11830558e-05,   7.49402000e-05,   2.49649517e-04,
         5.94021858e-04,   1.03935138e-03,   1.71728043e-03,
         2.80148712e-03,   4.08612150e-03,   6.48393352e-03,
         1.06228433e-02,   1.45502493e-02,   1.05510636e-02,
         7.85295875e-03,   5.05088521e-03,   3.37994761e-03,
         2.19889358e-03,   1.21959135e-03,   7.32311626e-04,
         4.51989426e-04,   1.71421624e-04,   4.89657728e-05,
         1.79866606e-05,   1.13052842e-05,   1.30733939e-05,
         1.23565514e-05,   2.74377983e-05,   7.79381764e-05,
         2.55326266e-04,   4.89074163e-04,   8.93153247e-04,
         1.49839420e-03,   2.46974967e-03,   3.89764720e-03,
         5.73526922e-03,   8.47317159e-03,   1.14423003e-02,
         1.02446863e-02,   6.91510231e-03,   4.62330276e-03,
         3.18817276e-03,   1.93961478e-03,   1.16989141e-03,
         6.53123242e-04,   3.84695236e-04,   1.69967941e-04,
         4.90517250e-05,   1.91104090e-05,   1.50555948e-05,
         1.97148815e-05,   1.61005713e-05,   4.51275309e-05,
         8.03734822e-05,   2.29583612e-04,   4.22660591e-04,
         7.55694271e-04,   1.25494673e-03,   2.14066763e-03,
         3.64584983e-03,   4.97246891e-03,   7.07624047e-03,
         9.36424092e-03,   8.29454675e-03,   5.96945405e-03,
         4.24975378e-03,   2.80028236e-03,   1.69743325e-03,
         1.02518005e-03,   6.06685006e-04,   3.54036234e-04,
         1.58804456e-04,   4.99076420e-05,   8.19475902e-06,
         8.68531592e-06,   1.90635351e-05,   2.18937566e-05,
         2.23877762e-05,   8.56805326e-05,   1.40603681e-04,
         3.26126853e-04,   5.86217322e-04,   1.01111482e-03,
         1.65821913e-03,   2.83650703e-03,   4.75657208e-03,
         6.01931159e-03,   8.14677018e-03,   6.78000317e-03,
         5.21931849e-03,   3.62976222e-03,   2.32736434e-03,
         1.25861909e-03,   7.29611739e-04,   4.58916773e-04,
         2.40752911e-04,   1.27561304e-04,   5.40134675e-05,
         1.70659497e-05,   1.05578153e-05,   1.59884611e-05,
         1.40091114e-05,   5.27094113e-05,   5.40233613e-05,
         1.11221556e-04,   2.16676078e-04,   3.92356440e-04,
         7.53016619e-04,   1.22320102e-03,   2.15434453e-03,
         3.55254925e-03,   5.35371242e-03,   6.97905215e-03,
         5.74782697e-03,   4.21328927e-03,   2.77827588e-03,
         1.64070909e-03,   8.79336085e-04,   5.23315801e-04,
         3.33984479e-04,   1.76649096e-04,   6.66617628e-05,
         2.82347057e-05,   1.53273563e-05,   9.19336988e-06,
         1.09371483e-05,   1.92987631e-05,   1.18267806e-05,
         2.36831304e-05,   4.31794927e-05,   1.05360129e-04,
         2.42065966e-04,   5.21858036e-04,   8.20248291e-04,
         1.53460638e-03,   2.51235077e-03,   3.93965217e-03,
         5.62801702e-03,   4.29519036e-03,   3.03632146e-03,
         1.68003078e-03,   9.10138049e-04,   5.20095216e-04,
         2.87958776e-04,   1.55206638e-04,   6.83387195e-05,
         3.38129545e-05,   2.11562498e-05,   4.42948344e-05,
         7.64815663e-06,   3.06268732e-05,   1.10230244e-05,
         3.73671574e-05,   4.66350586e-05,   3.44085877e-05,
         6.56589147e-05,   9.55555480e-05,   2.29379165e-04,
         5.07052918e-04,   9.14979150e-04,   1.52727542e-03,
         2.73061938e-03,   4.14616437e-03,   3.02379514e-03,
         1.71176178e-03,   8.59020331e-04,   4.44631157e-04,
         2.14828659e-04,   1.19160007e-04,   5.01189334e-05,
         2.55334008e-05,   1.52253440e-05,   9.74107864e-06,
         1.74135020e-05,   9.52146864e-06,   1.84737052e-05,
         9.99735819e-06,   1.68174729e-05,   7.21031530e-06,
         1.15260715e-05,   1.94648653e-05,   3.82493117e-05,
         1.08915046e-04,   2.25210032e-04,   4.65761508e-04,
         8.90676945e-04,   1.83035860e-03,   3.52268652e-03,
         2.15316056e-03,   9.95464932e-04,   4.89392373e-04,
         2.00328064e-04,   1.07095184e-04,   3.82745151e-05,
         1.73909140e-05,   8.70389178e-06,   1.29028693e-05,
         1.02130588e-05,   8.79584787e-06,   7.34557949e-06,
         7.16094783e-06,   1.04987047e-05,   1.13771216e-05,
         8.83034511e-06,   8.03750913e-06,   2.03031409e-05,
         7.66871392e-05,   5.90101114e-05,   1.19626044e-04,
         2.36448232e-04,   5.40574644e-04,   1.27278798e-03,
         3.06858986e-03,   1.50769986e-03,   6.78768452e-04,
         2.68626761e-04,   1.29305810e-04,   3.77548805e-05,
         4.56756618e-05,   7.88721825e-06,   2.07501436e-05,
         1.39019560e-05,   3.27507691e-05,   1.14299423e-05,
         7.82891865e-06,   2.33693412e-06,   1.66637986e-05,
         9.03424455e-06,   1.17140853e-05,   3.19183356e-06,
         1.18322963e-05,   1.37549364e-05,   2.40852772e-05,
         5.34684413e-05,   1.81941712e-04,   3.42071909e-04,
         9.43205386e-04,   2.76455774e-03,   1.04628506e-03,
         4.24831436e-04,   1.46485842e-04,   8.99603778e-05,
         2.17971113e-05,   1.23567450e-05,   8.50545414e-06,
         1.13713371e-05,   1.01791770e-05,   9.85235415e-06,
         1.06947196e-05,   8.61589783e-06,   1.12953455e-05,
         1.22093291e-05,   1.03367375e-05,   5.63895259e-06,
         6.45010755e-06,   9.68865494e-06,   1.79425934e-05,
         2.38202102e-05,   3.87733414e-05,   1.20251354e-04,
         2.69641960e-04,   6.86112805e-04,   2.39638781e-03,
         8.87099391e-04,   2.97601434e-04,   1.05470231e-04,
         3.59835810e-05,   1.22705699e-05,   7.09482759e-06,
         7.85116655e-06,   7.63341530e-06,   8.89630317e-06,
         8.99417079e-06,   7.81268092e-06,   1.68698297e-05,
         6.27569315e-06,   3.08999245e-06,   8.92497654e-06,
         8.90963068e-06,   1.02574505e-05,   6.44408914e-06,
         1.15497153e-05,   9.30069905e-06,   5.05236159e-05,
         9.51592899e-05,   1.76910304e-04,   5.67621171e-04,
         2.31726556e-03,   6.75708303e-04,   2.27062971e-04,
         7.20935681e-05,   2.62066762e-05,   9.73555985e-06,
         4.26644178e-06,   1.76393141e-05,   6.65908103e-06,
         9.40829589e-06,   4.70413275e-06,   9.65357135e-06,
         1.07222086e-05,   1.29582195e-02,   1.40933248e-02,
         1.37577187e-02,   1.41184583e-02,   1.61627758e-02,
         8.62097203e-03,   8.66408289e-03,   7.43389917e-03,
         7.64083435e-03,   7.42649641e-03,   5.98347753e-03,
         1.48792814e-02,   4.55421840e-03,   5.58621624e-03,
         5.25539771e-03,   5.23871718e-03,   6.79637200e-03,
         5.86441389e-03,   6.76622639e-03,   1.09624414e-02,
         1.10656940e-02,   6.17960178e-03,   9.42478651e-03,
         4.79382803e-03,   4.99173173e-03])

# get the classnames from the directory structure
directory_names = list(set(glob.glob(os.path.join("competition_data","train", "*"))\
 ).difference(set(glob.glob(os.path.join("competition_data","train","*.*")))))
 
# Example image
# This example was chosen for because it has two 
# noncontinguous pieces that will make the segmentation 
# example more illustrative
example_file = glob.glob(os.path.join(directory_names[79],"*.jpg"))[0]
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


def getMinorMajorRatio(image):
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
	lrdiff = 0.0
	tbdiff = 0.0
	hu1 = hu2 = hu3 = hu12 = hu13 = hu23 = 0.0
	whu1 = whu2 = whu3 = whu12 = whu13 = whu23 = 0.0
	extent = 0.0
	minintensity = maxintensity = meanintensity = 0.0
	intensityratio1 = intensityratio2 = intensityratio3 = 0.0
	if ((not maxregion is None) and  (maxregion.major_axis_length != 0.0)):
		corners = corner_peaks(corner_harris(maxregion.image), min_distance=5)
		corners_subpix = corner_subpix(maxregion.image, corners, window_size=13)
		cornercentercoords = np.nanmean(corners_subpix, axis=0)
		cornerstdcoords = np.nanstd(corners_subpix, axis=0)
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
		whu1 = 0.0 if maxregion is None else maxregion.weighted_moments_hu[1]
		whu2 = 0.0 if maxregion is None else maxregion.weighted_moments_hu[2]
		whu3 = 0.0 if maxregion is None else maxregion.weighted_moments_hu[3]
		whu12 = 0.0 if (maxregion is None or whu1==0.0) else whu2/whu1
		whu13 = 0.0 if (maxregion is None or whu1==0.0) else whu3/whu1
		whu23 = 0.0 if (maxregion is None or whu2==0.0) else whu3/whu2
		extent = 0.0 if maxregion is None else maxregion.extent
		minintensity = 0.0 if maxregion is None else maxregion.min_intensity
		meanintensity = 0.0 if maxregion is None else maxregion.mean_intensity
		maxintensity = 0.0 if maxregion is None else maxregion.max_intensity
		intensityratio1 = 0.0 if (maxregion is None or maxintensity==0.0) else meanintensity/maxintensity
		intensityratio2 = 0.0 if (maxregion is None or maxintensity==0.0) else minintensity/maxintensity
		intensityratio3 = 0.0 if (maxregion is None or meanintensity==0.0) else minintensity/meanintensity
		perimratio = 0.0 if (maxregion is None or maxregion.minor_axis_length==0.0) else maxregion.perimeter/(maxregion.minor_axis_length*4.0+maxregion.major_axis_length*4.0)
		eigenratio = 0.0 if largeeigen == 0.0 else smalleigen/largeeigen
		orientation = 0.0 if maxregion is None else maxregion.orientation
		centroid = (0.0,0.0) if maxregion is None else maxregion.centroid
		cornercentercoords = np.absolute(cornercentercoords - centroid) if maxregion.major_axis_length==0.0 else np.absolute(cornercentercoords - centroid)/maxregion.major_axis_length
		cornercenter = np.linalg.norm(cornercentercoords)
		if maxregion.major_axis_length!=0.0: cornerstdcoords = np.absolute(cornerstdcoords)/maxregion.major_axis_length
		cornerstd = np.linalg.norm(cornerstdcoords)
		left = np.sum(maxregion.image[:,maxregion.image.shape[1]/2:])
		if maxregion.image.shape[1] % 2 == 0:
			right = np.sum(maxregion.image[:,:maxregion.image.shape[1]/2])
		else:
			right = np.sum(maxregion.image[:,:maxregion.image.shape[1]/2+1])
		lrdiff = np.abs((right-left)/(right+left)) 
		top = np.sum(maxregion.image[maxregion.image.shape[0]/2:,:])
		if maxregion.image.shape[0] % 2 == 0:
			bottom = np.sum(maxregion.image[:maxregion.image.shape[0]/2,:])
		else:
			bottom = np.sum(maxregion.image[:maxregion.image.shape[0]/2+1,:])
		tbdiff = np.abs((top-bottom)/(top+bottom)) 
	else:
		cornercentercoords = (0.0,0.0)
		cornerstdcoords = (0.0,0.0)
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
	return minintensity,meanintensity,maxintensity,intensityratio1,intensityratio2,intensityratio3,extent,lrdiff,tbdiff,cornercenter,cornercentercoords,cornerstd,cornerstdcoords,ratio,fillratio,eigenratio,solidity,hu1,hu2,hu3,hu12,hu13,hu23,whu1,whu2,whu3,whu12,whu13,whu23,perimratio,arearatio,orientation,centroid

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
corr2dsize = 2401
corrsize = 49
num_features = imageSize + corr2dsize + 3*corrsize + 44  # for our ratio


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
			image = rotImage(image)
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
			(minintensity,meanintensity,maxintensity,intensityratio1,intensityratio2,intensityratio3,extent,lrdiff,tbdiff,cornercenter,cornercentercoords,cornerstd,cornerstdcoords,axisratio,fillratio,eigenratio,solidity,hu1,hu2,hu3,hu12,hu13,hu23,whu1,whu2,whu3,whu12,whu13,whu23,
			perimratio,arearatio,orientation,centroid) = getMinorMajorRatio(image)
			corners = getCorners(cvimage,orientation)
			image = resize(image, (maxPixel, maxPixel))
			horslice = image[:,maxPixel/2]
			vertslice = image[maxPixel/2]
			correlation = signal.correlate(image,image)
			horcorrelation = signal.correlate(horslice,horslice)
			vertcorrelation = signal.correlate(vertslice,vertslice)
			crosscorrelation = signal.correlate(horslice,vertslice)
			correlation = correlation/correlation[correlation.shape[0]/2,correlation.shape[0]/2]
			horcorrelation = horcorrelation/horcorrelation[horcorrelation.shape[0]/2]
			vertcorrelation = vertcorrelation/vertcorrelation[vertcorrelation.shape[0]/2]
			crosscorrelation = crosscorrelation/crosscorrelation[horcorrelation.shape[0]/2]
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
    
			peaklocalmax = np.nanmean(peak_local_max(image))
			felzen = np.nanmean(segmentation.felzenszwalb(image))
			if np.isnan(peaklocalmax):
				peaklocalmax = 0.0
			if np.isnan(felzen):
				felzen = 0.0
			# Store the rescaled image pixels and the axis ratio
			X[i, 0:imageSize] = np.reshape(image, (1, imageSize))
			#Zak's note: I think I need to add new features into this array after axisratio
			X[i, imageSize:imageSize+corr2dsize] = np.reshape(correlation, (1,corr2dsize))
			X[i, imageSize+corr2dsize:imageSize+corr2dsize+corrsize] = np.reshape(horcorrelation, (1,corrsize))
			X[i, imageSize+corr2dsize+corrsize:imageSize+corr2dsize+2*corrsize] = np.reshape(vertcorrelation, (1,corrsize))
			X[i, imageSize+corr2dsize+2*corrsize:imageSize+corr2dsize+3*corrsize] = np.reshape(crosscorrelation, (1,corrsize))
			X[i, imageSize+3*corrsize+corr2dsize] = axisratio
			X[i, imageSize+3*corrsize+corr2dsize+1] = fillratio
			X[i, imageSize+3*corrsize+corr2dsize+2] = eigenratio
			X[i, imageSize+3*corrsize+corr2dsize+3] = solidity
			X[i, imageSize+3*corrsize+corr2dsize+4] = corners
			X[i, imageSize+3*corrsize+corr2dsize+5] = hu1
			X[i, imageSize+3*corrsize+corr2dsize+6] = hu2
			X[i, imageSize+3*corrsize+corr2dsize+7] = hu3
			X[i, imageSize+3*corrsize+corr2dsize+8] = hu12
			X[i, imageSize+3*corrsize+corr2dsize+9] = hu13
			X[i, imageSize+3*corrsize+corr2dsize+10] = hu23
			X[i, imageSize+3*corrsize+corr2dsize+11] = perimratio
			X[i, imageSize+3*corrsize+corr2dsize+12] = arearatio
			X[i, imageSize+3*corrsize+corr2dsize+13] = cornercenter
			X[i, imageSize+3*corrsize+corr2dsize+14] = cornercentercoords[0]
			X[i, imageSize+3*corrsize+corr2dsize+15] = cornercentercoords[1]
			X[i, imageSize+3*corrsize+corr2dsize+16] = cornerstd
			X[i, imageSize+3*corrsize+corr2dsize+17] = cornerstdcoords[0]
			X[i, imageSize+3*corrsize+corr2dsize+18] = cornerstdcoords[1]
			X[i, imageSize+3*corrsize+corr2dsize+19] = np.nanmean(filter.vsobel(image))
			X[i, imageSize+3*corrsize+corr2dsize+20] = np.nanmean(filter.hsobel(image))
			X[i, imageSize+3*corrsize+corr2dsize+21] = felzen
			X[i, imageSize+3*corrsize+corr2dsize+22] = peaklocalmax
			X[i, imageSize+3*corrsize+corr2dsize+23] = lrdiff
			X[i, imageSize+3*corrsize+corr2dsize+24] = tbdiff
			X[i, imageSize+3*corrsize+corr2dsize+25] = whu1
			X[i, imageSize+3*corrsize+corr2dsize+26] = whu2
			X[i, imageSize+3*corrsize+corr2dsize+27] = whu3
			X[i, imageSize+3*corrsize+corr2dsize+28] = whu12
			X[i, imageSize+3*corrsize+corr2dsize+29] = whu13
			X[i, imageSize+3*corrsize+corr2dsize+30] = whu23
			X[i, imageSize+3*corrsize+corr2dsize+31] = extent 
			X[i, imageSize+3*corrsize+corr2dsize+32] = minintensity
			X[i, imageSize+3*corrsize+corr2dsize+33] = meanintensity
			X[i, imageSize+3*corrsize+corr2dsize+34] = maxintensity
			X[i, imageSize+3*corrsize+corr2dsize+35] = intensityratio1
			X[i, imageSize+3*corrsize+corr2dsize+36] = intensityratio2
			X[i, imageSize+3*corrsize+corr2dsize+37] = intensityratio3
			X[i, imageSize+3*corrsize+corr2dsize+38] = hormean
			X[i, imageSize+3*corrsize+corr2dsize+39] = horstd
			X[i, imageSize+3*corrsize+corr2dsize+40] = vertmean
			X[i, imageSize+3*corrsize+corr2dsize+41] = vertstd
			X[i, imageSize+3*corrsize+corr2dsize+42] = horcount
			X[i, imageSize+3*corrsize+corr2dsize+43] = vertcount
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
clf = RF(n_estimators=n_estim, n_jobs=3,compute_importances=True);
scores = cross_validation.cross_val_score(clf, X, y, cv=5, n_jobs=1);
print "Accuracy of all classes"
#print np.mean(scores)

kf = KFold(y, n_folds=10)
y_pred = y * 0
for train, test in kf:
	X_train, X_test, y_train, y_test = X[train,:], X[test,:], y[train], y[test]
	clf = RF(n_estimators=n_estim, n_jobs=3,compute_importances=True)
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
kf = KFold(y, n_folds=10)
# prediction probabilities number of samples, by number of classes
y_pred = np.zeros((len(y),len(set(y))))
for train, test in kf:
	X_train, X_test, y_train, y_test = X[train,:], X[test,:], y[train], y[test]
	clf = RF(n_estimators=n_estim, n_jobs=3,compute_importances=True)
	clf.fit(X_train, y_train)
	y_pred[test] = clf.predict_proba(X_test)
	
multiclass_log_loss(y, y_pred)
