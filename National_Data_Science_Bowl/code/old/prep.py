def prep():
	header = "acantharia_protist_big_center,acantharia_protist_halo,acantharia_protist,amphipods,appendicularian_fritillaridae,appendicularian_s_shape,appendicularian_slight_curve,appendicularian_straight,artifacts_edge,artifacts,chaetognath_non_sagitta,chaetognath_other,chaetognath_sagitta,chordate_type1,copepod_calanoid_eggs,copepod_calanoid_eucalanus,copepod_calanoid_flatheads,copepod_calanoid_frillyAntennae,copepod_calanoid_large_side_antennatucked,copepod_calanoid_large,copepod_calanoid_octomoms,copepod_calanoid_small_longantennae,copepod_calanoid,copepod_cyclopoid_copilia,copepod_cyclopoid_oithona_eggs,copepod_cyclopoid_oithona,copepod_other,crustacean_other,ctenophore_cestid,ctenophore_cydippid_no_tentacles,ctenophore_cydippid_tentacles,ctenophore_lobate,decapods,detritus_blob,detritus_filamentous,detritus_other,diatom_chain_string,diatom_chain_tube,echinoderm_larva_pluteus_brittlestar,echinoderm_larva_pluteus_early,echinoderm_larva_pluteus_typeC,echinoderm_larva_pluteus_urchin,echinoderm_larva_seastar_bipinnaria,echinoderm_larva_seastar_brachiolaria,echinoderm_seacucumber_auricularia_larva,echinopluteus,ephyra,euphausiids_young,euphausiids,fecal_pellet,fish_larvae_deep_body,fish_larvae_leptocephali,fish_larvae_medium_body,fish_larvae_myctophids,fish_larvae_thin_body,fish_larvae_very_thin_body,heteropod,hydromedusae_aglaura,hydromedusae_bell_and_tentacles,hydromedusae_h15,hydromedusae_haliscera_small_sideview,hydromedusae_haliscera,hydromedusae_liriope,hydromedusae_narco_dark,hydromedusae_narco_young,hydromedusae_narcomedusae,hydromedusae_other,hydromedusae_partial_dark,hydromedusae_shapeA_sideview_small,hydromedusae_shapeA,hydromedusae_shapeB,hydromedusae_sideview_big,hydromedusae_solmaris,hydromedusae_solmundella,hydromedusae_typeD_bell_and_tentacles,hydromedusae_typeD,hydromedusae_typeE,hydromedusae_typeF,invertebrate_larvae_other_A,invertebrate_larvae_other_B,jellies_tentacles,polychaete,protist_dark_center,protist_fuzzy_olive,protist_noctiluca,protist_other,protist_star,pteropod_butterfly,pteropod_theco_dev_seq,pteropod_triangle,radiolarian_chain,radiolarian_colony,shrimp_caridean,shrimp_sergestidae,shrimp_zoea,shrimp-like_other,siphonophore_calycophoran_abylidae,siphonophore_calycophoran_rocketship_adult,siphonophore_calycophoran_rocketship_young,siphonophore_calycophoran_sphaeronectes_stem,siphonophore_calycophoran_sphaeronectes_young,siphonophore_calycophoran_sphaeronectes,siphonophore_other_parts,siphonophore_partial,siphonophore_physonect_young,siphonophore_physonect,stomatopod,tornaria_acorn_worm_larvae,trichodesmium_bowtie,trichodesmium_multiple,trichodesmium_puff,trichodesmium_tuft,trochophore_larvae,tunicate_doliolid_nurse,tunicate_doliolid,tunicate_partial,tunicate_salp_chains,tunicate_salp,unknown_blobs_and_smudges,unknown_sticks,unknown_unclassified".split(',')

        with open('namesClasses.dat','rb') as f:

                namesClasses = cPickle.load(f)

	labels = map(lambda s: s.split('\\')[-1], namesClasses)

	#get the total test images
	fnames = glob.glob(os.path.join("competition_data", "test", "*.jpg"))
	numberofTestImages = len(fnames)

	X_test = np.zeros((numberofTestImages, num_features), dtype=float)

	#Get filename separate from prefix path
	images = map(lambda fileName: fileName.split('\\')[-1], fnames)

	i = 0
	# report progress for each 5% done  
	report = [int((j+1)*numberofTestImages/20.) for j in range(20)]
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
		PCAFeatures = pca.fit_transform(image[0])
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
		hormirror = image[:maxPixel/2]-image[maxPixel-1:maxPixel/2:-1]
		vertmirror = image[:,:maxPixel/2]-image[:,maxPixel-1:maxPixel/2:-1]		
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
		X[i, featcount+lbpcounts.size:featcount+lbpcounts.size+PCAsize] = pca.explained_variance_ratio_
		X[i, featcount+lbpcounts.size+PCAsize] = np.mean(PCAFeatures)
		y[i] = label
		i += 1
		if i in report: print np.ceil(i *100.0 / numberofTestImages), "% done"
		
		y_pred = clf.predict_proba(X_test)
		
		df = pd.DataFrame(y_pred, columns=labels, index=images)
		
		df.index.name = 'image'
		
		df = df[header]
		
		df.to_csv('competition_data/submission.csv')
		
		#!gzip competition_data/submission.csv
		
		#!ls -l competition_data/submission.csv.gz	
