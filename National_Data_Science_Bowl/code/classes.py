#Need to add identities under id_list
class Tree(object):
	def __init__(self, data, end_class = True, id_list = None, layer = 0):
		self.data = data
		self.end_class = end_class
		self.num_branches = 0
		self.branches = []
		self.layer = layer
		self.source = None
		if id_list is None:
			self.id_list = []
	def add_branch(self, tree):
		self.num_branches = self.num_branches + 1
		self.branches.append(tree)
		tree.set_layer(self.layer)
		if tree.id_list:
			self.id_list = self.id_list + tree.id_list
		tree.source = self
	def count_ends(self, layer = 0):
		count = 0
		for branch in self.branches:
			count = count + branch.count_ends(layer+1)
		if self.end_class:
			count = count + 1
		return count
	def set_layer(self, prev_layer):
		self.layer = prev_layer + 1
	def max_depth(self,depth=1):
		maxdepth=depth
		for branch in self.branches:
			maxdepth = max(maxdepth,branch.max_depth(depth+1))
		return maxdepth
	def data_search(self, term):
		if self.data == term:
			return self
		else:
			for branch in self.branches:
				search_result = branch.data_search(term)
				if search_result:
					return search_result
			return None
	def update_ids(self):
		for branch in self.branches:
			branch.update_ids()
			if branch.id_list:
				self.id_list = self.id_list + branch.id_list
		self.unduplicate_ids()
	def unduplicate_ids(self):
		uniques = []
		for x in self.id_list:
			if x not in uniques: 
				uniques.append(x)
		if uniques:
			self.id_list = uniques
	def find_layers(self, layers, layer = 0, max_depth = None):
		if max_depth is None:
			max_depth = self.max_depth()
		layers[layer].append(self)
		for branch in self.branches:
			layers = branch.find_layers(layers,layer+1,max_depth)
		return layers
	def list_ends(self, end_count = None, end_list = None):
		if end_list is None:
			if end_count is None:
				end_count = self.count_ends()
			end_list = [0 for i in range(end_count)]
		for branch in self.branches:
			end_list = branch.list_ends(end_count, end_list)
		if self.end_class:
			if self.id_list:
				end_list[self.id_list[0]] = self
			else:
				end_list.pop()
		return end_list
	def trim(self):
		branch_list = self.branches
		remove_list = []
		for branch in branch_list:
			if branch.trim():
				remove_list.append(branch)
		for branch in remove_list:
			self.branches.remove(branch)
		if not self.id_list:
			self.source.num_branches = self.source.num_branches - 1
			self.source = None
			return True
		return False
				
		

root_class = Tree(data = 'root', end_class = False)

plankton = Tree(data = 'plankton', end_class = False)
protists = Tree(data = 'protists', end_class = False)
trichodesmium = Tree(data = 'trichodesmium', end_class = False)
diatoms = Tree(data = 'diatoms', end_class = False)
gelatinous_zooplankton = Tree(data = 'gelatinous_zooplankton', end_class = False)
fish = Tree(data = 'fish', end_class = False)
crustaceans = Tree(data = 'crustaceans', end_class = False)
chaetognaths = Tree(data = 'chaetognaths', end_class = False)
gastropods = Tree(data = 'gastropods', end_class = False)
other_invert_larvae = Tree(data = 'other_invert_larvae', end_class = False)
detritus = Tree(data = 'detritus', end_class = False)
unknown = Tree(data = 'unknown', end_class = False)
acantharia_protist_big_center = Tree(data = 'acantharia_protist_big_center')
acantharia_protist_halo = Tree(data = 'acantharia_protist_halo')
acantharia_protist = Tree(data = 'acantharia_protist')
amphipods = Tree(data = 'amphipods')
appendicularian_fritillaridae = Tree(data = 'appendicularian_fritillaridae')
appendicularian_s_shape = Tree(data = 'appendicularian_s_shape')
appendicularian_slight_curve = Tree(data = 'appendicularian_slight_curve')
appendicularian_straight = Tree(data = 'appendicularian_straight')
artifacts_edge  = Tree(data = 'artifacts_edge')
artifacts = Tree(data = 'artifacts')
chaetognath_non_sagitta = Tree(data = 'chaetognath_non_sagitta')
chaetognath_other = Tree(data = 'chaetognath_other')
chaetognath_sagitta = Tree(data = 'chaetognath_sagitta')
chordate_type1 = Tree(data = 'chordate_type1')
copepod_calanoid_eggs = Tree(data = 'copepod_calanoid_eggs')
copepod_calanoid_eucalanus = Tree(data = 'copepod_calanoid_eucalanus')
copepod_calanoid_flatheads = Tree(data = 'copepod_calanoid_flatheads')
copepod_calanoid_frillyAntennae = Tree(data = 'copepod_calanoid_frillyAntennae')
copepod_calanoid_large_side_antennatucked = Tree(data = 'copepod_calanoid_large_side_antennatucked')
copepod_calanoid_large = Tree(data = 'copepod_calanoid_large')
copepod_calanoid_octomoms = Tree(data = 'copepod_calanoid_octomoms')
copepod_calanoid_small_longantennae = Tree(data = 'copepod_calanoid_small_longantennae')
copepod_calanoid = Tree(data = 'copepod_calanoid')
copepod_cyclopoid_copilia = Tree(data = 'copepod_cyclopoid_copilia')
copepod_cyclopoid_oithona_eggs = Tree(data = 'copepod_cyclopoid_oithona_eggs')
copepod_cyclopoid_oithona = Tree(data = 'copepod_cyclopoid_oithona')
copepod_other = Tree(data = 'copepod_other')
crustacean_other = Tree(data = 'crustacean_other')
ctenophore_cestid = Tree(data = 'ctenophore_cestid')
ctenophore_cydippid_no_tentacles = Tree(data = 'ctenophore_cydippid_no_tentacles')
ctenophore_cydippid_tentacles = Tree(data = 'ctenophore_cydippid_tentacles')
ctenophore_lobate = Tree(data = 'ctenophore_lobate')
decapods = Tree(data = 'decapods')
detritus_blob = Tree(data = 'detritus_blob')
detritus_filamentous = Tree(data = 'detritus_filamentous')
detritus_other = Tree(data = 'detritus_other')
diatom_chain_string = Tree(data = 'diatom_chain_string')
diatom_chain_tube = Tree(data = 'diatom_chain_tube')
echinoderm_larva_pluteus_brittlestar = Tree(data = 'echinoderm_larva_pluteus_brittlestar')
echinoderm_larva_pluteus_early = Tree(data = 'echinoderm_larva_pluteus_early')
echinoderm_larva_pluteus_typeC = Tree(data = 'echinoderm_larva_pluteus_typeC')
echinoderm_larva_pluteus_urchin = Tree(data = 'echinoderm_larva_pluteus_urchin')
echinoderm_larva_seastar_bipinnaria = Tree(data = 'echinoderm_larva_seastar_bipinnaria')
echinoderm_larva_seastar_brachiolaria = Tree(data = 'echinoderm_larva_seastar_brachiolaria')
echinoderm_seacucumber_auricularia_larva = Tree(data = 'echinoderm_seacucumber_auricularia_larva')
echinopluteus = Tree(data = 'echinopluteus')
ephyra = Tree(data = 'ephyra')
euphausiids_young = Tree(data = 'euphausiids_young')
euphausiids = Tree(data = 'euphausiids')
fecal_pellet = Tree(data = 'fecal_pellet')
fish_larvae_deep_body = Tree(data = 'fish_larvae_deep_body')
fish_larvae_leptocephali = Tree(data = 'fish_larvae_leptocephali')
fish_larvae_medium_body = Tree(data = 'fish_larvae_medium_body')
fish_larvae_myctophids = Tree(data = 'fish_larvae_myctophids')
fish_larvae_thin_body = Tree(data = 'fish_larvae_thin_body')
fish_larvae_very_thin_body = Tree(data = 'fish_larvae_very_thin_body')
heteropod = Tree(data = 'heteropod')
hydromedusae_aglaura = Tree(data = 'hydromedusae_aglaura')
hydromedusae_bell_and_tentacles = Tree(data = 'hydromedusae_bell_and_tentacles')
hydromedusae_h15 = Tree(data = 'hydromedusae_h15')
hydromedusae_haliscera_small_sideview = Tree(data = 'hydromedusae_haliscera_small_sideview')
hydromedusae_haliscera = Tree(data = 'hydromedusae_haliscera')
hydromedusae_liriope = Tree(data = 'hydromedusae_liriope')
hydromedusae_narco_dark = Tree(data = 'hydromedusae_narco_dark')
hydromedusae_narco_young = Tree(data = 'hydromedusae_narco_young')
hydromedusae_narcomedusae = Tree(data = 'hydromedusae_narcomedusae')
hydromedusae_other = Tree(data = 'hydromedusae_other')
hydromedusae_partial_dark = Tree(data = 'hydromedusae_partial_dark')
hydromedusae_shapeA_sideview_small = Tree(data = 'hydromedusae_shapeA_sideview_small')
hydromedusae_shapeA = Tree(data = 'hydromedusae_shapeA')
hydromedusae_shapeB = Tree(data = 'hydromedusae_shapeB')
hydromedusae_sideview_big = Tree(data = 'hydromedusae_sideview_big')
hydromedusae_solmaris = Tree(data = 'hydromedusae_solmaris')
hydromedusae_solmundella = Tree(data = 'hydromedusae_solmundella')
hydromedusae_typeD_bell_and_tentacles = Tree(data = 'hydromedusae_typeD_bell_and_tentacles')
hydromedusae_typeD = Tree(data = 'hydromedusae_typeD')
hydromedusae_typeE = Tree(data = 'hydromedusae_typeE')
hydromedusae_typeF = Tree(data = 'hydromedusae_typeF')
invertebrate_larvae_other_A = Tree(data = 'invertebrate_larvae_other_A')
invertebrate_larvae_other_B = Tree(data = 'invertebrate_larvae_other_B')
jellies_tentacles = Tree(data = 'jellies_tentacles')
polychaete = Tree(data = 'polychaete')
protist_dark_center = Tree(data = 'protist_dark_center')
protist_fuzzy_olive = Tree(data = 'protist_fuzzy_olive')
protist_noctiluca = Tree(data = 'protist_noctiluca')
protist_other = Tree(data = 'protist_other')
protist_star = Tree(data = 'protist_star')
pteropod_butterfly = Tree(data = 'pteropod_butterfly')
pteropod_theco_dev_seq = Tree(data = 'pteropod_theco_dev_seq')
pteropod_triangle = Tree(data = 'pteropod_triangle')
radiolarian_chain = Tree(data = 'radiolarian_chain')
radiolarian_colony = Tree(data = 'radiolarian_colony')
shrimp_caridean = Tree(data = 'shrimp_caridean')
shrimp_sergestidae = Tree(data = 'shrimp_sergestidae')
shrimp_zoea = Tree(data = 'shrimp_zoea')
shrimp_like_other = Tree(data = 'shrimp-like_other')
siphonophore_calycophoran_abylidae = Tree(data = 'siphonophore_calycophoran_abylidae')
siphonophore_calycophoran_rocketship_adult = Tree(data = 'siphonophore_calycophoran_rocketship_adult')
siphonophore_calycophoran_rocketship_young = Tree(data = 'siphonophore_calycophoran_rocketship_young')
siphonophore_calycophoran_sphaeronectes_stem = Tree(data = 'siphonophore_calycophoran_sphaeronectes_stem')
siphonophore_calycophoran_sphaeronectes_young = Tree(data = 'siphonophore_calycophoran_sphaeronectes_young')
siphonophore_calycophoran_sphaeronectes = Tree(data = 'siphonophore_calycophoran_sphaeronectes')
siphonophore_other_parts = Tree(data = 'siphonophore_other_parts')
siphonophore_partial = Tree(data = 'siphonophore_partial')
siphonophore_physonect_young = Tree(data = 'siphonophore_physonect_young')
siphonophore_physonect = Tree(data = 'siphonophore_physonect')
stomatopod = Tree(data = 'stomatopod')
tornaria_acorn_worm_larvae = Tree(data = 'tornaria_acorn_worm_larvae')
trichodesmium_bowtie = Tree(data = 'trichodesmium_bowtie')
trichodesmium_multiple = Tree(data = 'trichodesmium_multiple')
trichodesmium_puff = Tree(data = 'trichodesmium_puff')
trichodesmium_tuft = Tree(data = 'trichodesmium_tuft')
trochophore_larvae = Tree(data = 'trochophore_larvae')
tunicate_doliolid_nurse = Tree(data = 'tunicate_doliolid_nurse')
tunicate_doliolid = Tree(data = 'tunicate_doliolid')
tunicate_partial = Tree(data = 'tunicate_partial')
tunicate_salp_chains = Tree(data = 'tunicate_salp_chains')
tunicate_salp = Tree(data = 'tunicate_salp')
unknown_blobs_and_smudges = Tree(data = 'unknown_blobs_and_smudges')
unknown_sticks = Tree(data = 'unknown_sticks')
unknown_unclassified = Tree(data = 'unknown_unclassified')
		
root_class.add_branch(artifacts)
root_class.add_branch(artifacts_edge)
root_class.add_branch(plankton)

plankton.add_branch(protists)
plankton.add_branch(trichodesmium)
plankton.add_branch(diatoms)
plankton.add_branch(gelatinous_zooplankton)
plankton.add_branch(chordate_type1)
plankton.add_branch(fish)
plankton.add_branch(crustaceans)
plankton.add_branch(polychaete)
plankton.add_branch(chaetognaths)
plankton.add_branch(gastropods)
plankton.add_branch(other_invert_larvae)
plankton.add_branch(detritus)
plankton.add_branch(unknown)

protists.add_branch(acantharia_protist)
protists.add_branch(acantharia_protist_big_center)
protists.add_branch(acantharia_protist_halo)
protists.add_branch(protist_noctiluca)
protists.add_branch(protist_other)
protists.add_branch(protist_star)
protists.add_branch(protist_fuzzy_olive)
protists.add_branch(protist_dark_center)
protists.add_branch(radiolarian_colony)
protists.add_branch(radiolarian_chain)

trichodesmium.add_branch(trichodesmium_tuft)
trichodesmium.add_branch(trichodesmium_bowtie)
trichodesmium.add_branch(trichodesmium_puff)
trichodesmium.add_branch(trichodesmium_multiple)

diatoms.add_branch(diatom_chain_string)
diatoms.add_branch(diatom_chain_tube)

gelatinous_zooplankton.add_branch(jellies_tentacles)
gelatinous_zooplankton.add_branch(ephyra)
gelatinous_zooplankton.add_branch(ctenophore_cestid)
gelatinous_zooplankton.add_branch(ctenophore_cydippid_tentacles)
gelatinous_zooplankton.add_branch(ctenophore_cydippid_no_tentacles)
gelatinous_zooplankton.add_branch(ctenophore_lobate)
gelatinous_zooplankton.add_branch(appendicularian_fritillaridae)
gelatinous_zooplankton.add_branch(appendicularian_s_shape)
gelatinous_zooplankton.add_branch(appendicularian_slight_curve)
gelatinous_zooplankton.add_branch(appendicularian_straight)
gelatinous_zooplankton.add_branch(tunicate_doliolid)
gelatinous_zooplankton.add_branch(tunicate_doliolid_nurse)
gelatinous_zooplankton.add_branch(tunicate_salp)
gelatinous_zooplankton.add_branch(tunicate_partial)
gelatinous_zooplankton.add_branch(tunicate_salp_chains)
gelatinous_zooplankton.add_branch(siphonophore_partial) #This doesn't seem to be in the diagram, but added anyway
gelatinous_zooplankton.add_branch(siphonophore_other_parts)
gelatinous_zooplankton.add_branch(siphonophore_physonect)
gelatinous_zooplankton.add_branch(siphonophore_physonect_young)
gelatinous_zooplankton.add_branch(siphonophore_calycophoran_abylidae)
gelatinous_zooplankton.add_branch(siphonophore_calycophoran_rocketship_adult)
gelatinous_zooplankton.add_branch(siphonophore_calycophoran_rocketship_young)
gelatinous_zooplankton.add_branch(siphonophore_calycophoran_sphaeronectes)
gelatinous_zooplankton.add_branch(siphonophore_calycophoran_sphaeronectes_young)
gelatinous_zooplankton.add_branch(siphonophore_calycophoran_sphaeronectes_stem)
gelatinous_zooplankton.add_branch(hydromedusae_narcomedusae)
gelatinous_zooplankton.add_branch(hydromedusae_narco_dark)
gelatinous_zooplankton.add_branch(hydromedusae_solmundella)
gelatinous_zooplankton.add_branch(hydromedusae_solmaris)
gelatinous_zooplankton.add_branch(hydromedusae_narco_young)
gelatinous_zooplankton.add_branch(hydromedusae_liriope)
gelatinous_zooplankton.add_branch(hydromedusae_aglaura)
gelatinous_zooplankton.add_branch(hydromedusae_haliscera)
gelatinous_zooplankton.add_branch(hydromedusae_haliscera_small_sideview)
gelatinous_zooplankton.add_branch(hydromedusae_partial_dark)
gelatinous_zooplankton.add_branch(hydromedusae_typeD_bell_and_tentacles)
gelatinous_zooplankton.add_branch(hydromedusae_typeD)
gelatinous_zooplankton.add_branch(hydromedusae_typeE)
gelatinous_zooplankton.add_branch(hydromedusae_typeF)
gelatinous_zooplankton.add_branch(hydromedusae_other)
gelatinous_zooplankton.add_branch(hydromedusae_bell_and_tentacles)
gelatinous_zooplankton.add_branch(hydromedusae_shapeA)
gelatinous_zooplankton.add_branch(hydromedusae_shapeA_sideview_small)
gelatinous_zooplankton.add_branch(hydromedusae_sideview_big)
gelatinous_zooplankton.add_branch(hydromedusae_shapeB)
gelatinous_zooplankton.add_branch(hydromedusae_h15)

fish.add_branch(fish_larvae_leptocephali)
fish.add_branch(fish_larvae_myctophids)
fish.add_branch(fish_larvae_very_thin_body)
fish.add_branch(fish_larvae_thin_body)
fish.add_branch(fish_larvae_medium_body)
fish.add_branch(fish_larvae_deep_body)

crustaceans.add_branch(crustacean_other)
crustaceans.add_branch(stomatopod)
crustaceans.add_branch(copepod_cyclopoid_copilia)
crustaceans.add_branch(copepod_cyclopoid_oithona)
crustaceans.add_branch(copepod_cyclopoid_oithona_eggs)
crustaceans.add_branch(copepod_calanoid)
crustaceans.add_branch(copepod_other)
crustaceans.add_branch(copepod_calanoid_small_longantennae)
crustaceans.add_branch(copepod_calanoid_frillyAntennae)
crustaceans.add_branch(copepod_calanoid_flatheads)
crustaceans.add_branch(copepod_calanoid_eggs)
crustaceans.add_branch(copepod_calanoid_octomoms)
crustaceans.add_branch(copepod_calanoid_large)
crustaceans.add_branch(copepod_calanoid_large_side_antennatucked)
crustaceans.add_branch(copepod_calanoid_eucalanus)
crustaceans.add_branch(amphipods)
crustaceans.add_branch(shrimp_like_other)
crustaceans.add_branch(euphausiids)
crustaceans.add_branch(euphausiids_young)
crustaceans.add_branch(decapods)
crustaceans.add_branch(shrimp_zoea)
crustaceans.add_branch(shrimp_caridean)
crustaceans.add_branch(shrimp_sergestidae)

chaetognaths.add_branch(chaetognath_sagitta)
chaetognaths.add_branch(chaetognath_non_sagitta)
chaetognaths.add_branch(chaetognath_other)

gastropods.add_branch(heteropod)
gastropods.add_branch(pteropod_butterfly)
gastropods.add_branch(pteropod_triangle)
gastropods.add_branch(pteropod_theco_dev_seq)

other_invert_larvae.add_branch(trochophore_larvae)
other_invert_larvae.add_branch(invertebrate_larvae_other_A)
other_invert_larvae.add_branch(invertebrate_larvae_other_B)
other_invert_larvae.add_branch(tornaria_acorn_worm_larvae)
other_invert_larvae.add_branch(echinoderm_larva_seastar_bipinnaria)
other_invert_larvae.add_branch(echinoderm_larva_seastar_brachiolaria)
other_invert_larvae.add_branch(echinoderm_larva_pluteus_early)
other_invert_larvae.add_branch(echinoderm_larva_pluteus_urchin)
other_invert_larvae.add_branch(echinoderm_larva_pluteus_typeC)
other_invert_larvae.add_branch(echinoderm_larva_pluteus_brittlestar)
other_invert_larvae.add_branch(echinopluteus)
other_invert_larvae.add_branch(echinoderm_seacucumber_auricularia_larva)

detritus.add_branch(fecal_pellet)
detritus.add_branch(detritus_blob)
detritus.add_branch(detritus_filamentous)
detritus.add_branch(detritus_other)

unknown.add_branch(unknown_blobs_and_smudges)
unknown.add_branch(unknown_sticks)
unknown.add_branch(unknown_unclassified)

# classifications =  	{'artifacts':
						# ['artifacts', 'artifacts_edges']
					# 'plankton':
						# {'protists':
							# {'acantharia_protist':
								# ['acantharia_protist', 'acantharia_protist_halo', 'acantharia_protist_big_center']
							# }
							# {'protist_noctiluca':
								# ['protist_noctiluca']
							# }
							# {'protist_other':
								# ['protist_other', 'protist_star', 'protist_fuzzy_olive', 'protist_dark_center']
							# }
							# {'radiolarian_colony':
								# ['radiolarian_colony', 'radiolarian_chain']
							# }	
						# }
						# {'trichodesmium':
							# ['tricho_tuft', 'tricho_bowtie', 'tricho_puff', 'tricho_multiple']
						# }
						# {'diatoms':
							# ['diatom_chain_string', 'diatom_chain_tube']
						# }
						# {'gelatinous_zooplankton':
							# {'jellies_tentacles':
								# ['jellies_tentacles']
							# }
							# {'ctenophores':
								# ['ctenophore_cestid', 'ctenophore_cydippid_tentacles', 'ctenophore_cydippid_no_tentacles', 'ctenophore_lobate']
							# }
							# {ephyra
						# }
						# {'fish':
						# }
						# {'crustaceans':
						# }
						# {'gastropods':
						# }
						# {'detritus':
						# }
						# {'unknown':
						# }
					# }
					
					
# Classes:
# --------------------------------
# Root:
 	# artifacts_edge
 	# artifacts
	# plankton
		# protists
 			# acantharia_protist
 				# acantharia_protist_big_center
 				# acantharia_protist_halo	
			# protist_noctiluca
			# protist_other
				# protist_star
				# protist_fuzzy_olive
				# protist_dark_center
			# radiolarian_colony
				# radiolarian_chain
		# trichodesmium
			# tricho_tuft
			# tricho_bowtie
			# tricho_puff
			# tricho_multiple
		# diatoms
			# diatom_chain_string
			# diatom_chain_tube
		# gelatinous_zooplankton
			# jellies_tentacles
			# ctenophores
				# ctenophore_cestid
				# ctenophore_cydippid_tentacles
				# ctenophore_cydippid_no_tentacles
				# ctenophore_lobate
			# ephyra
			# hydromedusae
				# hydromedusae_narcomedusae
					# hydromedusae_narco_dark
					# hydromedusae_solmundella
					# hydromedusae_solmaris
						# hydromedusae_narco_young
				# hydromedusae_liriope
				# hydromedusae_aglaura
				# hydromedusae_haliscera
					# hydromedusae_haliscera_small_sideview
				# other_hydromedusae
					# hydromedusae_typeD_general (Added category)
						# hydromedusae_typeD_bell_and_tentacles
						# hydromedusae_typeD
					
					
		# fish
		# crustaceans
		# chaetognaths
		# gastropods
		# other invert larvae
		# detritus
		# unknown
 	 
