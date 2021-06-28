# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 10:48:10 2019

@author: gathu
"""


from otheralgorithms.FSSD.util.csvProcessing import writeCSV,readCSV,writeCSVwithHeader,readCSVwithHeader
from otheralgorithms.FSSD.enumerator.enumerator_attribute_complex import enumerator_complex_cbo_init_new_config,compute_support_complex_index_with_bitset,encode_sup,closed_complex_index,pattern_over_attributes,value_to_yield_complex,pattern_equal_pattern_on_a_single_attribute,decode_sup
from otheralgorithms.FSSD.enumerator.enumerator_attribute_themes2 import get_domain_from_dataset_theme
from otheralgorithms.FSSD.filterer.filter import filter_pipeline_obj

from time import time
import cProfile
import pstats
from copy import copy,deepcopy
import argparse
from os.path import basename, splitext, dirname
#from memory_profiler import profile
from subprocess import call,check_output
import sys
from itertools import combinations,chain
from sys import stdout
import pysubgroup as ps
import pandas as pd
import os
import shutil
import csv
from bisect import bisect_left
#from guppy import hpy 
def nb_bit_1(n):
	return bin(n).count('1')

def transform_dataset(dataset,attributes,class_attribute,wanted_label,verbose=False):
	new_dataset=[]
	statistics={}
	alpha_ratio_class=0.
	positive_extent=set()
	negative_extent=set()

	for k in range(len(dataset)):
		row=dataset[k]
		new_row={attr_name:row[attr_name] for attr_name in attributes}
		new_row['positive']=int(row[class_attribute]==wanted_label)
		new_row[class_attribute]=row[class_attribute]
		if new_row['positive']:
			positive_extent|={k}
			alpha_ratio_class+=1
		else:
			negative_extent|={k}

		new_dataset.append(new_row)
	# statistics['rows']=len(new_dataset)
	# statistics['alpha']=alpha_ratio_class/len(dataset)
	# nb_possible_intervals=1.
	# for attr in attributes:
	# 	statistics['|dom('+attr+')|']=float(len(set(x[attr] for x in dataset)))
	# 	nb_possible_intervals*=(statistics['|dom('+attr+')|']*(statistics['|dom('+attr+')|']+1))/2.
	# statistics['intervals']=nb_possible_intervals
	# statistics['intervalsGood']=transform(nb_possible_intervals)

	# if verbose:
	# 	print '------------------------------------------------------------'
	# 	for x in statistics:
	# 		print x, ' ',statistics[x]
	# 	print '------------------------------------------------------------'
		#raw_input('......')
	return new_dataset,positive_extent,negative_extent,alpha_ratio_class/len(dataset),statistics




def wracc(tpr,fpr,alpha):
	return alpha*(1-alpha)*(tpr-fpr)

def wracc_and_bound(tpr,fpr,alpha):
	return alpha*(1-alpha)*(tpr-fpr),alpha*(1-alpha)*(tpr)

def wracc_gain(tpr,fpr,alpha,current_pattern_set_tpr=0.,current_pattern_set_fpr=0.):
	return alpha*(1-alpha)*((1-current_pattern_set_tpr)*tpr-(1-current_pattern_set_fpr)*fpr)

def wracc_and_bound_gain(tpr,fpr,alpha,current_pattern_set_tpr=0.,current_pattern_set_fpr=0.):
	return alpha*(1-alpha)*((1-current_pattern_set_tpr)*tpr-(1-current_pattern_set_fpr)*fpr),alpha*(1-alpha)*((1-current_pattern_set_tpr)*tpr)


# def enumerating_closed_candidate_subgroups_with_cotp(dataset,attributes,types,positive_extent,negative_extent,alpha_ratio_class,threshold=1,indices_to_consider=None,infos_already_computed=[None,None,{'config':None}]):
	# 	attributes_types=[{'name':a, 'type':t} for a,t in zip(attributes,types)]
	# 	nobitset=True
	# 	if indices_to_consider is None:
	# 		indices_to_consider=set(range(len(dataset)))
	# 	nb_pos_extent=float(len(indices_to_consider&positive_extent))
	# 	nb_neg_extent=float(len(indices_to_consider&negative_extent))
		
		
	# 	attributes_full_index,allindex_full,initValues=infos_already_computed
	# 	if attributes_full_index is None:
	# 		#initValues={'config':None}
	# 		print 'CHECK BIATCH'
	# 		(_,_,cnf) = next(enumerator_complex_cbo_init_new_config(dataset, attributes_types,threshold=1,initValues=initValues,verbose=True,nobitset=nobitset,config_init={'indices':indices_to_consider}))#,config_init={'indices':indices_to_consider}))
	# 		attributes_full_index=cnf['attributes']
	# 		allindex_full=cnf['allindex']
	# 		infos_already_computed[0]=attributes_full_index
	# 		infos_already_computed[1]=allindex_full
	# 		infos_already_computed[2]=initValues
		
	# 	###############
	# 	closedPattern=closed_complex_index(attributes_full_index,set(),indices_to_consider&positive_extent,allindex_full,0)
		
	# 	attributeClosed=pattern_over_attributes(attributes_full_index, closedPattern)
	# 	sup_full_after_cotp=set(indices_to_consider)
	# 	sup_full_after_cotp_bitset=0#encode_sup(indices_to_consider,len(dataset))
	# 	for ai in range(len(attributes)):
	# 		sup_full_after_cotp_avant=set(sup_full_after_cotp)
	# 		_,sup_full_after_cotp,sup_full_after_cotp_bitset=compute_support_complex_index_with_bitset(attributeClosed,dataset,sup_full_after_cotp,sup_full_after_cotp_bitset,allindex_full,ai,wholeDataset=dataset,closed=False)
	# 		# print sup_full_after_cotp_avant-sup_full_after_cotp
	# 		# print attributes[ai]
	# 		# for x in sup_full_after_cotp_avant-sup_full_after_cotp:
	# 		# 	print [dataset[x][attributes[ai]]]
	# 		# print len(sup_full_after_cotp_avant-sup_full_after_cotp),[attributeClosed[ai]['pattern'][0],attributeClosed[ai]['pattern'][-1]]
	# 	#closedPatternXXXX=[[x[0],x[-1]] for x in closed_complex_index(attributes_full_index,set(),sup_full_after_cotp,allindex_full,0)]
	# 	#print closedPatternXXXX
	# 	#print '**************************************',[[x[0],x[-1]] for x in closedPattern],len(indices_to_consider),len(sup_full_after_cotp),len(indices_to_consider&negative_extent),len(indices_to_consider&positive_extent),len(sup_full_after_cotp&negative_extent),len(sup_full_after_cotp&positive_extent),'**************************************'
		
		
	# 	#indices_to_consider=sup_full_after_cotp
	# 	#################

	# 	positive_extent_to_consider=positive_extent&sup_full_after_cotp
	# 	negative_extent_to_consider=negative_extent&sup_full_after_cotp
	# 	#full_support=indices_to_consider
		
	# 	# nb_pos_extent=float(len(positive_extent_to_consider))
	# 	# nb_neg_extent=float(len(negative_extent_to_consider))


	# 	FIRST_ITERATION=True
	# 	#print len(positive_extent_to_consider),len(negative_extent_to_consider)
	# 	#'indices_bitset':encode_sup(positive_extent_to_consider,len(dataset)),
	# 	if nobitset:
	# 		config_init={'indices':positive_extent_to_consider,'FULL_SUPPORT':sup_full_after_cotp,'FULL_SUPPORT_BITSET':0,'FULL_SUPPORT_INFOS':dataset,'alpha':alpha_ratio_class,'positive_extent':positive_extent_to_consider,'negative_extent':negative_extent_to_consider}
	# 	else:
	# 		config_init={'indices':positive_extent_to_consider,'FULL_SUPPORT':sup_full_after_cotp,'FULL_SUPPORT_BITSET':encode_sup(sup_full_after_cotp,len(dataset)),'FULL_SUPPORT_INFOS':dataset,'alpha':alpha_ratio_class,'positive_extent':positive_extent_to_consider,'negative_extent':negative_extent_to_consider}
		

		
	# 	#raw_input('....')
	# 	for (p,l,cnf) in enumerator_complex_cbo_init_new_config(dataset, attributes_types,threshold=threshold,config_init=config_init,initValues=initValues,verbose=False,nobitset=nobitset):
	# 		# print p,len(cnf['indices'])
	# 		# raw_input('...')
	# 		##########COMPUTING_SUPPORT_FULL#############	
	# 		for id_attr,(a1,a2) in enumerate(zip(cnf['attributes'],attributeClosed)): 
	# 			a2['refinement_index']=a1['refinement_index']
	# 			a2['pattern']=p[id_attr]
			
	# 		#print len(indices_to_consider)
	# 		# cnf['FULL_SUPPORT']=indices_to_consider
	# 		# #print len(cnf['FULL_SUPPORT'])
	# 		# for ai in range(0,len(attributes)):
	# 		# 	cnf['FULL_SUPPORT_INFOS'],cnf['FULL_SUPPORT'],cnf['FULL_SUPPORT_BITSET']=compute_support_complex_index_with_bitset(attributeClosed,dataset,cnf['FULL_SUPPORT'],cnf['FULL_SUPPORT_BITSET'],allindex_full,ai,wholeDataset=dataset,closed=False)
			
	# 		# print len(cnf['FULL_SUPPORT'])
	# 		# raw_input('...')
	# 		for ai in range(cnf['refinement_index'],len(attributes)):
	# 			cnf['FULL_SUPPORT_INFOS'],cnf['FULL_SUPPORT'],cnf['FULL_SUPPORT_BITSET']=compute_support_complex_index_with_bitset(attributeClosed,dataset,cnf['FULL_SUPPORT'],cnf['FULL_SUPPORT_BITSET'],allindex_full,ai,wholeDataset=dataset,closed=False)
			

	# 		#cnf['FULL_SUPPORT_INFOS'],cnf['FULL_SUPPORT'],cnf['FULL_SUPPORT_BITSET']=compute_support_complex_index_with_bitset(attributeClosed,dataset,cnf['FULL_SUPPORT'],cnf['FULL_SUPPORT_BITSET'],allindex_full,cnf['refinement_index'],wholeDataset=dataset)
	# 		# for ai in range(len(attributes)):
	# 		# 	cnf['FULL_SUPPORT_INFOS'],cnf['FULL_SUPPORT'],cnf['FULL_SUPPORT_BITSET']=compute_support_complex_index_with_bitset(attributes_full_index,dataset,cnf['FULL_SUPPORT'],cnf['FULL_SUPPORT_BITSET'],allindex_full,ai,wholeDataset=dataset,closed=False)
				


	# 		#print cnf['FULL_SUPPORT']&cnf['indices']==cnf['indices']
			
	# 		#print len(cnf['FULL_SUPPORT'])

	# 		##########COMPUTING_SUPPORT_FULL#############
	# 		#cnt+=1
	# 		#raw_input('....')

	# 		pattern_infos={
	# 			'support_full':cnf['FULL_SUPPORT'],
	# 			'support_full_bitset':cnf['FULL_SUPPORT_BITSET'],
	# 			'support_positive':cnf['FULL_SUPPORT']&positive_extent,#cnf['indices'],
	# 			'support_positive_bitset':cnf['indices_bitset'],
	# 			'tpr':len(cnf['FULL_SUPPORT']&positive_extent)/nb_pos_extent,
	# 			'fpr':0. if nb_neg_extent==0 else (len(cnf['FULL_SUPPORT']&negative_extent))/nb_neg_extent,
	# 			'support_size':len(cnf['FULL_SUPPORT']),
	# 			'alpha':alpha_ratio_class
	# 		}
	# 		#print pattern_infos['tpr']
	# 		#print p,wracc(pattern_infos['tpr'],pattern_infos['fpr'],pattern_infos['alpha'])
	# 		yield p,l,pattern_infos,cnf
	# 	#raw_input('....')



def enumerating_closed_candidate_subgroups_with_cotp(dataset,attributes,types,positive_extent,negative_extent,alpha_ratio_class,threshold=1,indices_to_consider=None,infos_already_computed=[None,None,{'config':None}],depthmax=float('inf')):
	#attributes_types=[{'name':a, 'type':t} for a,t in zip(attributes,types)]
	attributes_types=[{'name':a, 'type':t} if t!='themes' else {'name':a, 'type':t,'widthmax':2} for a,t in zip(attributes,types)] #because dssd allows that each attributes may appear at most 2 times ( ... )
	nobitset=True
	if indices_to_consider is None:
		indices_to_consider=set(range(len(dataset)))
	nb_pos_extent=float(len(indices_to_consider&positive_extent))
	nb_neg_extent=float(len(indices_to_consider&negative_extent))
	
	attributes_full_index,allindex_full,initValues=infos_already_computed
	if attributes_full_index is None:
		(_,_,cnf) = next(enumerator_complex_cbo_init_new_config(dataset, attributes_types,threshold=1,initValues=initValues,verbose=False,nobitset=nobitset,config_init={'indices':indices_to_consider}))#,config_init={'indices':indices_to_consider}))
		attributes_full_index=cnf['attributes']
		allindex_full=cnf['allindex']
		infos_already_computed[0]=attributes_full_index
		infos_already_computed[1]=allindex_full
		infos_already_computed[2]=initValues
	
	###############
	closedPattern=closed_complex_index(attributes_full_index,set(),indices_to_consider&positive_extent,allindex_full,0)
	attributeClosed=pattern_over_attributes(attributes_full_index, closedPattern)
	sup_full_after_cotp=set(indices_to_consider)
	sup_full_after_cotp_bitset=0
	for ai in range(len(attributes)):
		_,sup_full_after_cotp,sup_full_after_cotp_bitset=compute_support_complex_index_with_bitset(attributeClosed,dataset,sup_full_after_cotp,sup_full_after_cotp_bitset,allindex_full,ai,wholeDataset=dataset,closed=False)
	#################

	positive_extent_to_consider=positive_extent&sup_full_after_cotp
	negative_extent_to_consider=negative_extent&sup_full_after_cotp
	config_init={'indices':positive_extent_to_consider,'FULL_SUPPORT':sup_full_after_cotp,'FULL_SUPPORT_BITSET':0,'FULL_SUPPORT_INFOS':dataset,'alpha':alpha_ratio_class,'positive_extent':positive_extent_to_consider,'negative_extent':negative_extent_to_consider,'parent':value_to_yield_complex(attributeClosed,0),'current_depth':0}
	
	for (p,l,cnf) in enumerator_complex_cbo_init_new_config(dataset, attributes_types,threshold=threshold,config_init=config_init,initValues=initValues,verbose=True,nobitset=nobitset):
		if cnf['current_depth']>depthmax:
			cnf['flag']=False
			continue

		#if 'parent' in cnf:

		# print 'pattern : ',p,'parent : ',cnf['parent']
		# raw_input('...')
		##########COMPUTING_SUPPORT_FULL#############	
		for id_attr,(a1,a2) in enumerate(zip(cnf['attributes'],attributeClosed)): 
			a2['refinement_index']=a1['refinement_index']
			a2['pattern']=p[id_attr]
		

		cnf['FULL_SUPPORT_INFOS'],cnf['FULL_SUPPORT'],cnf['FULL_SUPPORT_BITSET']=compute_support_complex_index_with_bitset(attributeClosed,dataset,cnf['FULL_SUPPORT'],cnf['FULL_SUPPORT_BITSET'],allindex_full,cnf['refinement_index'],wholeDataset=dataset,closed=False)
		for ai in range(cnf['refinement_index']+1,len(attributes)):
			parent_v=cnf['parent'][ai];pattern_v=p[ai];type_v=attributes_types[ai]['type']
			# if pattern_equal_pattern_on_a_single_attribute(pattern_v,parent_v,type_v):
			# 	continue
			cnf['FULL_SUPPORT_INFOS'],cnf['FULL_SUPPORT'],cnf['FULL_SUPPORT_BITSET']=compute_support_complex_index_with_bitset(attributeClosed,dataset,cnf['FULL_SUPPORT'],cnf['FULL_SUPPORT_BITSET'],allindex_full,ai,wholeDataset=dataset,closed=False)
		

		pattern_infos={
			'support_full':cnf['FULL_SUPPORT'],
			'support_full_bitset':cnf['FULL_SUPPORT_BITSET'],
			'support_positive':cnf['indices'],#cnf['indices'],
			'support_positive_bitset':cnf['indices_bitset'],
			'tpr':len(cnf['indices'])/nb_pos_extent,
			'fpr':0. if nb_neg_extent==0 else (len(cnf['FULL_SUPPORT']&negative_extent))/nb_neg_extent,
			'support_size':len(cnf['FULL_SUPPORT']),
			'alpha':alpha_ratio_class
		}
		yield p,l,pattern_infos,cnf
		
		cnf['current_depth']=cnf['current_depth']+1

	#print ('')

def post_processing_top_k(patterns_set,positive_extent,negative_extent,k=3,timebudget=3600):
	len_all_dataset=float(len(positive_extent)+len(negative_extent))
	FINISHED=True
	startus=time()
	Pattern_set=[]
	alpha=len(positive_extent)/len_all_dataset
	retrieved_top_k=0
	union_of_all_patterns=set()
	current_quality=0.
	while retrieved_top_k<k:
		
		maximizing=0
		current_best=None


		for p in patterns_set:
			current_support=p[1]['support_full']#-union_of_all_patterns
			test_union=union_of_all_patterns|current_support
			tpr_union=float(len(test_union&positive_extent))/len(positive_extent)
			fpr_union=float(len(test_union&negative_extent))/len(negative_extent)

			quality_union=wracc(tpr_union,fpr_union,alpha)
			#p[2]=quality_union-current_quality
			#print p[0],quality_union-current_quality
			if quality_union-current_quality>maximizing:
				current_best=(p[0],p[1],quality_union-current_quality)


				maximizing=quality_union-current_quality
				# if time()-startus>timebudget:
				# 	break

		


		if current_best is None:
			break

		current_best_support=current_best[1]['support_full']
		union_of_all_patterns|=	current_best_support
		Pattern_set.append(current_best)
		current_quality=wracc(len(union_of_all_patterns&positive_extent)/float(len(positive_extent)),len(union_of_all_patterns&negative_extent)/float(len(negative_extent)),alpha)
		retrieved_top_k+=1

		if time()-startus>timebudget:
			FINISHED=False
			break
	
	pattern_union_info={
		'support_full':union_of_all_patterns,
		'support_positive':union_of_all_patterns&positive_extent,#cnf['indices'],
		'tpr':len(union_of_all_patterns&positive_extent)/float(len(positive_extent)),
		'fpr':0. if len(negative_extent)==0 else len(union_of_all_patterns&negative_extent)/float(len(negative_extent)),
		'support_size':len(union_of_all_patterns),
		'alpha':alpha,
		'quality':wracc(len(union_of_all_patterns&positive_extent)/float(len(positive_extent)),len(union_of_all_patterns&negative_extent)/float(len(negative_extent)),alpha),
		'finished':FINISHED

	}
	return Pattern_set,pattern_union_info




def p_1_less_relevant_than_p_2(p1_pos,p1_neg,p2_pos,p2_neg):
	if p1_pos<=p2_pos and p1_neg>=p2_neg:
		return True
	return False

def del_from_list_by_index(l,del_indexes):
	del_indexes_new_indexes=[];del_indexes_new_indexes_append=del_indexes_new_indexes.append
	if len(del_indexes):
		#print (l[del_indexes[0]][0])
		del l[del_indexes[0]]
		del_indexes_new_indexes_append(del_indexes[0])
		for k in range(1,len(del_indexes)):
			del_indexes_new_indexes_append((del_indexes[k]-del_indexes[k-1])+del_indexes_new_indexes[k-1]-1)
			#print (l[del_indexes_new_indexes[-1]][0])
			del l[del_indexes_new_indexes[-1]]#l[del_indexes[k]-del_indexes[k-1]]

def iterator_combinations_needed(nb_patterns,k=3):
	return chain(*[combinations(range(nb_patterns),i) for i in range(1,k+1)])

def combin(n, k):
    if k > n//2:
        k = n-k
    x = 1
    y = 1
    i = n-k+1
    while i <= n:
        x = (x*i)//y
        y += 1
        i += 1
    return x


def post_processing_top_k_groundtruth(patterns_set,positive_extent,negative_extent,k=3,timebudget=3600):
	FINISHED=True
	start=time()
	len_all_dataset=float(len(positive_extent)+len(negative_extent))
	Pattern_set=[]
	alpha=len(positive_extent)/len_all_dataset
	retrieved_top_k=0
	union_of_all_patterns=set()
	current_quality=0.
	to_delete=[]
	#print (len(patterns_set))
	for i in range(0,len(patterns_set)-1):
		p_i_sup=patterns_set[i][1]['support_full']
		p_i_sup_pos=p_i_sup&positive_extent
		p_i_sup_neg=p_i_sup&negative_extent
		remove_i=False
		for j in range(i+1,len(patterns_set)):
			p_j_sup=patterns_set[j][1]['support_full']
			p_j_sup_pos=p_j_sup&positive_extent
			p_j_sup_neg=p_j_sup&negative_extent


			if p_1_less_relevant_than_p_2(p_i_sup_pos,p_i_sup_neg,p_j_sup_pos,p_j_sup_neg):
				remove_i=True
				#print (i,j)
				#print (patterns_set[i][0])
				to_delete.append(i)
				break
	#print (len(patterns_set))
	#print (to_delete)
	#del_from_list_by_index(patterns_set,to_delete)
	#print (len(patterns_set))

	current_quality=0
	union_of_all_patterns=set()
	nb_op_to_do= float(sum(combin(len(patterns_set),x) for x in range(1,k+1)))
	count=0
	for indices_patterns in iterator_combinations_needed(len(patterns_set),k):
		
		count+=1
		if count%100==0:
			stdout.write('%s\r' % ('Percentage Done : ' + ('%.2f'%((count/nb_op_to_do)*100))+ '%'));stdout.flush();
		
		test_union=set.union(*[patterns_set[x][1]['support_full'] for x in indices_patterns])
		tpr_union=float(len(test_union&positive_extent))/len(positive_extent)
		fpr_union=float(len(test_union&negative_extent))/len(negative_extent)
		quality_union=wracc(tpr_union,fpr_union,alpha)
		if quality_union>current_quality:
			current_best=indices_patterns
			current_quality=quality_union
			union_of_all_patterns=test_union
		if time()-start>timebudget:
			FINISHED=False
			break
	#print (current_quality)
	for i in current_best:
		p=patterns_set[i]
		Pattern_set.append((p[0],p[1],p[2]))
	Pattern_set.sort(key=lambda x:x[2],reverse=True)

	pattern_union_info={
		'support_full':union_of_all_patterns,
		'support_positive':union_of_all_patterns&positive_extent,#cnf['indices'],
		'tpr':len(union_of_all_patterns&positive_extent)/float(len(positive_extent)),
		'fpr':0. if len(negative_extent)==0 else len(union_of_all_patterns&negative_extent)/float(len(negative_extent)),
		'support_size':len(union_of_all_patterns),
		'alpha':alpha,
		'quality':wracc(len(union_of_all_patterns&positive_extent)/float(len(positive_extent)),len(union_of_all_patterns&negative_extent)/float(len(negative_extent)),alpha),
		'finished':FINISHED
	}
	return Pattern_set,pattern_union_info		
		#raw_input('......')
	# for p in Pattern_set:
	# 	print p[0],p[2],p[1]['tpr'],p[1]['fpr'],p[1]['support_size']
	# 	#print len(union_of_all_patterns)
	# print 'UNION PATTERNS QUALITY : ', wracc(len(union_of_all_patterns&positive_extent)/float(len(positive_extent)),len(union_of_all_patterns&negative_extent)/float(len(negative_extent)),alpha)


def find_top_k_subgroups_naive(dataset,attributes,types,class_attribute,wanted_label,k=3,threshold=0,timebudget=3600,depthmax=float('inf')):
	start=time()
	FINISHED=True
	new_dataset,positive_extent,negative_extent,alpha_ratio_class,statistics = transform_dataset(dataset,attributes,class_attribute,wanted_label)
	
	Pattern_set=[]
	union_of_all_patterns=set()
	current_considered_dataset_support=set(range(len(dataset)))
	current_pattern_set_tpr=0.
	current_pattern_set_fpr=0.

	retrieved_top_k=0

	#while retrieved_top_k<k:
	enum=enumerating_closed_candidate_subgroups_with_cotp(dataset,attributes,types,positive_extent,negative_extent,alpha_ratio_class,threshold=threshold,indices_to_consider=current_considered_dataset_support,infos_already_computed=[None,None,{'config':None}],depthmax=depthmax)
	
	(pattern,label,pattern_infos,config)=next(enum)
	best_pattern=(pattern,pattern_infos,wracc(pattern_infos['tpr'],pattern_infos['fpr'],pattern_infos['alpha'])) 
	#print best_pattern[2]
	Pattern_set.append(best_pattern)
	nb=1
	#raw_input('....')
	for (pattern,label,pattern_infos,config) in enum:
		nb+=1
		quality = wracc(pattern_infos['tpr'],pattern_infos['fpr'],pattern_infos['alpha'])

		#print pattern,pattern_infos['support_size'],quality
		Pattern_set.append((pattern,pattern_infos,quality))
		if nb%1000==0:
			if time()-start>timebudget:
				FINISHED=False
				break

		
	patterns_set_to_ret,pattern_union_info=post_processing_top_k(Pattern_set,positive_extent,negative_extent,k,timebudget=timebudget-(time()-start))	
	pattern_union_info['timespent']=time()-start
	pattern_union_info['nb_patterns']=nb
	pattern_union_info['finished']=FINISHED
	for best_pattern in patterns_set_to_ret:
		best_pattern[1]['timespent']=pattern_union_info['timespent']
		best_pattern[1]['nb_patterns']=nb
	return patterns_set_to_ret,pattern_union_info	
	# union_of_all_patterns|=	best_pattern[1]['support_full']

	# current_considered_dataset_support=current_considered_dataset_support-best_pattern[1]['support_full']
	# current_pattern_set_tpr=len(union_of_all_patterns&positive_extent)/float(len(positive_extent))
	# current_pattern_set_fpr=len(union_of_all_patterns&negative_extent)/float(len(negative_extent))

	# print nb,best_pattern[0],best_pattern[2],time()-start
	# print len(union_of_all_patterns),len(current_considered_dataset_support)
	# print wracc(current_pattern_set_tpr,current_pattern_set_fpr,alpha_ratio_class)


def find_top_k_subgroups_groundtruth(dataset,attributes,types,class_attribute,wanted_label,k=3,threshold=0,timebudget=3600,depthmax=float('inf')):
	start=time()
	FINISHED=True
	new_dataset,positive_extent,negative_extent,alpha_ratio_class,statistics = transform_dataset(dataset,attributes,class_attribute,wanted_label)
	
	Pattern_set=[]
	union_of_all_patterns=set()
	current_considered_dataset_support=set(range(len(dataset)))
	current_pattern_set_tpr=0.
	current_pattern_set_fpr=0.

	retrieved_top_k=0

	#while retrieved_top_k<k:
	enum=enumerating_closed_candidate_subgroups_with_cotp(dataset,attributes,types,positive_extent,negative_extent,alpha_ratio_class,threshold=threshold,indices_to_consider=current_considered_dataset_support,infos_already_computed=[None,None,{'config':None}],depthmax=depthmax)
	
	(pattern,label,pattern_infos,config)=next(enum)
	best_pattern=(pattern,pattern_infos,wracc(pattern_infos['tpr'],pattern_infos['fpr'],pattern_infos['alpha'])) 
	#print best_pattern[2]
	Pattern_set.append(best_pattern)
	nb=1
	#raw_input('....')
	for (pattern,label,pattern_infos,config) in enum:
		nb+=1
		quality = wracc(pattern_infos['tpr'],pattern_infos['fpr'],pattern_infos['alpha'])

		#print pattern,pattern_infos['support_size'],quality
		Pattern_set.append((pattern,pattern_infos,quality))
		if nb%1000==0:
			if time()-start>timebudget:
				FINISHED=False
				break

	
	patterns_set_to_ret,pattern_union_info=post_processing_top_k_groundtruth(Pattern_set,positive_extent,negative_extent,k)	
	

	pattern_union_info['timespent']=time()-start
	pattern_union_info['nb_patterns']=nb
	pattern_union_info['finished']=FINISHED
	for best_pattern in patterns_set_to_ret:
		best_pattern[1]['timespent']=pattern_union_info['timespent']
		best_pattern[1]['nb_patterns']=nb
	return patterns_set_to_ret,pattern_union_info	



def find_top_k_subgroups(dataset,attributes,types,class_attribute,wanted_label,k=3,threshold=1,timebudget=3600,depthmax=float('inf')):
	start=time()
	FINISHED=True
	new_dataset,positive_extent,negative_extent,alpha_ratio_class,statistics = transform_dataset(dataset,attributes,class_attribute,wanted_label)
	infos_already_computed=[None,None,{'config':None}]
	Pattern_set=[]
	union_of_all_patterns=set()
	current_considered_dataset_support=set(range(len(dataset)))
	current_pattern_set_tpr=0.
	current_pattern_set_fpr=0.

	retrieved_top_k=0
	nb_all=0
	
	while retrieved_top_k<k and len(current_considered_dataset_support&positive_extent)>0:

		enum=enumerating_closed_candidate_subgroups_with_cotp(dataset,attributes,types,positive_extent,negative_extent,alpha_ratio_class,threshold=threshold,indices_to_consider=current_considered_dataset_support,infos_already_computed=infos_already_computed,depthmax=depthmax)

		#raw_input('**')
		(pattern,label,pattern_infos,config)=next(enum)
		best_pattern=(pattern,pattern_infos,wracc_gain(pattern_infos['tpr'],pattern_infos['fpr'],alpha_ratio_class,current_pattern_set_tpr,current_pattern_set_fpr)) 
		nb=1
		nb_all+=1
		#raw_input('....')
		#print pattern, wracc_and_bound_gain(pattern_infos['tpr'],pattern_infos['fpr'],alpha_ratio_class,current_pattern_set_tpr,current_pattern_set_fpr),pattern_infos['tpr'],pattern_infos['fpr'],pattern_infos['support_size']
		for (pattern,label,pattern_infos,config) in enum:
			nb_all+=1
			nb+=1
			quality,bound = wracc_and_bound_gain(pattern_infos['tpr'],pattern_infos['fpr'],alpha_ratio_class,current_pattern_set_tpr,current_pattern_set_fpr)
			#print (pattern,quality,bound,pattern_infos['tpr'],pattern_infos['fpr'],pattern_infos['support_size'],quality)
			if quality > best_pattern[2]:
				best_pattern=(pattern,pattern_infos,quality)
			if bound <= best_pattern[2]:
				config['flag']=False
			if nb_all%1000==0:
				if time()-start>timebudget:
					FINISHED=False
					break

		if best_pattern[2]<0:
			break
		retrieved_top_k+=1
		best_pattern[1]['timespent']=time()-start
		best_pattern[1]['nb_patterns']=nb
		Pattern_set.append(best_pattern)

		union_of_all_patterns|=	best_pattern[1]['support_full']
		current_considered_dataset_support=current_considered_dataset_support-best_pattern[1]['support_full']

		current_pattern_set_tpr=len(union_of_all_patterns&positive_extent)/float(len(positive_extent))
		current_pattern_set_fpr=len(union_of_all_patterns&negative_extent)/float(len(negative_extent))
		if nb_all%1000==0:
			if time()-start>timebudget:
				FINISHED=False
				break
		#print best_pattern[0],best_pattern[2],len(best_pattern[1]['support_full']),time()-start,nb

		
	pattern_union_info={
		'support_full':union_of_all_patterns,
		'support_positive':union_of_all_patterns&positive_extent,#cnf['indices'],
		'tpr':len(union_of_all_patterns&positive_extent)/float(len(positive_extent)),
		'fpr':0. if len(negative_extent)==0 else len(union_of_all_patterns&negative_extent)/float(len(negative_extent)),
		'support_size':len(union_of_all_patterns),
		'alpha':alpha_ratio_class,
		'quality':wracc(len(union_of_all_patterns&positive_extent)/float(len(positive_extent)),len(union_of_all_patterns&negative_extent)/float(len(negative_extent)),alpha_ratio_class),
		'timespent':time()-start,
		'nb_patterns':nb_all,
		'finished':FINISHED
	}
	return Pattern_set,	pattern_union_info
	#print 'UNION PATTERNS QUALITY : ', wracc(len(union_of_all_patterns&positive_extent)/float(len(positive_extent)),len(union_of_all_patterns&negative_extent)/float(len(negative_extent)),alpha_ratio_class),retrieved_top_k,nb_all


def transform_pattern_set_results_to_print_dataset(dataset,patterns_set,pattern_union_info,attributes,types,class_attribute,wanted_label):
	find_minimal_top_k=True
	_,positive_extent,negative_extent,alpha_ratio_class,_ = transform_dataset(dataset,attributes,class_attribute,wanted_label)
	HEADER=['id_pattern','attributes','pattern','support_size','support_size_ratio','quality','quality_gain','tpr','fpr','nb_patterns','timespent']
	to_return=[]
	id_pattern=0


	dict_additional_labels={}
	for a,t in zip(attributes,types):
		if t=='themes':
			dom=set()
			for o in dataset: dom |=  {v for v in o[a]}
			#print (a,dom)
			dict_additional_labels[a]=get_domain_from_dataset_theme(dom)[1]

	for p in patterns_set:
		
		#print p[0],p[2],p[1]['tpr'],p[1]['fpr'],p[1]['support_size']
		#print(attributes,types)
		filtering_pipeline=[]
		for p_i,a_i,t_i in zip(p[0],attributes,types):
			if t_i=='numeric':
				filtering_pipeline.append({'dimensionName':a_i,'inInterval':p_i})
			elif t_i=='simple':
				if (len(p_i)==1):
					filtering_pipeline.append({'dimensionName':a_i,'inSet':p_i})
			elif t_i == 'themes':
				filtering_pipeline.append({'dimensionName':a_i,'contain_themes':p_i})
			else:
				filtering_pipeline.append({'dimensionName':a_i,'inSet':p_i})

		#filtering_pipeline=[{'dimensionName':a_i,'inInterval':p_i} if t_i=='numeric' else {'dimensionName':a_i,'inSet':p_i} for p_i,a_i,t_i in zip(p[0],attributes,types) if (t_i!='simple' or not (t_i=='simple' and len(p_i)>1 ) )]
		pattern_to_yield=[p_i if t_i=='numeric' else (p_i[0] if len(p_i)==1 else '*') if t_i=='simple' else ([dict_additional_labels[a_i][p_i_v] for p_i_v in p_i if p_i_v!=''] if len(p_i)>1 else '*') if t_i=='themes' else p_i for p_i,a_i,t_i in zip(p[0],attributes,types)]
		#print filtering_pipeline
		support_recomputed,support_recomputed_indices=filter_pipeline_obj(dataset, filtering_pipeline)
		#print (dataset[0])

		tpr = len(support_recomputed_indices&positive_extent)/float(len(positive_extent))
		fpr = len(support_recomputed_indices&negative_extent)/float(len(negative_extent))
		to_return.append({
			'id_pattern':id_pattern,
			'attributes':attributes,
			'pattern':pattern_to_yield,
			'support_size':len(support_recomputed_indices),
			'support_size_ratio':len(support_recomputed_indices)/float(len(dataset)),
			'quality' : wracc(tpr,fpr,alpha_ratio_class),
			'quality_gain' : p[2],
			'tpr':tpr,
			'fpr':fpr,
			'timespent':p[1]['timespent'],
			'real_support':encode_sup(support_recomputed_indices,len(dataset)),
			'support':support_recomputed_indices,
			'nb_patterns':p[1]['nb_patterns'],
			'finished':pattern_union_info.get('finished',True),
		})

		id_pattern+=1


	tpr=len(pattern_union_info['support_full']&positive_extent)/float(len(positive_extent))
	fpr=len(pattern_union_info['support_full']&negative_extent)/float(len(negative_extent))
	
	union_pattern={
			'id_pattern':'SubgroupSet',
			'attributes':attributes,
			'pattern':'-',
			'support_size':len(pattern_union_info['support_full']),
			'support_size_ratio':len(pattern_union_info['support_full'])/float(len(dataset)),
			'quality' : pattern_union_info['quality'],
			'quality_gain' : pattern_union_info['quality'],
			'tpr':tpr,
			'fpr':fpr,
			'timespent':pattern_union_info['timespent'],
			'real_support':encode_sup(pattern_union_info['support_full'],len(dataset)),
			'alpha':alpha_ratio_class,
			'support':pattern_union_info['support_full'],
			'nb_patterns':pattern_union_info['nb_patterns'],
			'finished':pattern_union_info.get('finished',True),

		}

	

	if find_minimal_top_k and len(to_return)>1:
		quality_union=union_pattern['quality']
		union_support=union_pattern['support']
		something_removed=True
		to_delete=None
		while something_removed:
			something_removed=False
			for i in range(len(to_return)):
				union_with_pattern_i_eliminated=set.union(*[to_return[x]['support'] for x in range(len(to_return)) if x!=i])
				# tpr=len(union_with_pattern_i_eliminated&positive_extent)/float(len(positive_extent))
				# fpr=len(union_with_pattern_i_eliminated&negative_extent)/float(len(negative_extent))
				# quality=wracc(tpr,fpr,alpha_ratio_class)
				#if quality_union==quality:
				if union_support==union_with_pattern_i_eliminated:
					something_removed=True
					to_delete=i
			if something_removed:
				del to_return[to_delete]


	to_return.append(union_pattern)

	return to_return,HEADER


def all_distinct(l):
	return len(set(l))==len(l)

#@profile(precision=10)
def find_top_k_subgroups_general(dataset,attributes,types,class_attribute,wanted_label,k=5,method='fssd',timebudget=3600,depthmax=float('inf')):
    method_to_use=find_top_k_subgroups if method=='fssd' else  find_top_k_subgroups_naive if method=='naive' else find_top_k_subgroups_groundtruth if method=='groundtruth' else find_top_k_subgroups
    patterns_set,pattern_union_info=method_to_use(dataset,attributes,types,class_attribute,wanted_label,k,timebudget=timebudget,depthmax=depthmax)
    returned_to_write,header=transform_pattern_set_results_to_print_dataset(dataset,patterns_set,pattern_union_info,attributes,types,class_attribute,wanted_label)
    return patterns_set,pattern_union_info,returned_to_write,header




def pre_treatement_for_depthmax_and_complex_categorical(dataset,attributes,types,class_attribute,wanted_label,k=5,method='fssd',consider_richer_categorical_language=False,timebudget=3600,depthmax=float('inf')):
	if consider_richer_categorical_language:
		types=['themes' if x=='simple' or x=='nominal' else x for x in types]
		for a,t in zip(attributes,types):
			if t=='themes':
				domain=sorted({row[a] for row in  dataset})
				domain_to_indices={v:i+1 for i,v in enumerate(domain)}
				domain_new_names={v:[str(domain_to_indices[v]).zfill(3)+' '+str(v)]+[str(100+domain_to_indices[vnot]).zfill(3)+' not '+str(vnot) for vnot in domain if vnot!=v] for v in domain}
				for row in dataset:
					row[a]=domain_new_names[row[a]]
	attributes_to_reconsider_descs=[False]*len(attributes)
	attributes_to_reconsider_descs_discretizations={}
	if depthmax< float('inf'):
		discretize_sophistically=True
		indice=0
		for a,t in zip(attributes,types):
			
			if t == 'numeric':
				_,positive_extent,negative_extent,_,_ = transform_dataset(dataset,attributes,class_attribute,wanted_label)
				domain=sorted({dataset[x][a] for x in range(len(dataset))})
				domain_pos=sorted({dataset[x][a] for x in positive_extent}|{domain[-1]})
				domain_pos_flattened=sorted([dataset[x][a] for x in positive_extent])
				domain_pos_and_its_next={domain_pos[i]:domain_pos[i+1] if i<len(domain_pos)-1 else domain_pos[i] for i in range(len(domain_pos))}
				if depthmax<len(domain_pos):
					discretized_domain_pos_tmp=[domain_pos_flattened[int(x/float(depthmax-1) * (len(domain_pos_flattened)))] for x in range(int(depthmax)-1)]+[domain_pos[-1]]
					
					nb_try=10
					while not all_distinct(discretized_domain_pos_tmp):
						nb_try=nb_try-1
						for i in range(len(discretized_domain_pos_tmp)-1):
							if discretized_domain_pos_tmp[i+1]==discretized_domain_pos_tmp[i]:
								discretized_domain_pos_tmp[i+1]=domain_pos_and_its_next[discretized_domain_pos_tmp[i+1]]
						if nb_try<=0: 
							break

						

					if discretize_sophistically:
						discretized_domain_pos=[domain_pos[int(x/float(depthmax-1) * (len(domain_pos)))] for x in range(int(depthmax)-1)]+[domain_pos[-1]]
					else:
						discretized_domain_pos=sorted(set(discretized_domain_pos_tmp))

					discretized_domain_pos_transform={ x:bisect_left(discretized_domain_pos,x) for x in domain}
					discretized_domain_pos_transform={x:discretized_domain_pos[y-1] if y>=1 else 0.  for x,y in discretized_domain_pos_transform.items()}
					discretized_domain_pos_transform={x:x if x in discretized_domain_pos else y for x,y in discretized_domain_pos_transform.items() }
					#print (discretized_domain_pos_transform)
					for row in dataset:
						#print (row[a],discretized_domain_pos_transform[row[a]])
						row[a]= discretized_domain_pos_transform[row[a]]
						#print (row[a])
					attributes_to_reconsider_descs[indice]=True
					discretized_domain_pos_transform_reversed={}
					for k,v in discretized_domain_pos_transform.items():
						discretized_domain_pos_transform_reversed[v]=discretized_domain_pos_transform_reversed.get(v,[])+[k]

					for k in discretized_domain_pos_transform_reversed:
						tmp_sorted=sorted(discretized_domain_pos_transform_reversed[k])
						discretized_domain_pos_transform_reversed[k]=[tmp_sorted[0],tmp_sorted[-1]]
					#print (discretized_domain_pos_transform_reversed)
					#input('....')
					attributes_to_reconsider_descs_discretizations[a]=discretized_domain_pos_transform_reversed #discretization, domain
				else:
					discretized_domain_pos=domain_pos
					

				#print (discretized_domain_pos)

				#input('.........')
			indice+=1
	return dataset,attributes,types,attributes_to_reconsider_descs_discretizations

def find_top_k_subgroups_general_precall(dataset,attributes,types,class_attribute,wanted_label,k=5,method='fssd',consider_richer_categorical_language=False,timebudget=3600,depthmax=float('inf')):
	# if consider_richer_categorical_language:
	# 	types=['themes' if x=='simple' or x=='nominal' else x for x in types]
	# 	for a,t in zip(attributes,types):
	# 		if t=='themes':
	# 			domain=sorted({row[a] for row in  dataset})
	# 			domain_to_indices={v:i+1 for i,v in enumerate(domain)}
	# 			domain_new_names={v:[str(domain_to_indices[v]).zfill(3)+' '+str(v)]+[str(100+domain_to_indices[vnot]).zfill(3)+' not '+str(vnot) for vnot in domain if vnot!=v] for v in domain}
	# 			for row in dataset:
	# 				row[a]=domain_new_names[row[a]]
	# attributes_to_reconsider_descs=[False]*len(attributes)
	# attributes_to_reconsider_descs_discretizations={}
	# if depthmax< float('inf'):
	# 	discretize_sophistically=True
	# 	indice=0
	# 	for a,t in zip(attributes,types):
			
	# 		if t == 'numeric':
	# 			_,positive_extent,negative_extent,_,_ = transform_dataset(dataset,attributes,class_attribute,wanted_label)
	# 			domain=sorted({dataset[x][a] for x in range(len(dataset))})
	# 			domain_pos=sorted({dataset[x][a] for x in positive_extent}|{domain[-1]})
	# 			domain_pos_flattened=sorted([dataset[x][a] for x in positive_extent])
	# 			domain_pos_and_its_next={domain_pos[i]:domain_pos[i+1] if i<len(domain_pos)-1 else domain_pos[i] for i in range(len(domain_pos))}
	# 			if depthmax<len(domain_pos):
	# 				discretized_domain_pos_tmp=[domain_pos_flattened[int(x/float(depthmax-1) * (len(domain_pos_flattened)))] for x in range(int(depthmax)-1)]+[domain_pos[-1]]
					
	# 				nb_try=10
	# 				while not all_distinct(discretized_domain_pos_tmp):
	# 					nb_try=nb_try-1
	# 					for i in range(len(discretized_domain_pos_tmp)-1):
	# 						if discretized_domain_pos_tmp[i+1]==discretized_domain_pos_tmp[i]:
	# 							discretized_domain_pos_tmp[i+1]=domain_pos_and_its_next[discretized_domain_pos_tmp[i+1]]
	# 					if nb_try<=0: 
	# 						break

						

	# 				if discretize_sophistically:
	# 					discretized_domain_pos=[domain_pos[int(x/float(depthmax-1) * (len(domain_pos)))] for x in range(int(depthmax)-1)]+[domain_pos[-1]]
	# 				else:
	# 					discretized_domain_pos=sorted(set(discretized_domain_pos_tmp))

	# 				discretized_domain_pos_transform={ x:bisect_left(discretized_domain_pos,x) for x in domain}
	# 				discretized_domain_pos_transform={x:discretized_domain_pos[y-1] if y>=1 else 0.  for x,y in discretized_domain_pos_transform.items()}
	# 				discretized_domain_pos_transform={x:x if x in discretized_domain_pos else y for x,y in discretized_domain_pos_transform.items() }
	# 				#print (discretized_domain_pos_transform)
	# 				for row in dataset:
	# 					#print (row[a],discretized_domain_pos_transform[row[a]])
	# 					row[a]= discretized_domain_pos_transform[row[a]]
	# 					#print (row[a])
	# 				attributes_to_reconsider_descs[indice]=True
	# 				discretized_domain_pos_transform_reversed={}
	# 				for k,v in discretized_domain_pos_transform.items():
	# 					discretized_domain_pos_transform_reversed[v]=discretized_domain_pos_transform_reversed.get(v,[])+[k]

	# 				for k in discretized_domain_pos_transform_reversed:
	# 					tmp_sorted=sorted(discretized_domain_pos_transform_reversed[k])
	# 					discretized_domain_pos_transform_reversed[k]=[tmp_sorted[0],tmp_sorted[-1]]
	# 				#print (discretized_domain_pos_transform_reversed)
	# 				#input('....')
	# 				attributes_to_reconsider_descs_discretizations[a]=discretized_domain_pos_transform_reversed #discretization, domain
	# 			else:
	# 				discretized_domain_pos=domain_pos
					

	# 			#print (discretized_domain_pos)

	# 			#input('.........')
	# 		indice+=1
	dataset,attributes,types,attributes_to_reconsider_descs_discretizations=pre_treatement_for_depthmax_and_complex_categorical(dataset,attributes,types,class_attribute,wanted_label,k,method,consider_richer_categorical_language,timebudget,depthmax)


	patterns_set,pattern_union_info,returned_to_write,header=find_top_k_subgroups_general(dataset,attributes,types,class_attribute,wanted_label,k,method,timebudget=timebudget,depthmax=depthmax+10)
	if len(attributes_to_reconsider_descs_discretizations):#any(attributes_to_reconsider_descs):
		for rowind in range(len(returned_to_write)-1):
			row=returned_to_write[rowind]
			for i,a in enumerate(row['attributes']):
				if a in attributes_to_reconsider_descs_discretizations:
					#print(attributes_to_reconsider_descs_discretizations[a])
					values_real_corresponding=attributes_to_reconsider_descs_discretizations[a]
					row['pattern'][i]=[values_real_corresponding[row['pattern'][i][0]][0],values_real_corresponding[row['pattern'][i][1]][-1]]
					#print([values_real_corresponding[row['pattern'][i][0]][0],values_real_corresponding[row['pattern'][i][1]][-1]])



	return patterns_set,pattern_union_info,returned_to_write,header


def transform_dataset_to_attributes(file,class_attribute,delimiter=',',SIMPLE_TO_NOMINAL=False):
	dataset,header=readCSVwithHeader(file,delimiter=delimiter)
	#################################################
	row=dataset[0]
	attribute_parsed=[]
	types_parsed=[]
	for k in header:
		v=row[k]
		if k != class_attribute:
			attribute_parsed.append(k)
			try:
				float(v)
				types_parsed.append('numeric')
			except Exception as e:
				if SIMPLE_TO_NOMINAL:
					types_parsed.append('nominal')

				else:
					types_parsed.append('simple')

	attributes=attribute_parsed
	types=types_parsed
	return attributes,types













def read_file_conf(source):
	
	with open(source, 'r') as csvfile:
		readfile = csv.reader(csvfile, delimiter='\t')
		results=[row for row in readfile if len(row)>0]
	return results



def get_stat_dataset(dataset_file):
	delimiter='\t'
	dataset,header=readCSVwithHeader(dataset_file,delimiter=delimiter)
	class_attribute=header[-1]
	attributes,types=transform_dataset_to_attributes(dataset_file,class_attribute,delimiter=delimiter)
	dataset,header=readCSVwithHeader(dataset_file,numberHeader=[a for a,t in zip(attributes,types) if t=='numeric'],delimiter=delimiter)
	
	statistics=[]
	

	classes=set(x[class_attribute] for x in dataset)
	for wanted_label in classes:
		alpha_ratio_class=0.
		positive_extent=set()
		negative_extent=set()
		statistics_one_dataset={}
		for k in range(len(dataset)):
			row=dataset[k]
			new_row={attr_name:row[attr_name] for attr_name in attributes}
			new_row['positive']=int(row[class_attribute]==wanted_label)
			new_row[class_attribute]=row[class_attribute]
			if new_row['positive']:
				positive_extent|={k}
				alpha_ratio_class+=1
			else:
				negative_extent|={k}
		statistics_one_dataset['dataset']=splitext(basename(dataset_file))[0]
		statistics_one_dataset['rows']=len(dataset)
		statistics_one_dataset['class_attribute']=class_attribute
		statistics_one_dataset['class']=wanted_label
		statistics_one_dataset['alpha']=alpha_ratio_class/float(len(dataset))
		statistics_one_dataset['nb_attributes']=len(attributes)
		statistics_one_dataset['categoric']=len([x for x,t in zip(attributes,types) if t=='simple'])
		statistics_one_dataset['numeric']=len([x for x,t in zip(attributes,types) if t=='numeric'])
		statistics.append(statistics_one_dataset)
	return statistics