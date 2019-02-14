'''
@author: Matt
@description: A set of helper functions for working with machine learning models 
'''

import csv
import math
import numpy as np

from matplotlib_venn import venn2
import matplotlib.pyplot as plt
plt.style.use('default')

def preprocess(filename, normalize=True):
	'''
	Args: filename (path string), normalize (Boolean, default True)
	Return: a tuple of numpy arrays: a matrix of the shape (entries, features) for features, labels with the shape (entries, 1)
	'''
	with open(filename, 'r', newline='') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		header = next(reader)
		#minus two because of client_# and classification
		num_features = len(header) - 2
		rows = []
		results = []

		for row in reader:
			#convert strings 'True' and 'False' to 1 or 0
			results.append(int(row.pop().lower() in ["true"]))
			rows.append(np.array(row, dtype='float64'))

	features = np.array(rows)
	features = features[:,1:]
	
	client_ids = features[:,0]
	labels = np.array(results).reshape(len(results), 1).astype('int')
	
	if normalize:
		means = features.mean(axis=0)
		stds = features.std(axis=0)

		#calculate the z-scores, this can be cleaned up with a numpy function
		for entry in features:
			for i in range(features.shape[1]):
				#set the z-score
				entry[i] = (entry[i] - means[i]) / stds[i]
				if math.isnan(entry[i]):
					entry[i] = 0.0
				
	return features, labels

def decision_boundary(prob):
	return 1 if prob >= 0.5 else 0

def classify(predictions):
	decision = np.vectorize(decision_boundary)
	return decision(predictions).flatten()

def confusion_matrix(predicted, actual):
	true_pos, true_neg ,false_pos, false_neg = 0, 0, 0, 0
		
	for i in range(len(predicted)):
		if predicted[i] == actual[i] and predicted[i] == 1:
			true_pos += 1
		elif predicted[i] == actual[i] and predicted[i] == 0:
			true_neg += 1
		elif predicted[i] != actual[i] and predicted[i] == 1:
			false_pos += 1
		elif predicted[i] != actual[i] and predicted[i] == 0:
			false_neg += 1
	
	return true_pos, true_neg, false_pos, false_neg

# aka cross_entropy
# penalizes the classifier based on how far off it is
def log_loss(predictions, labels):
    N = predictions.shape[0]
    ce = -np.sum(labels * np.log(predictions))/N
    return ce

def performance(estimated, actual, verbose=True, show_visual=False):
	'''
	Args: estimated: an array of estimated output probabilities, actual: actual output classification
	Returns: prints accuracy, and if verbose=True, show_visual=True, other metrics
	'''
	#loss = log_loss(estimated, actual)
	est_labels = classify(estimated)
	
	tp, tn, fp, fn = confusion_matrix(est_labels, actual)
	accuracy = (tp + tn) / actual.size
	error = (fp + fn) / actual.size
	
	# when it's actually 1, how often does it predict 1? (sensitivity / recall)
	actual_counts = np.unique(actual, return_counts=True) # (array([0, 1]), array([5529, 6471]))
	try:
		recall = tp / actual_counts[1][1] # TP/actual yes
	except:
		recall = tp / actual_counts[1][0] # TP/actual yes
	
	# when it predicts 1, how often is it correct?
	pred_counts = np.unique(est_labels, return_counts=True)
	try:
		precision = tp / pred_counts[1][1] # TP/predicted yes
	except:
		precision = tp / pred_counts[1][0] # TP/predicted yes
	
	if verbose:
		print ('Accuracy: ', accuracy)
		print ('Error Rate ', error)
		#print ('Log Loss: ', loss)
		print ('Recall: ', recall)
		print ('Precision: ', precision)
		print ('Total entries: ', actual.size)
		print ('True Positive: ', tp)
		print ('True Negative: ', tn)
		print ('False Positive: ', fp)
		print ('False Negative: ', fn)

	if show_visual:
		visualize(estimated, actual)

	return accuracy

#with 100ths place precision
def split_data(features, labels, train, test):
	'''
	Args: features, lables, train percentage, test percentage
	Returns: train_x, train_y, test_x, test_y
	'''
	if (train + test) != 100:
		raise Exception('Testing and training percentages should add to 100')

	rand_state = np.random.get_state()
	np.random.shuffle(features)
	np.random.set_state(rand_state)
	np.random.shuffle(labels)
	
	features = np.split(features, 100, axis=0)
	labels = np.split(labels, 100, axis=0)
	
	train_x = np.concatenate(features[0:train])
	train_y = np.concatenate(labels[0:train])
	test_x = np.concatenate(features[train:])
	test_y = np.concatenate(labels[train:])
	
	return train_x, train_y, test_x, test_y

def visualize(estimated, test_y):
	'''
	Args: estimated labels/probabilities, test_y
	Returns: prints a performance venn diagram
	'''
	tp, tn, fp, fn = confusion_matrix(classify(estimated), test_y)
	fn_circle = max(fn, fn-tp)
	fp_cicle = max(fp, fp-tp)
	diagram = venn2(subsets=(fn_circle, fp_cicle, tp), set_labels=('','',''), set_colors=('#5791c6','#d8b66e'))
	diagram.get_label_by_id('10').set_text('FN')
	diagram.get_label_by_id('11').set_text('TP')
	diagram.get_label_by_id('01').set_text('FP')


#roadmap before starting presentation
#have a story to tell
#try an make it understandable by those not as technical
#see the thinking process, sell our function
#be systematic abotu research approach