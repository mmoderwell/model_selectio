'''
@author: Matt
@description: A set of functions used for analyzing machine learning model performance
'''

import math
import numpy as np
import seaborn as sns

from matplotlib_venn import venn2
import matplotlib.pyplot as plt
plt.style.use('default')


def decision_boundary(prob):
	return 1 if prob >= 0.5 else 0


def classify(predictions):
	decision = np.vectorize(decision_boundary)
	return decision(predictions).flatten()


def confusion_matrix(predicted, actual):
	'''
	Args: predicted: an array of estimated output classifications, actual: actual output classification
	Returns: true_pos, true_neg, false_pos, false_neg
	'''
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


def performance_visual(estimated, test_y):
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


def zeroR(actual):
	'''
	Args: actual classifications of test set
	Returns: accuracy given the most common class is predicted every time
	'''
	counts = np.unique(actual, return_counts=True)
	index_max = np.argmax(counts[1])

	if counts[0][index_max] == 1:
		estimated = np.ones((len(actual), 1))
	elif counts[0][index_max] == 0:
		estimated = np.zeros((len(actual), 1))

	tp, tn, fp, fn = confusion_matrix(estimated, actual)
	accuracy = (tp + tn) / actual.size
	return accuracy

def density_plot(probs):
	ax = sns.distplot(probs, hist=True, kde=True, bins=50, color='cornflowerblue', hist_kws={'edgecolor':'cornflowerblue'},kde_kws={'linewidth': 4})
	ax.set(xlabel='Probability', ylabel='Density')
	return ax

def distribution(estimated, actual, precision, verbose=True):
	'''
	Args: estimated: an array of estimated output probabilities, actual: actual output classification, precision (int)
	Returns: plotted distribution of probabilities vs. # at the certainty
	'''
	pred_counts = clean_counts(estimated, precision)

	observations_mean = np.mean(pred_counts[1])
	observations_std = np.std(pred_counts[1])

	mean = np.mean(estimated)
	min = np.min(estimated)
	max = np.max(estimated)

	if verbose:
		print ("Mean Prediction: ", mean)
		print ("Min Prediction: ", min)
		print ("Max Prediction: ", max)

	#if precision is above 3, make points smaller
	def size_calc(precision):
		return 4 if precision >= 3 else 10
		
	size = size_calc(precision)

	# plot the points as scatter plot
	plt.figure(figsize=(12,8), frameon=False)
	plt.scatter(pred_counts[0], pred_counts[1], label= "Model", color ='k', marker = "o", s = size)

	plt.axhline(y=(observations_mean + observations_std), linewidth=2, color='#e8ecf2', label="Mean + std")
	plt.axhline(y=(observations_mean - observations_std), linewidth=2, color='#e8ecf2', label="Mean - std")
	plt.axhline(y=observations_mean, linewidth=2, color='#f7dcd2', label="Mean")

	# add the labels
	plt.xlabel('Probability')
	plt.ylabel('Number of predictions')
	plt.title('Probability vs. # of predictions')
	plt.legend()
	# show the plot
	plt.show()


def clean_counts(estimated, precision):
	'''
	Args: estimated: an array of estimated output probabilities, (int) decimal place precision
	Returns: a numpy array with two sub arrays, [0] array of probabilities, [1] array of corresponding counts at that probability
	'''
	rounded_est = np.around(estimated, precision)
	pred_counts = np.unique(rounded_est, return_counts=True)
	cleaned = [[],[]]

	# fill in missing bands and set number of observations to 0
	bands = [round(float(band), precision) for band in pred_counts[0]] # 0.0, 0.01, 0.02, ... , 1.0
	counts = [int(count) for count in pred_counts[1]]

	for i in range(0, ((10 ** precision) + 1)):
		if (i * (10 ** -precision)) not in bands:
			bands.append(round(float(i * (1 * 10 ** -precision)), 2))
			counts.append(0)

	pred_counts_dict = dict(zip(bands, counts)) 
	for key in sorted(pred_counts_dict):
		cleaned[0].append(key)
		cleaned[1].append(pred_counts_dict[key])

	return np.array([np.array(cleaned[0]), np.array(cleaned[1])])


def distribution_metric(estimated, actual, precision=2, visualize=True, verbose=True):
	'''
	Args: estimated: an array of estimated output probabilities, actual: actual output classification, precision (int), visualize=Bool
	Returns: the calculated percentage of predictions outside of 1 standard deviation from the mean number of predictions at each probability
	'''
	if visualize:
		distribution(estimated, actual, precision, verbose=bool(verbose))
		#density_plot(estimated)

	pred_counts = clean_counts(estimated, precision)

	#get the mean # of predictions across probability bands
	observations_mean = np.mean(pred_counts[1])
	observations_std = np.std(pred_counts[1])
	outside =  pred_counts[1][(pred_counts[1] > (observations_mean + observations_std)) | (pred_counts[1] < (observations_mean - observations_std))]
	percent_outside = (np.sum(outside) / len(actual)) * 100
	print ('{0}% of observations are outside of 1 standard deviation from the mean'.format(round(percent_outside, 2)))


def performance(estimated, actual, visualize=True, verbose=True):
	'''
	Args: estimated: an array of estimated output probabilities, actual: actual output classification
	Returns: returns accuracy, and if verbose=True and/or show_visual=True, other metrics
	'''
	#loss = log_loss(estimated, actual)
	zero = zeroR(actual)

	est_labels = classify(estimated)
	
	tp, tn, fp, fn = confusion_matrix(est_labels, actual)
	accuracy = (tp + tn) / actual.size
	error = (fp + fn) / actual.size
	fpr = fp / (fp + tn) * 100
	fnr = fn / (tp + fn) * 100
	
	# when it's actually 1, how often does it predict 1? (sensitivity / recall)
	actual_counts = np.unique(actual, return_counts=True) # (array([0, 1]), array([5529, 6471]))
	recall = tp / actual_counts[1][1] # TP/actual yes
	
	# when it predicts 1, how often is it correct?
	pred_counts = np.unique(est_labels, return_counts=True)
	try:
		precision = tp / pred_counts[1][1] # TP/predicted yes
	except:
		precision = tp / pred_counts[1][0] # TP/predicted yes
	
	if verbose:
		print ('Accuracy: ', accuracy)
		print ('ZeroR: ', zero)
		#print ('Error Rate ', error)
		#print ('Log Loss: ', loss)
		print ('Recall: ', recall)
		print ('Precision: ', precision)
		print ('Total entries: ', actual.size)
		print ('True Positive: ', tp, '    ({0}%)'.format(round((tp/actual.size) * 100), 1))
		print ('True Negative: ', tn, '    ({0}%)'.format(round((tn/actual.size) * 100), 1))
		print ('False Positive: ', fp, '   ({0}%)'.format(round((fp/actual.size) * 100), 1))
		print ('False Negative: ', fn, '   ({0}%)'.format(round((fn/actual.size) * 100), 1))
		print ('False Positive Rate: ', fpr)
		print ('False Negative Rate: ', fnr)

	if visualize:
		performance_visual(estimated, actual)

	return accuracy


def detailed(estimated, actual):
	'''
	Args: estimated: an array of estimated output probabilities, actual: actual output classification
	Returns: prints a summary of model metrics
	'''

	#loss = log_loss(estimated, actual)
	zero = zeroR(actual)
	est_labels = classify(estimated)
	
	tp, tn, fp, fn = confusion_matrix(est_labels, actual)
	accuracy = (tp + tn) / actual.size
	error = (fp + fn) / actual.size
	
	# when it's actually 1, how often does it predict 1? (sensitivity / recall)
	actual_counts = np.unique(actual, return_counts=True) # (array([0, 1]), array([5529, 6471]))
	recall = tp / actual_counts[1][1] # TP/actual yes
	
	# when it predicts 1, how often is it correct?
	pred_counts = np.unique(est_labels, return_counts=True)
	try:
		precision = tp / pred_counts[1][1] # TP/predicted yes
	except:
		precision = tp / pred_counts[1][0] # TP/predicted yes

	increase = (( accuracy - zero ) / zero) * 100
	print ('Beat ZeroR accuracy by:', round((accuracy - zero), 3))
	print ('    ', round(zero, 3), '-->', round(accuracy, 3), '     ({0}% increase)'.format(round(increase, 1)))
	print ('\n')

	metrics = {'recall': recall, 'precision': precision}
	best_metric = max(metrics, key=metrics.get)
	print ('Best metric:', best_metric)
	metric_diff = abs(precision - recall)
	if metric_diff <= .07:
		print ('     Precision and recall are balanced', '    ({0}% difference)'.format(round(abs((recall - precision) / precision) * 100, 1)))
	elif best_metric == 'recall':
		print ('     Model favors recall: casts a wide net - gets tp but also fp', '    ({0}% difference)'.format(round(abs((recall - precision) / precision) * 100, 1)))
	elif best_metric == 'precision':
		print ('     Model favors precision: casts a smaller, more specialized net - gets mostly tp', '    ({0}% difference)'.format(round(abs((recall - precision) / precision) * 100, 1)))
	print ('\n')