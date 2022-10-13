# CART on the Bank Note dataset
import enum
from random import seed
from random import randrange
from csv import reader 
import time



#Globales
c = 0
a = 0
lineas = 0

# Load a CSV file
def loadCsv(filename):
	file = open(filename, "rt")
	lines = reader(file)
	dataset = list(lines)
	return dataset 


# Convert string column to float
def strColumnToFloat(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Limit dataset lenght
def limitDataset(dataset, size):
	for i in range (size):
		dataset.append(fullDataSet[i])
	

#-----------------------------------------------------------------
# Split a dataset into k folds
def crossValidationSplit(dataset, n_folds):
	global a,c,lineas
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	a+=3
	for i in range(n_folds):
		c+=1
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
			lineas+=3
		dataset_split.append(fold)
		lineas+=2
	c+=1
	lineas+=4
	return dataset_split

# Calculate accuracy percentage
def accuracyMetric(actual, predicted):
	global a, c,lineas
	correct = 0
	a+=1
	lineas+=2
	for i in range(len(actual)): 
		lineas+=1
		c+=1
		if actual[i] == predicted[i]:
			correct += 1
			lineas+=2
			c+=1
			a+=1
	c+=1
	return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluateAlgorithm(dataset, algorithm, n_folds, *args):
	global a,c,lineas
	folds = crossValidationSplit(dataset, n_folds) 
	scores = list()
	lineas+=2
	a+=2
	for fold in folds:
		c+=1
		a+=1
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		a+=4
		lineas+=5
		for row in fold:
			c+=1
			a+=1

			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
			a+=3
			lineas+=4
		c+=1
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracyMetric(actual, predicted)
		scores.append(accuracy)
		a+=4
		lineas+=5
	return scores

# Split a dataset based on an attribute and an attribute value
def testSplit(index, value, dataset):
	global a,c,lineas
	left, right = list(), list()
	a+=2
	lineas+=1
	for row in dataset:
		c+=1
		a+=1
		lineas+=1
		if row[index] < value:
			c+=1
			left.append(row)
			a+=1
			lineas+=2
		else:
			c+=1
			right.append(row)
			a+=1
			lineas+=2
	return left, right

# Calculate the Gini index for a split dataset
def giniIndex(groups, classes):
	global a,c,lineas
	# count all samples at split point
	n_instances = float(sum([len(group) for group in groups]))
	# sum weighted Gini index for each group
	gini = 0.0
	a+=2
	lineas+=2
	for group in groups:
		c+=1
		a+=1
		size = float(len(group))
		a+=1
		lineas+=2
		# avoid divide by zero
		if size == 0:
			c+=1
			lineas+=1
			continue
		score = 0.0
		lineas+=2
		a+=1
		# score the group based on the score for each class
		for class_val in classes:
			c+=1
			a+=1
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
			a+=2
			lineas+=3
		# weight the group score by its relative size
		gini += (1.0 - score) * (size / n_instances)
		a+=1
		lineas+=1
	return gini

# Select the best split point for a dataset
def getSplit(dataset):
	global a,c,lineas
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	a+=5
	lineas+=2
	for index in range(len(dataset[0])-1):
		c+=1
		a+=1
		lineas+=1
		for row in dataset:
			c+=1
			a+=1
			groups = testSplit(index, row[index], dataset)
			gini = giniIndex(groups, class_values)
			a+=2
			c+=1
			lineas+=3
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
				a+=1
				lineas+=2
	return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Create a terminal node value
def toTerminal(group):
	global a,c,lineas
	outcomes = [row[-1] for row in group]
	a+=1
	lineas+=1
	return max(set(outcomes), key=outcomes.count)


# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
	global a,c,lineas
	left, right = node['groups']
	del(node['groups'])
	a+=3
	lineas+=2
	# check for a no split
	if not left or not right:
		c+=1
		node['left'] = node['right'] = toTerminal(left + right)
		a+=1
		lineas+=2
		return
	# check for max depth
	if depth >= max_depth:
		c+=2
		node['left'], node['right'] = toTerminal(left), toTerminal(right)
		a+=2
		lineas+=2
		return
	# process left child
	if len(left) <= min_size:
		c+=2
		node['left'] = toTerminal(left)
		lineas+=2
		a+=1
	else:
		c+=2
		node['left'] = getSplit(left)
		a+=1
		split(node['left'], max_depth, min_size, depth+1)
		lineas+=3
	# process right child
	if len(right) <= min_size:
		c+=2
		node['right'] = toTerminal(right)
		a+=1
		lineas+=2
	else:
		c+=2
		node['right'] = getSplit(right)
		split(node['right'], max_depth, min_size, depth+1)
		a+=1
		lineas+=3

# Build a decision tree
def buildTree(train, max_depth, min_size):
	global a,lineas
	root = getSplit(train)
	a+=1
	split(root, max_depth, min_size, 1)
	lineas+=2
	return root

# Make a prediction with a decision tree
def predict(node, row):
	global a,c,lineas
	if row[node['index']] < node['value']:
		c+=1
		if isinstance(node['left'], dict):
			c+=1
			lineas+=2
			return predict(node['left'], row)
		else:
			c+=1
			lineas+=1
			return node['left']
	else:
		lineas+=1
		c+=1
		if isinstance(node['right'], dict):
			c+=1
			lineas+=1
			return predict(node['right'], row)
		else:
			c+=1
			lineas+=1
			return node['right']

# Classification and Regression Tree Algorithm
def decisionTree(train, test, max_depth, min_size):
	global a,c,lineas
	tree = buildTree(train, max_depth, min_size)
	predictions = list()
	a+=2
	lineas+=2
	for row in test:
		c+=1
		prediction = predict(tree, row)
		predictions.append(prediction)
		a+=3
		lineas+=3
	lineas+=1
	c+=1
	return(predictions)

def setSize(i):
	if(i == 0):
		return 10
	elif(i == 1):
		return 50
	elif(i == 2):
		return 100
	elif(i == 3):
		return 200
	elif(i == 4):
		return 500
	elif(i == 5):
		return 1000
	elif(i == 6):
		return 7000
	else:
		return 10000

		

	

# Test CART on Bank Note dataset
seed(1)
# Load and prepare data
fullDataSet = loadCsv('dataFile.csv')

# Convert string attributes to integers
for i in range(len(fullDataSet[0])):
	strColumnToFloat(fullDataSet, i)

for i in range(8):
	#Reset Counters
	c = 0
	a = 0
	lineas = 0

	# Limit Data
	dataset = []
	size = setSize(i)
	limitDataset(dataset,size)

	# Timer Start
	startTime = time.time()

	# Evaluate algorithm
	n_folds = 5
	max_depth = 5
	min_size = 10
	scores = evaluateAlgorithm(dataset, decisionTree, n_folds, max_depth, min_size)

	# Timer End
	endTime = time.time()
	elapsedTime = endTime - startTime

	print("-----------------Corrida",i ,"---------------------")
	print("Tamaño del dataset: ",size)
	print("Comparaciones: ",c)
	print("Asignaciones: ",a)
	print("Lineas Ejecutadas: ",lineas)
	print("Execution time: ",elapsedTime)
	print("--------------------------------------------------")

