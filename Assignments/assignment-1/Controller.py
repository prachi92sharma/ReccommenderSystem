import os
import sys
import ast
import math

fileSize = 'full'
trainFile = '/scratch/ds222-2017/assignment-1/DBPedia.' + fileSize + '/' + fileSize + '_train.txt'
testFile = '/home/abhinava/Assignments/assignment-1/new_' + fileSize + '_test.txt'
noOfReducers = ''

vocabularyCount = 423121

labelMap = {}
labelMapList = []
noOfTestLines = 0

def createLabelMap():
	global labelMap
	global labelMapList
	global noOfTestLines
	with open("label-map.txt") as fp:
		for line in fp:
			key,value = line.strip().split('\t',1)
			labelMap[key] = ast.literal_eval(value)
	labelMapList = list(labelMap.keys())
	for key in labelMap:
		noOfTestLines += int(labelMap[key][0])


def getConditionalProbability(token, label, wordMapList):
	c = 0
	prob = 1.0 / len(labelMapList)

	if wordMapList:
		tempDict = {s[0]:int(s[1:][0]) for s in wordMapList}
		if label in tempDict:
			c = tempDict[label]
		prob = math.log((c + 1.0) / (1.0 + int(labelMap[label][1]) + vocabularyCount))
	
	return prob

def getPredictedLabel(tokens, wordEventList):
	probArray = []

	for label in labelMapList:
		s = 0
		for i in range(0, len(tokens)):
			s += getConditionalProbability(tokens[i], label, wordEventList[i])
		s = s + math.log(int(labelMap[label][0]) / float(noOfTestLines))
		probArray.append(s)
	return labelMapList[probArray.index(max(probArray))]

def preprocessLabelWord(label_word):
	t1 = label_word.split('\t',1)
	t2 = ast.literal_eval(t1[1])
	t3 = ast.literal_eval(t2[1])
	return [t1[0],[t2[0],t3]]

def startTesting():
	print('Starting Testing')
	trueLabel = None
	currDocId = None
	noOfCorrectPredictions = 0
	tokens = []
	word_events_list = []
	totalInputLines = 0

	with open("test-file.txt") as fp:
		for line in fp:
			docId,label_word = line.strip().split('@',1)
			label_word = preprocessLabelWord(label_word)

			if docId == currDocId:
				tokens.append(label_word[1][0])
				word_event_list.append(label_word[1][1])
			else:
				if currDocId != None:
					predLabel = getPredictedLabel(tokens, word_event_list)
					if predLabel in trueLabel.split(','):
						#print trueLabel,predLabel
						noOfCorrectPredictions += 1
				totalInputLines += 1
				trueLabel = label_word[0]
				currDocId = docId
				tokens = [label_word[1][0]]
				word_event_list = [label_word[1][1]]

	predLabel = getPredictedLabel(tokens, word_event_list)
	if predLabel in trueLabel.split(','):
		noOfCorrectPredictions += 1
	return (float(noOfCorrectPredictions) / totalInputLines)

noOfReducers = raw_input("Enter Number of Reducers: ")

#----------------
# Training Phase
#----------------

# Modify test file to have id generated test file
if fileSize == 'full':
	os.system('awk \'{print NR  "@" $s}\' /scratch/ds222-2017/assignment-1/DBPedia.' + fileSize + '/' + fileSize + '_test.txt > /home/abhinava/Assignments/assignment-1/new_' + fileSize + '_test.txt')
else:
	os.system('awk \'NR>3 {print NR  "@" $s}\' /scratch/ds222-2017/assignment-1/DBPedia.' + fileSize + '/' + fileSize + '_test.txt > /home/abhinava/Assignments/assignment-1/new_' + fileSize + '_test.txt')

# Empty Home Folder
os.system('hadoop fs -rm -r /user/abhinava/*')

# Copy Training and Test data to HDFS
os.system('hadoop fs -copyFromLocal '+trainFile+' /user/abhinava')
os.system('hadoop fs -copyFromLocal '+testFile+' /user/abhinava')

# Run Hadoop Streaming to generate Word Count File
os.system('hadoop jar /usr/hdp/2.6.1.0-129/hadoop-mapreduce/hadoop-streaming.jar -D mapred.reduce.tasks=' + noOfReducers + ' -file WordJobMapper.py -mapper WordJobMapper.py -file WordJobReducer.py -reducer WordJobReducer.py -input /user/abhinava/' + fileSize + '_train.txt -output word-count')

# Get Vocabulary Count

# Run Hadoop Streaming to generate Label Count File
os.system('hadoop jar /usr/hdp/2.6.1.0-129/hadoop-mapreduce/hadoop-streaming.jar -D mapred.reduce.tasks=' + noOfReducers + ' -file LabelJobMapper.py -mapper LabelJobMapper.py -file LabelJobReducer.py -reducer LabelJobReducer.py -input /user/abhinava/' + fileSize + '_train.txt -output label-count')

#------------
# Test Phase
#------------
print('-------------------------------------------Training Finished---------------------------------------------')
# Create folder for intermideiate input steps
os.system('hadoop fs -mkdir /user/abhinava/input2')
os.system('hadoop fs -chmod 777 /user/abhinava/input2')
os.system('hadoop fs -mkdir /user/abhinava/input3')
os.system('hadoop fs -chmod 777 /user/abhinava/input3')
os.system('hadoop fs -mkdir /user/abhinava/input4')
os.system('hadoop fs -chmod 777 /user/abhinava/input4')


# Concatenate Reducer Output Files to One File
os.system('hadoop fs -getmerge /user/abhinava/word-count /home/abhinava/word-map.txt')
os.system('hadoop fs -put /home/abhinava/word-map.txt /user/abhinava/input2')
os.system('rm -rf /home/abhinava/word-map.txt')
os.system('hadoop fs -rm -r /user/abhinava/word-count')

os.system('hadoop fs -getmerge /user/abhinava/label-count /home/abhinava/Assignments/assignment-1/label-map.txt')
os.system('hadoop fs -rm -r /user/abhinava/label-count')


os.system('hadoop jar /usr/hdp/2.6.1.0-129/hadoop-mapreduce/hadoop-streaming.jar -D mapred.reduce.tasks=' + noOfReducers + ' -file TestWordMapper.py -mapper TestWordMapper.py -file TestWordReducer.py -reducer TestWordReducer.py -input /user/abhinava/new_' + fileSize + '_test.txt -output test')

# Combine Test File Word outputs
os.system('hadoop fs -getmerge /user/abhinava/test /home/abhinava/test-word-count.txt')
os.system('hadoop fs -put /home/abhinava/test-word-count.txt /user/abhinava/input2')
os.system('rm -rf /home/abhinava/test-word-count.txt')
os.system('hadoop fs -rm -r /user/abhinava/test/part*')

# Sort this Input2( test-word-count + label-map ) file
os.system('hadoop jar /usr/hdp/2.6.1.0-129/hadoop-mapreduce/hadoop-streaming.jar -D mapred.reduce.tasks=' + noOfReducers + ' -file Input2Mapper.py -mapper Input2Mapper.py -file Input2Reducer.py -reducer Input2Reducer.py -input /user/abhinava/input2/* -output output2')

# Merge output2 and create Input3 File(<id,word,wordDetails>) 
os.system('hadoop fs -getmerge /user/abhinava/output2 /home/abhinava/input3.txt')
os.system('hadoop fs -put /home/abhinava/input3.txt /user/abhinava/input3')
os.system('rm -rf /home/abhinava/input3.txt')

# Delete input2 and output2
os.system('hadoop fs -rm -r /user/abhinava/input2')
os.system('hadoop fs -rm -r /user/abhinava/output2')

# Input3 File(<id,word,wordDetails>) 
os.system('hadoop jar /usr/hdp/2.6.1.0-129/hadoop-mapreduce/hadoop-streaming.jar -D mapred.reduce.tasks=' + noOfReducers + ' -file Input3Mapper.py -mapper Input3Mapper.py -file Input3Reducer.py -reducer Input3Reducer.py -input /user/abhinava/input3/* -output output3')

# Merge output3 and create Input4 File(<id,word,wordDetails>) 
os.system('hadoop fs -getmerge /user/abhinava/output3 /home/abhinava/input4.txt')
os.system('hadoop fs -put /home/abhinava/input4.txt /user/abhinava/input4')
os.system('rm -rf /home/abhinava/input4.txt')

# Delete input3 and output3
os.system('hadoop fs -rm -r /user/abhinava/input3')
os.system('hadoop fs -rm -r /user/abhinava/output3')

# Input4 File(<id,word,wordDetails>) out \put - sortred by docId 
os.system('hadoop jar /usr/hdp/2.6.1.0-129/hadoop-mapreduce/hadoop-streaming.jar -D mapred.reduce.tasks=' + noOfReducers + ' -file Input4Mapper.py -mapper Input4Mapper.py -file Input4Reducer.py -reducer Input4Reducer.py -input /user/abhinava/input4/* -output output4')

# Merge output4 and create final test file  
os.system('hadoop fs -getmerge /user/abhinava/output4 /home/abhinava/Assignments/assignment-1/test-file.txt')

createLabelMap()
print startTesting()