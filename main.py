# For Python 2 / 3 compatability
from __future__ import print_function
import pandas as pd

# Function to import .csv files to be used in the algorithm.
def import_file(location):
    file = pd.read_csv(location, sep=';')
    data = file.values
    return data

# Function to find unique values in dataset.
def unique_vals(rows, col):
    return set([row[col] for row in rows])

# Counting the amount of each of type of in a dataset.
def class_counts(rows):
    counts = {}
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

# Check if a value is numeric.
def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)

# Class that keeps track of a column number and a specific column value to ask 
# a question and compare the column value in the question with the feature value.
class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    # Method to compare the set value in the class with the feature value.
    def match(self, example):
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value
    
    # Helper method to print the question in a readable format.
    def __repr__(self):
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))

# Function to check whether each row matches the question.
# If yes, add it to true_rows. If not, add it to false_rows.
def partition(rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows

# Function to calculate the gini impurity.
def gini(rows):
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity

# Function to check the information gain per question.
def info_gain(left, right, current_uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)

# Function to find the best split in the rows available.
# Also looking for the best question to ask.
def find_best_split(rows):
    best_gain = 0  # keep track of the best information gain
    best_question = None  # keep train of the feature / value that produced it
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1  # number of columns

    for col in range(n_features):  # for each feature

        values = set([row[col] for row in rows])  # unique values in the column

        for val in values:  # for each value

            question = Question(col, val)

            true_rows, false_rows = partition(rows, question)

            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            gain = info_gain(true_rows, false_rows, current_uncertainty)

            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question

# Class that keeps track of how many times a feature value appears in the data.
class Leaf:
    def __init__(self, rows):
        self.predictions = class_counts(rows)

# Class that holds the question and the two branches it leads to.
class Decision_Node:
    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

# Recursion function which will build the decision tree.
def build_tree(rows):
    # Returning the question with the highest information gain.
    gain, question = find_best_split(rows)
    
    # If there is no information gain possible, end with a Leaf.
    if gain == 0:
        return Leaf(rows)

    true_rows, false_rows = partition(rows, question)

    true_branch = build_tree(true_rows)

    false_branch = build_tree(false_rows)

    return Decision_Node(question, true_branch, false_branch)

# Function to print the tree in the console.
def print_tree(node, spacing=""):
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    print (spacing + str(node.question))

    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")

# Function to classify how the test data should follow the tree that has been build.
def classify(row, node):
    if isinstance(node, Leaf):
        return node.predictions
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

# Function to print the predictions of a leaf.
def print_leaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs

# Function to calculate the succes rate of the test data used in the tree.
def calculate_success(test_result, row):
    success = 0
    positives = 0
    recall = 0
    
    for result in test_result:
        if (test_result[result] == '100%'):
           success += 1
           if (result == row[-1]):
               positives += 1
           else:
               recall +=1
            
        else:
            recall +=1
            break
        
    return success, positives, recall

# Function to calculate the classification accuracy of the test results            
def calculate_classification_accurary(total_success, testing_data):
    total_tests = len(testing_data)
    classification_accurary = total_success / total_tests * 100
    print ('Total predictions correct: ', (total_success))
    print ('Classification Accuracy Percentage: ',(classification_accurary),'%')

# Function to calculate the F1 Score of the test results 
def calculate_f1_score(total_success, total_positives, total_wrong, total_test_items):
    precision = total_positives / total_success
    actual_positives = total_test_items - total_wrong
    recall = total_positives / actual_positives
    
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print (' ')
    print ('Precision: ',(precision))
    print ('Recall: ',(recall))
    print ('F1 Score: ',(f1_score))     

print('########################################################')
print('Importing Training Data')
print('########################################################')
print (' ')

training_data = import_file("training_data.csv")
print(training_data)
print (' ')

header = ["audience", "keyword", "device","celebrity" , "platform"]
my_tree = build_tree(training_data)

print('########################################################')
print('Printing Decission Tree Design')
print('########################################################')
print (' ')
print_tree(my_tree)

print (' ')
print('########################################################')
print('Using Test Data On Decission Tree')
print('########################################################')
print (' ')
testing_data = import_file('testing_data.csv')

total_success = 0
total_positives = 0
total_wrong = 0

for row in testing_data:
    test_result = print_leaf(classify(row, my_tree))
    
    success, positives, wrong = calculate_success(test_result, row)
    
    total_success += success
    total_positives += positives
    total_wrong += wrong
        
    print ("Actual: %s. Predicted: %s" %
           (row[-1], test_result))

print (' ')  
print('########################################################')
print('Generating Test Rapport')
print('########################################################')
print (' ')
calculate_classification_accurary(total_success, testing_data)
calculate_f1_score(total_success, total_positives, total_wrong, len(testing_data))