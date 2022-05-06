import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
import numpy as np
import math
import csv
import copy
from matplotlib import pyplot as plt
from tensorflow import keras
from random import randint
from random import sample

'''Calculculates logartihm base 2 Keras backend.'''
def log2(input_tensor):
	numerator = keras.backend.log(input_tensor)
	denominator = keras.backend.log(keras.backend.constant(2, dtype = numerator.dtype))
	return numerator/denominator

'''Calculates custom loss using Keras backend.
   Combines binary crossentropy individually for each class'''
def multilabel_binary_crossentropy(y_true, y_pred):
	first = (y_true[0][0]*log2(y_pred[0][0]))+(1-y_true[0][0])*log2(1-y_pred[0][0])
	second = (y_true[0][1]*log2(y_pred[0][1]))+(1-y_true[0][1])*log2(1-y_pred[0][1])
	third = (y_true[0][2]*log2(y_pred[0][2]))+(1-y_true[0][2])*log2(1-y_pred[0][2])
	fourth = (y_true[0][3]*log2(y_pred[0][3]))+(1-y_true[0][3])*log2(1-y_pred[0][3])
	fifth = (y_true[0][4]*log2(y_pred[0][4]))+(1-y_true[0][4])*log2(1-y_pred[0][4])
	sixth = (y_true[0][5]*log2(y_pred[0][5]))+(1-y_true[0][5])*log2(1-y_pred[0][5])
	seventh = (y_true[0][6]*log2(y_pred[0][6]))+(1-y_true[0][6])*log2(1-y_pred[0][6])
	eighth = (y_true[0][7]*log2(y_pred[0][7]))+(1-y_true[0][7])*log2(1-y_pred[0][7])
	ninth = (y_true[0][8]*log2(y_pred[0][8]))+(1-y_true[0][8])*log2(1-y_pred[0][8])
	tenth = (y_true[0][9]*log2(y_pred[0][9]))+(1-y_true[0][9])*log2(1-y_pred[0][9])
	eleventh = (y_true[0][10]*log2(y_pred[0][10]))+(1-y_true[0][10])*log2(1-y_pred[0][10])
	twelfth = (y_true[0][11]*log2(y_pred[0][11]))+(1-y_true[0][11])*log2(1-y_pred[0][11])
	loss = -first-second-third-fourth-fifth-sixth-seventh-eighth-ninth-tenth-eleventh-twelfth+keras.backend.epsilon()
	return loss

'''Calculates macro-averaged accuracy.
   Takes the greatest n elements in the prediction vector as class predictions where n is the number of ground truths.
   Does not use Keras backend streamlining because counting number of ground truths is not possible with backend processes.'''
def generous_accuracy_macro(y_true, y_pred):
	y_true_arr = keras.backend.get_value(y_true)
	y_pred_arr = keras.backend.get_value(y_pred)
	for i in range(y_pred_arr.shape[0]):
		y_pred_arr[i][np.argmax(y_pred_arr[i])] = 1
		for j in range(y_pred_arr.shape[1]):
			if y_pred_arr[i][j] != 1:
				y_pred_arr[i][j] = 0
	for i in range(y_pred_arr.shape[1]):
		class_accuracy_sum = 0
		for j in range(y_pred_arr.shape[0]):
			if y_true_arr[j][i] == y_pred_arr[j][i] == 1:
				class_accuracy_sum += 1
	return class_accuracy_sum/y_pred_arr.shape[1]

'''Calculates macro-averaged precision.
   Takes the greatest n elements in the prediction vector as class predictions where n is the number of ground truths.
   Does not use Keras backend streamlining because counting number of ground truths is not possible with backend processes.'''
def precision_macro(y_true, y_pred):
	y_true_arr = keras.backend.get_value(y_true)
	y_pred_arr = keras.backend.get_value(y_pred)
	for i in range(y_pred_arr.shape[0]):
		true_indexes = y_pred_arr[i].argsort()[-int(np.sum(y_true[i])):][::-1]
		for j in range(true_indexes.shape[0]):
			y_pred_arr[i][true_indexes[j]] = 1
		for j in range(y_pred_arr.shape[1]):
			if y_pred_arr[i][j] != 1:
				y_pred_arr[i][j] = 0
	precision_sum = ignore_count = 0
	for i in range(y_pred_arr.shape[1]):
		true_positive_count = false_positive_count = 0
		for j in range(y_pred_arr.shape[0]):
			if y_true_arr[j][i] == 1:
				if y_pred_arr[j][i] == 1:
					true_positive_count += 1
			elif y_pred_arr[j][i] == 1:
				false_positive_count += 1
		if true_positive_count == 0 and false_positive_count == 0:
			ignore_count += 1
		else:
			precision_sum += true_positive_count/(true_positive_count+false_positive_count)
	return precision_sum/(y_pred_arr.shape[1]-ignore_count)

'''Calculates macro-averaged recall.
   Takes the greatest n elements in the prediction vector as class predictions where n is the number of ground truths.
   Does not use Keras backend streamlining because counting number of ground truths is not possible with backend processes.'''
def recall_macro(y_true, y_pred):
	y_true_arr = keras.backend.get_value(y_true)
	y_pred_arr = keras.backend.get_value(y_pred)
	for i in range(y_pred_arr.shape[0]):
		true_indexes = y_pred_arr[i].argsort()[-int(np.sum(y_true[i])):][::-1]
		for j in range(true_indexes.shape[0]):
			y_pred_arr[i][true_indexes[j]] = 1
		for j in range(y_pred_arr.shape[1]):
			if y_pred_arr[i][j] != 1:
				y_pred_arr[i][j] = 0
	recall_sum = ignore_count = 0
	for i in range(y_pred_arr.shape[1]):
		true_positive_count = false_negative_count = 0
		for j in range(y_pred_arr.shape[0]):
			if y_pred_arr[j][i] == 1:
				if y_true_arr[j][i] == 1:
					true_positive_count += 1
			elif y_true_arr[j][i] == 1:
				false_negative_count += 1
		if true_positive_count == 0 and false_negative_count == 0:
			ignore_count += 1
		else:
			recall_sum += true_positive_count/(true_positive_count+false_negative_count)
	return recall_sum/(y_pred_arr.shape[1]-ignore_count)

'''Calculates macro-averaged F1 score using precision and recall.
   Does not use Keras backend streamlining becaue precision and recall are used in the calculation.'''
def f1_macro(y_true, y_pred):
	precision = precision_macro(y_true, y_pred)
	recall = recall_macro(y_true, y_pred)
	if precision == 0 and recall == 0:
		return 0
	else:
		return 2*precision*recall/(precision+recall)

'''Creates model, defines parameters and hyper-parameters, and defines initialisation specifications.'''
def create_model():
	input_shape = (6, 3, 18)
	num_classes = 12
	kernel_initialiser = keras.initializers.RandomUniform(minval = -0.1, maxval = 0.1)
	bias_initialiser = keras.initializers.zeros()
	network = keras.models.Sequential()
	network.add(keras.layers.Conv2D(200, use_bias = True, kernel_size = (3, 3), strides = (1, 1), activation = "relu", input_shape = input_shape, padding = "same", kernel_initializer = kernel_initialiser, bias_initializer = bias_initialiser))
	network.add(keras.layers.Conv2D(100, use_bias = True, kernel_size = (3, 3), strides = (1, 1), activation = 'relu', kernel_initializer = kernel_initialiser, bias_initializer = bias_initialiser))
	network.add(keras.layers.Flatten())
	network.add(keras.layers.Dropout(0.1))
	network.add(keras.layers.Dense(num_classes, use_bias = True, activation = "sigmoid", kernel_initializer = kernel_initialiser, bias_initializer = bias_initialiser))
	network.compile(loss = multilabel_binary_crossentropy, optimizer = keras.optimizers.Adadelta(learning_rate = 1, rho = 0.99), metrics = [f1_macro, generous_accuracy_macro], run_eagerly = True)
	return network

'''Generates a random sequence for scrambling a Rubik's cube.
   Scrambles are between 0 and 26 moves inclusive.
   Scramble sequences are written to "data.txt" for use with solver.'''
def generate_scrambles():
	move_dict = {0: "L", 1: "L'", 2: "F", 3: "F'", 4: "R", 5: "R'", 6: "B", 7: "B'", 8: "D", 9: "D'", 10: "U", 11: "U'"}
	with open("maneuvers.txt", mode = "w", newline = '') as scrambles:
		writer = csv.writer(scrambles, delimiter = ' ')
		for i in range(0, 100000):
			scramble_string = ""
			for j in range(0, randint(1, 26)):
				scramble_move = move_dict[randint(0, len(move_dict)-1)]
				scramble_string += scramble_move + ' '
			writer.writerow([scramble_string])

'''Writes an ordered version the scrambles in data.txt to train.txt based on number of moves in the optimal solutions.
   Most curricula will have complexity level progression which increases with distance to an extent so this makes future restructuring more efficient.'''
def structure_num_moves():
	with open("train.txt", mode = 'a', newline = '') as destination:
		for complexity_level in range(1, 10):
			with open("data.txt", mode = 'r', newline = '') as source:
				for line in source:
					if len(line) > 0 and line[-3] == str(complexity_level) and line[-4] == ',':
						destination.write(line)
		for complexity_level in range(10, 23):
			with open("data.txt", mode = 'r', newline = '') as source:
				for line in source:
					if len(line) > 0 and line[-4]+line[-3] == str(complexity_level):
						destination.write(line)

'''Inserts data from a stream into an array.'''
def insert_data(arr, numbers):
	numbers_index = 0
	for r in range(3):
		for c in range(3):
			arr[0][r][c] = int(numbers[numbers_index])
			arr[1][r][c] = int(numbers[numbers_index+1])
			arr[2][r][c] = int(numbers[numbers_index+2])
			arr[3][r][c] = int(numbers[numbers_index+3])
			arr[4][r][c] = int(numbers[numbers_index+5])
			arr[5][r][c] = int(numbers[numbers_index+4])
			numbers_index += 6
	return arr

'''Transfers data streams from train.txt to training and ground truth arrays.'''
def read_data(usage):
	example_count = 0
	with open("{}.txt".format(usage), mode='r', newline='') as source:
		for line in source:
			if len(line) > 10:
				example_count += 1
	x_train = np.empty((example_count, 6, 3, 18))
	y_train = np.zeros((example_count, 12))
	label_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11}
	with open("{data_set}.txt".format(data_set = usage), mode = 'r', newline = '') as source:
		line_counter = 0
		line = str(source.readline())
		while line:
			if len(line) > 18*3*6:
				x_train[line_counter, 0:6, 0:3, 0:3] = insert_data(x_train[line_counter, 0:6, 0:3, 0:3], line[0:54])
				x_train[line_counter, 0:6, 0:3, 3:6] = insert_data(x_train[line_counter, 0:6, 0:3, 3:6], line[54:108])
				x_train[line_counter, 0:6, 0:3, 6:9] = insert_data(x_train[line_counter, 0:6, 0:3, 6:9], line[108:162])
				x_train[line_counter, 0:6, 0:3, 9:12] = insert_data(x_train[line_counter, 0:6, 0:3, 9:12], line[162:216])
				x_train[line_counter, 0:6, 0:3, 12:15] = insert_data(x_train[line_counter, 0:6, 0:3, 12:15], line[270:324])
				x_train[line_counter, 0:6, 0:3, 15:18] = insert_data(x_train[line_counter, 0:6, 0:3, 15:18], line[216:270])
				index_counter = 324
				y_train_value = ''
				while line[index_counter] != ',':
					y_train_value += line[index_counter]
					index_counter += 1
				y_train_value = int(y_train_value)
				y_train[line_counter][label_map[y_train_value]] = 1
				is_new_training_example = False
				while not is_new_training_example:
					current_position = source.tell()
					line = source.readline()
					if 18 * 3 * 6 > len(line) > 2:
						y_train_value = ''
						short_line_index_counter = 0
						while line[short_line_index_counter] != ',':
							y_train_value += line[short_line_index_counter]
							short_line_index_counter += 1
						y_train_value = int(y_train_value)
						y_train[line_counter][y_train_value] = 1
						y_train[line_counter][label_map[y_train_value]] = 1
					else:
						source.seek(current_position)
						is_new_training_example = True
			line = str(source.readline())
			line_counter += 1
	return x_train, y_train

'''Splits data split into exclusive training, validation, and testing sets.'''
def split_data(x, y, val_proportion, test_proportion = 0.0):
	if test_proportion > 0 and val_proportion+test_proportion < 1:
		print("{}% training data, {}% validation data, {}% test data".format(math.floor(100*(1-(val_proportion+test_proportion))), math.floor(100*val_proportion), math.floor(100*test_proportion)))
		val_size = math.floor(val_proportion*x.shape[0])
		test_size = math.floor(test_proportion*x.shape[0])
		selection_indexes = np.array(sample(range(x.shape[0]), val_size+test_size))
		val_selection_indexes = np.array(sample(selection_indexes.tolist(), val_size))
		test_selection_indexes = np.setdiff1d(selection_indexes, val_selection_indexes)
		x_val = np.empty((val_selection_indexes.shape[0], x.shape[1], x.shape[2], x.shape[3]))
		y_val = np.empty((val_selection_indexes.shape[0], y.shape[1]))
		x_test = np.empty((test_selection_indexes.shape[0], x.shape[1], x.shape[2], x.shape[3]))
		y_test = np.empty((test_selection_indexes.shape[0], y.shape[1]))
		for i in range(x_val.shape[0]):
			for j in range(x_val.shape[1]):
				for k in range(x_val.shape[2]):
					for l in range(x_val.shape[3]):
						x_val[i][j][k][l] = x[val_selection_indexes[i]][j][k][l]
		for i in range(y_val.shape[0]):
			for j in range(y_val.shape[1]):
				y_val[i][j] = y[val_selection_indexes[i]][j]
		for i in range(x_test.shape[0]):
			for j in range(x_test.shape[1]):
				for k in range(x_test.shape[2]):
					for l in range(x_test.shape[3]):
						x_test[i][j][k][l] = x[test_selection_indexes[i]][j][k][l]
		for i in range(y_test.shape[0]):
			for j in range(y_test.shape[1]):
				y_test[i][j] = y[test_selection_indexes[i]][j]
	else:
		print("{}% training data, {}% validation data, external test data used".format(math.floor(100*(1-val_proportion)), math.floor(100*val_proportion)))
		val_size = math.floor(val_proportion*x.shape[0])
		selection_indexes = np.array(sample(range(x.shape[0]), val_size))
		val_selection_indexes = selection_indexes
		x_val = np.empty((val_selection_indexes.shape[0], x.shape[1], x.shape[2], x.shape[3]))
		y_val = np.empty((val_selection_indexes.shape[0], y.shape[1]))
		alt_train = read_data("test")
		x_test, y_test = alt_train[0], alt_train[1]
		for i in range(x_val.shape[0]):
			for j in range(x_val.shape[1]):
				for k in range(x_val.shape[2]):
					for l in range(x_val.shape[3]):
						x_val[i][j][k][l] = x[val_selection_indexes[i]][j][k][l]
		for i in range(y_val.shape[0]):
			for j in range(y_val.shape[1]):
				y_val[i][j] = y[val_selection_indexes[i]][j]
	x_train = np.delete(x, selection_indexes, axis = 0)
	y_train = np.delete(y, selection_indexes, axis = 0)
	return x_train, y_train, x_val, y_val, x_test, y_test, selection_indexes

'''Saves a portion of the training data from each complexity level and distributes it among the subsequent stages.'''
def accumulate_data(x_train, y_train, change_indexes):
	denominators = np.zeros(change_indexes.shape[0]-1)
	for i in range(denominators.shape[0]):
		for j in range(1, denominators.shape[0]-(i+1)+1):
			denominators[i] += j
	denominators = denominators*2
	denominators[denominators.shape[0]-1] = 1
	train_dist = np.empty((change_indexes.shape[0]-1, change_indexes.shape[0]-1), dtype = object)
	for i in range(train_dist.shape[0]):
		lower_bound = change_indexes[i]
		data_range = change_indexes[i+1]-change_indexes[i]
		for j in range(train_dist.shape[1]-i, 0, -1):
			if j == train_dist.shape[1]-i:
				upper_bound = lower_bound+denominators[i]/2/denominators[i]*data_range
			else:
				upper_bound = lower_bound+j/denominators[i]*data_range
			train_dist[i][train_dist.shape[1]-i-j] = (math.floor(lower_bound), math.floor(upper_bound))
			lower_bound = upper_bound
	new_change_indexes = np.zeros(change_indexes.shape, dtype=int)
	lower_bound = upper_bound = 0
	x_train_cum = np.zeros(x_train.shape)
	y_train_cum = np.zeros(y_train.shape)
	for i in range(train_dist.shape[0]):
		for j in range(train_dist.shape[0]):
			for k in range(train_dist.shape[1]):
				if i == j+k:
					upper_bound = lower_bound + train_dist[j][k][1]-train_dist[j][k][0]
					x_train_cum[lower_bound:upper_bound] = x_train[train_dist[j][k][0]:train_dist[j][k][1]]
					y_train_cum[lower_bound:upper_bound] = y_train[train_dist[j][k][0]:train_dist[j][k][1]]
					lower_bound = upper_bound
		new_change_indexes[i+1] = upper_bound
	zeros_encountered = True
	trim_counter = 0
	index = x_train_cum.shape[0] - 1
	while zeros_encountered:
		if np.all((x_train_cum[index] == 0)):
			trim_counter += 1
			index -= 1
		else:
			zeros_encountered = False
	x_train_cum = x_train_cum[:-trim_counter]
	y_train_cum = y_train_cum[:-trim_counter]
	for i in range(new_change_indexes.shape[0]-1):
		selections = sample(range(new_change_indexes[i+1]-new_change_indexes[i]), new_change_indexes[i+1]-new_change_indexes[i])
		x_train_cum[new_change_indexes[i]:new_change_indexes[i+1]] = np.take(x_train_cum[new_change_indexes[i]:new_change_indexes[i+1]], selections, axis = 0)
		y_train_cum[new_change_indexes[i]:new_change_indexes[i+1]] = np.take(y_train_cum[new_change_indexes[i]:new_change_indexes[i+1]], selections, axis = 0)
	return x_train_cum, y_train_cum, new_change_indexes

'''Calculates the boundries between complexity level in a preordered curriculum.'''
def find_change_indexes(sorted_structure):
	num_complexity_lvls = 1
	for i in range(sorted_structure.shape[0]-1):
		if sorted_structure[i+1] != sorted_structure[i]:
			num_complexity_lvls += 1
	change_indexes = np.empty(num_complexity_lvls+1, dtype = np.int32)
	change_indexes[0] = 0
	current_index = 1
	for i in range(sorted_structure.shape[0]-1):
		if sorted_structure[i+1] != sorted_structure[i]:
			change_indexes[current_index] = i+1
			current_index += 1
	change_indexes[current_index] = sorted_structure.shape[0]-1
	return change_indexes

'''Trains model on a curriculum.
   Can perform data accumulation, complexity levels training range specification, streamlining, and plotting.'''
def learn_curriculum(x, y, sorted_structure, accumulate = False, lvl_range = (0, 0), streamline = False, plot = True):
	val_proportion, test_proportion = 0.2, 0.1
	x_train, y_train, x_val, y_val, x_test, y_test, selection_indexes = split_data(x, y, val_proportion, test_proportion)
	sorted_structure = np.delete(sorted_structure, selection_indexes, axis = 0)
	change_indexes = find_change_indexes(sorted_structure)
	if accumulate:
		accumulated_data = accumulate_data(x_train, y_train, change_indexes)
		x_train = accumulated_data[0]
		y_train = accumulated_data[1]
		change_indexes = accumulated_data[2]
	if 0 <= lvl_range[0] < lvl_range[1] <= change_indexes[change_indexes.shape[0]-1]:
		print("training on complexity levels {} through {}".format(lvl_range[0], lvl_range[1]))
		x_train = x_train[change_indexes[lvl_range[0]]:change_indexes[lvl_range[1]]]
		y_train = y_train[change_indexes[lvl_range[0]]:change_indexes[lvl_range[1]]]
	else:
		print("training on complexity levels {} through {}".format(change_indexes[0], change_indexes.shape[0]-1))
	if plot:
		bar_positions = np.zeros(change_indexes.shape[0])
		bar_positions[0] = 0
		bar_heights = np.empty(bar_positions.shape[0])
		bar_widths = np.empty(bar_positions.shape[0])
		bar_colours = np.empty(bar_positions.shape[0], dtype=object)
		bar_colours[0] = "grey"
		figure, axs = plt.subplots(2, 2)
		axs[0, 0].set_title("Validation F1 Score")
		axs[0, 1].set_title("Test F1 Score")
		axs[1, 0].set_title("Validation Loss")
		axs[1, 1].set_title("Test Loss")
		figure.suptitle('F1 Scores and Losses for Training Data and Validation Data with Non-productive Training Disregarded')
	train_step_size = 100
	updates_per_lvl = np.empty(change_indexes.shape[0]-1, dtype = int)
	for i in range(updates_per_lvl.shape[0]):
		updates_per_lvl[i] = math.ceil((change_indexes[i+1]-change_indexes[i])/train_step_size)
	total_num_updates = np.sum(updates_per_lvl)
	val_step_size = math.floor(x.shape[0]*val_proportion/total_num_updates)
	network = create_model()
	print("initial performance")
	initial_results = network.evaluate(x = x_test, y = y_test, batch_size = x_test.shape[0])
	param_backup = network.get_weights()
	prev_threshold = 0
	threshold = 0
	prev_train_f1 = initial_results[1]
	prev_val_f1 = initial_results[1]
	prev_train_loss = initial_results[0]
	prev_val_loss = initial_results[0]
	end_point = 0
	counted_updates = 0
	for i in range(change_indexes.shape[0]-1):
		num_updates = updates_per_lvl[i]
		train_f1_scores = np.empty(num_updates+1)
		val_f1_scores = np.empty(num_updates+1)
		train_losses = np.empty(num_updates+1)
		val_losses = np.empty(num_updates+1)
		start_point = end_point
		disregard_count = 0
		print("complexity level {} of {} with at most {} updates".format(i+1, change_indexes.shape[0]-1, num_updates))
		j = 0
		while j < num_updates:
			if change_indexes[i]+(j+1)*train_step_size < change_indexes[i+1]:
				update = network.fit(x = x_train[change_indexes[i]+j*train_step_size:change_indexes[i]+(j+1)*train_step_size],
									 y = y_train[change_indexes[i]+j*train_step_size:change_indexes[i]+(j+1)*train_step_size],
									 validation_data = (x_val[j*val_step_size:(j+1)*val_step_size], y_val[j*val_step_size:(j+1)*val_step_size]),
									 batch_size = 1, validation_batch_size = val_step_size, verbose = 0, shuffle = False)
			else:
				update = network.fit(x = x_train[change_indexes[i]+j*train_step_size:change_indexes[i+1]],
									 y = y_train[change_indexes[i]+j*train_step_size:change_indexes[i+1]],
									 validation_data = (x_val[j*val_step_size:(j+1)*val_step_size], y_val[j*val_step_size:(j+1)*val_step_size]),
									 batch_size = 1, validation_batch_size = val_step_size, verbose = 0, shuffle = False)
			val_f1 = update.history["val_f1_macro"][0]
			train_f1_scores[j+1] = update.history["f1_macro"][0]
			val_f1_scores[j+1] = val_f1
			train_losses[j+1] = update.history["loss"][0]
			val_losses[j+1] = update.history["val_loss"][0]
			end_point += 1
			print("general loss: {}, general F1: {}".format(update.history["val_loss"][0], val_f1))
			new_threshold = ((threshold*counted_updates)/(counted_updates+1))+((val_f1/(counted_updates+1)))
			counted_updates += 1
			j += 1
			if streamline and val_f1 > threshold:
				threshold = val_f1
			elif not streamline and new_threshold > threshold:
				threshold = new_threshold
				param_backup = network.get_weights()
				disregard_count = 0
			else:
				disregard_count += 1
		if plot:
			bar_positions[i] = start_point
			if i != 0:
				if prev_threshold != threshold:
					if bar_colours[i-1] != "grey":
						bar_colours[i] = "grey"
					else:
						bar_colours[i] = "silver"
				else:
					if bar_colours[i-1] == "grey":
						bar_colours[i] = "grey"
					else:
						bar_colours[i] = "silver"
			x_range = np.empty(end_point-start_point+1)
			x_range[0] = start_point-1
			for j in range(x_range.shape[0]):
				x_range[j] = start_point+j
		train_f1_scores[0] = prev_train_f1
		val_f1_scores[0] = prev_val_f1
		train_losses[0] = prev_train_loss
		val_losses[0] = prev_val_loss
		if plot:
			axs[0, 0].plot(x_range, val_f1_scores)
			axs[0, 1].plot(x_range, train_f1_scores)
			axs[1, 0].plot(x_range, val_losses)
			axs[1, 1].plot(x_range, train_losses)
		print("disregarding {} updates to prevent overfitting\n".format(disregard_count))
		network.set_weights(param_backup)
		end_point -= disregard_count
		counted_updates -= disregard_count
		if plot:
			if disregard_count < x_range.shape[0]-1:
				prev_train_f1 = train_f1_scores[train_f1_scores.shape[0]-1-disregard_count]
				prev_val_f1 = val_f1_scores[val_f1_scores.shape[0]-1-disregard_count]
				prev_train_loss = train_losses[train_losses.shape[0]-1-disregard_count]
				prev_val_loss = val_losses[val_losses.shape[0]-1-disregard_count]
			bar_heights[i] = threshold
			bar_widths[i] = end_point-start_point
		prev_threshold = threshold
	print("testing on unseen data")
	evaluation = network.evaluate(x = x_test, y = y_test, batch_size = x_test.shape[0])
	print("end_point: {}".format(end_point))
	final_f1 = evaluation[1]
	if plot:
		bar_positions[bar_positions.shape[0] - 1] = end_point
		bar_heights[bar_heights.shape[0] - 1] = final_f1
		bar_widths[bar_widths.shape[0] - 1] = total_num_updates - end_point
		bar_colours[bar_colours.shape[0] - 1] = "red"
		axs[0, 0].set_xlim(([0, 2*end_point+disregard_count]))
		axs[0, 1].set_xlim(([0, 2*end_point+disregard_count]))
		axs[1, 0].set_xlim(([0, 2*end_point+disregard_count]))
		axs[1, 1].set_xlim(([0, 2*end_point+disregard_count]))
		axs[0, 0].bar(bar_positions, height = bar_heights, width = bar_widths, align = "edge", color = bar_colours, alpha = 0.2)
		axs[0, 0].set_title("Vaildation F1 Score")
		axs[0, 1].set_title("Training F1 Score")
		axs[1, 0].set_title("Validation Loss")
		axs[1, 1].set_title("Training Loss")
	if plot:
		plt.show()
	return network

'''Counts the amount of data in each complexity level of a curriculum.'''
def calculate_complexity_frequencies(change_indexes):
	frequencies = np.empty(change_indexes.shape[0]-1)
	for i in range(change_indexes.shape[0]-1):
		frequencies[i] = change_indexes[i+1]-change_indexes[i]
	return frequencies

'''Counts colour clusters in a cube state to calculate a cluster score.'''
def update_score(example, score_sum):
	for cha in range(6):
		for row in range(3):
			for col in range(18):
				if example[cha][row][col] == 1:
					if col != 0 and col != 3 and col != 5 and col != 9 and col != 12 and col != 15 and row != 0:
						if example[cha][row-1][col-1] == 1:  # top left
							score_sum += 1
					if row != 0:
						if example[cha][row-1][col] == 1:  # top middle
							score_sum += 1
					if col != 2 and col != 5 and col != 8 and col != 11 and col != 14 and col != 17 and row != 0:
						if example[cha][row-1][col+1] == 1:  # top right
							score_sum += 1
					if col != 17:
						if example[cha][row][col+1] == 1:  # middle right
							score_sum += 1
					if col != 2 and col != 5 and col != 8 and col != 11 and col != 14 and col != 17 and row != 2:
						if example[cha][row+1][col+1] == 1:  # bottom right
							score_sum += 1
					if row != 2:
						if example[cha][row+1][col] == 1:  # bottom middle
							score_sum += 1
					if col != 0 and col != 3 and col != 5 and col != 9 and col != 12 and col != 15 and row != 2:
						if example[cha][row+1][col-1] == 1:  # bottom left
							score_sum += 1
					if col != 0:
						if example[cha][row][col-1] == 1:  # middle left
							score_sum += 1
	return score_sum

'''Applies a specified move to a cube state to produce a new cube state.'''
def move(example, moved_example, label_index):
	if label_index%2 == 0:
		is_inverted = False
	else:
		is_inverted = True
	if label_index < 2:
		a, b, c, d, e, pattern_type = 0, 3, 15, 11, 12, 0
	elif label_index < 4:
		a, b, c, d, e, pattern_type = 3, 6, 15, 2, 12, 2
	elif label_index < 6:
		a, b, c, d, e, pattern_type = 6, 9, 17, 5, 14, 5
	elif label_index < 8:
		a, b, c, d, e, pattern_type = 9, 0, 15, 8, 12, 1
	elif label_index < 10:
		a, b, c, d, e, pattern_type = 12, 6, 3, 0, 9, 4
	else:
		a, b, c, d, e, pattern_type = 15, 6, 9, 0, 3, 3
	a0, a1, a2 = a, a+1, a+2
	if pattern_type == 0:
		b0, b1, b2 = b, b, b
		b3, b4, b5 = 0, 1, 2
		c0, c1, c2 = c, c, c
		c3, c4, c5 = 0, 1, 2
		d0, d1, d2 = d, d, d
		d3, d4, d5 = 2, 1, 0
		e0, e1, e2 = e, e, e
		e3, e4, e5 = 0, 1, 2
	elif pattern_type == 5:
		b0, b1, b2 = b, b, b
		b3, b4, b5 = 0, 1, 2
		c0, c1, c2 = c, c, c
		c3, c4, c5 = 2, 1, 0
		d0, d1, d2 = d, d, d
		d3, d4, d5 = 2, 1, 0
		e0, e1, e2 = e, e, e
		e3, e4, e5 = 2, 1, 0
	elif pattern_type == 1:
		b0, b1, b2 = b, b, b
		b3, b4, b5 = 2, 1, 0
		c0, c1, c2 = c, c+1, c+2
		c3, c4, c5 = 0, 0, 0
		d0, d1, d2 = d, d, d
		d3, d4, d5 = 0, 1, 2
		e0, e1, e2 = e+2, e+1, e
		e3, e4, e5 = 2, 2, 2
	elif pattern_type == 2:
		b0, b1, b2 = b, b, b
		b3, b4, b5 = 0, 1, 2
		c0, c1, c2 = c, c+1, c+2
		c3, c4, c5 = 2, 2, 2
		d0, d1, d2 = d, d, d
		d3, d4, d5 = 2, 1, 0
		e0, e1, e2 = e+2, e+1, e
		e3, e4, e5 = 0, 0, 0
	elif pattern_type == 3:
		b0, b1, b2 = b, b+1, b+2
		b3, b4, b5 = 0, 0, 0
		c0, c1, c2 = c, c+1, c+2
		c3, c4, c5 = 0, 0, 0
		d0, d1, d2 = d, d+1, d+2
		d3, d4, d5 = 0, 0, 0
		e0, e1, e2 = e, e+1, e+2
		e3, e4, e5 = 0, 0, 0
	else:
		b0, b1, b2 = b, b+1, b+2
		b3, b4, b5 = 2, 2, 2
		c0, c1, c2 = c, c+1, c+2
		c3, c4, c5 = 2, 2, 2
		d0, d1, d2 = d, d+1, d+2
		d3, d4, d5 = 2, 2, 2
		e0, e1, e2 = e, e+1, e+2
		e3, e4, e5 = 2, 2, 2
	if is_inverted:
		for channel in range(6):
			moved_example[channel][0][a0] = example[channel][0][a2]
			moved_example[channel][0][a1] = example[channel][1][a2]
			moved_example[channel][0][a2] = example[channel][2][a2]
			moved_example[channel][1][a2] = example[channel][2][a1]
			moved_example[channel][2][a2] = example[channel][2][a0]
			moved_example[channel][2][a1] = example[channel][1][a0]
			moved_example[channel][2][a0] = example[channel][0][a0]
			moved_example[channel][1][a0] = example[channel][0][a1]
			moved_example[channel][b3][b0] = example[channel][e3][e0]
			moved_example[channel][b4][b1] = example[channel][e4][e1]
			moved_example[channel][b5][b2] = example[channel][e5][e2]
			moved_example[channel][c3][c0] = example[channel][b3][b0]
			moved_example[channel][c4][c1] = example[channel][b4][b1]
			moved_example[channel][c5][c2] = example[channel][b5][b2]
			moved_example[channel][d3][d0] = example[channel][c3][c0]
			moved_example[channel][d4][d1] = example[channel][c4][c1]
			moved_example[channel][d5][d2] = example[channel][c5][c2]
			moved_example[channel][e3][e0] = example[channel][d3][d0]
			moved_example[channel][e4][e1] = example[channel][d4][d1]
			moved_example[channel][e5][e2] = example[channel][d5][d2]
	else:
		for channel in range(6):
			moved_example[channel][0][a0] = example[channel][2][a0]
			moved_example[channel][0][a1] = example[channel][1][a0]
			moved_example[channel][0][a2] = example[channel][0][a0]
			moved_example[channel][1][a2] = example[channel][0][a1]
			moved_example[channel][2][a2] = example[channel][0][a2]
			moved_example[channel][2][a1] = example[channel][1][a2]
			moved_example[channel][2][a0] = example[channel][2][a2]
			moved_example[channel][1][a0] = example[channel][2][a1]
			moved_example[channel][b3][b0] = example[channel][c3][c0]
			moved_example[channel][b4][b1] = example[channel][c4][c1]
			moved_example[channel][b5][b2] = example[channel][c5][c2]
			moved_example[channel][c3][c0] = example[channel][d3][d0]
			moved_example[channel][c4][c1] = example[channel][d4][d1]
			moved_example[channel][c5][c2] = example[channel][d5][d2]
			moved_example[channel][d3][d0] = example[channel][e3][e0]
			moved_example[channel][d4][d1] = example[channel][e4][e1]
			moved_example[channel][d5][d2] = example[channel][e5][e2]
			moved_example[channel][e3][e0] = example[channel][b3][b0]
			moved_example[channel][e4][e1] = example[channel][b4][b1]
			moved_example[channel][e5][e2] = example[channel][b5][b2]

'''Orders training data to fit a curriculum's sctructure.'''
def order(x, y, structure, reverse = False):
	sorted_structure = np.sort(structure)
	x_structured = np.empty(x.shape)
	y_structured = np.empty(y.shape)
	max_difficulty = np.amax(structure)
	min_difficulty = np.amin(structure)
	current_index = 0
	for complexity in range(min_difficulty, max_difficulty+1):
		for i in range(structure.shape[0]):
			if structure[i] == complexity:
				x_structured[current_index] = x[i]
				y_structured[current_index] = y[i]
				current_index += 1
	if reverse:
		x_structured = np.flip(x_structured, axis = 0)
		y_structured = np.flip(y_structured, axis = 0)
		sorted_structure = np.flip(sorted_structure, axis = 0)
	return x_structured, y_structured, sorted_structure

'''Randomly shuffles training data and makes call to train a baseline model.'''
def train_baseline(x, y, streamline = False, plot = True):
	print("training baseline model")
	permutation = np.random.permutation(x.shape[0])
	x = x[permutation]
	y = y[permutation]
	sorted_structure = np.zeros(x.shape[0])
	return learn_curriculum(x, y, sorted_structure, accumulate = False, streamline = streamline, plot = plot)

'''Orders training data based on distance from the solved state and makes call to train a model with a distance based curriculum.'''
def train_distance(x_structured, y_structured, reverse = False, accumulate = False, lvl_range = (0, 0), streamline = False, plot = True):
	print("training with the distance curriculum")
	sorted_structure = np.empty(x_structured.shape[0])
	current_index = 0
	with open("train.txt", mode = 'r', newline = '') as source:
		for line in source:
			if len(line) > 18*3*6:
				if line[325] == ',':
					sorted_structure[current_index] = int(line[326:-2])
				else:
					sorted_structure[current_index] = int(line[327:-2])
				current_index += 1
	if reverse:
		x_structured = np.flip(x_structured, axis = 0)
		y_structured = np.flip(y_structured, axis = 0)
		sorted_structure = np.flip(sorted_structure, axis = 0)
	return learn_curriculum(x_structured, y_structured, sorted_structure, accumulate = accumulate, lvl_range = lvl_range, streamline = streamline, plot = plot)

'''Orders training data based on the ratio of cluster scores from the scrambled cube state to the cube state after a predicted move is make and makes call to train a model with a cluster based curriculum.'''
def train_clusters(x, y, metric = 'd', reverse = False, accumulate = False, lvl_range = (0, 0), streamline = False, plot = True):
	print("training with the clusters curriculum")
	if metric == 'd':
		print("using the division metric")
	elif metric == 's':
		print("using the subtraction metric")
	else:
		print("invalid metric so defaulting to division")
	moved_representation = copy.deepcopy(x)
	structure = np.empty(x.shape[0], dtype = int)
	for example in range(x.shape[0]):
		score = update_score(x[example], 0)
		score_sum = 0
		num_optimal_moves = 0
		for i in range(y.shape[1]):
			label = y[example][i]
			if label == 1:
				num_optimal_moves += 1
				move(x[example], moved_representation[example], i)
				score_sum = update_score(moved_representation[example], score_sum)
				moved_representation[example] = copy.deepcopy(x[example])
		new_score = int(score_sum/num_optimal_moves)
		if metric == 's':
			difficulty = new_score-score
		else:
			difficulty = int(10*new_score/score)
		structure[example] = difficulty
	structured_data = order(x, y, structure, reverse = reverse)
	x_structured = structured_data[0]
	y_structured = structured_data[1]
	sorted_structure = structured_data[2]
	return learn_curriculum(x_structured, y_structured, sorted_structure, accumulate = accumulate, lvl_range = lvl_range, streamline = streamline, plot = plot)

'''Orders training data based on the number of truth vaues for each example and makes the call to train a model with a label based curriculum.'''
def train_labels(x, y, reverse = False, accumulate = False, lvl_range = (0, 0), streamline = False, plot = True):
	if reverse:
		reverse = False
	else:
		reverse = True
	print("training with the labels curriculum")
	structure = np.sum(y, axis = 1, dtype = int)
	structure_data = order(x, y, structure, reverse = reverse)
	x, y, sorted_structure = structure_data[0], structure_data[1], structure_data[2]
	return learn_curriculum(x, y, sorted_structure, accumulate = accumulate, lvl_range = lvl_range, streamline = streamline, plot = plot)

'''Generates a solution sequence for an input scramble by making predictions using a saved trained model.
   Stops if the cube state is not the solved state after 100 predicted moves'''
def solve(sequence):
	move_dict = {0: "L", 1: "L'", 2: "F", 3: "F'", 4: "R", 5: "R'", 6: "B", 7: "B'", 8: "D", 9: "D'", 10: "U", 11: "U'"}
	move_dict_reversed = {"L": 0, "L'": 1, "F": 2, "F'": 3, "R": 4, "R'": 5, "B": 6, "B'": 7, "D": 8, "D'": 9, "U": 10, "U'": 11}
	trained_network = keras.models.load_model('trained_network', custom_objects={"f1_macro": f1_macro, "generous_accuracy_macro": generous_accuracy_macro, "multilabel_binary_crossentropy": multilabel_binary_crossentropy})
	solved_state = np.array([[[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							  [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							  [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
							 [[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							  [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							  [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
							 [[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							  [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							  [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
							 [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
							  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
							  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]],
							 [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
							  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
							  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]],
							 [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
							  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
							  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]]])
	scramble = []
	scramble_index = 0
	for i in range(len(sequence)):
		if sequence[i] == "'":
			if scramble[scramble_index-1]%2 == 0:
				scramble[scramble_index-1] += 1
			else:
				scramble[scramble_index-1] -= 1
		else:
			scramble.append(move_dict_reversed[sequence[i]])
			scramble_index += 1
	next_state = copy.deepcopy(solved_state)
	for i in range(len(scramble)):
		prev_state = copy.deepcopy(next_state)
		move(prev_state, next_state, scramble[i])
	solution_string = ""
	prev_prev_move = 12
	prev_move = 12
	num_predictions_made = 0
	while not np.array_equal(next_state, solved_state) and num_predictions_made < 100:
		prev_state = copy.deepcopy(next_state)
		move_choices = trained_network.predict(np.expand_dims(prev_state, axis = 0), batch_size = 1)
		prediction = np.argmax(move_choices)
		if (prev_move%2 == 0 and prediction == prev_move+1) or (prev_move%2 != 0 and prediction == prev_move-1) or (prev_prev_move == prev_move == prediction):
			if prediction == 11:
				prediction = np.argmax(move_choices[0][:prediction])
			elif prediction == 0:
				prediction = np.argmax(move_choices[0][prediction+1:])
			else:
				prediction = np.argmax(np.concatenate((move_choices[:, :prediction], move_choices[:, prediction+1:]), axis = 1)[0])
		move(prev_state, next_state, prediction)
		prev_prev_move = prev_move
		prev_move = prediction
		solution_string += move_dict[prediction]
		num_predictions_made += 1
	if num_predictions_made < 100:
		print(solution_string)
	else:
		print("unable to solve")

'''Working with raw training data before import.'''
#generate_scrambles()
#structure_num_moves()

'''Getting data.'''
input_data, labels = read_data("train")

'''Training a baseline model.'''
#trained_network = train_baseline(input_data, labels, plot = False)

'''Training a model using a curriculum.'''
for i in range(4):
#trained_network = train_distance(input_data, labels)
#trained_network = train_clusters(input_data, labels)
	trained_network = train_labels(input_data, labels, plot = False)

'''Saving and loading a trained model.'''
#trained_network.save("trained_network")
#trained_network = keras.models.load_model('trained_network', custom_objects={"f1_macro": f1_macro, "generous_accuracy_macro": generous_accuracy_macro, "multilabel_binary_crossentropy": multilabel_binary_crossentropy})

'''Solving an input scramble using the model.'''
#solve("RF'F'BL")

