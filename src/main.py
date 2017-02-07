import scipy.io as sio
import numpy as np
from numpy import linalg as la
from PIL import Image
import math
from collections import Counter
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time


# Function to display sample grid_size^2 digits as a combined image
# digits matrix should be of dimension at least 28*28*grid_size
def display_digit_images(digits_matrix, grid_size, image_name):
    # grid_size = 2
    # grid_size += 1
    group_image = np.zeros((28 * grid_size, 28 * grid_size))

    # print get_time_string(), group_image.shape
    # print get_time_string(), digits_matrix.shape

    for i in range(1, grid_size+1):
        for j in range(1, grid_size+1):
            if (i - 1) * grid_size + j - 1 >= digits_matrix.shape[2]:
                break
            group_image[(i - 1) * 28: i * 28, (j - 1) * 28: j * 28] = digits_matrix[:, :, (i - 1) * grid_size + j - 1]
    img = Image.fromarray(group_image)
    # print get_time_string(), image_name
    img_rgb = img.convert('RGB')
    img_rgb.save(image_name, 'jpeg')
    # img.show()
    return


# Takes training set images as input and returns the eigen values and eigen vectors of the set in decreasing order
def hw1FindEigendigits(training_set_images):
    # Compute the mean of the columns in the chosen sample images
    training_set_images_mean = np.mean(training_set_images, axis=1)

    # Normalize the sample images by subtracting the mean
    training_set_images_without_mean = (
        training_set_images.transpose() - training_set_images_mean).transpose()  # (784 x K)

    # Since computing eigen vectors of smaller dimension matrix is computationally cheaper
    smaller_dim_mat = np.dot(training_set_images_without_mean.transpose(), training_set_images_without_mean)  # (K x K)
    eigen_values, eigen_vectors_temp = la.eig(smaller_dim_mat)  # (K x K)

    # Calculate the actual eigen values by multiplying with A
    eigen_vectors = np.dot(training_set_images_without_mean, eigen_vectors_temp)  # (784 x K)

    sort_indices = np.argsort(eigen_values)
    num_eigen_values = eigen_values.size

    sorted_eigen_values = np.zeros(num_eigen_values)
    sorted_eigen_vectors = np.zeros((eigen_vectors.shape[0], eigen_vectors.shape[1]))  # (784 x K)

    for i in range(num_eigen_values - 1, 0, -1):
        sorted_eigen_values[num_eigen_values - 1 - i] = eigen_values[sort_indices[i]]
        sorted_eigen_vectors[:, num_eigen_values - 1 - i] = eigen_vectors[:, sort_indices[i]]


    # Normalize the eigen vectors obtained
    eigen_vectors_normalized = normalize_columns(sorted_eigen_vectors)

    return training_set_images_mean, sorted_eigen_vectors, eigen_vectors_normalized


# Given a matrix, normalize each column (each eigen vector) of the matrix
def normalize_columns(A):
    r = A.shape[0]
    c = A.shape[1]
    column_norms = np.zeros(c)
    A_normalized = np.zeros((r, c))
    for j in range(0, c - 1):
        sq = 0
        for i in range(0, r - 1):
            sq = sq + A[i, j] * A[i, j]
        norm2 = math.sqrt(sq)
        column_norms[j] = norm2

    for i in range(0, r - 1):
        for j in range(0, c - 1):
            A_normalized[i, j] = A[i, j] / column_norms[j]
    return A_normalized


# Given a set of eigen vectors and training set of images A, project the training set to eigen space
def project_to_eigen_space(eigen_vectors, A):
    # Find mean of the column vectors
    mean = np.mean(A, axis=1)

    # Normalize the sample images by subtracting the mean
    A_without_mean = (A.transpose() - mean).transpose()

    eigen_projected_vectors = np.dot(eigen_vectors.transpose(), A_without_mean)
    return eigen_projected_vectors


def pause():
    # raw_input('Press <ENTER> to continue...')
    return


# Cosine based
def k_nearest_neighbours_cosine(train_set_eigen_projected, test_set_eigen_projected, k):
    test_set_size = test_set_eigen_projected.shape[1]
    train_set_size = train_set_eigen_projected.shape[1]

    # Nearest neighbour Matrix 'N' where each row is for each test image and columns are the indices of train_images
    # in increasing order of nearness
    N = np.zeros((test_set_size, k))

    for i in range(0, test_set_size - 1):
        cosine_dists = np.zeros(train_set_size)
        test_img = test_set_eigen_projected[:, i]
        for j in range(train_set_size):
            train_img = train_set_eigen_projected[:, j]
            cos_theta = float(np.dot(test_img, train_img)) / float((la.norm(test_img) * la.norm(train_img)))
            cosine_dists[j] = cos_theta
        sort_indices = np.argsort(cosine_dists)[::-1]  # Sort in descending order of acos
        N[i, :] = sort_indices[0: k]

    return N


# Manhattan distance based
def k_nearest_neighbours_manhattan(train_set_eigen_projected, test_set_eigen_projected, k):
    test_set_size = test_set_eigen_projected.shape[1]
    train_set_size = train_set_eigen_projected.shape[1]

    # Nearest neighbour Matrix 'N' where each row is for each test image and columns are the indices of train_images
    # in increasing order of nearness
    N = np.zeros((test_set_size, k))

    for i in range(0, test_set_size - 1):
        dists = np.zeros(train_set_size)
        test_img = test_set_eigen_projected[:, i]
        for j in range(train_set_size):
            train_img = train_set_eigen_projected[:, j]
            diff_img = train_img - test_img
            dists[j] = np.sum(np.absolute(diff_img))
        sort_indices = np.argsort(dists)
        N[i, :] = sort_indices[0: k]

    return N


# Euclidean distance based
def k_nearest_neighbours_euclidean(train_set_eigen_projected, test_set_eigen_projected, k):
    test_set_size = test_set_eigen_projected.shape[1]
    train_set_size = train_set_eigen_projected.shape[1]

    # Nearest neighbour Matrix 'N' where each row is for each test image and columns are the indices of train_images
    # in increasing order of nearness
    N = np.zeros((test_set_size, k))

    for i in range(0, test_set_size - 1):
        dists = np.zeros(train_set_size)
        test_img = test_set_eigen_projected[:, i]
        for j in range(train_set_size):
            train_img = train_set_eigen_projected[:, j]
            diff_img = train_img - test_img
            dists[j] = la.norm(diff_img)
        sort_indices = np.argsort(dists)
        N[i, :] = sort_indices[0: k]

    return N


def k_nearest_neighbours(train_set_eigen_projected, test_set_eigen_projected, k, type):
    types = ['Euclidean', 'Manhattan', 'Cosine']
    if type == 'Euclidean':
        return k_nearest_neighbours_euclidean(train_set_eigen_projected, test_set_eigen_projected, k)
    elif type == 'Manhattan':
        return k_nearest_neighbours_manhattan(train_set_eigen_projected, test_set_eigen_projected, k)
    elif type == 'Cosine':
        return k_nearest_neighbours_cosine(train_set_eigen_projected, test_set_eigen_projected, k)
    else:
        print 'Invalid type of distance for classification! Please select one among ', types
    return -1


def most_frequent_value(arr):
    counts = Counter(arr)
    return int(counts.most_common(1)[0][0])


def assign_labels_to_test_set(N, train_set_labels):
    test_set_size = N.shape[0]
    test_set_labels = np.zeros(test_set_size)

    for i in range(0, test_set_size - 1):
        index = most_frequent_value(N[i, :])
        train_set_label = train_set_labels[:, index]
        test_set_labels[i] = train_set_label

    return test_set_labels


def get_accuracy(test_set_labels, actual_test_set_labels):
    test_set_size = test_set_labels.shape[0]
    match = 0
    for i in range(0, test_set_size - 1):
        if test_set_labels[i] == actual_test_set_labels[0, i]:
            match += 1
    print get_time_string(), 'Matches: %d/%d' % (match, test_set_size)
    accuracy = (float)(match * 100) / (float)(test_set_size)
    print get_time_string(), 'Accuracy: %f' % accuracy
    return accuracy


def eigen_digits_classify(arg_training_set_size, arg_test_set_size, arg_knn_size, arg_num_eigen_vectors, difficulty,
                          experiment_number, sub_experiment_number, trial_number, knn_type):
    dir_name = '../images_' + `experiment_number`
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # print get_time_string(), dir_name
    file_prefix = knn_type + '_' + `arg_training_set_size` + '_' + `arg_test_set_size` + '_' + `arg_knn_size` + '_' + `arg_num_eigen_vectors` + '_' + `difficulty` + '_' + `sub_experiment_number` + '_' + `trial_number` + '_'

    print get_time_string(), 'Loading contents of matrix...'
    input_contents = sio.loadmat('../data/digits.mat')

    # Read training data
    train_all_labels = input_contents['trainLabels']
    train_all_images_2D = input_contents['trainImages']  # 2D refers to data present as (28 x 28) matrix for each sample
    train_all_images_2D = train_all_images_2D.astype(float)

    # Read test data
    test_all_lables = input_contents['testLabels']
    test_all_images_2D = input_contents['testImages']
    test_all_images_2D = test_all_images_2D.astype(float)

    train_all_images = np.reshape(train_all_images_2D, (784, 60000))
    test_all_images = np.reshape(test_all_images_2D, (784, 10000))

    # Choose some number of random columns from the train_images_2D
    num_eigen_vectors = arg_num_eigen_vectors
    print get_time_string(), '%d random images from the training set is chosen to construct the eigen space. ' % num_eigen_vectors
    eigen_space_training_set_indices = np.random.randint(0, train_all_images.shape[1], num_eigen_vectors)
    eigen_space_training_set_images = train_all_images[:, eigen_space_training_set_indices]

    # Display the sample training images used for eigen space construction
    eigen_space_training_set_images_2D = np.reshape(eigen_space_training_set_images, (28, 28, num_eigen_vectors))
    print get_time_string(), 'Displaying some sample images from the training set chosen to construct eigen space... '
    file_name = dir_name + '/' + file_prefix + 'sample_eigen_space_training_images.jpeg'
    display_digit_images(eigen_space_training_set_images_2D, 5, file_name)
    pause()

    # Find the eigen values and eigen vectors of the co-variance matrix
    training_set_images_mean, eigen_vectors, eigen_vectors_normalized = hw1FindEigendigits(eigen_space_training_set_images)  # eigen_vectors: (784 x K)

    # Display the first 100 eigen vectors as images
    eigen_vectors_2D = np.reshape(eigen_vectors, (28, 28, num_eigen_vectors))
    print get_time_string(), 'Displaying few eigen vectors of the covariance matrix in decreasing order of eigen values...'
    file_name = dir_name + '/' + file_prefix + 'eigen_vectors_images.jpeg'
    display_digit_images(eigen_vectors_2D, 5, file_name)
    pause()

    # Construct a set of images as training set
    training_set_size = arg_training_set_size
    print get_time_string(), '%d random images from the training set is chosen. ' % training_set_size
    training_set_indices = np.random.randint(0, train_all_images.shape[1], training_set_size)
    training_set_images = train_all_images[:, training_set_indices]
    training_set_labels = train_all_labels[:, training_set_indices]

    # Display a few sample training images
    training_set_images_2D = np.reshape(training_set_images, (28, 28, training_set_size))
    print get_time_string(), 'Displaying some sample images from the training set chosen to construct eigen space... '
    file_name = dir_name + '/' + file_prefix + 'sample_training_images.jpeg'
    display_digit_images(training_set_images_2D, 5, file_name)
    pause()

    # Project the sample training set to eigen space and reconstruct them
    print get_time_string(), 'Projecting the training set to eigen space and reconstructing...'
    training_set_images_eigen_projected = project_to_eigen_space(eigen_vectors_normalized, training_set_images)
    training_set_images_reconstructed = np.dot(eigen_vectors_normalized, training_set_images_eigen_projected)
    training_set_images_reconstructed_2D = np.reshape(training_set_images_reconstructed, (28, 28, training_set_size))
    print get_time_string(), 'Displaying a few reconstructed images from the training set...'
    file_name = dir_name + '/' + file_prefix + 'sample_training_images_reconstructed.jpeg'
    display_digit_images(training_set_images_reconstructed_2D, 5, file_name)
    pause()

    # Select a test set
    test_set_size = arg_test_set_size
    print get_time_string(), '%d random images from the test set is chosen. ' % test_set_size
    if difficulty == 0:  # easy
        test_set_indices = np.random.randint(0, 5000, test_set_size)
    else:  # difficult
        test_set_indices = np.random.randint(5000, 10000, test_set_size)
    test_set_images = test_all_images[:, test_set_indices]
    test_set_labels = test_all_lables[:, test_set_indices]

    # Display the sample test images
    test_set_images_2D = np.reshape(test_set_images, (28, 28, test_set_size))
    print get_time_string(), 'Displaying some sample images from the test set chosen... '
    file_name = dir_name + '/' + file_prefix + 'sample_test_images.jpeg'
    display_digit_images(test_set_images_2D, 5, file_name)
    pause()

    # Project the sample test set to eigen space and reconstruct them
    print get_time_string(), 'Projecting the test set to eigen space and reconstructing...'
    test_set_images_eigen_projected = project_to_eigen_space(eigen_vectors_normalized, test_set_images)
    test_set_images_reconstructed = np.dot(eigen_vectors_normalized, test_set_images_eigen_projected)
    test_set_images_reconstructed_2D = np.reshape(test_set_images_reconstructed, (28, 28, test_set_size))
    print get_time_string(), 'Displaying a few reconstructed images from the test set...'
    file_name = dir_name + '/' + file_prefix + 'sample_test_images_reconstructed.jpeg'
    display_digit_images(test_set_images_reconstructed_2D, 5, file_name)
    pause()

    k = arg_knn_size
    print get_time_string(), 'Finding nearest neighbour (k=%d) indices for test set with the training set...' % k
    N = k_nearest_neighbours(training_set_images_eigen_projected, test_set_images_eigen_projected, k, knn_type)

    # Assign labels to test set based on nearest neighbours and their frequency
    print get_time_string(), 'Assigning labels to test images...'
    obtained_test_set_labels = assign_labels_to_test_set(N, training_set_labels)

    print get_time_string(), 'Getting accuracy for the test images...'
    accuracy = get_accuracy(obtained_test_set_labels, test_set_labels)
    # print get_time_string(), 'Accuracy: %d' % accuracy

    return accuracy


def get_min_axis_for_accuracy(min_accuracy):
    min_accuracy = int(min_accuracy)
    min_accuracy -= 10
    if min_accuracy <= 0:
        return min_accuracy
    min_accuracy -= min_accuracy % 10
    return min_accuracy


def print_divider():
    print '-------------------------------------------------------------------------------------------------------------------------------------'
    return


def print_divider_with_text(text):
    print '------------------------------------------------', text, '---------------------------------------------------------------------------'
    return


def print_usage_batch():
    print 'python -u main.py batch [<experiment_number>] [<number_of_trials_for_each_sub_experiment>] [<test>]'
    return


def get_cumulative_plt_time(plt_times):
    cum_plt_times = []
    for i in range(len(plt_times[0])):
        cum_plt_times.extend([(plt_times[0][i] + plt_times[1][i]) / 2.0])
    return cum_plt_times


def get_time_string():
    return time.strftime('%c') + ' '


def parse_command_line_args(sys_arg):
    print get_time_string(), 'Number of arguments: %d' % len(sys_arg)
    experiment_number = -1
    number_of_trials = 0
    test = 1
    if len(sys_arg) == 2 or len(sys_arg) == 1:  # Default values
        experiment_number = 1
        number_of_trials = 5
        test = 1
    if len(sys_arg) == 3:
        experiment_number = int(sys_arg[2])
        number_of_trials = 5
        test = 0
    if len(sys_arg) == 4:
        experiment_number = int(sys_arg[2])
        number_of_trials = int(sys_arg[3])
        test = 0
    if len(sys_arg) == 5:
        experiment_number = int(sys_arg[2])
        number_of_trials = int(sys_arg[3])
        if sys_arg[4] == 'test':
            test = 1
        else:
            print_usage_batch()
            return
    if experiment_number == -1:
        print_usage_batch()
        return
    driver(experiment_number, number_of_trials, test)


def parse_command_line_args_adhoc(sys_arg):
    # print get_time_string(), 'Number of arguments: %d' % len(sys_arg)
    if len(sys_arg) != 9:
        print_usage_adhoc()
        return -1, -1, -1, -1, -1, -1, -1
    return sys_arg[2], sys_arg[3], sys_arg[4], sys_arg[5], sys_arg[6], sys_arg[7], sys_arg[8]


def plot_eigen_value_vs_accurace(experiment_number, sub_experiment_number, knn_size, training_set_size, test_set_size,
                                 eigen_vector_sizes, total_trials, difficulty_string, plot_dir_name,
                                 eigen_vector_sizes_plt_extra_ul, knn_type):
    print_divider_with_text('Sub-experiment ' + `sub_experiment_number`)
    print get_time_string(), knn_type, ' Finding optimal value of num_eigen_vectors by fixing knn_size to %d, training_set_size to %d and test_set_size to %d' % (
    knn_size, training_set_size, test_set_size)
    print_divider()

    max_num_eigen_vectors = [0, 0]
    max_accuracies = [0.0, 0.0]
    min_accuracy = 100.0
    plt_accuracies = []
    for i in range(2):
        plt_accuracies.append([])
    plt_time_taken = []
    for i in range(2):
        plt_time_taken.append([])

    for difficulty in [0, 1]:
        print_divider_with_text('Difficulty ' + `difficulty`)

        for num_eigen_vectors in eigen_vector_sizes:
            print_divider_with_text('Num_eigen_vectors ' + `num_eigen_vectors`)
            total_accuracy = 0.0
            total_time = 0.0
            num_trials = 0
            for trial in range(total_trials):
                print_divider_with_text('Trial ' + `trial`)
                start_time = time.time()
                total_accuracy += eigen_digits_classify(training_set_size, test_set_size, knn_size, num_eigen_vectors,
                                                        difficulty, experiment_number, sub_experiment_number, trial,
                                                        knn_type)
                total_time += time.time() - start_time
                print_divider()
                num_trials += 1
            accuracy = total_accuracy / float(num_trials)
            avg_time = total_time / float(num_trials)
            if accuracy > max_accuracies[difficulty]:
                max_accuracies[difficulty] = accuracy
                max_num_eigen_vectors[difficulty] = num_eigen_vectors
            if accuracy < min_accuracy:
                min_accuracy = accuracy
            plt_accuracies[difficulty].extend([accuracy])
            plt_time_taken[difficulty].extend([avg_time])

        print get_time_string(), 'Difficulty: ', difficulty_string[difficulty], ' Accuracies obtained: ', \
        plt_accuracies[difficulty]
        print get_time_string(), 'Difficulty: ', difficulty_string[difficulty], ' max_num_eigen_vectors: ', \
        max_num_eigen_vectors[difficulty]
        print get_time_string(), 'Difficulty: ', difficulty_string[difficulty], ' Average time taken for one run: ', \
        plt_time_taken[difficulty]

    # Plot as graph and save to file
    file_name = plot_dir_name + '/' + `experiment_number` + '_' + `sub_experiment_number` + '_' + knn_type + '_' + 'num_eigen_vectors_vs_accuracy.png'
    cum_time_taken = get_cumulative_plt_time(plt_time_taken)
    plot_graph_with_time(eigen_vector_sizes, plt_accuracies, cum_time_taken, 'Num eigen vectors', 'Accuracy (%)',
                         'Time taken (sec)',
                         'Effect of number of eigen vectors on accuracy',
                         [0, eigen_vector_sizes[-1] + eigen_vector_sizes_plt_extra_ul,
                          get_min_axis_for_accuracy(min_accuracy), 100, max(0, cum_time_taken[0] - 20),
                          cum_time_taken[-1] + 20], file_name)

    print get_time_string(), knn_type,  ' max_num_eigen_vectors: ', max_num_eigen_vectors
    print get_time_string(), knn_type,  ' max_accuracies: ', max_accuracies
    print get_time_string(), knn_type,  ' plt_accuracies: ', plt_accuracies
    print get_time_string(), knn_type,  ' plt_time_taken: ', plt_time_taken

    return max_num_eigen_vectors, max_accuracies, plt_accuracies, plt_time_taken


def plot_training_set_size_vs_accuracy(experiment_number, sub_experiment_number, knn_size, test_set_size,
                                       max_num_eigen_vectors, training_set_sizes, total_trials, difficulty_string,
                                       plot_dir_name, training_set_sizes_plt_extra_ul, knn_type):
    # Fix all parameters but training_set_size and find optimal value
    print_divider_with_text('Sub-experiment ' + `sub_experiment_number`)
    print get_time_string(), knn_type, ' Finding optimal value of training_set_size by fixing knn_size to %d, num_eigen_vectors to (%d, %d) and test_set_size to %d' % (
        knn_size, max_num_eigen_vectors[0], max_num_eigen_vectors[1], test_set_size)
    print_divider()

    max_training_set_size = [0, 0]
    max_accuracies = [0.0, 0.0]
    min_accuracy = 100.0
    plt_accuracies = []
    for i in range(2):
        plt_accuracies.append([])
    plt_time_taken = []
    for i in range(2):
        plt_time_taken.append([])

    for difficulty in [0, 1]:
        print_divider_with_text('Difficulty ' + `difficulty`)

        for training_set_size in training_set_sizes:
            print_divider_with_text('Training set size ' + `training_set_size`)
            total_accuracy = 0.0
            total_time = 0.0
            num_trials = 0
            for trial in range(total_trials):
                print_divider_with_text('Trial ' + `trial`)
                start_time = time.time()
                total_accuracy += eigen_digits_classify(training_set_size, test_set_size, knn_size,
                                                        max_num_eigen_vectors[difficulty], difficulty,
                                                        experiment_number, sub_experiment_number, trial, knn_type)
                total_time += time.time() - start_time
                print_divider()
                num_trials += 1
            accuracy = total_accuracy / float(num_trials)
            avg_time = total_time / float(num_trials)
            if accuracy > max_accuracies[difficulty]:
                max_accuracies[difficulty] = accuracy
                max_training_set_size[difficulty] = training_set_size
            if accuracy < min_accuracy:
                min_accuracy = accuracy
            plt_accuracies[difficulty].extend([accuracy])
            plt_time_taken[difficulty].extend([avg_time])

        print get_time_string(), 'Difficulty: ', difficulty_string[difficulty], ' Accuracies obtained: ', \
            plt_accuracies[difficulty]
        print get_time_string(), 'Difficulty: ', difficulty_string[difficulty], ' max_training_set_size: ', \
            max_training_set_size[difficulty]
        print get_time_string(), 'Difficulty: ', difficulty_string[difficulty], ' Average time taken for one run: ', \
            plt_time_taken[difficulty]

    # Plot as graph and save to file
    file_name = plot_dir_name + '/' + `experiment_number` + '_' + `sub_experiment_number` + '_' + knn_type + '_' + 'training_set_size_vs_accuracy.png'
    cum_time_taken = get_cumulative_plt_time(plt_time_taken)
    plot_graph_with_time(training_set_sizes, plt_accuracies, cum_time_taken, 'Training set size', 'Accuracy (%)',
                         'Time taken (sec)',
                         'Effect of training_set_size on accuracy',
                         [0, training_set_sizes[-1] + training_set_sizes_plt_extra_ul,
                          get_min_axis_for_accuracy(min_accuracy), 100, max(0, cum_time_taken[0] - 20),
                          cum_time_taken[-1] + 20], file_name)

    print get_time_string(), knn_type,  ' max_training_set_size: ', max_training_set_size
    print get_time_string(), knn_type,  ' max_accuracies: ', max_accuracies
    print get_time_string(), knn_type,  ' plt_accuracies: ', plt_accuracies
    print get_time_string(), knn_type,  ' plt_time_taken: ', plt_time_taken

    return max_training_set_size, max_accuracies, plt_accuracies, plt_time_taken


def plot_knn_size_vs_accuracy(experiment_number, sub_experiment_number, test_set_size, max_num_eigen_vectors,
                              max_training_set_size, knn_sizes, total_trials, difficulty_string, plot_dir_name,
                              knn_sizes_plt_extra_ul, knn_type):
    # Fix all parameters but knn size and find optimal value
    print_divider_with_text('Sub-experiment ' + `sub_experiment_number`)
    print get_time_string(), knn_type, ' Finding optimal value of knn_size by fixing test_set_size to %d, num_eigen_vectors to (%d, %d) and training_set_size to (%d, %d)' % (
    test_set_size, max_num_eigen_vectors[0], max_num_eigen_vectors[1], max_training_set_size[0],
    max_training_set_size[1])
    print_divider()

    max_knn_size = [0, 0]
    max_accuracies = [0.0, 0.0]
    min_accuracy = 100.0
    plt_accuracies = []
    for i in range(2):
        plt_accuracies.append([])
    plt_time_taken = []
    for i in range(2):
        plt_time_taken.append([])

    for difficulty in [0, 1]:
        print_divider_with_text('Difficulty ' + `difficulty`)

        for knn_size in knn_sizes:
            print_divider_with_text('Knn size ' + `knn_size`)
            total_accuracy = 0.0
            total_time = 0.0
            num_trials = 0
            for trial in range(total_trials):
                print_divider_with_text('Trial ' + `trial`)
                start_time = time.time()
                total_accuracy += eigen_digits_classify(max_training_set_size[difficulty],
                                                        test_set_size, knn_size,
                                                        max_num_eigen_vectors[difficulty], difficulty,
                                                        experiment_number, sub_experiment_number, trial, knn_type)
                total_time += time.time() - start_time
                print_divider()
                num_trials += 1
            accuracy = total_accuracy / float(num_trials)
            avg_time = total_time / float(num_trials)
            if accuracy > max_accuracies[difficulty]:
                max_accuracies[difficulty] = accuracy
                max_knn_size[difficulty] = knn_size
            if accuracy < min_accuracy:
                min_accuracy = accuracy
            plt_accuracies[difficulty].extend([accuracy])
            plt_time_taken[difficulty].extend([avg_time])

        print get_time_string(), 'Difficulty: ', difficulty_string[difficulty], ' Accuracies obtained: ', \
        plt_accuracies[difficulty]
        print get_time_string(), 'Difficulty: ', difficulty_string[difficulty], ' max_knn_size: ', max_knn_size[
            difficulty]
        print get_time_string(), 'Difficulty: ', difficulty_string[difficulty], ' Average time taken for one run: ', \
        plt_time_taken[difficulty]

    # Plot as graph and save to file
    file_name = plot_dir_name + '/' + `experiment_number` + '_' + `sub_experiment_number` + '_' + knn_type + '_' + 'knn_size_vs_accuracy.png'
    cum_time_taken = get_cumulative_plt_time(plt_time_taken)
    plot_graph_with_time(knn_sizes, plt_accuracies, cum_time_taken, 'Knn size', 'Accuracy (%)', 'Time taken (sec)',
                         'Effect of knn size on accuracy',
                         [0, knn_sizes[-1] + knn_sizes_plt_extra_ul, get_min_axis_for_accuracy(min_accuracy), 100,
                          max(0, cum_time_taken[0] - 20), cum_time_taken[-1] + 20], file_name)

    print get_time_string(), knn_type,  ' max_knn_size: ', max_knn_size
    print get_time_string(), knn_type,  ' max_accuracies: ', max_accuracies
    print get_time_string(), knn_type,  ' plt_accuracies: ', plt_accuracies
    print get_time_string(), knn_type,  ' plt_time_taken: ', plt_time_taken

    return max_knn_size, max_accuracies, plt_accuracies, plt_time_taken


def driver(arg_experiment_number, arg_total_trials, arg_test):
    experiment_number = arg_experiment_number

    plot_dir_name = '../plots_' + `experiment_number`
    if not os.path.exists(plot_dir_name):
        os.makedirs(plot_dir_name)

    if arg_test == 0:
        eigen_vector_sizes = [5, 10, 15, 20, 25, 35, 40, 45, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500]
        training_set_sizes = [5, 10, 15, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
        knn_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 17, 20, 22, 25, 28, 30, 35, 40, 45, 50, 60, 75, 100]
        total_trials = arg_total_trials

        training_set_sizes_plt_extra_ul = 1000
        knn_sizes_plt_extra_ul = 10
        eigen_vector_sizes_plt_extra_ul = 100
    else:
        training_set_sizes = [10, 50]
        knn_sizes = [1, 2]
        eigen_vector_sizes = [10, 50]
        total_trials = 1

        training_set_sizes_plt_extra_ul = 10
        knn_sizes_plt_extra_ul = 2
        eigen_vector_sizes_plt_extra_ul = 10

    difficulty_string = ['hard', 'easy']
    sub_experiment_number = 1
    print_divider_with_text('Experiment ' + `experiment_number`)
    print get_time_string(), 'Experiment starting with %d and sub_experiment %d' % (
    experiment_number, sub_experiment_number)
    print_divider()

    # Fix all parameters except num_eigen_vectors and find optimal value
    knn_size = 5
    if arg_test == 0:
        training_set_size = 5000
        test_set_size = 100
    else:
        training_set_size = 100
        test_set_size = 50

    euclidean_max_num_eigen_vectors, euclidean_max_accuracies_eigen, euclidean_plt_accuracies_eigen, euclidean_plt_time_taken_eigen = plot_eigen_value_vs_accurace(
        experiment_number, sub_experiment_number, knn_size, training_set_size, test_set_size, eigen_vector_sizes,
        total_trials, difficulty_string, plot_dir_name, eigen_vector_sizes_plt_extra_ul, 'Euclidean')
    sub_experiment_number += 1
    euclidean_max_training_set_size, euclidean_max_accuracies_training, euclidean_plt_accuracies_training, euclidean_plt_time_taken_training = plot_training_set_size_vs_accuracy(
        experiment_number, sub_experiment_number, knn_size, test_set_size, euclidean_max_num_eigen_vectors,
        training_set_sizes, total_trials, difficulty_string, plot_dir_name, training_set_sizes_plt_extra_ul,
        'Euclidean')
    sub_experiment_number += 1
    euclidean_max_knn_size, euclidean_max_accuracies_knn, euclidean_plt_accuracies_knn, euclidean_plt_time_taken_knn = plot_knn_size_vs_accuracy(
        experiment_number, sub_experiment_number, test_set_size, euclidean_max_num_eigen_vectors,
        euclidean_max_training_set_size, knn_sizes, total_trials, difficulty_string, plot_dir_name,
        knn_sizes_plt_extra_ul, 'Euclidean')
    sub_experiment_number += 1


    cosine_max_num_eigen_vectors, cosine_max_accuracies_eigen, cosine_plt_accuracies_eigen, cosine_plt_time_taken_eigen = plot_eigen_value_vs_accurace(
        experiment_number, sub_experiment_number, knn_size, training_set_size, test_set_size, eigen_vector_sizes,
        total_trials, difficulty_string, plot_dir_name, eigen_vector_sizes_plt_extra_ul, 'Cosine')
    sub_experiment_number += 1
    cosine_max_training_set_size, cosine_max_accuracies_training, cosine_plt_accuracies_training, cosine_plt_time_taken_training = plot_training_set_size_vs_accuracy(
        experiment_number, sub_experiment_number, knn_size, test_set_size, cosine_max_num_eigen_vectors,
        training_set_sizes, total_trials, difficulty_string, plot_dir_name, training_set_sizes_plt_extra_ul, 'Cosine')
    sub_experiment_number += 1
    cosine_max_knn_size, cosine_max_accuracies_knn, cosine_plt_accuracies_knn, cosine_plt_time_taken_knn = plot_knn_size_vs_accuracy(
        experiment_number, sub_experiment_number, test_set_size, cosine_max_num_eigen_vectors,
        cosine_max_training_set_size, knn_sizes, total_trials, difficulty_string, plot_dir_name, knn_sizes_plt_extra_ul,
        'Cosine')
    sub_experiment_number += 1

    manhattan_max_num_eigen_vectors, manhattan_max_accuracies_eigen, manhattan_plt_accuracies_eigen, manhattan_plt_time_taken_eigen = plot_eigen_value_vs_accurace(
        experiment_number, sub_experiment_number, knn_size, training_set_size, test_set_size, eigen_vector_sizes,
        total_trials, difficulty_string, plot_dir_name, eigen_vector_sizes_plt_extra_ul, 'Manhattan')
    sub_experiment_number += 1
    manhattan_max_training_set_size, manhattan_max_accuracies_training, manhattan_plt_accuracies_training, manhattan_plt_time_taken_training = plot_training_set_size_vs_accuracy(
        experiment_number, sub_experiment_number, knn_size, test_set_size, manhattan_max_num_eigen_vectors,
        training_set_sizes, total_trials, difficulty_string, plot_dir_name, training_set_sizes_plt_extra_ul,
        'Manhattan')
    sub_experiment_number += 1
    manhattan_max_knn_size, manhattan_max_accuracies_knn, manhattan_plt_accuracies_knn, manhattan_plt_time_taken_knn = plot_knn_size_vs_accuracy(
        experiment_number, sub_experiment_number, test_set_size, manhattan_max_num_eigen_vectors,
        manhattan_max_training_set_size, knn_sizes, total_trials, difficulty_string, plot_dir_name,
        knn_sizes_plt_extra_ul, 'Manhattan')
    sub_experiment_number += 1

    for difficulty in [0, 1]:
        file_name = plot_dir_name + '/' + `experiment_number` + '_' + `sub_experiment_number` + '_knn_compare_eigen_vary_' + \
                    difficulty_string[difficulty] + '.png'
        accuracy_min_scale = get_min_axis_for_accuracy(np.min(
            [np.min(euclidean_plt_accuracies_eigen[difficulty]), np.min(cosine_plt_accuracies_eigen[difficulty]),
             np.min(manhattan_plt_accuracies_eigen[difficulty])]))
        plot_graph_compare_knn(eigen_vector_sizes, euclidean_plt_accuracies_eigen[difficulty],
                               cosine_plt_accuracies_eigen[difficulty],
                               manhattan_plt_accuracies_eigen[difficulty], 'Number of eigen vectors', 'Accuracy (%)',
                               'Comparsion of accuracy for types of distance in KNN (' + difficulty_string[
                                   difficulty] + ')',
                               [0, eigen_vector_sizes[-1] + eigen_vector_sizes_plt_extra_ul, accuracy_min_scale, 100],
                               file_name)

        file_name = plot_dir_name + '/' + `experiment_number` + '_' + `sub_experiment_number` + '_knn_compare_training_set_vary_' + \
                    difficulty_string[difficulty] + '.png'
        accuracy_min_scale = get_min_axis_for_accuracy(np.min(
            [np.min(euclidean_plt_accuracies_training[difficulty]), np.min(cosine_plt_accuracies_training[difficulty]),
             np.min(manhattan_plt_accuracies_training[difficulty])]))
        plot_graph_compare_knn(training_set_sizes, euclidean_plt_accuracies_training[difficulty],
                               cosine_plt_accuracies_training[difficulty],
                               manhattan_plt_accuracies_training[difficulty], 'Training set size', 'Accuracy (%)',
                               'Comparsion of accuracy for types of distance in KNN (' + difficulty_string[
                                   difficulty] + ')',
                               [0, training_set_sizes[-1] + training_set_sizes_plt_extra_ul, accuracy_min_scale, 100],
                               file_name)

        file_name = plot_dir_name + '/' + `experiment_number` + '_' + `sub_experiment_number` + '_knn_compare_knn_size_vary_' + \
                    difficulty_string[difficulty] + '.png'
        accuracy_min_scale = get_min_axis_for_accuracy(np.min(
            [np.min(euclidean_plt_accuracies_knn[difficulty]), np.min(cosine_plt_accuracies_knn[difficulty]),
             np.min(manhattan_plt_accuracies_knn[difficulty])]))
        plot_graph_compare_knn(knn_sizes, euclidean_plt_accuracies_knn[difficulty],
                               cosine_plt_accuracies_knn[difficulty],
                               manhattan_plt_accuracies_knn[difficulty], 'Knn size', 'Accuracy (%)',
                               'Comparsion of accuracy for types of distance in KNN (' + difficulty_string[
                                   difficulty] + ')',
                               [0, knn_sizes[-1] + knn_sizes_plt_extra_ul, accuracy_min_scale, 100],
                               file_name)

    # for i in range(0, len(training_set_sizes)):
    #     for j in range(0, len(test_set_sizes)):
    #         for l in range(0, len(eigen_vector_sizes)):
    #             for k in range(0, len(knn_sizes)):
    #                 for difficulty in [0, 1]: # 0 for easy and 1 for difficult test set
    #                     accuracy = 0.0
    #                     for trial in range (1, 10): # do ten trials and take average accuracy
    #                         # print get_time_string(), 'Calling eigen digits classification for parameters: training_set_size: %d test_set_size: %d knn_size: %d num_eigen_vectors: %d ' % (training_set_sizes[i], test_set_sizes[j], knn_sizes[k], eigen_vector_sizes[l])
    #                         # eigen_digits_classify(training_set_sizes[i], test_set_sizes[j], knn_sizes[k], eigen_vector_sizes[l], difficulty, experiment_number, trial)
    # print get_time_string(), 'Accuracy: %d' % eigen_digits_classify(10000, 500, 5, 500, 1, 1, 1)

    return


def tune_num_eigen_vectors(knn_size, experiment_number):
    training_set_sizes = [10, 20, 30, 40, 50, 75, 100, 150, 200, 250, 350, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000]
    training_set_sizes_plt_extra_ul = 1000
    eigen_vector_sizes = [10, 50, 100, 250, 500]
    # knn_size = 5
    # experiment_number = 51
    difficulty_string = ['hard', 'easy']
    total_trials = 1
    plot_dir_name = '../plots_' + `experiment_number`
    if not os.path.exists(plot_dir_name):
        os.makedirs(plot_dir_name)
    plot_tune_num_eigen_vectors(experiment_number, 1, 500, eigen_vector_sizes, training_set_sizes, knn_size, total_trials, difficulty_string, plot_dir_name, training_set_sizes_plt_extra_ul, knn_type='Euclidean')


def plot_tune_num_eigen_vectors(experiment_number, sub_experiment_number, test_set_size, eigen_vector_sizes,
                              training_set_sizes, knn_size, total_trials, difficulty_string, plot_dir_name, training_set_sizes_plt_extra_ul, knn_type):
    # Plot accuracy vs training set size for different values of num_eigen_vectors
    print_divider_with_text('Sub-experiment ' + `sub_experiment_number`)
    print get_time_string(), knn_type, ' Finding plots of accuracy vs training set for different values of eigen vectors for knn size: ', knn_size
    print_divider()

    min_accuracy = 100.0
    num_plots = len(eigen_vector_sizes)
    num_training_set_sizes = len(training_set_sizes)
    plt_accuracies = [[[0.0 for k in xrange(num_training_set_sizes)] for j in xrange(num_plots)] for i in xrange(2)]

    colors = ['blue', 'green', 'red', 'magenta', 'cyan']
    linestyles = ['b-o', 'g-o', 'r-o', 'm-o', 'c-o']

    for difficulty in [0, 1]:
        print_divider_with_text('Difficulty ' + `difficulty`)
        j = 0
        legends = []
        for num_eigen_vectors in eigen_vector_sizes:
            k = 0
            legends.extend([`num_eigen_vectors` + ' eigen vectors'])
            print_divider_with_text('Num eigen vectors ' + `num_eigen_vectors`)
            for training_set_size in training_set_sizes:
                print_divider_with_text('Training set size ' + `training_set_size`)
                total_accuracy = 0.0
                # total_time = 0.0
                num_trials = 0
                for trial in range(total_trials):
                    print_divider_with_text('Trial ' + `trial`)
                    # start_time = time.time()
                    total_accuracy += eigen_digits_classify(training_set_size,
                                                            test_set_size, knn_size,
                                                            num_eigen_vectors, difficulty,
                                                            experiment_number, sub_experiment_number, trial, knn_type)
                    # total_time += time.time() - start_time
                    print_divider()
                    num_trials += 1
                accuracy = total_accuracy / float(num_trials)
                # avg_time = total_time / float(num_trials)
                plt_accuracies[difficulty][j][k] = accuracy
                if accuracy < min_accuracy:
                    min_accuracy = accuracy
                k += 1
            j += 1

        print get_time_string(), 'Difficulty: ', difficulty_string[difficulty], ' Accuracies obtained: ', plt_accuracies[difficulty]

        # Plot as graph and save to file
        file_name = plot_dir_name + '/' + `experiment_number` + '_' + `sub_experiment_number` + '_' + knn_type + '_' + difficulty_string[difficulty] + '_' + `knn_size` + '_' + 'tune_num_eigen_vectors.png'
        print 'Plotting tuning of num_eigen_vectors for difficulty ', difficulty
        plot_tune_graph(training_set_sizes, plt_accuracies[difficulty], linestyles, legends, colors, 'Training set size', 'Accuracy (%)',
                             'Training set size vs Accuracy for different eigen vectors',
                             [0, training_set_sizes[-1] + training_set_sizes_plt_extra_ul, get_min_axis_for_accuracy(min_accuracy), 100], file_name, 1.5, 1.0)
        print get_time_string(), knn_type,  ' plt_accuracies: ', plt_accuracies

    return


def tune_training_set_size(num_eigen_vectors, experiment_number):
    training_set_sizes = [100, 250, 500, 1000, 5000]
    knn_sizes_plt_extra_ul = 5
    knn_sizes = [1, 2, 3, 4, 5, 6, 7, 9, 13, 16, 20, 25, 35, 45, 50]
    # num_eigen_vectors = 500
    # experiment_number = 51
    difficulty_string = ['hard', 'easy']
    total_trials = 1
    plot_dir_name = '../plots_' + `experiment_number`
    if not os.path.exists(plot_dir_name):
        os.makedirs(plot_dir_name)
    plot_tune_training_set_size(experiment_number, 1, 500, num_eigen_vectors, training_set_sizes, knn_sizes, total_trials, difficulty_string, plot_dir_name, knn_sizes_plt_extra_ul, knn_type='Euclidean')


def plot_tune_training_set_size(experiment_number, sub_experiment_number, test_set_size, num_eigen_vectors,
                              training_set_sizes, knn_sizes, total_trials, difficulty_string, plot_dir_name, knn_sizes_plt_extra_ul, knn_type):
    # Plot accuracy vs training set size for different values of num_eigen_vectors
    print_divider_with_text('Sub-experiment ' + `sub_experiment_number`)
    print get_time_string(), knn_type, ' Finding plots of accuracy vs knn size for different values of trainin set size for num_eigen_vectors: ', num_eigen_vectors
    print_divider()

    min_accuracy = 100.0
    num_plots = len(training_set_sizes)
    num_knn_sizes = len(knn_sizes)
    plt_accuracies = [[[0.0 for k in xrange(num_knn_sizes)] for j in xrange(num_plots)] for i in xrange(2)]

    colors = ['blue', 'green', 'red', 'magenta', 'cyan']
    linestyles = ['b-o', 'g-o', 'r-o', 'm-o', 'c-o']

    for difficulty in [0, 1]:
        print_divider_with_text('Difficulty ' + `difficulty`)
        j = 0
        legends = []
        for training_set_size in training_set_sizes:
            k = 0
            legends.extend([`training_set_size` + ' training set'])
            print_divider_with_text('Training set ' + `training_set_size`)
            for knn_size in knn_sizes:
                print_divider_with_text('Knn size ' + `knn_size`)
                total_accuracy = 0.0
                # total_time = 0.0
                num_trials = 0
                for trial in range(total_trials):
                    print_divider_with_text('Trial ' + `trial`)
                    # start_time = time.time()
                    total_accuracy += eigen_digits_classify(training_set_size,
                                                            test_set_size, knn_size,
                                                            num_eigen_vectors, difficulty,
                                                            experiment_number, sub_experiment_number, trial, knn_type)
                    # total_time += time.time() - start_time
                    print_divider()
                    num_trials += 1
                accuracy = total_accuracy / float(num_trials)
                # avg_time = total_time / float(num_trials)
                plt_accuracies[difficulty][j][k] = accuracy
                if accuracy < min_accuracy:
                    min_accuracy = accuracy
                k += 1
            j += 1

        print get_time_string(), 'Difficulty: ', difficulty_string[difficulty], ' Accuracies obtained: ', plt_accuracies[difficulty]

        # Plot as graph and save to file
        file_name = plot_dir_name + '/' + `experiment_number` + '_' + `sub_experiment_number` + '_' + knn_type + '_' + difficulty_string[difficulty] + '_' + `num_eigen_vectors` + '_' + 'tune_training_set_size.png'
        print 'Plotting tuning of training set size for difficulty ', difficulty
        plot_tune_graph(knn_sizes, plt_accuracies[difficulty], linestyles, legends, colors, 'Knn size', 'Accuracy (%)',
                             'Knn size vs Accuracy for different training set sizes',
                             [0, knn_sizes[-1] + knn_sizes_plt_extra_ul, get_min_axis_for_accuracy(min_accuracy), 100], file_name, 1.5, 1.0)
        print get_time_string(), knn_type,  ' plt_accuracies: ', plt_accuracies

    return


def tune_knn_size(num_eigen_vectors, experiment_number):
    training_set_sizes = [10, 20, 30, 40, 50, 75, 100, 150, 200, 250, 350, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000]
    training_set_sizes_plt_extra_ul = 1000
    knn_sizes = [1, 2, 3, 5, 8]
    # knn_size = 5
    # experiment_number = 51
    difficulty_string = ['hard', 'easy']
    total_trials = 1
    plot_dir_name = '../plots_' + `experiment_number`
    if not os.path.exists(plot_dir_name):
        os.makedirs(plot_dir_name)
    plot_tune_knn_sizes(experiment_number, 1, 500, num_eigen_vectors, training_set_sizes, knn_sizes, total_trials, difficulty_string, plot_dir_name, training_set_sizes_plt_extra_ul, knn_type='Euclidean')


def plot_tune_knn_sizes(experiment_number, sub_experiment_number, test_set_size, num_eigen_vectors,
                              training_set_sizes, knn_sizes, total_trials, difficulty_string, plot_dir_name, training_set_sizes_plt_extra_ul, knn_type):
    # Plot accuracy vs training set size for different values of num_eigen_vectors
    print_divider_with_text('Sub-experiment ' + `sub_experiment_number`)
    print get_time_string(), knn_type, ' Finding plots of accuracy vs training set for different values of knn for num_eigen_vectors: ', num_eigen_vectors
    print_divider()

    min_accuracy = 100.0
    num_plots = len(knn_sizes)
    num_training_set_sizes = len(training_set_sizes)
    plt_accuracies = [[[0.0 for k in xrange(num_training_set_sizes)] for j in xrange(num_plots)] for i in xrange(2)]

    colors = ['blue', 'green', 'red', 'magenta', 'cyan']
    linestyles = ['b-o', 'g-o', 'r-o', 'm-o', 'c-o']

    for difficulty in [0, 1]:
        print_divider_with_text('Difficulty ' + `difficulty`)
        j = 0
        legends = []
        for knn_size in knn_sizes:
            k = 0
            legends.extend([`knn_size` + ' k-NN'])
            print_divider_with_text('Knn size ' + `knn_size`)
            for training_set_size in training_set_sizes:
                print_divider_with_text('Training set size ' + `training_set_size`)
                total_accuracy = 0.0
                # total_time = 0.0
                num_trials = 0
                for trial in range(total_trials):
                    print_divider_with_text('Trial ' + `trial`)
                    # start_time = time.time()
                    total_accuracy += eigen_digits_classify(training_set_size,
                                                            test_set_size, knn_size,
                                                            num_eigen_vectors, difficulty,
                                                            experiment_number, sub_experiment_number, trial, knn_type)
                    # total_time += time.time() - start_time
                    print_divider()
                    num_trials += 1
                accuracy = total_accuracy / float(num_trials)
                # avg_time = total_time / float(num_trials)
                plt_accuracies[difficulty][j][k] = accuracy
                if accuracy < min_accuracy:
                    min_accuracy = accuracy
                k += 1
            j += 1

        print get_time_string(), 'Difficulty: ', difficulty_string[difficulty], ' Accuracies obtained: ', plt_accuracies[difficulty]

        # Plot as graph and save to file
        file_name = plot_dir_name + '/' + `experiment_number` + '_' + `sub_experiment_number` + '_' + knn_type + '_' + difficulty_string[difficulty] + '_' + `num_eigen_vectors` + '_' + 'tune_knn_size.png'
        print 'Plotting tuning of knn_size for difficulty ', difficulty
        plot_tune_graph(training_set_sizes, plt_accuracies[difficulty], linestyles, legends, colors, 'Training set size', 'Accuracy (%)',
                             'Training set size vs Accuracy for different eigen vectors',
                             [0, training_set_sizes[-1] + training_set_sizes_plt_extra_ul, get_min_axis_for_accuracy(min_accuracy), 100], file_name, 1.4, 1.0)
        print get_time_string(), knn_type,  ' plt_accuracies: ', plt_accuracies

    return


def plot_tune_graph(x_axis, y_axis, linestyles, legends, colors, x_label, y_label, title, axis, file_name, bba_x, bba_y):
    print get_time_string(), 'Plotting %s to file %s' % (title, file_name)

    print get_time_string(), 'x_axis: ', x_axis
    print get_time_string(), 'y_axis: ', y_axis
    print get_time_string(), 'linestyles: ', linestyles
    print get_time_string(), 'legends: ', legends
    print get_time_string(), 'colors: ', colors
    print get_time_string(), 'x_label: ', x_label
    print get_time_string(), 'y_label: ', y_label
    print get_time_string(), 'axis: ', axis

    num_plots = len(legends)
    patches = []
    for i in range(num_plots):
        plt.plot(x_axis, y_axis[i], linestyles[i], label=legends[i])
        patches.extend([mpatches.Patch(color=colors[i], label=legends[i], linestyle='solid', linewidth=0.1)])

    # plt.legend(handles=[red_euclidean_patch, blue_cosine_patch, green_manhattan_patch], loc=2)

    lgd = plt.legend(handles=patches, loc='upper right', bbox_to_anchor=(bba_x, bba_y))

    plt.grid(True)
    plt.axis(axis)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    fig = plt.figure(1)
    fig.savefig(file_name, dpi=300, format='png', bbox_extra_artists=(lgd,), bbox_inches='tight')

    # plt.show()
    # plt.savefig(file_name)
    plt.close()


def plot_graph_compare_knn(x_axis, y_axis1, y_axis2, y_axis3, x_label, y_label, title, axis, file_name):
    print get_time_string(), 'Plotting %s to file %s' % (title, file_name)

    print get_time_string(), 'x_axis: ', x_axis
    print get_time_string(), 'y_axis1: ', y_axis1
    print get_time_string(), 'y_axis2: ', y_axis2
    print get_time_string(), 'y_axis3: ', y_axis3
    print get_time_string(), 'x_label: ', x_label
    print get_time_string(), 'y_label: ', y_label
    print get_time_string(), 'axis: ', axis

    plt.plot(x_axis, y_axis1, 'r-o', label='euclidean')
    plt.plot(x_axis, y_axis2, 'b-o', label='cosine')
    plt.plot(x_axis, y_axis3, 'g-o', label='manhattan')

    red_euclidean_patch = mpatches.Patch(color='red', label='Euclidean', linestyle='solid', linewidth=0.1)
    blue_cosine_patch = mpatches.Patch(color='blue', label='Cosine', linestyle='solid', linewidth=0.1)
    green_manhattan_patch = mpatches.Patch(color='green', label='Manhattan', linestyle='solid', linewidth=0.1)

    # plt.legend(handles=[red_euclidean_patch, blue_cosine_patch, green_manhattan_patch], loc=2)

    lgd = plt.legend(handles=[red_euclidean_patch, blue_cosine_patch, green_manhattan_patch], loc='upper right', bbox_to_anchor=(1.4, 1.0))

    plt.grid(True)
    plt.axis(axis)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    fig = plt.figure(1)
    fig.savefig(file_name, dpi=300, format='png', bbox_extra_artists=(lgd,), bbox_inches='tight')

    # plt.show()
    # plt.savefig(file_name)
    plt.close()


# def plot_graph(x_axis, y_axis, x_label, y_label, title, axis, file_name):
#     print get_time_string(), 'Plotting %s to file %s' % (title, file_name)
#     plt.plot(x_axis, y_axis[0], 'r-o', label='hard')
#     plt.plot(x_axis, y_axis[1], 'b-o', label='easy')
#
#     red_hard_patch = mpatches.Patch(color='red', label='Hard', linestyle='solid', linewidth=0.1)
#     blue_easy_patch = mpatches.Patch(color='blue', label='Easy', linestyle='solid', linewidth=0.1)
#     plt.legend(handles=[blue_easy_patch, red_hard_patch], loc=2)
#
#     plt.grid(True)
#     plt.axis(axis)
#     plt.xlabel(x_label)
#     plt.ylabel(y_label)
#     plt.title(title)
#     # plt.show()
#     plt.savefig(file_name)
#     plt.close()


def plot_graph_with_time(x_axis, y_axis, y2_axis, x_label, y_label, y2_label, title, axis, file_name):
    print get_time_string(), 'Plotting %s to file %s' % (title, file_name)

    print get_time_string(), 'x_axis: ', x_axis
    print get_time_string(), 'y_axis: ', y_axis
    print get_time_string(), 'y2_axis: ', y2_axis
    print get_time_string(), 'x_label: ', x_label
    print get_time_string(), 'y_label: ', y_label
    print get_time_string(), 'y2_label: ', y2_label
    print get_time_string(), 'axis: ', axis

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(x_axis, y_axis[0], 'r-o', label='hard')
    ax1.plot(x_axis, y_axis[1], 'b-o', label='easy')
    ax2.plot(x_axis, y2_axis, 'g-.o', label='time')

    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax2.set_ylabel(y2_label, color='g')

    ax1.set_xlim(left=axis[0], right=axis[1])
    ax1.set_ylim(bottom=axis[2], top=axis[3])
    ax2.set_ylim(bottom=axis[4], top=axis[5])
    # ax1.set_xticks(x_ticks)
    # ax1.set_yticks([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100])

    red_hard_patch = mpatches.Patch(color='red', label='Hard', linestyle='solid', linewidth=0.1)
    blue_easy_patch = mpatches.Patch(color='blue', label='Easy', linestyle='solid', linewidth=0.1)
    green_time_patch = mpatches.Patch(color='green', label='Time', linestyle='solid', linewidth=0.1)

    lgd = plt.legend(handles=[blue_easy_patch, red_hard_patch, green_time_patch], loc='upper right', bbox_to_anchor=(1.3, 1.0))

    ax1.grid(True)
    plt.title(title)

    fig = plt.figure(1)
    fig.savefig(file_name, dpi=300, format='png', bbox_extra_artists=(lgd,), bbox_inches='tight')

    # plt.savefig(file_name)
    # fig.savefig('image_output.png', dpi = 400, format = 'png', bbox_extra_artists = (lgd,), bbox_inches = 'tight')
    plt.close()


def test_pyplot2():
    plot_graph_compare_knn([50, 100, 500, 1000, 5000, 10000, 25000, 50000, 60000], [23.5, 34.5, 45.5, 67, 76, 79, 84, 87, 92],  [22, 33, 44, 63, 71, 75, 80, 84, 89], [18, 30, 41, 60, 69, 72, 28, 81, 86], 'Training set size', 'Accuracy (%)', 'Training set size vs Accuracy', [0, 70000, 0, 100], 'sample2.png')


def test_pyplot():
    accuracies = [[56, 65, 76, 84, 98.5, 78.5, 89.2, 98.3], [43, 54, 64, 76, 93.2, 45.6, 74.5, 82.3]]
    time_taken = [10, 20, 25, 28, 30, 50, 70, 90]
    training_set_sizes = [10, 50, 100, 500, 1000, 15000, 30000, 50000]
    xticks = []
    i = 0
    while i <= 60000:
        xticks.extend([i])
        i += 5000

    xlabel = 'Training set size'
    ylabel = 'Accuracy (%)'
    y2label = 'Time taken (sec)'
    title = 'Effect of training set size on accuracy'
    axis = [0, 60000, 30, 100, 0, 100]
    plot_graph_with_time(training_set_sizes, xticks, accuracies, time_taken, xlabel, ylabel, y2label, title, axis, 'sample.png')


def print_usage_adhoc():
    print 'python -u main.py single <training_set_size> <test_set_size> <knn_size> <num_eigen_vectors> <difficulty> <experiment_number> <knn_distance_type>'
    return


def adhoc(sys_arg):
    training_set_size, test_set_size, knn_size, num_eigen_vectors, difficulty, experiment_number, knn_distance_type = parse_command_line_args_adhoc(sys_arg)
    if training_set_size == -1:
        return
    accuracy = eigen_digits_classify(int(training_set_size), int(test_set_size), int(knn_size), int(num_eigen_vectors), int(difficulty), int(experiment_number), 1, 1, 'Euclidean')
    print 'Accuracy obtained: ', accuracy
    # plt_times = [[3, 5, 6], [4, 1, 3]]
    # print get_time_string(), get_cumulative_plt_time(plt_times)

# experiment_number = 70
# tune_num_eigen_vectors(1, experiment_number)
# experiment_number += 1
# tune_num_eigen_vectors(2, experiment_number)
# experiment_number += 1
# tune_num_eigen_vectors(3, experiment_number)
# experiment_number += 1
# tune_num_eigen_vectors(5, experiment_number)
# experiment_number += 1
#
# tune_training_set_size(100, experiment_number)
# experiment_number += 1
# tune_training_set_size(200, experiment_number)
# experiment_number += 1
# tune_training_set_size(300, experiment_number)
# experiment_number += 1
# tune_training_set_size(500, experiment_number)
# experiment_number += 1

# experiment_number = 78
# tune_knn_size(100, experiment_number)
# experiment_number += 1
# tune_knn_size(200, experiment_number)
# experiment_number += 1
# tune_knn_size(300, experiment_number)
# experiment_number += 1
# tune_knn_size(500, experiment_number)
# experiment_number += 1

# parse_command_line_args(sys.argv)
if sys.argv[1] == 'single':
    adhoc(sys.argv)
elif sys.argv[1] == 'batch':
    parse_command_line_args(sys.argv)
else:
    print 'Valid arguments are \'single\' and \'batch\''
# test_pyplot2()