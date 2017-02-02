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


# Function to display sample grid_size^2 digits as a combined image
# digits matrix should be of dimension at least 28*28*grid_size
def display_digit_images(digits_matrix, grid_size, image_name):
    # grid_size = 2
    grid_size += 1
    group_image = np.zeros((28 * grid_size, 28 * grid_size))

    # print group_image.shape
    # print digits_matrix.shape

    for i in range(1, grid_size):
        for j in range(1, grid_size):
            if (i - 1) * grid_size + j - 1 >= digits_matrix.shape[2]:
                break
            group_image[(i - 1) * 28: i * 28, (j - 1) * 28: j * 28] = digits_matrix[:, :, (i - 1) * grid_size + j - 1]
    img = Image.fromarray(group_image)
    # print image_name
    img_rgb = img.convert('RGB')
    img_rgb.save(image_name, 'bmp')
    # img.show()
    return


# Takes training set images as input and returns the eigen values and eigen vectors of the set in decreasing order
def find_eigen_vectors(training_set_images):
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

    return sorted_eigen_values, sorted_eigen_vectors


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


def k_nearest_neighbours(train_set_eigen_projected, test_set_eigen_projected, k):
    test_set_size = test_set_eigen_projected.shape[1]
    train_set_size = train_set_eigen_projected.shape[1]

    # Nearest neighbour Matrix 'N' where each row is for each test image and columns are the indices of train_images
    # in increasing order of nearness
    N = np.zeros((test_set_size, k))

    for i in range(0, test_set_size - 1):
        dists = np.zeros(train_set_size)
        test_img = test_set_eigen_projected[:, i]
        for j in range(0, train_set_size - 1):
            train_img = train_set_eigen_projected[:, j]
            diff_img = train_img - test_img
            dists[j] = la.norm(diff_img)
        sort_indices = np.argsort(dists)
        N[i, :] = sort_indices[0: k]

    return N


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
    print 'Matches: %d/%d' % (match, test_set_size)
    accuracy = (float)(match * 100) / (float)(test_set_size)
    print 'Accuracy: %f' % accuracy
    return accuracy


def eigen_digits_classify(arg_training_set_size, arg_test_set_size, arg_knn_size, arg_num_eigen_vectors, difficulty,
                          experiment_number, sub_experiment_number, trial_number):
    dir_name = '../images_' + `experiment_number`
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # print dir_name
    file_prefix = `arg_training_set_size` + '_' + `arg_test_set_size` + '_' + `arg_knn_size` + '_' + `arg_num_eigen_vectors` + '_' + `difficulty` + '_' + `sub_experiment_number` + '_' + `trial_number` + '_'

    print 'Loading contents of matrix...'
    input_contents = sio.loadmat('/home/pandian/studies/semester2/ML/hw1/digits.mat')

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
    print '%d random images from the training set is chosen to construct the eigen space. ' % num_eigen_vectors
    eigen_space_training_set_indices = np.random.randint(0, train_all_images.shape[1], num_eigen_vectors)
    eigen_space_training_set_images = train_all_images[:, eigen_space_training_set_indices]

    # Display the sample training images used for eigen space construction
    eigen_space_training_set_images_2D = np.reshape(eigen_space_training_set_images, (28, 28, num_eigen_vectors))
    print 'Displaying some sample images from the training set chosen to construct eigen space... '
    file_name = dir_name + '/' + file_prefix + 'sample_eigen_space_training_images.bmp'
    display_digit_images(eigen_space_training_set_images_2D, 5, file_name)
    pause()

    # Find the eigen values and eigen vectors of the co-variance matrix
    eigen_values, eigen_vectors = find_eigen_vectors(eigen_space_training_set_images)  # eigen_vectors: (784 x K)

    # Display the first 100 eigen vectors as images
    eigen_vectors_2D = np.reshape(eigen_vectors, (28, 28, num_eigen_vectors))
    print 'Displaying few eigen vectors of the covariance matrix in decreasing order of eigen values...'
    file_name = dir_name + '/' + file_prefix + 'eigen_vectors_images.bmp'
    display_digit_images(eigen_vectors_2D, 5, file_name)
    pause()

    # Normalize the eigen vectors obtained
    eigen_vectors_normalized = normalize_columns(eigen_vectors)

    # Construct a set of images as training set
    training_set_size = arg_training_set_size
    print '%d random images from the training set is chosen. ' % training_set_size
    training_set_indices = np.random.randint(0, train_all_images.shape[1], training_set_size)
    training_set_images = train_all_images[:, training_set_indices]
    training_set_labels = train_all_labels[:, training_set_indices]

    # Display a few sample training images
    training_set_images_2D = np.reshape(training_set_images, (28, 28, training_set_size))
    print 'Displaying some sample images from the training set chosen to construct eigen space... '
    file_name = dir_name + '/' + file_prefix + 'sample_training_images.bmp'
    display_digit_images(training_set_images_2D, 5, file_name)
    pause()

    # Project the sample training set to eigen space and reconstruct them
    print 'Projecting the training set to eigen space and reconstructing...'
    training_set_images_eigen_projected = project_to_eigen_space(eigen_vectors_normalized, training_set_images)
    training_set_images_reconstructed = np.dot(eigen_vectors_normalized, training_set_images_eigen_projected)
    training_set_images_reconstructed_2D = np.reshape(training_set_images_reconstructed, (28, 28, training_set_size))
    print 'Displaying a few reconstructed images from the training set...'
    file_name = dir_name + '/' + file_prefix + 'sample_training_images_reconstructed.bmp'
    display_digit_images(training_set_images_reconstructed_2D, 5, file_name)
    pause()

    # Select a test set
    test_set_size = arg_test_set_size
    print '%d random images from the test set is chosen. ' % test_set_size
    if difficulty == 0:  # easy
        test_set_indices = np.random.randint(0, 5000, test_set_size)
    else:  # difficult
        test_set_indices = np.random.randint(5000, 10000, test_set_size)
    test_set_images = test_all_images[:, test_set_indices]
    test_set_labels = test_all_lables[:, test_set_indices]

    # Display the sample test images
    test_set_images_2D = np.reshape(test_set_images, (28, 28, test_set_size))
    print 'Displaying some sample images from the test set chosen... '
    file_name = dir_name + '/' + file_prefix + 'sample_test_images.bmp'
    display_digit_images(test_set_images_2D, 5, file_name)
    pause()

    # Project the sample test set to eigen space and reconstruct them
    print 'Projecting the test set to eigen space and reconstructing...'
    test_set_images_eigen_projected = project_to_eigen_space(eigen_vectors_normalized, test_set_images)
    test_set_images_reconstructed = np.dot(eigen_vectors_normalized, test_set_images_eigen_projected)
    test_set_images_reconstructed_2D = np.reshape(test_set_images_reconstructed, (28, 28, test_set_size))
    print 'Displaying a few reconstructed images from the test set...'
    file_name = dir_name + '/' + file_prefix + 'sample_test_images_reconstructed.bmp'
    display_digit_images(test_set_images_reconstructed_2D, 5, file_name)
    pause()

    k = arg_knn_size
    print 'Finding nearest neighbour (k=%d) indices for test set with the training set...' % k
    N = k_nearest_neighbours(training_set_images_eigen_projected, test_set_images_eigen_projected, k)

    # Assign labels to test set based on nearest neighbours and their frequency
    print 'Assigning labels to test images...'
    obtained_test_set_labels = assign_labels_to_test_set(N, training_set_labels)

    print 'Getting accuracy for the test images...'
    accuracy = get_accuracy(obtained_test_set_labels, test_set_labels)
    # print 'Accuracy: %d' % accuracy

    return accuracy


def get_min_axis_for_accuracy(min_accuracy):
    min_accuracy = int(min_accuracy)
    min_accuracy -= 10
    if min_accuracy <= 0:
        return min_accuracy
    min_accuracy -= min_accuracy % 10
    return min_accuracy


def print_divider():
    print '--------------------------------------------------------------------------------------------------------------------------------'
    return


def print_divider_with_text(text):
    print '-----------------------------------------', text, '--------------------------------------------------------------------------'
    return


def driver():
    experiment_number = 1

    plot_dir_name = '../plots_' + `experiment_number`
    if not os.path.exists(plot_dir_name):
        os.makedirs(plot_dir_name)

    eigen_vector_sizes = [10, 25, 50, 75, 100, 150, 200, 350, 500, 750]
    training_set_sizes = [10, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 20000, 30000, 40000, 50000, 60000]
    test_set_sizes = [100, 250, 500]
    knn_sizes = [1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50, 75, 100]
    total_trials = 5

    # training_set_sizes = [10, 50]
    # test_set_sizes = [10, 50]
    # knn_sizes = [1, 2]
    # eigen_vector_sizes = [10, 50]
    # total_trials = 1

    training_set_sizes_plt_extra_ul = 5000
    test_set_sizes_plt_extra_ul = 100
    knn_sizes_plt_extra_ul = 10
    eigen_vector_sizes_plt_extra_ul = 100

    difficulty_string = ['easy', 'hard']
    sub_experiment_number = 1
    print_divider_with_text('Experiment ' + `experiment_number`)
    print 'Experiment starting with %d and sub_experiment %d' % (experiment_number, sub_experiment_number)
    print_divider()

    # Fix all parameters except num_eigen_vectors and find optimal value
    knn_size = 10
    training_set_size = 50000
    test_set_size = 500

    print_divider_with_text('Sub-experiment ' + `sub_experiment_number`)
    print 'Finding optimal value of num_eigen_vectors by fixing knn_size to %d, training_set_size to %d and test_set_size to %d' % (knn_size, training_set_size, test_set_size)
    print_divider()

    max_num_eigen_vectors = [0, 0]
    max_accuracies = [0.0, 0.0]
    min_accuracy = 100.0
    plt_accuracies = []
    for i in range(2):
        plt_accuracies.append([])

    for difficulty in [0, 1]:
        print_divider_with_text('Difficulty ' + `difficulty`)

        for num_eigen_vectors in eigen_vector_sizes:
            print_divider_with_text('Num_eigen_vectors ' + `num_eigen_vectors`)
            total_accuracy = 0.0
            num_trials = 0
            for trial in range(total_trials):
                print_divider_with_text('Trial ' + `trial`)
                total_accuracy += eigen_digits_classify(training_set_size, test_set_size, knn_size, num_eigen_vectors,
                                                        difficulty, experiment_number, sub_experiment_number, trial)
                print_divider()
                num_trials += 1
            accuracy = total_accuracy / float(num_trials)
            if accuracy > max_accuracies[difficulty]:
                max_accuracies[difficulty] = accuracy
                max_num_eigen_vectors[difficulty] = num_eigen_vectors
            if accuracy < min_accuracy:
                min_accuracy = accuracy
            plt_accuracies[difficulty].extend([accuracy])

        print 'Difficulty: ', difficulty_string[difficulty], ' Accuracies obtained: ', plt_accuracies[difficulty]
        print 'Difficulty: ', difficulty_string[difficulty], ' max_num_eigen_vectors: ', max_num_eigen_vectors[difficulty]

    # Plot as graph and save to file
    file_name = plot_dir_name + '/' + `experiment_number` + '_' + `sub_experiment_number` + '_' + 'num_eigen_vectors_vs_accuracy.png'
    plot_graph(eigen_vector_sizes, plt_accuracies, 'Num eigen vectors', 'Accuracy',
               'Effect of number of eigen vectors on accuracy',
               [0, eigen_vector_sizes[-1] + eigen_vector_sizes_plt_extra_ul, get_min_axis_for_accuracy(min_accuracy), 100], file_name)

    # Fix all parameters but training_set_size and find optimal value
    sub_experiment_number += 1
    knn_size = 10
    test_set_size = 500
    print_divider_with_text('Sub-experiment ' + `sub_experiment_number`)
    print 'Finding optimal value of training_set_size by fixing knn_size to %d, num_eigen_vectors to (%d, %d) and test_set_size to %d' % (knn_size, max_num_eigen_vectors[0], max_num_eigen_vectors[1], test_set_size)
    print_divider()

    max_training_set_size = [0, 0]
    max_accuracies = [0.0, 0.0]
    min_accuracy = 100.0
    plt_accuracies = []
    for i in range(2):
        plt_accuracies.append([])

    for difficulty in [0, 1]:
        print_divider_with_text('Difficulty ' + `difficulty`)

        for training_set_size in training_set_sizes:
            print_divider_with_text('Training set size ' + `training_set_size`)
            total_accuracy = 0.0
            num_trials = 0
            for trial in range(total_trials):
                print_divider_with_text('Trial ' + `trial`)
                total_accuracy += eigen_digits_classify(training_set_size, test_set_size, knn_size,
                                                        max_num_eigen_vectors[difficulty], difficulty,
                                                        experiment_number, sub_experiment_number, trial)
                print_divider()
                num_trials += 1
            accuracy = total_accuracy / float(num_trials)
            if accuracy > max_accuracies[difficulty]:
                max_accuracies[difficulty] = accuracy
                max_training_set_size[difficulty] = training_set_size
            if accuracy < min_accuracy:
                min_accuracy = accuracy
            plt_accuracies[difficulty].extend([accuracy])

        print 'Difficulty: ', difficulty_string[difficulty], ' Accuracies obtained: ', plt_accuracies[difficulty]
        print 'Difficulty: ', difficulty_string[difficulty], ' max_training_set_size: ', max_training_set_size[difficulty]

    # Plot as graph and save to file
    file_name = plot_dir_name + '/' + `experiment_number` + '_' + `sub_experiment_number` + '_' + 'training_set_size_vs_accuracy.png'
    plot_graph(training_set_sizes, plt_accuracies, 'Training set size', 'Accuracy',
               'Effect of training_set_size on accuracy',
               [0, training_set_sizes[-1] + training_set_sizes_plt_extra_ul, get_min_axis_for_accuracy(min_accuracy), 100], file_name)

    # Fix all parameters but test_set_size and find optimal value
    sub_experiment_number += 1
    knn_size = 10
    print_divider_with_text('Sub-experiment ' + `sub_experiment_number`)
    print 'Finding optimal value of test_set_size by fixing knn_size to %d, num_eigen_vectors to (%d, %d) and training_set_size to (%d, %d)' % (knn_size, max_num_eigen_vectors[0], max_num_eigen_vectors[1], max_training_set_size[0], max_training_set_size[1])
    print_divider()

    max_test_set_size = [0, 0]
    max_accuracies = [0.0, 0.0]
    min_accuracy = 100.0
    plt_accuracies = []
    for i in range(2):
        plt_accuracies.append([])

    for difficulty in [0, 1]:
        print_divider_with_text('Difficulty ' + `difficulty`)

        for test_set_size in test_set_sizes:
            print_divider_with_text('Test set size ' + `test_set_size`)
            total_accuracy = 0.0
            num_trials = 0
            for trial in range(total_trials):
                print_divider_with_text('Trial ' + `trial`)
                total_accuracy += eigen_digits_classify(max_training_set_size[difficulty], test_set_size, knn_size,
                                                        max_num_eigen_vectors[difficulty], difficulty,
                                                        experiment_number, sub_experiment_number, trial)
                print_divider()
                num_trials += 1
            accuracy = total_accuracy / float(num_trials)
            if accuracy > max_accuracies[difficulty]:
                max_accuracies[difficulty] = accuracy
                max_test_set_size[difficulty] = test_set_size
            if accuracy < min_accuracy:
                min_accuracy = accuracy
            plt_accuracies[difficulty].extend([accuracy])

        print 'Difficulty: ', difficulty_string[difficulty], ' Accuracies obtained: ', plt_accuracies[difficulty]
        print 'Difficulty: ', difficulty_string[difficulty], ' max_test_set_size: ', max_test_set_size[difficulty]

    # Plot as graph and save to file
    file_name = plot_dir_name + '/' + `experiment_number` + '_' + `sub_experiment_number` + '_' + 'test_set_size_vs_accuracy.png'
    plot_graph(test_set_sizes, plt_accuracies, 'Test set size', 'Accuracy',
               'Effect of test_set_size on accuracy',
               [0, test_set_sizes[-1] + test_set_sizes_plt_extra_ul, get_min_axis_for_accuracy(min_accuracy), 100], file_name)

    # Fix all parameters but knn size and find optimal value
    sub_experiment_number += 1
    print_divider_with_text('Sub-experiment ' + `sub_experiment_number`)
    print 'Finding optimal value of knn_size by fixing test_set_size to (%d, %d), num_eigen_vectors to (%d, %d) and training_set_size to (%d, %d)' % (max_test_set_size[0], max_test_set_size[1], max_num_eigen_vectors[0], max_num_eigen_vectors[1], max_training_set_size[0], max_training_set_size[1])
    print_divider()

    max_knn_size = [0, 0]
    max_accuracies = [0.0, 0.0]
    min_accuracy = 100.0
    plt_accuracies = []
    for i in range(2):
        plt_accuracies.append([])

    for difficulty in [0, 1]:
        print_divider_with_text('Difficulty ' + `difficulty`)

        for knn_size in knn_sizes:
            print_divider_with_text('Knn size ' + `knn_size`)
            total_accuracy = 0.0
            num_trials = 0
            for trial in range(total_trials):
                print_divider_with_text('Trial ' + `trial`)
                total_accuracy += eigen_digits_classify(max_training_set_size[difficulty],
                                                        max_test_set_size[difficulty], knn_size,
                                                        max_num_eigen_vectors[difficulty], difficulty,
                                                        experiment_number, sub_experiment_number, trial)
                print_divider()
                num_trials += 1
            accuracy = total_accuracy / float(num_trials)
            if accuracy > max_accuracies[difficulty]:
                max_accuracies[difficulty] = accuracy
                max_knn_size[difficulty] = knn_size
            if accuracy < min_accuracy:
                min_accuracy = accuracy
            plt_accuracies[difficulty].extend([accuracy])

        print 'Difficulty: ', difficulty_string[difficulty], ' Accuracies obtained: ', plt_accuracies[difficulty]
        print 'Difficulty: ', difficulty_string[difficulty], ' max_knn_size: ', max_knn_size[difficulty]

    # Plot as graph and save to file
    file_name = plot_dir_name + '/' + `experiment_number` + '_' + `sub_experiment_number` + '_' + 'knn_size_vs_accuracy.png'
    plot_graph(knn_sizes, plt_accuracies, 'Knn size', 'Accuracy',
               'Effect of knn size on accuracy',
               [0, knn_sizes[-1] + knn_sizes_plt_extra_ul, get_min_axis_for_accuracy(min_accuracy), 100], file_name)

    # for i in range(0, len(training_set_sizes)):
    #     for j in range(0, len(test_set_sizes)):
    #         for l in range(0, len(eigen_vector_sizes)):
    #             for k in range(0, len(knn_sizes)):
    #                 for difficulty in [0, 1]: # 0 for easy and 1 for difficult test set
    #                     accuracy = 0.0
    #                     for trial in range (1, 10): # do ten trials and take average accuracy
    #                         # print 'Calling eigen digits classification for parameters: training_set_size: %d test_set_size: %d knn_size: %d num_eigen_vectors: %d ' % (training_set_sizes[i], test_set_sizes[j], knn_sizes[k], eigen_vector_sizes[l])
    #                         # eigen_digits_classify(training_set_sizes[i], test_set_sizes[j], knn_sizes[k], eigen_vector_sizes[l], difficulty, experiment_number, trial)
    # print 'Accuracy: %d' % eigen_digits_classify(10000, 500, 5, 500, 1, 1, 1)

    return


def plot_graph(x_axis, y_axis, x_label, y_label, title, axis, file_name):
    print 'Plotting %s to file %s' % (title, file_name)
    easy = plt.plot(x_axis, y_axis[0], 'r-o', label='easy')
    hard = plt.plot(x_axis, y_axis[1], 'b-o', label='hard')

    red_easy_patch = mpatches.Patch(color='red', label='Easy', linestyle='solid', linewidth=0.1)
    blue_hard_patch = mpatches.Patch(color='blue', label='Hard', linestyle='solid', linewidth=0.1)
    plt.legend(handles=[red_easy_patch, blue_hard_patch], loc=2)

    plt.grid(True)
    plt.axis(axis)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    # plt.show()
    plt.savefig(file_name)
    plt.close()


def test_pyplot():
    accuracies = [[78.5, 89.2, 98.3], [45.6, 74.5, 82.3]]
    training_set_sizes = [15000, 30000, 50000]
    xlabel = 'Training set size'
    ylabel = 'Accuracy'
    title = 'Effect of training set size on accuracy'
    axis = [0, 60000, 50, 100]
    plot_graph(training_set_sizes, accuracies, xlabel, ylabel, title, axis, 'sample.png')


driver()
# test_pyplot()
