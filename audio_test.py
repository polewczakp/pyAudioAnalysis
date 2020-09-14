"""
Script by Pawel Polewczak 2020

Required configuration for this script:
-Python 3.8+
-Python packages:
 open cmd in project folder and enter:
 pip install -e .
 pip install --upgrade --force-reinstall eyed3 plotly python-magic-bin psutil
-ffmpeg (copy bin files to %USERPROFILE%\\AppData\\Local\\Microsoft\\WindowsApps)

The original library is available on Github:
git clone https://github.com/tyiannak/pyAudioAnalysis.git
"""

import inspect
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import psutil

from pyAudioAnalysis import audioTrainTest

TRAIN = True
# available classifiers: ["svm", "svm_rbf", "knn", "randomforest", "gradientboosting", "extratrees"]
CLASSIFIERS = [
    "svm",
    "svm_rbf",
    "knn",
    "randomforest",
    "gradientboosting",
    "extratrees"
]

process_count = psutil.cpu_count(logical=False)  # physical core number is the optimal value


def init():
    def float_format(float_num):
        return "{0:0.2f}".format(float_num)
    np.set_printoptions(formatter={'float': float_format})
    print("Using {} physical cores.".format(process_count))


# debug methods
def retrieve_var_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]


def print_var(var):
    print("{}: {}".format(retrieve_var_name(var)[0], var))
# END debug methods


def multiprocessing_fast(func, args, workers=process_count):
    with ProcessPoolExecutor(workers) as ex:
        result = ex.map(func, args)
    return list(result)


def multiply_by_100(array):
    np_array = np.array(array) * 100.0
    np_array = np.around(np_array, 2)
    return np_array


def get_last_key_of_dict(dic):
    return list(dic.keys())[-1]


def get_classification_result(result_class_id, class_names):
    return str(class_names[int(result_class_id)]).split('/')[-1]


def test_spot_wrong_files(test_files_results):
    for path, classifiers in test_files_results.items():
        count_wrong = 0
        for _, cls in classifiers.items():
            if "WRONG" in cls:
                count_wrong += 1
        if count_wrong > len(classifiers) // 2:
            print("{}: {} fail(s)".format(path, count_wrong))


def test_result_summary(sample_id, result_as_class_name, result_correct, probabilities, classifier):
    probabilities = multiply_by_100(probabilities)
    print("{}: {} classifier probabilities: {}; "
          .format(sample_id + 1, classifier, probabilities), end="")
    print("result: {} ({})".format(result_as_class_name, bool(result_correct)))


def test_save_classifier_scores(classifier, classifiers_results,
                                total_score, time_diff):
    classifiers_results[classifier] = {"total_score": total_score,
                                       "time": time_diff}


def test_sort_results_by_score(result_dict):
    list_of_sorted_clfs = sorted(result_dict, key=lambda x:
                                 (result_dict[x]["total_score"], -result_dict[x]["time"]))
    new_dict = {}
    for clf in list_of_sorted_clfs:
        new_dict[clf] = result_dict[clf]
    return new_dict


def test_print_final_summary(test_filenames, test_files_results, classifiers_results):
    print("\nClassification results for each test file:")
    print(json.dumps(test_files_results, indent=4))

    print("\nFiles that struggle to be recognized by current models:")
    test_spot_wrong_files(test_files_results)

    print("\nScores for each classifier (max = {}):".format(len(test_filenames)))
    classifiers_results = test_sort_results_by_score(classifiers_results)
    test_sort_results_by_score(classifiers_results)
    print(json.dumps(classifiers_results, indent=4))

    largest_score_classifier = get_last_key_of_dict(classifiers_results)
    best_classifier = largest_score_classifier if largest_score_classifier else None
    best_result = classifiers_results[get_last_key_of_dict(classifiers_results)]["total_score"]
    best_classifier_rate = round(best_result / len(test_filenames) * 100, 2)
    print("The best classifier for this test set: {} ({}/{} = {}%)"
          .format(best_classifier, best_result, len(test_filenames), best_classifier_rate))

    tests_count = len(test_filenames) * len(CLASSIFIERS)
    passed_tests = sum(cl["total_score"] for cl in classifiers_results.values() if cl)
    success_rate = round(passed_tests / tests_count * 100, 2)
    print("Total tests passed: {}/{} ({}%)".format(passed_tests, tests_count, success_rate))


def get_class_list_from_folder(folder_name):
    class_list = []
    for _, dir_names, _ in os.walk('./' + folder_name):
        for class_name in dir_names:
            class_list.append(folder_name + '/' + class_name)
    class_list.sort()
    pretty_classes = ", ".join([x.rsplit('/', 1)[1] for x in class_list])
    print("Sound classes: {}".format(pretty_classes))
    return class_list


def get_file_list_from_folder(folder_name, train_folder_name="_mgr"):
    """

    :param folder_name: directory that contains test sounds (each name of the test files contains
        one of class names, eg. "alert_fireworks_12345.wav")
    :param train_folder_name: name of directory that contains sounds used to training
        categorized in class folders,
        Example: folder "training" that contains:
        "training/alert_fireworks/alert_fireworks_12345.wav"
        "training/alert_engine/engine_start.mp3"
    :return: list relative paths of each test file (typically "<training_folder>/<file>")
    """
    class_paths = get_class_list_from_folder(train_folder_name)
    file_list = []
    for dir_path, _, file_names in os.walk('./' + folder_name):
        for file_name in file_names:
            good_test_file_name = False
            file_path = dir_path.replace('\\', '/')[2:] + '/' + file_name
            for class_name in class_paths:
                if (class_name.split('/')[-1] in file_name) and file_path.count('/') < 2:
                    good_test_file_name = True
            if good_test_file_name:
                file_list.append(file_path)
            elif "/skip/" not in file_path:
                print("Warning: Test file \"{}\" doesn't contain any recognized class name"
                      "or it's not placed in the correct directory.".format(file_path))
    # file_list.sort()
    random.shuffle(file_list)  # workaround for multithreading I/O performance issue
    return file_list


def test_extract_features_and_classify(sample_enumerate, classifier, test_size):
    sample_id, sample_path = sample_enumerate
    print("Test {}/{}: {}".format(sample_id + 1, test_size, sample_path))
    [result_class_id, probabilities, class_names] = \
        audioTrainTest.file_classification(sample_path, classifier + "_model", classifier)
    if not isinstance(probabilities, np.ndarray):
        print("Error! File classification was not performed!")
        return
    result_as_class_name = get_classification_result(result_class_id, class_names)
    result_correct = result_as_class_name in sample_path
    test_result_summary(sample_id, result_as_class_name, result_correct, probabilities,
                        classifier)
    return {"sample_path": sample_path, "classifier": classifier,
            "result_as_class_name": result_as_class_name, "result_correct": result_correct}


def extract_features_and_train(classifier, train_folder_name="_mgr"):
    class_paths = get_class_list_from_folder(train_folder_name)
    random.shuffle(class_paths)
    audioTrainTest.extract_features_and_train(
        paths=class_paths, mid_window=1, mid_step=1, short_window=0.050, short_step=0.050,
        classifier_type=classifier, model_name=classifier + "_model"
        # , compute_beat=False, train_percentage=0.90
        )


def train_model():
    start = time.time()
    if process_count == 1:
        for classifier in CLASSIFIERS:
            extract_features_and_train(classifier)
    elif process_count > 1:
        multiprocessing_fast(extract_features_and_train, CLASSIFIERS)
    else:
        print("Error. Thread count should be a positive integer.")

    end = time.time()
    print("\n# Model training took total time of", round((end - start), 3), "s.")
    print("threads:", process_count)


def test_model(test_folder="_test", train_folder_name="_mgr"):
    start = time.time()
    test_filenames = get_file_list_from_folder(test_folder, train_folder_name)
    if not test_filenames:
        print("Error: Test file list is empty. "
              "Please ensure that file names contain expected class names.")
        return
    classifiers_results = {}
    test_files_results = {}
    print("# Analyzing sounds with classifiers: {}".format(CLASSIFIERS))
    for classifier in CLASSIFIERS:
        cl_start = time.time()
        classifier_score = 0
        results = []
        if process_count == 1:
            for sample_id, sample_path in enumerate(test_filenames):
                sample_enumerate = (sample_id, sample_path)
                results += test_extract_features_and_classify(
                    sample_enumerate, classifier, len(test_filenames))
        elif process_count > 1:
            results = multiprocessing_fast(
                partial(test_extract_features_and_classify,
                        classifier=classifier, test_size=len(test_filenames)),
                enumerate(test_filenames))
        else:
            print("Error. Thread count should be a positive integer.")
            return
        for result_dict in results:
            sample_path = result_dict["sample_path"]
            classifier_score += int(result_dict["result_correct"])
            result = result_dict["result_as_class_name"] + \
                (" <-- WRONG" if not result_dict["result_correct"] else "")
            if result_dict["sample_path"] in test_files_results:
                test_files_results[sample_path][classifier] = result
            else:
                test_files_results[sample_path] = {classifier: result}
        time_diff = round(time.time() - cl_start, 3)
        test_save_classifier_scores(classifier, classifiers_results, classifier_score, time_diff)

    # Summary of results
    test_print_final_summary(test_filenames, test_files_results, classifiers_results)
    total_classification_time = round(time.time() - start, 3)
    print("\n# Classification took total time of", total_classification_time, "s.")


if __name__ == "__main__":
    init()
    if TRAIN:
        train_model()
    test_model()
