import os
from collections import defaultdict
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_conversions import *
from sklearn import preprocessing
import time

def histograms_user_data(user_information):
    def plot_hist(variable_name, n_bins):
        vals = []
        num_NoneType = 0
        num_Users = len([_ for _ in user_information])
        for user in user_information:
            if user_information[user][variable_name] != None:
                vals.append(user_information[user][variable_name])
            else:
                num_NoneType += 1
        print(variable_name + ": percent NoneType = {:.4}%".format((float(num_NoneType) / num_Users) * 100.))
        plt.figure()
        plt.hist(vals, histtype='barstacked', normed=True, bins = n_bins)
        sns.kdeplot(vals)
        plt.show()

    # analyse user data
    # birth year
    plot_hist('BirthYear', 10)
    plot_hist('Gender', 2)
    plot_hist('Parkinsons', 2)
    plot_hist('DiagnosisYear', 10)

    # diagnosis age
    diagnosis_age = []
    for user in user_information:
        if user_information[user]['BirthYear'] != None and user_information[user]['DiagnosisYear'] != None:
            diagnosis_age.append(user_information[user]['DiagnosisYear'] - user_information[user]['BirthYear'])
    plt.hist(diagnosis_age, histtype='barstacked', normed=True)
    sns.kdeplot(diagnosis_age)
    plt.show()

    plot_hist('Sided', 2)
    plot_hist('UPDRS', 2)
    plot_hist('Impact', 3)
    plot_hist('Levadopa', 2)
    plot_hist('DA', 2)
    plot_hist('MAOB', 2)
    plot_hist('Other', 2)

    return


def remove_outliers(data, m=2):
    data = np.asarray(data)
    mean = np.mean(data, axis = 0)
    std_dev = np.std(data, axis = 0)
    print(mean, std_dev)
    ret_data = [d_point for d_point in data if (abs(d_point - mean) < (m * std_dev)).all()]
    return np.asarray(ret_data)


def load_user_info():
    # load all user information into initial user dictionary
    user_information = {}
    for file in os.listdir("./Archived users/"):
        f = open("./Archived users/" + file, 'r')
        uname = file[5:-4]
        user_information[uname] = build_profile(f)

    print("loaded user information")
    return user_information


def load_full_data(user_information, n_files = len([_ for _ in os.listdir("./Tappy Data/")])):
    # loads the full data set
    # ignores uses with no user information
    # ignores files that have incorrect or corrupt data
    data_dict = defaultdict(list)
    for num, file in enumerate(os.listdir("./Tappy Data/")):
        print ("{:.4}%".format((float(num) / n_files) * 100.), file)
        f = open("./Tappy Data/" + file, 'r')
        add_to_data_dict(user_information, data_dict, f)
        f.close()
        if num >= n_files:
            break
    print("loaded full data")
    return data_dict


def fill_NoneType_dates(user_information):
    user_information = fill_NoneTypes(user_information)
    return user_information


def build_data_labels_array(all_data, label='Parkinsons', exclude_data = []):
    # builds an array of n*m data elements, and n labels based on the label choice
    ret_data = []
    ret_labels = []
    data_keys = [k for k in all_data[0] if (k not in [j for j in exclude_data]) and k != label]
    print data_keys
    for d_point in all_data:
        d_point_list = [d_point[k] for k in data_keys]
        if not None in d_point_list:
            ret_labels.append(d_point[label])
            ret_data.append(d_point_list)

    return np.asarray(ret_labels), np.asarray(ret_data)


def plot_pca(labels, data):
    pca = PCA(n_components=2)
    pca.fit(data)
    data = pca.transform(data)
    if np.shape(data)[1] == 2:
        plot_data1, plot_data2 = data.transpose()


    plt.scatter(plot_data1, plot_data2, c=labels, alpha=0.5)
    plt.colorbar()
    plt.show()
    return


def get_pca_variences(data):
    pca = PCA()
    pca.fit_transform(data)
    return pca.explained_variance_ratio_


def plot_corrcoef(data):
    shape = np.shape(data)
    if shape[0] > shape[1]:
        data = data.transpose()
    corr = np.corrcoef(data)
    plt.matshow(corr)
    plt.colorbar()
    plt.show()
    return


def augment_data(user_info, full_data):
    data = []
    for user in full_data:
        for d_point in full_data[user]:
            data.append(augmement_data(d_point, user_info[user]))
    return data


if __name__ == '__main__':
    user_info = load_user_info()
    user_info = estimate_birthyear(user_info)
    full_data = load_full_data(user_info, n_files = 40)
    full_data = augment_data(user_info, full_data)


    # plot parkinsons vs not parkinsons with all useful data
    labels, data = build_data_labels_array(full_data, label='Parkinsons', exclude_data=[
        'T1', 'BirthYear', 'Gender', 'Hand', 'Movement',
        'Levadopa', 'MAOB', 'Tremors', 'Sided', 'DA', 'DiagnosisAge', 'Other', 'DaysSinceDiagnosis'])

    #data = preprocessing.scale(data)
    #plot_pca(labels, data)

    labels, data = build_data_labels_array(full_data, label='DiagnosisAge', exclude_data=[
    ])
    data = preprocessing.scale(data)
    plot_pca(labels, data)


