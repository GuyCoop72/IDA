"""
file containing functions to convert from textual data to floats, approximately centered around 0.
all outputs are given as floats
"""
import random
from sklearn import mixture
import numpy as np
from collections import defaultdict

#---------------------------------------------------
# getters
def get_LR_value(LR):
    if LR == 'L':
        return -1.
    elif LR == 'R':
        return 1.
    elif LR == 'S':
        return 0.
    else:
        return None


def get_LLRR_value(LLRR):
    if LLRR[0] == 'L':
        ret_val = -3.
    elif LLRR[0] == 'R':
        ret_val = 3.
    elif LLRR[0] == 'S':
        ret_val = 0.
    else:
        return None

    if LLRR[1] == 'L':
        return ret_val - 1
    elif LLRR[1] == 'R':
        return ret_val + 1
    elif LLRR[1] == 'S':
        return ret_val
    else:
        return None


def get_parkinsons(parkinsons):
    if parkinsons == 'True':
        return 1.
    elif parkinsons == 'False':
        return -1.
    else:
        return None


def get_gender(gender):
    if gender == 'Female':
        return -1.
    elif gender == 'Male':
        return 1.
    else:
        return None


def get_birthyear(birth_year):
    try:
        return float(birth_year)
    except:
        return None


def get_diagnosisyear(diagnosis_year):
    try:
        return float(diagnosis_year)
    except:
        return None


def get_diagnoisis_age(birth_year, diagnosis_year):
    return float(diagnosis_year) - float(birth_year)


def get_tremors(tremors):
    if tremors == 'True':
        return 1.
    elif tremors == 'False':
        return -1.
    else:
        return None


def get_sided(sided):
    if sided == 'Left':
        return -1.
    elif sided == 'Right':
        return 1.
    else:
        return None


def get_updrs(updrs):
    try:
        return float(updrs[0])
    except:
        return None


def get_impact(impact):
    if impact == 'Mild':
        return -1.
    elif impact == 'Medium':
        return 0.
    elif impact == 'Severe':
        return 1.
    else:
        return None


def get_levadopa(levadopa):
    if levadopa == 'True':
        return 1.
    elif levadopa == 'False':
        return -1.
    else:
        return None


def get_da(da):
    if da == 'True':
        return 1.
    elif da == 'False':
        return -1.
    else:
        return None


def get_maob(maob):
    if maob == 'True':
        return 1.
    elif maob == 'False':
        return -1.
    else:
        return None


def get_other(other):
    if other == 'True':
        return 1.
    elif other == 'False':
        return -1.
    else:
        return None

# --------------------------------------------------
# data augmentation
def augmement_data(d_point, user_d):
    new_d_point = {}

    # current age
    new_d_point['BirthYear'] = user_d['BirthYear']
    # diagnosis age
    if user_d['DiagnosisYear'] != None:
        new_d_point['DiagnosisAge'] = user_d['DiagnosisYear'] - user_d['BirthYear']
        # days since diagnoisis
        # print(user_d['DiagnosisYear'], d_point['Date'])
        days_since_diagnosis = (
            (365 * ((float(d_point['Date'][:2]) + 2000) - user_d['DiagnosisYear'])) +
            ((365. / 12.) * (float(d_point['Date'][2:4]) - 6)) +
            float(d_point['Date'][4:])
        )
        new_d_point['DaysSinceDiagnosis'] = days_since_diagnosis
    else:
        new_d_point['DiagnosisAge'] = None
        new_d_point['DaysSinceDiagnosis'] = None
    # gender
    new_d_point['Gender'] = user_d['Gender']
    # parkinsons
    new_d_point['Parkinsons'] = user_d['Parkinsons']
    # tremors
    new_d_point['Tremors'] = user_d['Tremors']
    # sided
    new_d_point['Sided'] = user_d['Sided']
    # levadopa
    new_d_point['Levadopa'] = user_d['Levadopa']
    # DA
    new_d_point['DA'] = user_d['DA']
    # MAOB
    new_d_point['MAOB'] = user_d['MAOB']
    # Other
    new_d_point['Other'] = user_d['Other']

    # hand
    new_d_point['Hand'] = d_point['Hand']
    # movement
    new_d_point['Movement'] = d_point['Movement']

    new_d_point['T1'] = d_point['T1']
    new_d_point['T2'] = d_point['T2']
    new_d_point['T3'] = d_point['T3']

    return new_d_point


# --------------------------------------------------
# generators
# discrete
def gen_impact(mild, mid, severe):
    r = random.random()
    if r <= float(mild) / (mild + mid + severe):
        return -1.
    elif r <= float(mild + mid) / (mild + mid + severe):
        return 0.
    else:
        return 1.

# semi-continuous
def gen_birthyear(birthyear_dist):
    return birthyear_dist.sample()


def gen_diagnosisyear(birthyear, diagnosis_age_dist):
    return birthyear + diagnosis_age_dist.sample()


# build profiles
def build_profile(f):
    # build a dictionary of possible user information
    profile = {}

    # handle birthyear
    temp_birthyear = f.readline().split()
    if len(temp_birthyear) > 1:
        profile['BirthYear'] = get_birthyear(temp_birthyear[1])
    else:
        profile['BirthYear'] = None

    # handle gender
    temp_gender = f.readline().split()
    if len(temp_gender) > 1:
        profile['Gender'] = get_gender(temp_gender[1])
    else:
        profile['Gender'] = None

    # handle parkinsons
    temp_parkinsons = f.readline().split()
    if len(temp_parkinsons) > 1:
        profile['Parkinsons'] = get_parkinsons(temp_parkinsons[1])
    else:
        profile['Parkinsons'] = None

    # handle tremors
    temp_tremors = f.readline().split()
    if len(temp_tremors) > 1:
        profile['Tremors'] = get_tremors(temp_tremors[1])
    else:
        profile['Tremors'] = None

    # handle diagnosis_year
    temp_diagnosisyear = f.readline().split()
    if len(temp_diagnosisyear) > 1:
        profile['DiagnosisYear'] = get_diagnosisyear(temp_diagnosisyear[1])
    else:
        profile['DiagnosisYear'] = None

    # handle Sided
    temp_sided = f.readline().split()
    if len(temp_sided) > 1:
        profile['Sided'] = get_sided(temp_sided[1])
    else:
        profile['Sided'] = None

    # handle UPDRS
    temp_updrs = f.readline().split()
    if len(temp_updrs) > 1:
        profile['UPDRS'] = get_updrs(temp_updrs[1:])
    else:
        profile['UPDRS'] = None

    # handle Impact
    temp_impact = f.readline().split()
    if len(temp_impact) > 1:
        profile['Impact'] = get_impact(temp_impact[1])
    else:
        profile['Impact'] = None

    # handle Levadopa
    temp_levadopa = f.readline().split()
    if len(temp_levadopa) > 1:
        profile['Levadopa'] = get_levadopa(temp_levadopa[1])
    else:
        profile['Levadopa'] = None

    # handle DA
    temp_da = f.readline().split()
    if len(temp_da) > 1:
        profile['DA'] = get_da(temp_da[1])
    else:
        profile['DA'] = None

    # handle MAOB
    temp_maob = f.readline().split()
    if len(temp_maob) > 1:
        profile['MAOB'] = get_maob(temp_maob[1])
    else:
        profile['MAOB'] = None

    # handle other
    temp_other = f.readline().split()
    if len(temp_other) > 1:
        profile['Other'] = get_other(temp_other[1])
    else:
        profile['Other'] = None

    return profile


# build data item
def add_to_data_dict(user_information, data_dict, f):
    def read_data_line(line):
        ret_line = []
        temp_line = line.split()
        uname = temp_line[0]
        if uname not in user_information:
            return None
        try:
            float(temp_line[4])
            float(temp_line[6])
            float(temp_line[7])
            if float(temp_line[4]) > 5000:
                return None
        except:
            return None

        return (uname, temp_line[1], temp_line[3], float(temp_line[4]), temp_line[5], float(temp_line[6]), float(temp_line[7]))

    for l in f:
        line = read_data_line(l)
        if line != None:
            line_dict = {'Date':line[1], 'Hand':get_LR_value(line[2]), 'T1':line[3], 'Movement':get_LLRR_value(line[4]), 'T2':line[5], 'T3':line[6]}
            data_dict[line[0]].append(line_dict)
    return data_dict


# find distributions of each data point
def estimate_birthyear(user_data):
    # fills the NoneType values for birthyear and diagnosis year based on the distribution in the data
    # build models
    birthyear_dist = mixture.GMM(n_components=1)
    birthyear_data = []
    for user in user_data:
        if user_data[user]['BirthYear'] != None:
            birthyear_data.append(user_data[user]['BirthYear'])
    birthyear_dist.fit(X=np.reshape(birthyear_data, (-1, 1)))

    #generate new data
    for user in user_data:
        if user_data[user]['BirthYear'] == None:
            user_data[user]['BirthYear'] = gen_birthyear(birthyear_dist)

    return user_data


