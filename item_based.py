import pickle as pk
import pandas as pd
import numpy as np

train_file_path = "./train.csv"
csv_data = pd.read_csv(train_file_path)
train_file_list = csv_data.values.tolist()
user_length = 2967
item_length = 4125


def transfer_to_tuple(list):
    tuple = []
    for i in range(len(list)):
        tuple.append((i, list[i]))
    return tuple


def load_all_file():
    data = np.zeros((item_length,user_length))
    for line in range(len(train_file_list)):
        user = int(train_file_list[line][0])
        item = int(train_file_list[line][1])
        rating = train_file_list[line][2]
        data[item][user] = rating

    return data


def load_train_file():
    data = np.zeros((item_length,user_length))
    for line in range(78274):
        user = int(train_file_list[line][0])
        item = int(train_file_list[line][1])
        rating = train_file_list[line][2]
        data[item][user] = rating

    return data


def save_item_user():
    data = load_all_file()
    f = open("./item_user.txt","wb")
    pk.dump(data,f,protocol=None)


save_item_user()
open_item_file = open("./item_user.txt","rb")
item_user_data = pk.load(open_item_file)
open_item_file.close()


def calculate_sim(i, j):
    ind = np.nonzero(item_user_data[i] * item_user_data[j])
    if len(ind[0]) == 0:
        return 0
    numerator = np.sum((item_user_data[i][ind]) * (item_user_data[j][ind]))
    sqrt_i = np.sqrt(np.sum(np.square(item_user_data[i][ind])))
    sqrt_j = np.sqrt(np.sum(np.square(item_user_data[j][ind])))
    denominator = sqrt_j * sqrt_i

    return numerator/denominator


def calculate_item_similarity():
    item_similarity = np.zeros((item_length,item_length))
    for i in range(item_length):
        for j in range(i,item_length):
            if i == j:
                item_similarity[i][j] = 0
            else:
                item_similarity[i][j]=item_similarity[j][i]=calculate_sim(i,j)
    f = open("./item_similarity.txt","wb")
    pk.dump(item_similarity,f,protocol=None)


# calculate_item_similarity()
f_similarity = open("./item_similarity.txt","rb")
item_similarity = pk.load(f_similarity)
f_similarity.close()


def calculate_average():
    mean_item = np.zeros(item_length)
    for i in range(item_length):
        average = 0
        num = 0
        for j in range(user_length):
            if item_user_data[i][j] != 0.0:
                average += item_user_data[i][j]
                num += 1
        if num != 0:
            average /= num
        mean_item[i] = average
    return mean_item


item_average = calculate_average()


def get_test_index_csv(path):
    csv_data = pd.read_csv(path)
    list = csv_data.values.tolist()
    test = []
    for i in range(len(list)):
        test.append(list[i])
    return test


def rating_prediction(item_index,user_index):
    k = 20
    numerator = 0
    denominator = 0
    similarity_tuple = sorted(transfer_to_tuple(item_similarity[item_index]), key=lambda x: x[1], reverse=True)
    for i in range(k):
        if item_user_data[similarity_tuple[i][0]][user_index] != 0:
            numerator += similarity_tuple[i][1]*item_user_data[similarity_tuple[i][0]][user_index]
            denominator += abs(similarity_tuple[i][1])
    if denominator != 0:
        prediction_rating = numerator/denominator
    else:
        prediction_rating = item_average[item_index]

    return prediction_rating


testList = get_test_index_csv("./test_index.csv")
result=[]
for i in range(len(testList)):
    userId = int(testList[i][0])
    itemId = int(testList[i][1])
    preRating = rating_prediction(itemId,userId)
    result.append([userId, itemId, preRating])
    print(preRating)
np.savetxt("./item_user_result.txt",np.array(result),delimiter=",",fmt="%d %d %.18f")