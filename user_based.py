import pandas as pd
import numpy as np
import pickle as pk


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
    data = np.zeros((user_length,item_length))
    for line in range(len(train_file_list)):
        user = int(train_file_list[line][0])
        item = int(train_file_list[line][1])
        rating = train_file_list[line][2]
        data[user][item] = rating

    return data


def load_train_file():
    data = np.zeros((user_length, item_length))
    for line in range(78274):
        user = int(train_file_list[line][0])
        item = int(train_file_list[line][1])
        rating = train_file_list[line][2]
        data[user][item] = rating

    return data


def save_user_item():
    data = load_all_file()
    f = open("./user_item.txt","wb")
    pk.dump(data,f,protocol=None)


save_user_item()
open_user_file = open("./user_item.txt","rb")
user_item_data = pk.load(open_user_file)
open_user_file.close()


def calculate_average():
    mean_user = np.average(user_item_data, 1, weights=np.int64(user_item_data > 0))
    return mean_user


user_average = calculate_average()


def calculate_user_similarity():
    user_similarity = np.zeros((user_length,user_length))
    for i in range(user_length):
        for j in range(i,user_length):
            if i == j:
                user_similarity[i][j] = 0
            else:
                user_similarity[i][j]=user_similarity[j][i]=calculate_sim(i,j)
    f = open("./user_similarity.txt","wb")
    pk.dump(user_similarity,f,protocol=None)


def calculate_sim(i, j):
    ind = np.nonzero(user_item_data[i] * user_item_data[j])
    if len(ind[0]) == 0:
        return 0
    numerator = np.sum((user_item_data[i][ind] - user_average[i]) * (user_item_data[j][ind] - user_average[j]))
    sqrt_i = np.sqrt(np.sum(np.square(user_item_data[i][ind] - user_average[i])))
    sqrt_j = np.sqrt(np.sum(np.square(user_item_data[j][ind] - user_average[j])))
    denominator = sqrt_j * sqrt_i
    if denominator == 0:
        if user_item_data[i][ind[0][0]] == user_item_data[j][ind[0][0]]:
            return 1.0
        return 0
    else:
        return numerator/denominator


# calculate_user_similarity()
f_similarity = open("./user_similarity.txt","rb")
user_similarity = pk.load(f_similarity)
f_similarity.close()


def get_test_index_csv(path):
    csv_data = pd.read_csv(path)
    list = csv_data.values.tolist()
    test = []
    for i in range(len(list)):
        test.append(list[i])
    return test


def rating_prediction(user_index,item_index):
    k = 5
    numerator = 0
    denominator = 0
    similarity_tuple = sorted(transfer_to_tuple(user_similarity[user_index]), key=lambda x: x[1], reverse=True)
    for i in range(k):
        if user_item_data[similarity_tuple[i][0]][item_index] != 0:
            numerator += similarity_tuple[i][1]*(user_item_data[similarity_tuple[i][0]][item_index]-user_average[similarity_tuple[i][0]])
            denominator += abs(similarity_tuple[i][1])
    if denominator != 0:
        prediction_rating = user_average[user_index] + numerator/denominator
    else:
        prediction_rating = user_average[user_index]

    return prediction_rating


testList = get_test_index_csv("./test_index.csv")
result = []
for i in range(len(testList)):
    userId = int(testList[i][0])
    itemId = int(testList[i][1])
    preRating = rating_prediction(userId,itemId)
    result.append([userId,itemId,preRating])
np.savetxt("./user_item_result.txt",np.array(result),delimiter=",",fmt="%d %d %.18f")