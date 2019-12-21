import numpy as np
import pandas as pd

k = 20
user_length = 2967
item_length = 4125


def get_test_index_csv(path):
    csv_data = pd.read_csv(path)
    list = csv_data.values.tolist()
    test = []
    for i in range(len(list)):
        test.append(list[i])
    return test


def output(average, Q,B_U, B_I,RY):
    testList = get_test_index_csv("./test_index.csv")
    result = []
    for i in range(len(testList)):
        user_index = int(testList[i][0])
        item_index = int(testList[i][1])
        preRating = average + B_U[user_index] + B_I[item_index] + np.dot(RY[user_index].T,Q[:, item_index])
        result.append([i, preRating])
    np.savetxt("./out_6.csv", np.array(result), delimiter=",", fmt="%d,%.18f")


def get_all_parameter():
    train_file_path = "./train.csv"
    csv_data = pd.read_csv(train_file_path)
    train_file_list = csv_data.values.tolist()
    data = np.zeros((user_length, item_length))
    user_item_dictionary = {}
    sum1 = 0.0
    for line in range(len(train_file_list)):
        user = int(train_file_list[line][0])
        item = int(train_file_list[line][1])
        rating = train_file_list[line][2]
        user_item_dictionary.setdefault(user, []).append(item)
        data[user][item] = rating
        sum1 += rating
    average1 = sum1/len(train_file_list)
    return data,average1,user_item_dictionary


if __name__ == '__main__':
    R, average_train, user_item_dictionary = get_all_parameter()

    a = 0.01
    l1 = 0.05
    l2 = 0.1
    num_epochs = 45

    user_set, item_set = R.nonzero()

    P = np.random.randn(k,user_length)*0.1
    Q = np.random.randn(k,item_length)*0.1
    Y = np.random.randn(k,item_length)*0.1
    B_U = np.zeros(user_length)
    B_I = np.zeros(item_length)

    RY = {}

    for u in range(user_length):
        tmp = np.zeros(k)
        for j in user_item_dictionary[u]:
            tmp = np.add(tmp, Y[:,j])
        RY[u] = tmp

    for epoch in range(num_epochs):
        print("epoch"+str(epoch))
        for u, i in zip(user_set, item_set):
            user_number = len(user_set[user_set == u])
            RY[u] = np.add(RY[u] / np.sqrt(user_number), P[:, u])
            r = R[u, i] - (average_train + B_U[u]+B_I[i]+np.dot(RY[u].T,Q[:, i]))
            P[:, u] += a * (r * Q[:, i] - l2 * P[:, u])
            Q[:, i] += a * (r * (P[:, u]+ 1 / np.sqrt(user_number) * RY[u]) - l2 * Q[:, i])

            for item in user_item_dictionary[u]:
                Y[:,item] += a * (r * 1 / np.sqrt(user_number) * Q[:, item] - l2 * Y[:,item])
            B_U[u] += a * (r - l1 * B_U[u])
            B_I[i] += a * (r - l1 * B_I[i])

    output(average_train,Q,B_U,B_I,RY)