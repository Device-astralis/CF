import numpy as np
import pandas as pd


def load_all_file(train_file_path):
    csv_data = pd.read_csv(train_file_path)
    train_file_list = csv_data.values.tolist()
    data = np.zeros((user_length,item_length))
    for line in range(len(train_file_list)):
        user = int(train_file_list[line][0])
        item = int(train_file_list[line][1])
        rating = train_file_list[line][2]
        data[user][item] = rating

    return data


def load_train_file(train_file_path):
    csv_data = pd.read_csv(train_file_path)
    train_file_list = csv_data.values.tolist()
    data = np.zeros((user_length, item_length))
    for line in range(78274):
        user = int(train_file_list[line][0])
        item = int(train_file_list[line][1])
        rating = train_file_list[line][2]
        data[user][item] = rating

    return data


def calculate_rmse(predict_rating_UI_matrix,train_file_path):
    csv_data = pd.read_csv(train_file_path)
    train_file_list = csv_data.values.tolist()
    error = 0.0
    for line in range(78274,len(train_file_list)):
        user = int(train_file_list[line][0])
        item = int(train_file_list[line][1])
        rating = train_file_list[line][2]
        error += (predict_rating_UI_matrix[user,item] - rating)**2

    rmse = (error/(len(train_file_list)-78274))**0.5
    print("rmse:"+str(rmse))


def get_test_index_csv(path):
    csv_data = pd.read_csv(path)
    list = csv_data.values.tolist()
    test = []
    for i in range(len(list)):
        test.append(list[i])
    return test


def output(predict_rating_UI_matrix):
    testList = get_test_index_csv("./test_index.csv")
    result = []
    for i in range(len(testList)):
        user_index = int(testList[i][0])
        item_index = int(testList[i][1])
        preRating = predict_rating_UI_matrix[user_index,item_index]
        result.append([user_index, item_index,preRating])
    np.savetxt("./out_3.csv", np.array(result), delimiter=",", fmt="%d,%d,%.18f")


if __name__ == "__main__":
    user_length = 2967
    item_length = 4125
    train_file_path="./train.csv"
    f = 10
    epoch_num = 10
    E = np.identity(f)
    # fxf的单位阵
    l = 0.16
    user_item_data = load_all_file(train_file_path)
    k_user_matrix = np.mat(np.random.randn(f,user_length))
    k_item_matrix = np.mat(np.random.randn(f,item_length))
    # I = user_item_data.copy()
    # I[user_item_data > 0] = 1
    for i in range(epoch_num):
        for user_index in range(user_length):
            left_result = np.mat(np.zeros((f,f)))
            right_result = np.mat(np.zeros((f,1)))

            for item_index in np.nonzero(user_item_data[user_index])[0]:
                if user_item_data[user_index][item_index] != 0.0:
                    left_result += k_item_matrix[:,item_index].dot(k_item_matrix[:,item_index].T)
                    right_result += k_item_matrix[:,item_index] * user_item_data[user_index][item_index]
            
            k_user_matrix [:,user_index] = (l * E + left_result).I.dot(right_result)
        
        for item_index in range(item_length):
            left_result = np.mat(np.zeros((f, f)))
            right_result = np.mat(np.zeros((f, 1)))

            for user_index in np.nonzero(user_item_data[:,item_index])[0]:
                if user_item_data[user_index][item_index] != 0.0:
                    left_result += k_user_matrix[:, user_index].dot(k_user_matrix[:, user_index].T)
                    right_result += k_user_matrix[:, user_index] * user_item_data[user_index][item_index]

            k_item_matrix[:, item_index] = (l * E + left_result).I.dot(right_result)

    predict_rating_UI_matrix = k_user_matrix.T.dot(k_item_matrix)
    output(predict_rating_UI_matrix)
    # calculate_rmse(predict_rating_UI_matrix,train_file_path)
