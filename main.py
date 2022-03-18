from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch

import dataProcessing as data

import Networks


def build_sequences(capacity_list, window_size):
    # Text = List of capacity
    x, y = [], []
    for i in range(len(capacity_list) - window_size):
        sequence = capacity_list[i: i + window_size]
        target = capacity_list[i + 1: i + 1 + window_size]

        x.append(sequence)
        y.append(target)

    return np.array(x), np.array(y)


def get_train_test(data_dict, name, window_size=8):
    data_sequence = data_dict[name]['capacity']
    train_data, test_data = data_sequence[:window_size + 1], data_sequence[window_size + 1:]
    train_x, train_y = build_sequences(capacity_list=train_data, window_size=window_size)
    for k, v in data_dict.items():
        if k != name:
            data_x, data_y = build_sequences(capacity_list=v['capacity'], window_size=window_size)
            train_x, train_y = np.r_[train_x, data_x], np.r_[train_y, data_y]

    return train_x, train_y, list(train_data), list(test_data)


def train(data_list, learning_rate, feature_size, hidden_dim, num_layers, weight_decay, EPOCH, dataset, Rated_Capacity, net):
    global model
    score_list, result_list = [], []
    for i in range(4):
        name = data_list[i]
        train_x, train_y, train_data, test_data = get_train_test(
            dataset, name, window_size=feature_size
        )
        train_size = len(train_x)
        print('Sample size: '+str(train_size))

        if net == 'LSTM':
            model = Networks.LSTMNet(input_size=feature_size,
                                     hidden_dim=hidden_dim,
                                     num_layers=num_layers,
                                     is_bidirectional=False,
                                     )
        elif net == 'GRU':
            model = Networks.GRUNet(input_size=feature_size,
                                    hidden_dim=hidden_dim,
                                    num_layers=num_layers,
                                    )

        model = model.to('cpu')
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=learning_rate,
                                     weight_decay=weight_decay)
        loss_list, y_ = [], []
        mae, rmse, re = 1, 1, 1
        last_score ,score = {1, 1, 1}, {1, 1, 1}
        last_result = []
        count = 0


        for epoch in range(EPOCH):
            count+=1
            # print("epoch"+str(epoch), end="  ")
            # (batch_size, seq_len, input_size)
            x = np.reshape(train_x/Rated_Capacity, (-1, 1, feature_size)).astype(np.float32)
            # (batch_size, 1)
            y = np.reshape(train_y[:, -1]/Rated_Capacity, (-1,1)).astype(np.float32)

            x, y = torch.from_numpy(x).to('cpu'), torch.from_numpy(y).to('cpu')
            output = model(x)
            output = output.reshape(-1, 1)
            loss = criterion(output, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (epoch)%50 == 0:    # Predict every 50 epoch
                test_x = train_data.copy()
                point_list = []
                while len(point_list) < len(Battery[name].capacity):
                    x1 = np.reshape(np.array(test_x[-feature_size:]) / Rated_Capacity,
                                    (-1, 1, feature_size)).astype(np.float32)
                    x1 = torch.from_numpy(x1).to('cpu')
                    pred = model(x1)
                    next_point = pred.data.numpy()[0,0] * Rated_Capacity
                    test_x.append(next_point)
                    point_list.append(next_point)
                y_.append(point_list)
                loss_list.append(loss)
                #mae, mse, rmse = data.evaluate(y_test=test_data, y_predict=y_[-1])

                #print('epoch: {:<2d} | loss: {:6.6f} | MAE: {:6.6f} | RMSE: {:6.6f}'.format(epoch, loss, mae, rmse))
                print('| epoch: {:<4d} | loss: {:6.6f} |'.format(epoch, loss))

                plt.figure('test')
                plt.scatter(Battery[name].cycle, Battery[name].capacity, label='Measured Value',s=1 , color='b')
                plt.plot(range(1, len(y_[-1]) + 1), y_[-1], label="Predicted Value", color='r')
                plt.title('Battery ' + name + ' Prediction and Measured Capacity Value')
                plt.xlabel('Cycle', fontsize='large')
                plt.ylabel('Capacity', fontsize='large')
                plt.xticks(range(0, len(Battery[name].cycle) + 100, 100))
                plt.yticks(np.arange(0, 1.2, 0.1))
                plt.legend(loc='upper right', shadow=True, fontsize='large')
                plt.show()

            score = [1,1,1]
            if loss_list[-1] < 2e-4 :#and (last_score[1] < score[1] and last_score[2] < score[2]):
                break
            # score = [re, mae, rmse]
            last_score = score.copy()
            last_result = y_.copy()
        score_list.append(last_score)
        result_list.append(last_result[-1])
        plt.figure(0)
        plt.plot(range(1,5*len(loss_list),5), loss_list, label="Loss", color='r')
        plt.title('Loss Of the Network')
        plt.xlabel('Epoch', fontsize='large')
        plt.legend(loc='upper right', shadow=True, fontsize='large')
        plt.show()
        plt.figure(i)
        plt.scatter(Battery[name].cycle, Battery[name].capacity, label='Measured Value',s=5 ,color='b')
        plt.plot(range(1, len(result_list[i]) + 1), result_list[i], label="Predicted Value", color='r')
        plt.title('Battery '+name+' Prediction and Measured Capacity Value')
        plt.xlabel('Cycle', fontsize='large')
        plt.ylabel('Capacity', fontsize='large')
        plt.xticks(range(0,len(Battery[name].cycle)+100,100))
        plt.yticks(np.arange(0,1.2,0.1))
        plt.legend(loc='upper right', shadow=True, fontsize='large')
        plt.show()
        '''
        plt.figure(name + 'loss')
        plt.plot(count, loss_list, label=name + " Training loss", color='r')
        '''
    return score_list, result_list


if __name__ == "__main__":
    data_path = 'dataset/CALCE_Batteries/'
    data_list = ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
    #Battery = data.load_data(data_list,data_path)

    Battery = np.load('dataset/CALCE_Batteries.npy', allow_pickle=True)
    Battery = Battery.item()


    window_size = 64
    EPOCH = 800
    lr = 0.0001
    hidden_dim = 512
    num_layers = 2
    weight_decay = 1e-4

    SCORE = []
    train(data_list=data_list,
                                    learning_rate=lr,
                                    feature_size=window_size,
                                    hidden_dim=hidden_dim,
                                    num_layers=num_layers,
                                    weight_decay=weight_decay,
                                    EPOCH=EPOCH,
                                    dataset=Battery,
                                    Rated_Capacity=1.1,
                                    net='GRU'   # GRU or LSTM
                                    )
    print('------------------------------------------------------------------')




