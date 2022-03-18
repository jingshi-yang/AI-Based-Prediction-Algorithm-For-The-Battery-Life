import numpy as np
import torch

from dataProcessing import evaluate
import Networks


def train(learning_rate, feature_size, hidden_dim, num_layers, weight_decay, EPOCH, dataset, Rated_Capacity, seed=0):
    score_list, result_list = [], []
    Battery_list = ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
    for i in range(4):
        name = Battery_list[i]
        train_x, train_y, train_data, test_data = get_train_test(
            dataset, name, window_size=feature_size
        )
        train_size = len(train_x)
        print('Sample size: '+str(train_size))

        model = neuralNetwork.NeuralNetwork(input_size=feature_size,
                                            hidden_dim=hidden_dim,
                                            num_layers=num_layers,
                                            is_bidirectional=True,
                                            )
        model = model.to('cpu')
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(),
                                              lr=learning_rate,
                                              weight_decay=weight_decay)

        loss_list, y_ = [0], []
        mae, rmse, re = 1, 1, 1
        score_ ,score = 1, 1
        for epoch in range(EPOCH):
            # (batch_size, seq_len, input_size)
            x = np.reshape(train_x/Rated_Capacity, (-1, 1, feature_size)).astype(np.float32)
            # (batch_size, 1)
            y = np.reshape(train_y[:-1]/Rated_Capacity, (-1,1)).astype(np.float32)

            x, y = torch.from_numpy(x).to('cpu'), torch.from_numpy(y).to('cpu')
            output = model(x)
            output = output.reshape(-1, 1)
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1)%100 == 0:    # Predict every 100 epoch
                test_x = train_data.copy()
                point_list = []
                while (len(test_x) - len(train_data)) < len(test_data):
                    x1 = np.reshape(np.array(test_x[-feature_size:]) / Rated_Capacity,
                                    (-1, 1, feature_size)).astype(np.float32)
                    x1 = torch.from_numpy(x1).to('cpu')
                    pred = model(x1)
                    next_point = pred.data.numpy()[0,0] * Rated_Capacity
                    test_x.append(next_point)
                    point_list.append(next_point)
                y_.append(point_list)
                loss_list.append(loss)
                mae, mse, rmse = evaluate(y_test=test_data, y_predict=y_[-1])
                # re
                print('epoch: {:<2d} | loss: {:6.4f} | MAE: {:6.4f} | RMSE: {:6.4f}'.format(epoch, loss, mae, rmse))
            score = [re, mae, rmse]
            if loss < 1e-3 and score_[0] < score[0]:
                break
            score_ = score.copy()
        score_list.append(score_)
        result_list.append(y_[-1])
    return score_list, result_list