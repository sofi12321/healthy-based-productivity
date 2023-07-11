import copy

def out_to_features(X, Y_pred):
    X_new = copy.deepcopy(X.clone().detach())
    Y_pred_new = copy.deepcopy(Y_pred.clone().detach())


    old_curr_time = X_new[5]
    new_start_time = Y_pred_new[0]

    # Set the new start time
    X_new[13] = new_start_time

    if new_start_time >= 1:

        X_new[13] = new_start_time - 1
        # Need to increase day of the week
        for i, val in enumerate(X[14:21]):
            if val == 1:
                X_new[14 + i] = 0

                if i == 6:
                    X_new[14] = 1

                else:
                    X_new[14 + i + 1] = 1

                break

    return X_new


#
# X = [0, 0, 0, 0, 0,    0.5,     0, 0, 0, 0, 0, 0, 0,    0.6,    0, 0, 1, 0, 0, 0, 0,    0]
# Y_pred = [0.4, 0, 0]
# print(out_to_features(X, Y_pred))