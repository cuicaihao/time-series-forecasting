
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def normalise_data(df_csv, column_name="TOTALDEMAND"):
    """
    Description: normalise total-demand data
    Input: df_csv
    Output: 
        X: an array (time series) of normalised_demand
    """
    data_demand = df_csv["TOTALDEMAND"].to_numpy().reshape(-1, 1)

    scaler_demand = StandardScaler()
    scaler_demand.fit(data_demand)
    X = scaler_demand.transform(data_demand)

    return X, scaler_demand


def vec2mat(V, din, dout):
    import numpy as np
    """
    Convert the vector to matrix
    V: input vector, shape(N,1), numpy array/ 2D Matrix
    din:  input demensionality. datatype: integer 
    dout: output demensionality. datatype: integer
    M: matrix with shape (N-din, din+dout)
    """
    V = V.reshape(-1, 1)
    N, p = V.shape
    din = np.uint32(din)
    dout = np.uint32(dout)
    if N < din+dout or p != 1:
        print(f"check {din}+{dout}>={N} ?")
        print(f"check {p} == 1 ?")
        return "[ERROR] Check: Row din+dout <=Number and V should be column matrix"
    M = np.zeros([N-din-dout+1, din+dout])
    for i in range(N-din-dout+1):
        M[i, :] = V[i:din+dout + i].flat  # add the window to the output matrix
    return M


def generate_dataset(df_csv, n_dim_in, n_dim_out, test_ratio):
    """
    Description: Split training and testing data based on the input test_ratio by order. First part is assigned to train set and last part is assigned to test part.
    Input:
        + df_data: data in data frame format includes target in last column, features in remain columns.
        + test_ratio
    Output: X_train, y_train, X_test, y_test
    """
    X_demand, scaler_demand = normalise_data(df_csv)
    # convert vector to array n_dim (number of previous demands used to predict)

    M_demand = vec2mat(X_demand, n_dim_in, n_dim_out)
    # print(M_demand.shape)
    n_row, n_col = M_demand.shape

    X = M_demand[:, 0:n_col-n_dim_out]
    y = M_demand[:, n_col-n_dim_out:]

    train_ratio = 1-test_ratio
    n_train = int(n_row*train_ratio)

    # extract train data
    X_train = X[0:n_train][:]
    y_train = y[0:n_train][:]

    # extract test data
    X_test = X[n_train:][:]
    y_test = y[n_train:][:]

    return X_train, y_train, X_test, y_test, scaler_demand


# Visualisation
def visualise_train_test(X_train, y_train, X_test, y_test, y_pred):
    """
    Description: Visualise the training, testing, and predicted data
    """
    n_train = X_train.shape[0]
    # print(n_train)
    t = np.arange(0, n_train, 1)
    n_test = X_test.shape[0]
    plt.figure(figsize=(20, 10))
    #plt.plot(np.arange(0,n_train,1),y_train,"blue",label="train data")
    plt.plot(np.arange(n_train, n_train+n_test, 1),
             y_test, "green", label="ground truth")
    plt.plot(np.arange(n_train, n_train+n_test, 1),
             y_pred, "r--", label="prediction")
    plt.title("ENERGY DEMAND PREDICTION")
    plt.legend(loc=0)
    return


def mean_absolute_error(y_true, y_pred):
    # or from sklearn.metrics import mean_absolute_error.
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    errors = np.abs(y_true - y_pred)
    return np.mean(errors)


# be carefully with the zero division issues!!!
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)  # list to array
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def plot_result(y_pred, y_test, scaler, length=100):
    t = np.arange(1, length)
    fig, ax = plt.subplots(figsize=(15, 3))
    line1, = ax.plot(t, scaler.inverse_transform(
        y_pred[t]), label='model prediction')
    line2, = ax.plot(t, scaler.inverse_transform(
        y_test[t]), label='real demand records')
    plt.xlabel('sample index')
    plt.ylabel('Elec-Demand')
    ax.legend()
    fig.tight_layout()
    plt.grid()
    plt.show()


def report_results(y_true, y_pred, scaler):
    # y_true and y_pred is in 1d shape (N,).
    mae = mean_absolute_error(scaler.inverse_transform(
        y_true), scaler.inverse_transform(y_pred))
    mape = mean_absolute_percentage_error(
        scaler.inverse_transform(y_true), scaler.inverse_transform(y_pred))
    return mae, mape
