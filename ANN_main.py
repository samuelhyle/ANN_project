
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPRegressor

weights_1 = np.array([1., 2., 1.])
bias = np.array([0., -2., -2.])
weights_2 = np.array([1, -1, 1])


def hidden(x, weights_1, bias):

    hidden_x = x * weights_1 + bias
    return np.maximum(hidden_x, 0)


def network(x, weights_1, bias, weights_2):

    hidden_x = hidden(x, weights_1, bias)
    final_x = hidden_x * weights_2
    return sum(final_x)

x_to_plot = np.linspace(-0.5, 2.5, 99)

plt.figure(figsize=(10, 12))
for i in range(3):
    plt.subplot(3, 1, i + 1)
    hidden_to_plot = [hidden(x, weights_1, bias)[i] for x in x_to_plot]
    plt.xlim(-0.5, 2.5)
    plt.ylim(-0.5, 1.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f'Hidden {i + 1} Plot')
    plt.xlabel('x')
    plt.ylabel(f'h{i + 1}')
    plt.plot(x_to_plot, hidden_to_plot)

plt.tight_layout()

y_to_plot = [network(x, weights_1, bias, weights_2) for x in x_to_plot]
plt.figure()
plt.title('Neural Network Toy Example Hypothesis Function')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x_to_plot, y_to_plot)
plt.show()

assert np.isclose(network(0, weights_1, bias, weights_2), 0.)
assert np.isclose(network(1, weights_1, bias, weights_2), 1.)
assert np.isclose(network(2, weights_1, bias, weights_2), 0.)
assert np.isclose(y_to_plot[0], 0.)
assert np.isclose(y_to_plot[-1], 0.)
assert np.isclose(y_to_plot[len(y_to_plot) // 2], 1.)


def main():
    data = pd.read_csv('FMIData_ann.csv')
    data.drop(columns=['Time zone', 'Precipitation amount (mm)', 'Snow depth (cm)',
                       'Ground minimum temperature (degC)', 'Maximum temperature (degC)', 'Minimum temperature (degC)'], inplace=True)
    data.columns = ['year', 'm', 'd', 'time', 'air temperature']

    data = data[data['time'] == '00:00']

    data['pre_1'] = data['air temperature'].shift(1)
    data['pre_2'] = data['air temperature'].shift(2)
    data['pre_3'] = data['air temperature'].shift(3)
    data['pre_4'] = data['air temperature'].shift(4)
    data['pre_5'] = data['air temperature'].shift(5)

    data = data.iloc[5:]

    X = data[['pre_1', 'pre_2', 'pre_3', 'pre_4', 'pre_5']]
    y = data['air temperature']


    print("Feature Matrix X:")
    print(X.head())
    print("\nLabel Vector y Shape:", y.shape)

    assert X.ndim == 2, "Wrong dimension of X"
    assert X.shape[1] == 5, "Wrong shape of X"
    assert y.ndim == 1, "Wrong dimension of y"
    assert y.shape[0] == 708, "Wrong shape of y"

main()