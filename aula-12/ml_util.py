import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

def regression_metrics(model, X, y, label="teste", plt=False, distro_residues=False):
    
    y_pred = model.predict(X)

    print(f"\nMetricas de avaliação (dados de {label}):\n")

    if plt:
        plt.scatter.real.pred(y, y_pred)

    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mape = mean_absolute_percentage_error(y, y_pred)

    print(f"R^2: {r2:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2%}")

    if distro_residues:
        residuos = y - y_pred
        print(f"\nDistribuição de residuos de {label}:\n")
        print(residuos.describe())

    if plt:
        sns.histplot(residuos, kde=True)
        plt.show()

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

def reg_metrics_train_test(
    model, X_train, y_train, X_test, y_test,
    calc_metrics_train=True, plt=False, distro_residues=False
):
    if calc_metrics_train:
        metrics_train = regression_metrics( 
            model, X_train, y_train,
            label="treino", plt=plt, distro_residues=distro_residues
        )

        print("------------------------------------")

    else:
        metrics_train: {}

    metrics_test = regression_metrics(
        model, X_test, y_test,
        label="teste", plt=plt, distro_residues=distro_residues
    )

    return metrics_train, metrics_test

def plot_reglin_poly_feature(modelo, X_train, y_train, X_test, y_test):

    title = "Modelo de Regressão Linear"


    if isinstance(modelo, Pipeline):
        modelo = modelo[-1]

    n_features = len(modelo.coef_)
      

    if n_features > 1:
        title += f" - com features polinomiais de grau {n_features}"

    plt.title(title)

    plt.scatter(X_train, y_train, label="treino")
    plt.scatter(X_test, y_test, label="teste")

    x_plot_modelo = np.linspace(X_train.min(), X_train.max(), 10000)

    y_plot_modelo = modelo.intercept_ 

    for i in range(n_features):
        y_plot_modelo += modelo.coef_[i]*(x_plot_modelo**(i+1))


    plt.plot(x_plot_modelo, y_plot_modelo, color="red", label="modelo")

    plt.legend()
    plt.show()