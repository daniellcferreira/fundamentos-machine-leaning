import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

def regression_metrics(model, X, y, label="teste", plt=False, distro_residues=False):
    
    y_pred = model.predict(X)

    print(f"\nMetricas de avaliação (dados de {label}):\n")

    if plot:
        plot_scatter_real(y, y_pred)

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

    if plot:
        sns.histplot(residuos, kde=True)
        plt.show()

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

def reg_metrics_train_test(
    model, X_train, y_train, X_test, y_test,
    calc_metrics_train=True, plot=False, distro_residues=False
):
    if calc_metrics_train_test:
        metrics_train = regrisson_metrics( 
            model, X_train, y_train,
            label="treino", plot=plot, distro_residues=distro_residues
        )

        print("------------------------------------")

    else:
        metrics_train: {}

    metrics_test = regression_metrics(
        model, X_test, y_test,
        label="teste", plot=plot, distro_residues=distro_residues
    )

    return metrics_train, metrics_test