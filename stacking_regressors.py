#!/usr/bin/env python

import warnings
from glob import glob

import janitor.chemistry
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from tqdm import tqdm
from xgboost import XGBRegressor

# Ignore future warnings from XGBoost. I know, it's wrong to do this, but ...
warnings.simplefilter(action='ignore', category=FutureWarning)


def test_regressor(dataset_name, regressor, name, cycle, x_train, x_test, y_train, y_test):
    regressor.fit(x_train, y_train)
    pred = regressor.predict(x_test)
    result = []
    for p, y in zip(pred, y_test):
        result.append([dataset_name, name, cycle, y, p])
    return result


def generate_regressors():
    # first stacking regressor
    estimators_1 = [
        ('ridge', RidgeCV()),
        ('xgb', XGBRegressor(objective='reg:squarederror'))
    ]
    stack_1 = StackingRegressor(
        estimators=estimators_1,
        final_estimator=RandomForestRegressor(n_estimators=10))

    # second stacking regressor
    estimators_2 = [('ridge', RidgeCV()),
                    ('lasso', LassoCV(tol=0.03)),
                    ('svr', SVR(C=1, gamma=1e-6))]
    stack_2 = StackingRegressor(
        estimators=estimators_2,
        final_estimator=GradientBoostingRegressor())

    # XGBoost regressor
    xgb = XGBRegressor(objective='reg:squarederror')

    # Random Forest regressor
    rf = RandomForestRegressor()

    regressor_list = [
        (stack_1, "stack_1"),
        (stack_2, "stack_2"),
        (xgb, "xgb"),
        (rf, "rf")
    ]

    return regressor_list


def main():
    out_list = []
    for smiles_filename in sorted(glob("data/*.smi")):
        print(smiles_filename)
        df = pd.read_csv(smiles_filename, sep=" ", header=None)
        df.columns = ['smiles', 'name', 'pIC50']

        # Generate the descriptors wit PyJanitor
        fp = janitor.chemistry.morgan_fingerprint(
            df=df.smiles2mol('smiles', 'mols'),
            mols_column_name='mols',
            radius=3,
            nbits=2048,
            kind='counts'
        )

        X = fp
        y = df.pIC50

        cv_cycles = 10
        dataset_name = smiles_filename.replace("data/", "").replace(".smi", "")
        for i in tqdm(range(0, cv_cycles)):
            X_train, X_test, y_train, y_test = train_test_split(X, y)
            for regressor, name in generate_regressors():
                out_list += test_regressor(dataset_name, regressor, name, i, X_train, X_test, y_train, y_test)

    out_df = pd.DataFrame(out_list, columns=["dataset", "method", "cycle", "exp", "pred"])
    out_df.to_csv("comparison.csv", index=False)


main()
