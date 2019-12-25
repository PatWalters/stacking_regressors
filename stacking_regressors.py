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

out_list = []
for smiles_filename in sorted(glob("*.smi")):
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

    for i in tqdm(range(0, 10)):
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        # first stacking regressor
        estimators_1 = [
            ('ridge', RidgeCV()),
            ('xgb', XGBRegressor(objective='reg:squarederror'))
        ]
        reg_1 = StackingRegressor(
            estimators=estimators_1,
            final_estimator=RandomForestRegressor(n_estimators=10))
        stack_1_r2 = reg_1.fit(X_train, y_train).score(X_test, y_test)

        # second stacking regressor
        estimators_2 = [('ridge', RidgeCV()),
                        ('lasso', LassoCV(tol=0.03)),
                        ('svr', SVR(C=1, gamma=1e-6))]
        reg_2 = StackingRegressor(
            estimators=estimators_2,
            final_estimator=GradientBoostingRegressor())
        stack_2_r2 = reg_2.fit(X_train, y_train).score(X_test, y_test)

        # XGBoost regressor
        xgb = XGBRegressor(objective='reg:squarederror')
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict(X_test)
        xgb_r2 = r2_score(xgb_pred, y_test)

        # Random Forest regressor
        rf = RandomForestRegressor()
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_r2 = r2_score(rf_pred, y_test)

        current_res = [smiles_filename, stack_1_r2, stack_2_r2, xgb_r2, rf_r2]
        print(current_res)
        out_list.append(current_res)

out_df = pd.DataFrame(out_list, columns=["dataset", "stack", "xgb"])
out_df.to_csv("stack.csv", index=False)
