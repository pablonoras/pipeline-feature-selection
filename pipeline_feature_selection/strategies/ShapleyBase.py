import xgboost as xgb
import numpy as np
import pandas as pd
import shap


class XGBoostClassifier:
    def __init__(self, max_depth=5):
        self.classifier = xgb.XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=max_depth,
                                            min_child_weight=3, gamma=0.2, subsample=0.6, colsample_bytree=1.0,
                                            objective='binary:logistic', nthread=20, scale_pos_weight=1, seed=27,
                                            use_label_encoder=False, eval_metric='mlogloss')

    def fit(self, X, y):
        self.classifier.fit(X, y)
        self.X = X

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)[:, 1]

    def get_ranked_features(self):
        mybooster = self.classifier.get_booster()
        model_bytearray = mybooster.save_raw()[4:]

        def myfun(self=None):
            return model_bytearray

        mybooster.save_raw = myfun

        explainer = shap.TreeExplainer(mybooster)
        shap_values = explainer.shap_values(self.X)
        shap_features = self.X.columns[np.argsort(-np.abs(shap_values).mean(0))].tolist()
        shap_values = -np.sort(-np.abs(shap_values).mean(0))

        feature = pd.DataFrame({'feature': list(shap_features)})
        importance = pd.DataFrame({'importance': list(shap_values)})

        return pd.concat([feature, importance], join='outer', axis=1)