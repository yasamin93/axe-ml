import mlflow

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

mlflow.autolog()

db = load_diabetes()
X_train, X_test, Y_train, Y_test = train_test_split(db.data, db.target)


def create_and_train_model():
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
    rf.fit(X_train, Y_train)
    return rf


rf = create_and_train_model()
predictions = rf.predict(X_test)
