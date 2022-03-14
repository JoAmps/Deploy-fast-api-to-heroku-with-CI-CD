# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import logging
from data import process_data
from clean_data import load_data, cleaned_data
from model_functions import train_model, \
    compute_model_metrics, model_predictions
from joblib import dump


logging.basicConfig(
    filename='./log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(cleaned_data):
    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        logging.info('SUCCESS!: Data has rows and columns!')
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't\
                 appear to have rows and columns")
        raise err

# Optional enhancement, use K-fold cross validation instead of a
# train-test split.


def split_data(data):
    try:
        train, test = train_test_split(data, test_size=0.20, random_state=0)
        logging.info('SUCCESS!:Data split successfully')
        return train, test
    except BaseException:
        logging.info('Error!:Error whiles splitting data')


cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def test_model(train_model):
    try:
        if not X_train.shape[0] == y_train.shape[0]:
            raise AssertionError(
                (X_train.shape[0],
                 y_train.shape[0]))
        if not X_test.shape[0] == y_test.shape[0]:
            raise AssertionError(
                (X_test.shape[0],
                 y_test.shape[0]))
        logging.info(
            'SUCCESS: dependent and independent training\
                 and testing data have equal lengths')
    except AssertionError as err:
        logging.error(
            "Lengths of the Independent and dependent training\
                 and testing data mismatch")
        raise err


def test_metrics(compute_model_metrics):
    try:
        if not y_test.shape[0] == predictions.shape[0]:
            raise AssertionError(
                (y_test.shape[0],
                 predictions.shape[0]))

        logging.info(
            'SUCCESS: predictions and output test data \
            have equal lengths')
    except AssertionError as err:
        logging.error(
            "Lengths of predictions and output test \
                 data mismatch")
        raise err


def model_slicing(data):
    """
    Slice model for categorical features
    """
    slice_values = []

    for cat in cat_features:
        for cls in test[cat].unique():
            df_temp = test[test[cat] == cls]
            X_test_temp, y_test_temp, _, _ = process_data(
                df_temp, categorical_features=cat_features,
                label="salary", encoder=encoder, lb=lb, training=False)
            y_preds = model.predict(X_test_temp)
            precision_temp, recall_temp, fbeta_temp = compute_model_metrics(
                y_test_temp, y_preds)
            results = "[%s->%s] Precision: %s " \
                "Recall: %s FBeta: %s" % (
                    cat,
                    cls,
                    precision_temp,
                    recall_temp,
                    fbeta_temp)
            slice_values.append(results)

    with open('slice_model_output.txt', 'w') as out:
        for slice_value in slice_values:
            out.write(slice_value + '\n')


if __name__ == '__main__':
    df = load_data('/Users/hyacinthampadu/Documents/Jos Folder/Data Science/Udacity mL devops engineer/project_3_rearrangements/project 3/Project_3/data/census_cleaned.csv')
    test_import(cleaned_data)
    train, test = split_data(df)
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features,
        label="salary", training=True)
    X_test, y_test, encoder_t, lb_t = process_data(
        test, categorical_features=cat_features,
        label="salary", training=False, encoder=encoder, lb=lb)
    test_model(train_model)
    model = train_model(X_train, y_train)
    predictions = model_predictions(X_test, model)
    test_metrics(compute_model_metrics)
    precision, recall, fbeta = compute_model_metrics(y_test, predictions)
    model_slicing(df)
