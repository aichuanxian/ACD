import numpy as np
import pandas as pd


def format_eval_output(rows):
    # tweets, targets, labels, predictions = zip(*rows)
    labels, predictions = zip(*rows)

    # Convert labels and predictions to lists
    labels_list = list(labels)
    predictions_list = list(predictions)

    # tweets = np.vstack(tweets)
    # targets = np.vstack(targets)
    # labels = np.vstack(labels_list)
    # predictions = np.vstack(predictions_list)


    results_df = pd.DataFrame()
    # results_df["tweet"] = tweets.reshape(-1).tolist()
    # results_df["target"] = targets.reshape(-1).tolist()
    results_df["label"] = labels
    results_df["prediction"] = predictions
    return results_df