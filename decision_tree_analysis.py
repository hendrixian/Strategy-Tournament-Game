# decision_tree_analysis.py

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np


def prepare_decision_tree_data(radar_metrics):
    import pandas as pd

    # Convert nested dict → DataFrame
    df = pd.DataFrame.from_dict(radar_metrics, orient='index')
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'strategy'}, inplace=True)

    # Create classification label based on avg_payoff
    median_score = df['avg_payoff'].median()
    df['label'] = df['avg_payoff'].apply(
        lambda x: "High" if x >= median_score else "Low"
    )

    return df



from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def train_decision_tree(df):
    """
    Train Decision Tree Classifier
    """

    features = [
        'cooperation_rate',
        'retaliation',
        'forgiveness',
        'stability',
        'win_rate'
    ]

    X = df[features]
    y = df['label']

    # 🔥 Train-test split (NEW)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = DecisionTreeClassifier(
        max_depth=3,
        random_state=42
    )

    clf.fit(X_train, y_train)

    # 🔥 Return everything needed
    return clf, features, X_train, X_test, y_train, y_test




def plot_decision_tree(clf, feature_names):
    """
    Plot tree structure
    """
    fig, ax = plt.subplots(figsize=(22, 12), dpi=120)
    tree.plot_tree(
        clf,
        feature_names=feature_names,
        class_names=["Low", "High"],
        filled=True,
        rounded=True,
        fontsize=18,
        ax=ax
    )
    return fig
