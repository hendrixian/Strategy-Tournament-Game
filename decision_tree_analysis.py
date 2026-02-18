# decision_tree_analysis.py

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
#import numpy as np

def assign_behavior_label(row):
    if row['cooperation_rate'] < 0.5 and row['retaliation'] > 0.5:
        return "Aggressive"
    elif row['cooperation_rate'] > 0.55 and row['forgivingness'] <0.3:
        return "Cooperative"
    elif row['robustness'] > 0.7:
        return "Defensive"
    else:
        return "Balanced"

def prepare_decision_tree_data(radar_metrics):
    #import pandas as pd

    # Convert nested dict → DataFrame
    df = pd.DataFrame.from_dict(radar_metrics, orient='index')
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'strategy'}, inplace=True)

    # Create classification label based on avg_payoff
    #median_score = df['avg_payoff'].median()
    #df['label'] = df['avg_payoff'].apply(
     #   lambda x: "High" if x >= median_score else "Low"
    #)
    if 'forgiveness' in df.columns and 'forgivingness' not in df.columns:
        df['forgivingness'] = df['forgiveness']
    if 'stability' in df.columns and 'robustness' not in df.columns:
        df['robustness']=df['stability']
    if 'avg_payoff' in df.columns and 'score' not in df.columns:
        df['score']=df['avg_payoff']
    if 'score' not in df.columns:
        raise ValueError("Decision tree requires a 'score' metric from radar data.")
   # median_score = df['score'].median()
    #df['label'] = df['score'].apply(
    #lambda x: "High" if x >= median_score else "Low"
#)
    df['label'] = df.apply(assign_behavior_label, axis=1)
    

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
        #'forgiveness',
        #'stability',
        #'win_rate'
        'forgivingness',
        'robustness',
        'score'
    ]
    missing_features = [feature for feature in features if feature not in df.columns]
    if missing_features:
        raise ValueError(
            f"Missing required feature columns for decision tree: {missing_features}"
        )

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
        class_names=["Aggressive", "Cooperative", "Defensive", "Balanced"],
        filled=True,
        rounded=True,
        fontsize=18,
        ax=ax
    )
    return fig
