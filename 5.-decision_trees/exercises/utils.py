import os
import pandas as pd
import matplotlib.pyplot as plt


def ou():
    print(f"We are working on {os.getcwd()}")


def q_stats(df: pd.DataFrame):
    """This function returns basic statistics of df"""
    print(f"Df size: {df.shape}")
    # print(f"Df cols {df.columns}")
    print(df.dtypes)


def r_file(name: str) -> pd.DataFrame:
    assert isinstance(name, str), "Name must be a valid file name"
    ip = 'C:\\Users\\f.gonzalez\\Desktop\\inria\\INRIA_SCIKITLEARN\\5.-decision_trees\\'
    f = os.path.join(ip, "input_files", name)
    print(f)
    return pd.read_csv(f)


def plot_dataframe(df: pd.DataFrame):
    """This function plots all the variables of a df wether they are categorical or continuos"""
    num_cols = df.dtypes.loc[df.dtypes == "float64"].index
    cat_cols = df.dtypes.loc[~pd.Series(df.dtypes == "float64")].index
    for col in num_cols:
        df[col].plot()
        plt.title(col)
        plt.show()
    for col in cat_cols:
        df[col].hist()
        plt.title(col)
        plt.show()
