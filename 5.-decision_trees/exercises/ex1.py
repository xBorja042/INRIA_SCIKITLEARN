from exercises.utils import *


def ex1():
    df = r_file("penguins_lter.csv")
    q_stats(df)
    # plot_dataframe(df)
    # print(df.describe())
    df["Species"].hist()
    plt.show()