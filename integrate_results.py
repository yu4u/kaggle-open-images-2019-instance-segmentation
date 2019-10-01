import argparse
import better_exceptions
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input1", type=str, required=True,
                        help="input layer 0 submissison file")
    parser.add_argument("--input2", type=str, required=True,
                        help="input layer 1 submissison file")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    df1 = pd.read_csv(args.input1, index_col="ImageID")
    df1.sort_index(axis=0, inplace=True)
    df2 = pd.read_csv(args.input2, index_col="ImageID")
    df2.sort_index(axis=0, inplace=True)
    results = []

    for i, ((index1, row1), (index2, row2)) in enumerate(zip(df1.iterrows(), df2.iterrows())):
        assert(index1 == index2)
        s1 = row1["PredictionString"]
        s2 = row2["PredictionString"]

        if isinstance(s1, float) and isinstance(s2, float):
            results.append("")
        elif isinstance(s1, float):
            results.append(s2)
        elif isinstance(s2, float):
            results.append(s1)
        else:
            results.append(" ".join([s1, s2]))

    df1.PredictionString = results
    df1.to_csv("integrated_result.csv")


if __name__ == '__main__':
    main()
