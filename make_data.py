"""
This script makes a tabular data set that a simple nerual network should be
able to make predictions on easily.

by Neil Campbell
July 30, 2023
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


def main():

    num_neg = 20000
    num_pos = 4000
    num_col = 8

    xs_neg = [np.random.normal(-3+2*i, 2, num_neg) for i in range(num_col)]
    xs_pos = [np.random.normal(-2+2*i, 2, num_pos) for i in range(num_col)]

    pos_idx = random.sample(range(1,num_pos+num_neg-1), num_pos)

    i_n = 1
    i_p = 0
    data = pd.DataFrame({f"col{i}":elem[0] for i, elem in enumerate(xs_neg)},
                        index=[0])

    print("About to assemble DF")
    for j in range(1, num_neg+num_pos):
        
        if j in pos_idx:

            new_df = pd.DataFrame({f"col{i}": elem[i_p] 
                                   for i, elem in enumerate(xs_pos)},
                                  index=[j])

            data = pd.concat((data, new_df))
            i_p += 1

        else:
        
            new_df = pd.DataFrame({f"col{i}": elem[i_n] 
                                   for i, elem in enumerate(xs_neg)},
                                  index=[j])

            data = pd.concat((data, new_df))
            i_n += 1

    # Add label
    lbs = np.zeros(num_neg + num_pos)
    lbs[pos_idx] = 1
    data["label"] = lbs

    print(data["label"].sum())
    print(pos_idx[:6])
    print(data.head(20))


    data.to_csv("data.txt", index=False)



if __name__ == "__main__":
    main()
