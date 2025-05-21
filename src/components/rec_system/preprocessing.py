import pandas as pd
import os

def preprocessing(gender: str, des: str):
    dt = pd.read_csv("travel-insurance.csv");

    # drop non-necessary column
    dt=dt.drop(['Net Sales', 'Commision (in value)', "Distribution Channel", "Claim"], axis=1)

    # drop record where age >= 0 and < 10
    index_age = dt[(dt['Age'] >= 0) & (dt['Age'] < 10) ].index
    dt.drop(index_age, inplace = True)

    # # drop record where age > 80
    index_age = dt[dt['Age'] > 80].index
    dt.drop(index_age, inplace = True)

    #Filter gender
    dt = dt[dt['Gender'] == gender].reset_index(drop=True)

    # #Filter destination
    dt = dt[dt['Destination'] == des].reset_index(drop=True)

    # populate id
    dt['id'] = range(len(dt))
    dt = dt[['id'] + [col for col in dt.columns if col != 'id']]
    print(dt)

    return dt

#test
if __name__ == "__main__":
    preprocessing("F", "SINGAPORE")