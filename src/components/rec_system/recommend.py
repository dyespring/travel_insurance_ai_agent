import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

from preprocessing import preprocessing

def recommend(gender:str, des:str, dur:int, age:int):
    dt = preprocessing(gender, des)

    y=dt.iloc[:,[0]]
    X=dt.iloc[:,[-4, -1]]

    X_new=pd.DataFrame()
    to_scale = X.columns
    scaler = StandardScaler()
    X_new[to_scale] = scaler.fit_transform(X[to_scale])

    # Define the input values
    data = [[dur, age]]  # a list of lists (rows)

    # Create the DataFrame
    input_df = pd.DataFrame(data, columns=X.columns)
    input_df = scaler.transform(input_df)

    #use nearest neighbors algorithm
    knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=3, n_jobs=-1)
    knn.fit(X_new, y);

    # get the recommend package
    distances , indices = knn.kneighbors(input_df,n_neighbors=3)  
    rec_indices = list(zip(indices.squeeze().tolist(), distances.squeeze().tolist()))
    recommend_frame = []
    for val in rec_indices:
        idx = dt[dt['id'] == val[0]].index
        recommend_frame.append({
            'Duration':dt.iloc[idx]['Duration'].values[0],
            'Age':dt.iloc[idx]['Age'].values[0],
            'Distance':val[1]
        })
        df = pd.DataFrame(recommend_frame)

    print(df)   

#test
if __name__ == "__main__":
    recommend("F", "SINGAPORE", 70, 30)