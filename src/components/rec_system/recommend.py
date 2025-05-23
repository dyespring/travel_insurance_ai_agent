import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .preprocessing import preprocessing

def recommend(gender:str, des:str, dur:int, age:int):
    n = 10
    dt = preprocessing(gender, des)

    if (len(dt.index) <= 1):
        return [];   
    if (len(dt.index) < n):
        n = len(dt.index) - 1

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
    knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=n, n_jobs=-1)
    knn.fit(X_new, y);

    # get the recommend package
    distances , indices = knn.kneighbors(input_df,n_neighbors=n) 
    
    rec_indices = list(zip(indices.squeeze().tolist(), distances.squeeze().tolist()))
    recommend_frame = []
    for val in rec_indices:
        idx = dt[dt['id'] == val[0]].index
        recommend_frame.append({
            'Product Name':dt.iloc[idx]['Product Name'].values[0],
            'Agency':dt.iloc[idx]['Agency'].values[0],
            'Agency Type':dt.iloc[idx]['Agency Type'].values[0],
            'Gender':dt.iloc[idx]['Gender'].values[0],
            'Destination':dt.iloc[idx]['Destination'].values[0],
            'Duration':dt.iloc[idx]['Duration'].values[0],
            'Age':dt.iloc[idx]['Age'].values[0],
            'Distance':val[1]
        })
        # df = pd.DataFrame(recommend_frame)
        # result = dt.to_json(orient='records')[1:-1].replace('},{', '} {')
    seen = set()
    filtered = []

    for item in recommend_frame:
        key = (item['Product Name'], item['Agency'])
        if key not in seen:
            seen.add(key)
            filtered.append(item)
        if len(filtered) == 3:
            break

    # `filtered` now contains the top 3 unique (Product Name, Agency) combinations
    print(filtered)
    return filtered   

#test
if __name__ == "__main__":
    recommend("F", "SINGAPORE", 70, 30)