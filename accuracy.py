import pandas as pd
df = pd.read_csv('/Users/shanmukhamundra/Desktop/House_Clsfcation/house_class.csv')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
X = df[['Area', 'Room', 'Lon', 'Lat', 'Zip_area', 'Zip_loc']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=X['Zip_loc'].values, random_state=1
)
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoder.fit(X_train[['Zip_area', 'Zip_loc', 'Room']])
encoded_train = encoder.transform(X_train[['Zip_area', 'Zip_loc', 'Room']])
encoded_test = encoder.transform(X_test[['Zip_area', 'Zip_loc', 'Room']])
encoded_train_df = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(), index=X_train.index)
encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(), index=X_test.index)
X_train_final = X_train[['Area', 'Lon', 'Lat']].join(encoded_train_df)
X_test_final = X_test[['Area', 'Lon', 'Lat']].join(encoded_test_df)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(
    criterion='entropy',
    max_features=3,
    splitter='best',
    max_depth=6,
    min_samples_split=4,
    random_state=3
)
model.fit(X_train_final, y_train)
predictions = model.predict(X_test_final)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print(round(accuracy, 2))