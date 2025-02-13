import pandas as pd
df = pd.read_csv('/Users/shanmukhamundra/Desktop/House_Clsfcation/house_class.csv')
from sklearn.model_selection import train_test_split
X = df[['Area', 'Room', 'Lon', 'Lat', 'Zip_area', 'Zip_loc']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=X['Zip_loc'].values, random_state=1
)
#----OneHotEncoding----#
from sklearn.preprocessing import OneHotEncoder
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
predictions_ohe = model.predict(X_test_final)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions_ohe)
#print(accuracy)
#-----Ordinal Encoding-----#
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder()
encoder.fit(X_train[['Zip_area', 'Zip_loc', 'Room']])
encoded_train = encoder.transform(X_train[['Zip_area', 'Zip_loc', 'Room']])
encoded_test = encoder.transform(X_test[['Zip_area', 'Zip_loc', 'Room']])
encoded_train_df = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out())
encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out())
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
predictions_ord = model.predict(X_test_final)
from sklearn.metrics import accuracy_score
accuracy_ordinal = accuracy_score(y_test, predictions_ord)
#print(accuracy_ordinal)
#----Target Encoding----#
from category_encoders import TargetEncoder
encoder = TargetEncoder(cols=['Zip_area', 'Room', 'Zip_loc'])
encoder.fit(X_train, y_train)
X_train = encoder.transform(X_train)
X_test = encoder.transform(X_test)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(
    criterion='entropy',
    max_features=3,
    splitter='best',
    max_depth=6,
    min_samples_split=4,
    random_state=3
)
model.fit(X_train, y_train)
predictions_target = model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_target = accuracy_score(y_test, predictions_target)
#print(accuracy_target)
from sklearn.metrics import classification_report
report1 = classification_report(y_test, predictions_ohe, output_dict=True)
#print(report1)
report2 = classification_report(y_test, predictions_ord, output_dict=True)
#print(report2)
report3 = classification_report(y_test, predictions_target, output_dict=True)
#print(report3)
f1_macro_ohe = report1['macro avg']['f1-score']
f1_macro_ord = report2['macro avg']['f1-score']
f1_macro_tar = report3['macro avg']['f1-score']
print(f'OneHotEncoder:{f1_macro_ohe:.2f}')
print(f'OrdinalEncoder:{f1_macro_ord:.2f}')
print(f'TargetEncoder:{f1_macro_tar:.2f}')