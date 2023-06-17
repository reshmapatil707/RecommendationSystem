import pandas
import numpy
from sklearn.ensemble import RandomForestClassifier

import utilities

MAX_ROWS = 10000
df_raw_train = pandas.read_csv("../data/raw/train_ver2.csv", nrows=MAX_ROWS)
df_raw_test = pandas.read_csv("../data/raw/test_ver2.csv", nrows=MAX_ROWS )

columns = ['fecha_dato', 'ncodpers', 'sexo', 'antiguedad', 'age', 'renta'] + list(df_raw_train.columns[24:-4])
df_train = df_raw_train[columns].copy()

# Cleaning on training data

# Cleaning antiguedad
df_train = utilities.impute_median(df_train, 'antiguedad')
# Cleaning age
df_train = utilities.impute_median(df_train, 'age')

# cleaning sex
df_train['sexo'] = df_train['sexo'].fillna('H')
# clean income imputing median
df_train['renta'] = df_train['renta'].fillna(df_train['renta'].dropna().median())

# make dummies for sexo
df_train['sexo'] = pandas.get_dummies(df_train['sexo'], drop_first=True)

product_columns = [c for c in df_train.columns if c.startswith('ind') and c.endswith('ult1')]

models = []

known_variable = []
for p in product_columns:
    X_train = df_train.loc[:, ['sexo', 'antiguedad', 'age', 'renta']]
    y_train = df_train[p]

    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_train, y_train)
    models.append(clf)
    known_variable = X_train.columns


# Load and Clean the test data
df_test = df_raw_test[known_variable].copy()
df_test.isnull().sum()

# Cleaning antiguedad
# df_test = impute_median(df_test, 'antiguedad')
# Cleaning age
# df_test = impute_median(df_test, 'age')
# cleaning sex
df_test['sexo'] = df_test['sexo'].fillna('H')
# clean income imputing median
df_test = utilities.impute_median(df_test, 'renta')

# make dummies for sexo
df_test['sexo'] = pandas.get_dummies(df_test['sexo'], drop_first=True)

# DO the predictions
collect_predictions = []
for m in models:
    collect_predictions.append(m.predict(df_test[known_variable]))

df_predictions = pandas.DataFrame(numpy.array(collect_predictions).T, columns=product_columns)
df_predictions['ncodpers'] = df_raw_test['ncodpers']
result = []

for df_p in df_predictions.iterrows():
    cur_row = [df_p[1]['ncodpers'], ' '.join(list(df_p[1][df_p[1] == 1].keys()))]
    result.append(cur_row)

df_result = pandas.DataFrame(result, columns=['ncodpers', 'added_products'])
df_result.to_csv("../data/final/submission.csv", index=False)
print("Run Completed!")
