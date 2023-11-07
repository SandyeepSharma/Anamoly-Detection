import numpy as np
import pandas as pd


df = pd.read_csv(r"/home/sandeep/disk_C/csv_file/creditcardfraud/creditcard.csv")
df.isnull().sum()

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


from sklearn.preprocessing import StandardScaler
nrm = StandardScaler()
X_train = nrm.fit_transform(X_train)
X_test = nrm.transform(X_test)

# =============================================================================
# from imblearn.over_sampling import SMOTE
# sm = SMOTE(random_state=12, ratio = 1.0)
# x_res, y_res = sm.fit_sample(X_train, y_train)
# =============================================================================
#y.value_counts(), np.bincount(y_res)



from sklearn.ensemble import IsolationForest
IF = IsolationForest(n_estimators= 150  , random_state= 42)
IF.fit(x_res)
y_pred = IF.predict(x_res)

y_pred[y_pred==-1] =1
y_pred[y_pred==1] =0

from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators= 150  , random_state= 42)
RF.fit(X_train, y_train)
y_pred = RF.predict(X_test)


from sklearn.svm import SVC
svc = SVC(kernel='rbf')
svc.fit(X_train, y_train)
y_pred = svc.predict(y_test)

from sklearn.neighbors import LocalOutlierFactor
LOF = LocalOutlierFactor(n_neighbors= 20 )
y_pred =LOF.fit_predict(x_res)


y_pred[y_pred==-1] =1
y_pred[y_pred==1] =0


from sklearn.metrics import classification_report , accuracy_score , confusion_matrix
cr = classification_report(y_test , y_pred)
ac_s = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)