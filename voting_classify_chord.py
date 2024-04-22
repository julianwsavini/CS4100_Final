import pandas as pd
from chroma import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle
import matplotlib.pyplot as plt
import seaborn as sns 

all_chromas = pd.concat([get_chromagram(sample, midi_data) for sample in list(midi_data.keys())])
outputs = all_chromas['Chord Actual']
inputs = all_chromas.drop(labels=['Chord Actual'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2)

# step 2: Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Assuming X_train, X_test are your input data and y_train, y_test are the corresponding labels
# Initialize and train the Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=65)
dt_classifier.fit(X_train_scaled, y_train)

# Initialize base classifiers
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train_scaled, y_train)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=65)
rf_classifier.fit(X_train_scaled, y_train)

# Create a Voting Classifier with the base classifiers
voting_classifier = VotingClassifier(estimators=[('knn', knn_classifier), ('dt', dt_classifier), ('rf', rf_classifier)], voting='soft')
voting_classifier.fit(X_train_scaled, y_train)

# write model to pkl file
pickle.dump(voting_classifier, open('voting.pkl', "wb"))

# make predictions on the test set
y_pred = voting_classifier.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
conf_matrix = confusion_matrix(y_test, y_pred)
labels = list(np.unique(y_pred))
# plot confusion matrix 
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, cmap='viridis', fmt='d', cbar=False, xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# t_z = np.zeros(12)
# t_z[11] = 80
# t_z[2] = 90
# t_z[7] = 127
# test = pd.DataFrame(t_z).transpose()
# test.columns = notes
# new_scaled_test = scaler.transform(test)
# print(voting_classifier.predict(new_scaled_test))