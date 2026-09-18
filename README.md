import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

df = pd.read_csv('thyroid_dataset.csv')

df.head()

X = df.drop(columns = ['Outlier_label'], axis = 1)
y = df['Outlier_label']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


clf = IsolationForest(n_estimators=200, contamination='auto', random_state=42)
labels = clf.fit_predict(X_scaled, y)


labels

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 8))
sns.scatterplot(x = X_pca[:, 0], y = X_pca[:, 1], c = labels)
plt.xlabel('PC label 1')
plt.ylabel('PC label 2')
plt.title('Anamoly Detection + PCA + Random Forest')
plt.show()

normal = np.sum(labels == 1)
outlier = np.sum(labels == -1)

print(f'The normal data = {normal}')
print(f'Outlier in the data = {outlier}')
