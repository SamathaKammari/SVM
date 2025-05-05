#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from io import StringIO


# In[2]:


df = pd.read_csv('breast-cancer.csv')


# In[3]:


# Features and labels
X = df.drop(columns=['id', 'diagnosis']).values
y = df['diagnosis'].map({'M': 1, 'B': 0}).values  # Convert labels: M -> 1, B -> 0


# In[4]:


X_2d = X[:, [0, 1]]  # Columns 0 and 1


# In[5]:


# Split data
X_train, X_test, y_train, y_test = train_test_split(X_2d, y, test_size=0.3, random_state=42)


# In[6]:


# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[7]:


# Step 2: Train SVM with linear and RBF kernel
svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

svm_linear.fit(X_train_scaled, y_train)
svm_rbf.fit(X_train_scaled, y_train)


# In[8]:


# Step 3: Visualize decision boundary
def plot_decision_boundary(model, X, y, title):
    h = 0.02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.title(title)
    plt.xlabel('Radius Mean (Scaled)')
    plt.ylabel('Texture Mean (Scaled)')
    # Plot decision boundaries
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plot_decision_boundary(svm_linear, X_train_scaled, y_train, 'Linear SVM')

plt.subplot(1, 2, 2)
plot_decision_boundary(svm_rbf, X_train_scaled, y_train, 'RBF SVM')

plt.tight_layout()
plt.show()


# In[9]:


# Step 4: Tune hyperparameters (C and gamma for RBF) on full dataset
X_full_train, X_full_test, y_full_train, y_full_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_full_train_scaled = scaler.fit_transform(X_full_train)
X_full_test_scaled = scaler.transform(X_full_test)

C_values = [0.1, 1, 10]
gamma_values = ['scale', 0.1, 1]
best_score = 0
best_params = {}

for C in C_values:
    for gamma in gamma_values:
        svm = SVC(kernel='rbf', C=C, gamma=gamma, random_state=42)
        svm.fit(X_full_train_scaled, y_full_train)
        score = svm.score(X_full_test_scaled, y_full_test)
        if score > best_score:
            best_score = score
            best_params = {'C': C, 'gamma': gamma}

print(f"Best parameters: {best_params}")
print(f"Best test accuracy: {best_score:.4f}")


# In[10]:


# Step 5: Cross-validation to evaluate performance
svm_best = SVC(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'], random_state=42)
cv_scores = cross_val_score(svm_best, X_full_train_scaled, y_full_train, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


# In[ ]:




