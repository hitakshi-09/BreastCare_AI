#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the essental libraries 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[2]:


from sklearn.datasets import load_breast_cancer


# In[3]:


cancer = load_breast_cancer()


# In[4]:


cancer


# In[5]:


cancer.keys()


# In[6]:


cancer.values()


# In[7]:


print(cancer['DESCR'])


# In[8]:


print(cancer['target'])


# In[9]:


print(cancer['target_names'])


# In[10]:


print(cancer['feature_names'])


# In[11]:


cancer['data'].shape


# In[12]:


df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns= np.append(cancer['feature_names'],['target']))


# In[13]:


df_cancer.head()


# In[14]:


df_cancer.tail()


# # *VISUALISING THE DATA*

# In[15]:


sns.pairplot(df_cancer , vars =['mean radius','mean texture', 'mean perimeter', 'mean area',
 'mean smoothness', 'mean compactness' ,'mean concavity',
 'mean concave points', 'mean symmetry', 'mean fractal dimension',
 'radius error', 'texture error', 'perimeter error', 'area error',
 'smoothness error', 'compactness error', 'concavity error',
 'concave points error', 'symmetry error', 'fractal dimension error',
 'worst radius', 'worst texture', 'worst perimeter', 'worst area',
 'worst smoothness', 'worst compactness', 'worst concavity',
 'worst concave points', 'worst symmetry', 'worst fractal dimension'])                                                                                                                                                                          


# In[16]:


sns.pairplot(df_cancer ,hue ='target', vars =['mean radius','mean texture', 'mean perimeter', 'mean area',
 'mean smoothness', 'mean compactness' ,'mean concavity']) 


# In[17]:


sns.scatterplot(x='mean area',y='mean smoothness',hue='target',data =df_cancer)


# In[18]:


plt.figure(figsize =(20,10))
sns.heatmap(df_cancer.corr(), annot =True)


# # *SPLITTING THE DATASET*

# In[19]:


x = df_cancer.drop(['target'],axis =1)


# In[20]:


x


# In[21]:


y= df_cancer['target']


# In[22]:


y


# In[23]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)


# In[ ]:


x_train


# In[26]:


y_train


# In[27]:


x_test


# In[28]:


y_test


# # *TRAINING THE MODEL USING SVM*

# In[29]:


from sklearn.svm import SVC


# In[30]:


from sklearn.metrics import classification_report , confusion_matrix


# In[31]:


svc_model= SVC()


# In[32]:


svc_model.fit(x_train,y_train)


# # *EVALUATING THE MODEL*
# 

# In[33]:


y_predict =svc_model.predict(x_test)


# In[34]:


y_predict


# In[35]:


cm = confusion_matrix(y_test,y_predict)


# In[36]:


sns.heatmap(cm ,annot=True)


# # Model Improvisation
# 

# In[37]:


min_train =x_train.min()


# In[38]:


range_train =(x_train - min_train).max()


# In[39]:


x_train_scaled =(x_train-min_train)/range_train


# In[40]:


sns.scatterplot(x = x_train['mean area'], y= x_train['mean smoothness'],hue =y_train)


# In[41]:


sns.scatterplot(x = x_train_scaled['mean area'], y= x_train_scaled['mean smoothness'],hue =y_train)


# In[42]:


min_test =x_test.min()
range_test =(x_test - min_test).max()
x_test_scaled =(x_test-min_test)/range_test


# In[43]:


svc_model.fit(x_train_scaled,y_train)


# In[44]:


y_predict =svc_model.predict(x_test_scaled)


# In[45]:


cn = confusion_matrix(y_test,y_predict)


# In[46]:


sns.heatmap(cn,  annot = True)


# In[47]:


print(classification_report(y_test,y_predict))


# ### An accuracy of 96% has been achieved after appying the technique of Normalization for Improvisation

# In[48]:


param_grid ={'C':[0.1,1,10,100],'gamma':[1,0.1,0.01,0.001],'kernel':['rbf']}


# In[49]:


from sklearn.model_selection import GridSearchCV


# In[50]:


grid=GridSearchCV(SVC(),param_grid,refit=True,verbose=4)


# In[51]:


grid.fit(x_train_scaled,y_train)


# In[52]:


grid.best_params_


# In[53]:


grid_predictions=grid.predict(x_test_scaled)


# In[54]:


cn =confusion_matrix(y_test,grid_predictions)


# In[55]:


sns.heatmap(cn , annot =True)


# In[56]:


print(classification_report(y_test,grid_predictions))


# ### Accuracy of 97% has been achieved by further Improvisation by optimization of C and Gamma Parameters

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import tkinter as tk
from tkinter import messagebox
import os
import csv

# Load the dataset
cancer = load_breast_cancer()

# Create a DataFrame
df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns=np.append(cancer['feature_names'], ['target']))

# Select all features
features = cancer['feature_names']
X = df_cancer[features]
y = df_cancer['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Scale the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
svc_model = SVC()
svc_model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = svc_model.predict(X_test_scaled)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

def write_to_csv(user_data, result):
    # Define the file name
    file_name = 'predictions.csv'

    # Check if the file exists and write the header if not
    file_exists = os.path.isfile(file_name)

    # Open the file in append mode
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write the header if the file does not exist
        if not file_exists:
            writer.writerow(features + ['prediction'])

        # Write the user data and result
        writer.writerow(list(user_data.flatten()) + [result])

def predict_cancer():
    try:
        # Get user input
        user_data = np.array([float(entries[feature].get()) for feature in features]).reshape(1, -1)

        # Create a DataFrame with the correct feature names
        user_data_df = pd.DataFrame(user_data, columns=features)

        # Scale the input data
        user_data_scaled = scaler.transform(user_data_df)

        # Make a prediction
        prediction = svc_model.predict(user_data_scaled)
        result = "Benign (not detected)" if prediction[0] == 1 else "Malignant (detected)"

        # Write the user data and prediction result to CSV
        write_to_csv(user_data, result)

        # Display the result in a custom window
        result_window = tk.Toplevel(root)
        result_window.title("Prediction Result")

        # Set the size of the result window
        result_window.geometry("600x200")

        # Center the result window on the screen
        result_window.update_idletasks()
        width = result_window.winfo_width()
        height = result_window.winfo_height()
        x = (result_window.winfo_screenwidth() // 2) - (width // 2)
        y = (result_window.winfo_screenheight() // 2) - (height // 2)
        result_window.geometry(f'{width}x{height}+{x}+{y}')

        tk.Label(result_window, text=f"The breast cancer prediction is {result}", font=("Georgia", 16)).pack(expand=True, pady=20)
        tk.Button(result_window, text="OK", command=result_window.destroy, font=("Georgia", 14), bg="green", fg="white").pack(pady=10)
    except Exception as e:
        messagebox.showerror("Input Error", "Please enter valid data for all fields")

# Create the main window
root = tk.Tk()
root.title("Breast Cancer Prediction")
root.attributes('-fullscreen', True)

# Set the background color for the main window
root.configure(bg='#f0f0f0')

# Create a header frame with a background color
header_frame = tk.Frame(root, bg='#accddb', padx=20, pady=20)
header_frame.pack(fill=tk.X)

# Add a title and slogan in the header frame
title = tk.Label(header_frame, text="Breast Cancer Prediction Tool", font=("Georgia", 24, "bold"), bg='#accddb', fg='black')
title.pack(pady=30)

slogan = tk.Label(header_frame, text="Early detection is the key to better health.", font=("Georgia", 14), fg="#003d67", bg='#accddb')
slogan.pack(pady=20)

# Create a frame for the content and center it
content_frame = tk.Frame(root, padx=20, pady=20, bg='#f0f0f0')
content_frame.pack(expand=True)

# Create and place the input fields in a 3-column grid
entries = {}
for i, feature in enumerate(features):
    row = (i // 3) + 2  # Start from row 2
    col = i % 3
    tk.Label(content_frame, text=feature, font=("Georgia", 12), bg='#f0f0f0').grid(row=row, column=col*2, sticky=tk.E, padx=15, pady=10)
    entry = tk.Entry(content_frame, font=("Georgia", 12))
    entry.grid(row=row, column=col*2 + 1, padx=10, pady=10)
    entries[feature] = entry

# Create and place the predict button
predict_button = tk.Button(content_frame, text="Predict", command=predict_cancer, font=("Georgia", 14), bg="green", fg="white")
predict_button.grid(row=12, columnspan=7, pady=20)

# Add a close button
close_button = tk.Button(content_frame, text="Close", command=root.destroy, font=("Georgia", 14), bg="red", fg="white")
close_button.grid(row=13, columnspan=7, pady=10)

# Run the main event loop
root.mainloop()

