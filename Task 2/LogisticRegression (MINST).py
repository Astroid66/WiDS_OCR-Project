#!/usr/bin/env python
# coding: utf-8

# In[8]:


from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, cache=True)


# In[10]:


print(mnist.data.shape)
print(mnist.target.shape)


# In[11]:


from sklearn.model_selection import train_test_split

train_img, test_img, train_lbl, test_lbl = train_test_split(
mnist.data, mnist.target, test_size=1/7.0, random_state=0)


# In[21]:


import numpy as np
import matplotlib.pyplot as plt

# plt.figure(figsize=(20,4))
# for index, (image, label) in enumerate(zip(train_img[0:5],train_lbl[0:5])):
#     plt.subplot(1,5,index+1)
#     plt.imshow(np.reshape(image,(28,28)),cmap=plt.cm.gray)
#     plt.title('Training: %i\n' % label, fontsize = 20)


# In[23]:


from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression(solver = 'lbfgs')


# In[26]:


logisticRegr.fit(train_img, train_lbl)


# In[28]:


predictions = logisticRegr.predict(test_img)


# In[29]:


score = logisticRegr.score(test_img, test_lbl)
print(score)


# In[30]:


import numpy as np
import matplotlib.pyplot as plt

index=0
misclassifiedIndexes = []
for label, predict in zip(test_lbl, predictions):
    if label != predict:
        misclassifiedIndexes.append(index)
        index+=1


# In[32]:


plt.figure(figsize=(20,4))

for plotIndex, badIndex in enumerate(misclassifiedIndexes[0:5]):
 plt.subplot(1, 5, plotIndex + 1)
 plt.imshow(np.reshape(test_img[badIndex], (28,28)), cmap=plt.cm.gray)
 plt.title('Predicted: {}, Actual: {}'.format(predictions[badIndex], test_lbl[badIndex]), fontsize = 15)


# In[ ]:




