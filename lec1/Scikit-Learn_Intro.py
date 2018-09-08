
# coding: utf-8

# # Intro to Scikit-Learn 

# What is Scikit-Learn:
# 
# A free library that provides tools that makes it super easy to perform data analysis. Great for getting started with learning the basics of machine learning models!
# 

# To DO:
#     *Data Representation
#     *Load Data
#     *Choose a model
#     *Intantiate Model
#     *Pick Hyperparamters
#     *Graph 

# Lets start off by loading one of the most famous scikit learn datasets and see what it looks like. Go ahead and take a moment to look up the Iris Dataset with scikit learn.
# 

# *waits a few minute* 

# ## Loading Data from a Library

# In[1]:


#insert dataset name after import statement
from sklearn import datasets


# Great now that we've imported the data we can load it. 

# In[2]:


iris = datasets.load_iris()


# Try printing out iris and examine what information we have in the data set and explore how it's stored

# In[3]:


print(iris)


# In[4]:


# print the iris data
print(iris.data)


# #### Some quick terminology
# The data above just loooks like just a bunch of numbers right? Let's try to better understand what it means. 
# 
#     *Each row is a sample (also known as: observation, example, instance, record)
#     *Each column is a feature (also known as: predictor, attribute, independent variable, input, regressor, covariate)

# Let's try printing out the features and samples.

# In[5]:


# print the names of the four features
print(iris.feature_names)
# print integers representing the species of each observation
print(iris.target)
# print the encoding scheme for species: 0 = setosa, 1 = versicolor, 2 = virginica
print(iris.target_names)


# So now we see that each column represents a different feature, in this case sepal lenght, sepal width, petal length and petal width! We also know each row represents one example. 
# 
# Run the next cell to see a snippet of the code in a table that may look more familiar

# In[7]:


from IPython.display import Image
Image(filename='iris.png')


# Remember all that stuff we learned with numpy? Comes in handy again here since scikit-learn is built on top of numpy, so we can use all the stuff we learned on our data here! Let's try playing around with the data. 

# In[8]:


#print out the type of data 
print(type(iris))
#print out the type of the feature_names
print(type(iris.feature_names))
#print out the type of the targets 
print(type(iris.target_names))


# In[3]:


#print out the shape of the data
print(iris.data.shape)

#print out the shape of the target_name
print(iris.target_names.shape)


# Notice if we try and print out the shape of feature_names we get an error! The type is a list, therefore cannot call shape on the data

# Usually when we have data we'll save it to a variable (X) and the labeled response to (Y). In this case what would we assign as X? What would we assign as Y? 

# In[5]:


#Fil in here how we'd load the data 
X = iris.data
Y = iris.target
X, Y


# ## Scikit-Learn API Basics 

# Great! Now that we know how to load data, let's dive a little deeper and look at some of the basics of the API. 
#     1. Choose a class of model by importing the appropriate estimator class from Scikit-Learn.
#     2. Choose model hyperparameters by instantiating this class with desired values.
#     3. Arrange data into a features matrix and target vector following the discussion above.
#     4. Fit the model to your data by calling the fit() method of the model instance
#     5. Apply the Model to new data, can use predict() or transform() 
# 
# Lets go ahead and look at a simple example, loading a logistic regression classifier with some random made up data.

# In[6]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

rng = np.random.RandomState(42)
x = 10 * rng.rand(50)
y = 2 * x - 1 + rng.randn(50)
plt.scatter(x, y);
plt.show()


# Cool now we have some data to work with. Now lets go ahead and choose a model, remember how we imported a dataset before? With the same intuition let's import a model

# In[7]:


#Try looking at this documentation and figuring out how to import the Logistic Regression Model: 
# http://scikit-learn.org/stable/supervised_learning.html
from sklearn.linear_model import LinearRegression


# Once we have decided on our model class, there are still some options open to us. Depending on the model class we are working with, we might need to answer one or more questions like the following:
#     
#     *Would we like to fit for the offset (i.e., y-intercept)?
#     *Would we like the model to be normalized?
#     *Would we like to preprocess our features to add model flexibility?
#     *What degree of regularization would we like to use in our model?
#     *How many model components would we like to use?
# 
# These are examples of the important choices that must be made once the model class is selected. These choices are often represented as hyperparameters, or parameters that must be set before the model is fit to data. 
# 
# Throughout the course of this decal you're going to learn more about each of these questions and how to answer them! For now don't worry too much we're just showing you an example!

# In[8]:


model = LinearRegression(fit_intercept=True)
model 


# Here our target variable y is already in the correct form (a length-n_samples array). Think about how iris.target looked? Same idea! 

# In[14]:


print(y)


# Although think back to the Iris dataset, every example was its own row. Currently our X data is just one long row. We'll need to fix this and make the data x a matrix of [n_samples, n_features]. 
# 
# In this case we have 50 samples, and 1 feature so we'll need a matrix of shape (50, 1)! Can you figure out how to reshape the data? 

# In[9]:


print(x)
# X = x[:, np.newaxis]
X = x[:, None]
print(X)
print(X.shape)


# Now lets apply the model to our data, we can do this by calling model.fit() 

# In[10]:


model.fit(X, y)


# That's it, and yep it's that easy! Now let's take a look at what the model actually fit to. 

# In[11]:


model.coef_


# In[12]:


model.intercept_


# These two parameters represent the slope and intercept of the simple linear fit to the data. Comparing to the data definition, we see that they are very close to the input slope of 2 and intercept of -1.
# 
# Now lets plot these results and see what it looks like!

# In[13]:


xfit = np.linspace(-1, 11)
Xfit = xfit[:, np.newaxis]
yfit = model.predict(Xfit)

plt.scatter(x, y)
plt.plot(xfit, yfit);
plt.show()


# Cool! Now let's take the basics we just learned and see if we can classify digits!

# # Classifying MNIST Digits

# The MNIST digit data set is a very famous and widely used dataset! Lucky for us Scikit learn has already preformatted the data, so it's super easy to use! Let's go ahead and load the data. 

# In[14]:


from sklearn.datasets import load_digits
digits = load_digits()
digits.images.shape


# In[15]:


X = digits.data
X.shape
y = digits.target
y.shape


# Now lets actually visualize what this data looks like, run the cell below! 

# In[22]:


import matplotlib.pyplot as plt

fig, axes = plt.subplots(10, 10, figsize=(8, 8),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
    ax.text(0.05, 0.05, str(digits.target[i]),
            transform=ax.transAxes, color='green')
    
plt.show()


# Cool,  Now that we have some data! Unlike before though, this data cannot be directly plotted on a 2D graph. It's pretty hard to visualize this data on a 64 dimensional paramter space so lets use a technique called Manifold Learning. Don't worry too much about this, the gist is that we can we can reduce the data to a 2D space to better visuaize it. 

# In[16]:


from sklearn.manifold import Isomap
iso = Isomap(n_components=2)
iso.fit(digits.data)
data_projected = iso.transform(digits.data)
data_projected.shape


# Hey! Now our data is 2D, coolio let's plot this bad boy. 

# In[17]:


plt.scatter(data_projected[:, 0], data_projected[:, 1], c=digits.target,
            edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('spectral', 10))
plt.colorbar(label='digit label', ticks=range(10))
plt.clim(-0.5, 9.5);
plt.show()


# Wowza what a world of color. Now what does this mean? Lets compare two digs, zeros and ones!
# 
# Zeros (in black) and ones (in purple) have very little overlap in parameter space. Intuitively, this makes sense: a zero is empty in the middle of the image, while a one will generally have ink in the middle. 
# 
# Now thik about 5 (bright green) and 6 (lime colored green). Their data points seem to be really close! Why is this? Think about a 5 and a 6, they definitely share some overlap. 

# Cool! Now that we've better visualized our data lets actually clasify these digits. 
# 
# We're going to split up our data. We'll use some of it to train a model and some of it to test the accurarcy of the model. Again don't worry too much about these concepts they'll be covered way more in depth throughout the course of this decal! 

# In[18]:


from sklearn.model_selection import train_test_split

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=1)


# Now let's choose a model. For this we're going ot choose a Gaussian naive Bayes model with model.fit()

# In[19]:


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()

model.fit(Xtrain, ytrain)


# Now let's use the model.predict method to see what values our model predict for the data. We'll use the X test data we set aside a few cells before. 

# In[20]:


y_model = model.predict(Xtest)


# Great, now lets use y_model and compare it to the test data we set aside and see how our model performed!

# In[21]:


from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_model)


# Not too shabby of results for just a few lines of code! One fun thing we can use to visualize results are a confusion matrix. This will tell us which digits were misclassified the most in a nice visually pleasing chart. 

# In[22]:


from sklearn.metrics import confusion_matrix

mat = confusion_matrix(ytest, y_model)

sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value');

plt.show()


# Cool! Now we can see where our model messes up. We can see for example, ones and twos are sometimes predicted as eights! 

# ###  Congrats! You made it!
# It's okay if you don't understand everything that happend in this lab! What we wanted was for everyone to get familiar with using a library and the process of of loading data, visualizing that data, picking a model and actually applying the model! 
