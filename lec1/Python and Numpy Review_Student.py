
# coding: utf-8

# # Jupyter Notebook How To: 

# Below is a code cell. Code cells allow you to enter and run code
# Run a code cell using Shift-Enter or pressing the  button in the toolbar above:

# In[1]:


message = "Hello Python world!"
print(message)


# There are two other keyboard shortcuts for running code:
# 
# Alt-Enter runs the current cell and inserts a new one below.
# Ctrl-Enter run the current cell and enters command mode.

# Its helpful to get to know the different cell running options in the Cell menu
# The "Cell" menu has a number of menu items for running code in different ways. Go ahead and take a look! 

# ### Restarting the kernels
# The kernel maintains the state of a notebook's computations. You can reset this state by restarting the kernel. This is done by clicking on the  in the toolbar above. Once you restart your kernel you will need to rerun all the cells you previously had run.

# The best way to best understand Jupyter Notebooks is to play around with one! Keep going through this tutorial and feel free to raise your hand and ask any questions if something comes up or post on piazza! 

# # Helpful Basic Python How To: 
# 
# Feel free to skip on to the Numpy section if you have 61A+ eqivalent python programming knowledge

# ### Variables
# We can use variables to hold values. These variables can hold different types of values. You can use type() to see what the variable's type. 

# In[2]:


x = 1.0
type(x)


# In[3]:


x = 1
type(x)


# In[4]:


x = "Machine Learning Decal is hella dope"
type(x)


# In[7]:


w = 3 
y = 8
z = w + y 
print(z)


# Try assigning a variable to the letter 'c' that's the sum of two variable '

# In[8]:


## Your answer here ## 
a = 1
b = 2.0
c = a+b


# ### Dictionaries 
# Think of a dictionary as an unordered set of key: value pairs, with the requirement that the keys are unique (within one dictionary).
# You can use keys to retrieve the values associated with that key. 

# In[9]:


params = {"parameter1" : 1.0,
          "parameter2" : 2.0,
          "parameter3" : 3.0,}

print(type(params))
print(params)
print(params["parameter1"])
print(params["parameter2"])
print(params["parameter3"])


# ### Lists
# Lists are the most commonly used data structure. Think of it as a sequence of data that is enclosed in square brackets and data are separated by a comma. Each of these data can be accessed by calling it's index value.
# 
# Lists are declared by just equating a variable to '[ ]' or list. 

# In[10]:


a = [1,2,3]
print(type(a))


# You can diretly assign values to a list like below: 

# In[11]:


x = ['machine learning', 'data science', 'berkeley']


# ### Indexing
# In python, Indexing starts from 0. Thus now the list x, which has two elements will have apple at 0 index and orange at 1 index.
# 

# In[12]:


x[0] 


# You can also directly change the value of an index: 

# In[ ]:


x[0] = "ai buzzword!"


# In[14]:


x


# You can also use negative indices to go backwards in a list

# In[15]:


x[-1]


# You can even have lists inside of lists, this is the same ideas of how we'd store a matrix. 

# In[17]:


y = [x]


# In[19]:


y


# Now try and access the original list x from y! 

# In[20]:


## FILL THIS IN! ##
y[0]


# Now try and access the the word 'berkeley' from y! 

# In[21]:


## FILL THIS IN ##
y[0][2]


# ### Slicing
# Indexing was only limited to accessing a single element, Slicing on the other hand is accessing a sequence of data inside the list. In other words "slicing" the list.
# 
# Slicing is done by defining the index values of the first element and the last element from the parent list that is required in the sliced list. It is written as parentlist[ a : b ] where a,b are the index values from the parent list. If a or b is not defined then the index value is considered to be the first value for a if a is not defined and the last value for b when b is not defined.

# In[22]:


num = [0,1,2,3,4,5,6,7,8,9]
print(num[0:4])
print(num[4:])


# In[25]:


coolio_list = [2,9,3,8,4,7,5,6]


# Try and use slicing to get the list [2,9,3,8,4,7,5] or just play around with slicing and see what kind of results you get! 

# In[27]:


## FILL THIS IN### 
coolio_list[2:4]


# ### Built in List Functions
# 

# In[28]:


#Find the length
print(len(num))

#Find the min value
print(min(num))

#Find the max value
print(max(num))

#Concetate two lists 
x = [1,2,3] + [5,4,7]
print(x)

#Check if an element is in a list
1 in x


# ### For Loops 
# For loops lets you iterate through data. 

# In[30]:


for x in range(4):
    print(x)


# In[32]:


a = "aren't for loops fun!" 
#iterate through the sentence letter by letter
for x in a:
    print(x)


# In[33]:


#iterate through the key value pairs from the dictionary we defined earlier
for key, value in params.items():
    print(key + " = " + str(value))


# Iterate through the the following sentence and print out every characters 

# In[35]:


sentence = "I am so excited for the Data Science Decal!!!!"
## FILL THIS IN ###
for i in sentence:
    print(i)


# ### Define a Function
# We can create helpful functions where we can save code to be reused. Let's make some helpful functions!

# In[36]:


def square(x):
    return x*x 

# You can return multiple  values
def powers(x):
    return x ** 2, x ** 3, x ** 4

#You don't have to have a return value
def split_up_string(x):
    for s in x:
        print(s)


# Your turn! Try and define your own function below: 

# In[38]:


#Define your function here


# # Numpy
# 
# What is Numpy and why do we use it? 
# 
# It's an awesome python package that adds support for large, multi-dimensional arrays. Really good for vector operations, matrix operations because its super parallelized so its super fast! 
# 

# Why not Python arrays? 
# 
# Python arrays has certain limitations: they don’t support “vectorized” operations like elementwise addition and multiplication, and the fact that they can contain objects of differing types mean that Python must store type information for every element, and must execute type dispatching code when operating on each element. This also means that very few list operations can be carried out by efficient C loops – each iteration would require type checks and other Python API bookkeeping.
# 
# 

# ### Importing numpy
# Functions for numerical computiing are provided by a separate module called numpy which we must import.
# 
# By convention, we import numpy using the alias np.
# 
# Once we have done this we can prefix the functions in the numpy library using the prefix np.

# In[41]:


# This is the de facto way to import NumPy. You probably don't want to write numpy.whatever every time
import numpy as np


# ### Numpy Arrays
# NumPy arrays are the workhorse of the library. A NumPy array is essentially a bunch of data coupled with some metadata:
# 
# type: the type of objects in the array. This will typically be floating-point numbers for our purposes, but other types can be stored. The type of an array can be accessed via the dtype attribute.
# 
# shape: the dimensions of the array. This is given as a tuple, where element $i$ of the tuple tells you how the "length" of the array in the $i$th dimension. For example, a 10-dimensional vector would have shape (10,), a 32-by-100 matrix would have shape (32,100), etc. The shape of an array can be accessed via the shape attribute.
# 
# Let's see some examples! There are number of ways to construct arrays. One is to pass in a Python sequence (such as list or tuple) to the np.array function:

# In[43]:


np.array([1, 2.3, -6])


# We can also easily create ordered numerical lists!

# In[45]:


# Remember we zero index so you will actually get 0 to 6!
print(np.arange(7))
# Rmember the list wont include 9
print(np.arange(3, 9))


# We can also customize these list with a third paramter that specifices step size

# In[46]:


np.arange(0.0, 100.0, 10.0)


# To create a multi-dimensional array, we'll need to nest the sequences:

# In[47]:


np.array([[1, 2.3, -6], [7, 8, 9]])


# Neat! 
# 
# There are also many convenience functions for constructing special arrays. Here are some that might be useful: 

# In[49]:


# The identity matrix of given size
np.eye(7)


# In[51]:


# A matrix with the given vector on the diagonal
np.diag([1.1,2.2,3.3])


# In[52]:


#An array of all zeros or ones with the given shape
np.zeros((8,4)), np.ones(3)


# In[ ]:


# An array with a given shape full of a specified value
np.full((3,4), 2.1)


# In[ ]:


# A random (standard normal) array with the given shape
np.random.randn(5,6)


# Okay your turn! In the cell belows try and create:
# 
# 
# -A diagonal matrix with values from 1-20 (try and create this and only type one number!)

# In[ ]:


#Your answer here


# Okay now let's suppose we have some data in an array so we can start doing stuff with it.
# 

# In[53]:


A = np.random.randn(10,5); x = np.random.randn(5)
A


# One useful thing that NumPy lets us do efficiently is apply the same function to every element in an array. You'll often need to e.g. exponentiate a bunch of values, but if you use a list comprehension or map with the builtin Python math functions it may be really slow. Instead just write

# In[55]:


# log, sin, cos, etc. work similarly
np.exp(A)


# We can take the sum/mean/standard deviation/etc. of all the elements in an array:

# In[56]:


np.sum(x), np.mean(x), np.std(x)


# You can also specify an axis over which to compute the sum if you want a vector of row/column sums (again, sum here can be replaced with mean or other operations):

# In[60]:


# Create an array with numbers in the range 0,...,3 (similar to the normal Python range function,
# but it returns a NumPy array) and then reshape it to a 2x2 matrix
B = np.arange(4).reshape((2,2))

# Original matrix, column sum, row sum
B, np.sum(B, axis=0), np.sum(B, axis=1)


# ### Linear Algebra
# By now we have a pretty good idea of how data is stored and accessed within NumPy arrays. But we typically want to do something more "interesting", which for our purposes usually means linear algebra operations. Fortunately NumPy has good support for such routines. Let's see some examples!

# In[61]:


# Matrix-vector product. The dimensions have to match, of course
A.dot(x)
# Note that in Python3 there is also a slick notation A @ x which does the same thing


# In[62]:


# Transpose a matrix
A.T


# Now that you're familiar with numpy feel free to check out the documentation and see what else you can do! Documentation can be found here: https://docs.scipy.org/doc/

# ## Excercises 
# Lets try out all the new numpy stuff we just learned! Even if you have experience in numpy we suggest trying these out. 

# 1) Create a vector of size 10 containing zeros 

# In[66]:


## FILL IN YOUR ANSWER HERE ##
my_vec=np.zeros(10)


# 2) Now change the fith value to be 5 

# In[68]:


## FILL IN YOUR ANSWER HERE ##
my_vec[4]=5
my_vec


# 3) Create a vector with values ranging from 10 to 49

# In[70]:


## FILL IN YOUR ANSWER HERE ##
my_vec_2 = np.arange(10, 50)


# 4) Reverse the previous vector (first element becomes last)

# In[71]:


## FILL IN YOUR ANSWER HERE ##
my_vec_2=reversed(my_vec_2)
my_vec_2=my_vec_2[::-1]
np.flip(my_vec_2)


# 5) Create a 3x3 matrix with values ranging from 0 to 8. Create a 1D array first and then resshape it. 

# In[72]:


## FILL IN YOUR ANSWER HERE ##
my_vec_3=np.arange(0, 9).reshape(3, 3)
my_vec_3


# 6) Create a 3x3x3 array with random values

# In[76]:


## FILL IN YOUR ANSWER HERE ##
np.random.random((3, 3, 3))


# 7) Create a random array and find the sum, mean, and standard deviation

# In[73]:


## FILL IN YOUR ANSWER HERE ##
summ=np.sum(my_vec_3)
mean=np.mean(my_vec_3)
std=np.std(my_vec_3)

