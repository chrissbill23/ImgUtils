#!/usr/bin/env python
# coding: utf-8

# In[2]:


from imgutils import *


# # Load Dataset

# ## Paralellized

# ### RGB

# No resize

# In[2]:


get_ipython().run_line_magic('time', 'X, Y,enc = loadimgdataset("./Data/Training",parallel=True)')
del X
del Y


# Resize

# In[3]:


get_ipython().run_line_magic('time', 'X, Y,enc = loadimgdataset("./Data/Training",size=(64,64,3),parallel=True)')
X.shape
del X
del Y


# ### Gray

# No resize

# In[3]:


get_ipython().run_line_magic('time', 'X, Y,enc = loadimgdataset("./Data/Training",gray=True,parallel=True)')
del X
del Y


# Resize

# In[4]:


get_ipython().run_line_magic('time', 'X, Y,enc = loadimgdataset("./Data/Training",gray=True,size=(64,64),parallel=True)')
X.shape
del X
del Y


# ## Not Paralellized

# ### RGB

# No resize

# In[5]:


get_ipython().run_line_magic('time', 'X, Y,enc = loadimgdataset("./Data/Training")')
del X
del Y


# Resize

# In[6]:


get_ipython().run_line_magic('time', 'X, Y,enc = loadimgdataset("./Data/Training",size=(64,64,3))')
X.shape
del X
del Y


# ### Gray

# No resize

# In[7]:


get_ipython().run_line_magic('time', 'X, Y,enc = loadimgdataset("./Data/Training",gray=True)')
del X
del Y


# Resize

# In[8]:


get_ipython().run_line_magic('time', 'X, Y,enc = loadimgdataset("./Data/Training",gray=True,size=(64,64))')
X.shape
del X
del Y


# In[ ]:




