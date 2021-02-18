# Starbucks Capstone Challenge

![](https://raw.githubusercontent.com/gustavovenegas2010/Proyect_4/main/Imagenes/P1.jpg)
## 1. Introduction

This data set contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks. 

Not all users receive the same offer, and that is the challenge to solve with this data set.

Your task is to combine transaction, demographic and offer data to determine which demographic groups respond best to which offer type. This data set is a simplified version of the real Starbucks app because the underlying simulator only has one product whereas Starbucks actually sells dozens of products.

Every offer has a validity period before the offer expires. As an example, a BOGO offer might be valid for only 5 days. You'll see in the data set that informational offers have a validity period even though these ads are merely providing information about a product; for example, if an informational offer has 7 days of validity, you can assume the customer is feeling the influence of the offer for 7 days after receiving the advertisement.

You'll be given transactional data showing user purchases made on the app including the timestamp of purchase and the amount of money spent on a purchase. This transactional data also has a record for each offer that a user receives as well as a record for when a user actually views the offer. There are also records for when a user completes an offer. 

Keep in mind as well that someone using the app might make a purchase through the app without having received an offer or seen an offer.

### Example

To give an example, a user could receive a discount offer buy 10 dollars get 2 off on Monday. The offer is valid for 10 days from receipt. If the customer accumulates at least 10 dollars in purchases during the validity period, the customer completes the offer.

However, there are a few things to watch out for in this data set. Customers do not opt into the offers that they receive; in other words, a user can receive an offer, never actually view the offer, and still complete the offer. For example, a user might receive the "buy 10 dollars get 2 dollars off offer", but the user never opens the offer during the 10 day validity period. The customer spends 15 dollars during those ten days. There will be an offer completion record in the data set; however, the customer was not influenced by the offer because the customer never viewed the offer.


## 2. Library and Import database

import pandas as pd
import numpy as np
import math
import seaborn as sns
import json
import re
import os
import datetime
from time import time

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import ttest_ind
from scipy.stats import norm

import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['image.cmap'] = "bwr"
#plt.rcParams['figure.dpi'] = "100"
plt.rcParams['savefig.bbox'] = "tight"

# read in the json files

portfolio = pd.read_json('portfolio.json', orient='records', lines=True)

profile = pd.read_json('profile.json', orient='records', lines=True)

transcript = pd.read_json('transcript.json', orient='records', lines=True)

## 3. Database description

### 3.1 Portfolio Data 

Containing offer ids and meta data about each offer (duration, type, etc.):

**portfolio.json**
* id (string) - offer id
* offer_type (string) - type of offer ie BOGO, discount, informational
* difficulty (int) - minimum required spend to complete an offer
* reward (int) - reward given for completing an offer
* duration (int) - time for offer to be open, in days
* channels (list of strings)

The following reflects that there are 10 different types of id, with 3 types of offer: BOGO, discount, informational, which communicate with different channels such as web, email, mobile and social. 

![](https://raw.githubusercontent.com/gustavovenegas2010/Proyect_4/main/Imagenes/1.PNG)

![](https://raw.githubusercontent.com/gustavovenegas2010/Proyect_4/main/Imagenes/2.PNG)

**It was found that the discount offer type has the highest minimum required to spend to complete an offer, that the offer with the shortest duration is informational and that the highest reward amount is given to BOGO type offers.**

### 3.2 Profile Data

This database contains the demographic data of each client, with the following information:

**profile.json**
* age (int) - age of the customer 
* became_member_on (int) - date when customer created an app account
* gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
* id (str) - customer id
* income (float) - customer's income

**The following is an analysis of the variables in the database whats contains data from 17,000 customers:**

![](https://raw.githubusercontent.com/gustavovenegas2010/Proyect_4/main/Imagenes/3.PNG)

### Null value analysis

![](https://raw.githubusercontent.com/gustavovenegas2010/Proyect_4/main/Imagenes/4.PNG)

**It is evident that there are 2,175 clients with null values in the gender and income columns.**

### Gender Description

![](https://raw.githubusercontent.com/gustavovenegas2010/Proyect_4/main/Imagenes/5.PNG)

**It is evident that there is no null data, also that most of the clients are male with 57.3%, followed by 41.3% and 1.4% of other gender.** 

### Age Description

![](https://raw.githubusercontent.com/gustavovenegas2010/Proyect_4/main/Imagenes/6.PNG)

**As shown in the graph above, most of the customers are between 50 and 70 years old. It is also evident that there are outliers, which would later be targeted for cleaning.**

### Income Description

![](https://raw.githubusercontent.com/gustavovenegas2010/Proyect_4/main/Imagenes/7.PNG)

**It is evident that the income variable does not have atypical data, that the average income of the clients is 65,404, that 50% of the clients have incomes from 49,000 to 80,000 and that 25% of the clients have incomes over 120,000.**

### 3.3 Transcript Data

This database contains records of transactions, bids received, bids viewed and bids completed, the variables are as follows::

**transcript.json**
* event (str) - record description (ie transaction, offer received, offer viewed, etc.)
* person (str) - customer id
* time (int) - time in hours since start of test. The data begins at time t=0
* value - (dict of strings) - either an offer id or transaction amount depending on the record

**The following database contains 306,534 records.**

![](https://raw.githubusercontent.com/gustavovenegas2010/Proyect_4/main/Imagenes/8.PNG)

### Null value analysis

![](https://raw.githubusercontent.com/gustavovenegas2010/Proyect_4/main/Imagenes/9.PNG)

**It is evident that this base does not contain null values.**

### Gender Description

![](https://raw.githubusercontent.com/gustavovenegas2010/Proyect_4/main/Imagenes/10.PNG)

**Shows that 45.3% are transactions, followed by 24.8% offer received, after offer viewed 18.8% and offer completed 10.9%.**

## 4. Data Cleaning and Preparing

In this section we clean and prepare the data for each database supplied, let's start with the first dataset, 


### 4.1 Cleaning  Portfolio Dataset

**In the Portfolio dataset, the communication channel variable is pivoted, as well as the type of offer and the latter is united in a single database.**

![](https://raw.githubusercontent.com/gustavovenegas2010/Proyect_4/main/Imagenes/11.PNG)

### 4.2 Cleaning Profile Data

**In the Profile dataset, the atypical data of the variable age, i.e. ages equal to 118, are eliminated and the variable memberdays is transformed into date format.**

![](https://raw.githubusercontent.com/gustavovenegas2010/Proyect_4/main/Imagenes/12.PNG)

### 4.3 Cleaning Transcript Data

In the Transcript dataset, the rows related to the offer action are extracted, their id and the variables are renamed, in order to subsequently perform the database joins.

![](https://raw.githubusercontent.com/gustavovenegas2010/Proyect_4/main/Imagenes/13.PNG)

**After the cleaning process, it is evident how the atypical data of the variable age is already corrected.**

![](https://raw.githubusercontent.com/gustavovenegas2010/Proyect_4/main/Imagenes/14.PNG)

## 5. Final basis

Next, the three databases are merged together

![](https://raw.githubusercontent.com/gustavovenegas2010/Proyect_4/main/Imagenes/15.PNG)

## Information analysis

**In order to analyze the best offers by demographic group, we will proceed to perform a cross analysis of two demographic variables vs. the type of offer.**

### Type of offer by _Record description_.

![](https://raw.githubusercontent.com/gustavovenegas2010/Proyect_4/main/Imagenes/16.PNG)

**The above graph shows that the most important offers are Bogo and Discount, and that the most frequently completed offer is Discount.**

![](https://raw.githubusercontent.com/gustavovenegas2010/Proyect_4/main/Imagenes/17.PNG)

**The most received offers are Discount and Bogo, the most viewed and completed is Bogo, followed by Discount. From this we conclude that the best offer is Bogo.**

### Type of offer by _Gender_.

![](https://raw.githubusercontent.com/gustavovenegas2010/Proyect_4/main/Imagenes/18.PNG)

![](https://raw.githubusercontent.com/gustavovenegas2010/Proyect_4/main/Imagenes/19.PNG)

**After reviewing the proportion of gender by type of supply, it is evident that there is no difference by gender, that is to say that gender does not generate a tendency towards one type of supply.**

**It is also observed that in the completed offers there is a slight preference for women.**

### Type of offer by _Income_.

![](https://raw.githubusercontent.com/gustavovenegas2010/Proyect_4/main/Imagenes/20.PNG)

**After reviewing the income of clients by type of offer, it is observed that there are no differences, i.e., that clients earning more or less do not generate a tendency towards one type of offer.**

**It is also observed that in the completed offers there is a slight preference for customers with higher incomes.**

## 6.  Data Modeling

**Next, we create the pipelines for the data transformation, with the objective of building a solid database to create a logistic regression model.**

![](https://raw.githubusercontent.com/gustavovenegas2010/Proyect_4/main/Imagenes/21.PNG)

**We select only the variables of interest to be studied.**

![](https://raw.githubusercontent.com/gustavovenegas2010/Proyect_4/main/Imagenes/22.PNG)

**Finally, the null data was deleted, leaving a total of 148,805 records.**

![](https://raw.githubusercontent.com/gustavovenegas2010/Proyect_4/main/Imagenes/23.PNG)

### Logistic regression model

With the consolidated database, we partition the training base and the test base, 70% and 30% respectively.

![](https://raw.githubusercontent.com/gustavovenegas2010/Proyect_4/main/Imagenes/24.PNG)

![](https://raw.githubusercontent.com/gustavovenegas2010/Proyect_4/main/Imagenes/25.PNG)

**78% of the cases were well classified by the model.**

## 7. Conclusion

With 90% confidence in the study, it is evidenced that the variables that are significantly statistical are: age, male gender, income, web, modile and social.

From here it is evident that:

`*` Older customers are more likely to complete the offer.

`*` Men are less likely to complete the offer.

`*` The more income customers have, the more likely they are to complete the offer.

`*` The higher the minimum required ("difficulty") for spending, the probability of completing the offer decreases.

`*` The Web, Email, Mobile and Social communication channels are not effective so that the probability of completing the offer increases 


-------------------------------

**Given that the accuracy criterion is not greater than 80%, it is recommended for future studies to implement different modeling methodologies in order to improve their predictions and the behavior of customers to be identified with greater accuracy. Which methodology is better for modeling this data?** 

![](https://raw.githubusercontent.com/gustavovenegas2010/Proyect_4/main/Imagenes/PF.jpeg)









