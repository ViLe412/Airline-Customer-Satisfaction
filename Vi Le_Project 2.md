# **Project 2 - Airline Customer Satisfaction**

## Vi Le | DAT 402: Machine Learning- Data Science, Fall 2024

---

Customer satisfaction is a cornerstone of success for businesses in industries such as retail, telecommunications, and airlines. Satisfied customers are more likely to become loyal advocates, reflecting a brand’s quality and trustworthiness. 

With over four years of experience in retail and customer service, I have seen firsthand how critical it is to meet and exceed customer expectations. Inspired by this, I chose to analyze and predict customer satisfaction levels in the airline industry for this project. By leveraging data insights, my goal is to better understand the factors that drive satisfaction and loyalty, as well as to build a model that could classify happy and unhappy customers based on different factors.

Credits:
- The following data set is from TJ Klein (link: https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)
- The data set provided is modified from John D. (link: https://www.kaggle.com/datasets/johndddddd/customer-satisfaction)

**In this project, I will use 3 different classification methods on the same data set, and compare their peformances:**
- **Naive Bayes**
- **Logistic Regression**
- **Decision Tree**

First thing first, I will import all necessary libraries and packages.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.naive_bayes import MultinomialNB, CategoricalNB, GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
```

Then, I will load the data set, and see a first few rows of it.


```python
df = pd.read_csv("airline.csv")
df = pd.DataFrame(df)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>id</th>
      <th>Gender</th>
      <th>Customer Type</th>
      <th>Age</th>
      <th>Type of Travel</th>
      <th>Class</th>
      <th>Flight Distance</th>
      <th>Inflight wifi service</th>
      <th>Departure/Arrival time convenient</th>
      <th>...</th>
      <th>Inflight entertainment</th>
      <th>On-board service</th>
      <th>Leg room service</th>
      <th>Baggage handling</th>
      <th>Checkin service</th>
      <th>Inflight service</th>
      <th>Cleanliness</th>
      <th>Departure Delay in Minutes</th>
      <th>Arrival Delay in Minutes</th>
      <th>satisfaction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>70172</td>
      <td>Male</td>
      <td>Loyal Customer</td>
      <td>13</td>
      <td>Personal Travel</td>
      <td>Eco Plus</td>
      <td>460</td>
      <td>3</td>
      <td>4</td>
      <td>...</td>
      <td>5</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>25</td>
      <td>18.0</td>
      <td>neutral or dissatisfied</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>5047</td>
      <td>Male</td>
      <td>disloyal Customer</td>
      <td>25</td>
      <td>Business travel</td>
      <td>Business</td>
      <td>235</td>
      <td>3</td>
      <td>2</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>6.0</td>
      <td>neutral or dissatisfied</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>110028</td>
      <td>Female</td>
      <td>Loyal Customer</td>
      <td>26</td>
      <td>Business travel</td>
      <td>Business</td>
      <td>1142</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>5</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>0.0</td>
      <td>satisfied</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>24026</td>
      <td>Female</td>
      <td>Loyal Customer</td>
      <td>25</td>
      <td>Business travel</td>
      <td>Business</td>
      <td>562</td>
      <td>2</td>
      <td>5</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>11</td>
      <td>9.0</td>
      <td>neutral or dissatisfied</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>119299</td>
      <td>Male</td>
      <td>Loyal Customer</td>
      <td>61</td>
      <td>Business travel</td>
      <td>Business</td>
      <td>214</td>
      <td>3</td>
      <td>3</td>
      <td>...</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0.0</td>
      <td>satisfied</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>




```python
print(df.dtypes)
```

    Unnamed: 0                             int64
    id                                     int64
    Gender                                object
    Customer Type                         object
    Age                                    int64
    Type of Travel                        object
    Class                                 object
    Flight Distance                        int64
    Inflight wifi service                  int64
    Departure/Arrival time convenient      int64
    Ease of Online booking                 int64
    Gate location                          int64
    Food and drink                         int64
    Online boarding                        int64
    Seat comfort                           int64
    Inflight entertainment                 int64
    On-board service                       int64
    Leg room service                       int64
    Baggage handling                       int64
    Checkin service                        int64
    Inflight service                       int64
    Cleanliness                            int64
    Departure Delay in Minutes             int64
    Arrival Delay in Minutes             float64
    satisfaction                          object
    dtype: object


Explanation of column names:
- Gender: Gender of the passengers (Female, Male)
- Customer Type: The customer type (Loyal customer, disloyal customer) 
- Age: The actual age of the passengers
- Type of Travel: Purpose of the flight of the passengers (Personal Travel, Business Travel)
- Class: Travel class in the plane of the passengers (Business, Eco, Eco Plus)
- Flight distance: The flight distance of this journey
- Inflight wifi service: Satisfaction level of the inflight wifi service (0:Not Applicable;1-5)
- Departure/Arrival time convenient: Satisfaction level of Departure/Arrival time convenient
- Ease of Online booking: Satisfaction level of online booking
- Gate location: Satisfaction level of Gate location
- Food and drink: Satisfaction level of Food and drink
- Online boarding: Satisfaction level of online boarding
- Seat comfort: Satisfaction level of Seat comfort
- Inflight entertainment: Satisfaction level of inflight entertainment
- On-board service: Satisfaction level of On-board service
- Leg room service: Satisfaction level of Leg room service
- Baggage handling: Satisfaction level of baggage handling
- Check-in service: Satisfaction level of Check-in service
- Inflight service: Satisfaction level of inflight service
- Cleanliness: Satisfaction level of Cleanliness
- Departure Delay in Minutes: Minutes delayed when departure
- Arrival Delay in Minutes: Minutes delayed when Arrival
- Satisfaction: Airline satisfaction level(Satisfaction, neutral or dissatisfaction)

----

### **Step 2: Data Preprocessing**


```python
# Check for NA values
print(df.isna().sum())
```

    Unnamed: 0                             0
    id                                     0
    Gender                                 0
    Customer Type                          0
    Age                                    0
    Type of Travel                         0
    Class                                  0
    Flight Distance                        0
    Inflight wifi service                  0
    Departure/Arrival time convenient      0
    Ease of Online booking                 0
    Gate location                          0
    Food and drink                         0
    Online boarding                        0
    Seat comfort                           0
    Inflight entertainment                 0
    On-board service                       0
    Leg room service                       0
    Baggage handling                       0
    Checkin service                        0
    Inflight service                       0
    Cleanliness                            0
    Departure Delay in Minutes             0
    Arrival Delay in Minutes             310
    satisfaction                           0
    dtype: int64



```python
df = df.dropna()
```


```python
# Check statistical aspect of data
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>id</th>
      <th>Age</th>
      <th>Flight Distance</th>
      <th>Inflight wifi service</th>
      <th>Departure/Arrival time convenient</th>
      <th>Ease of Online booking</th>
      <th>Gate location</th>
      <th>Food and drink</th>
      <th>Online boarding</th>
      <th>Seat comfort</th>
      <th>Inflight entertainment</th>
      <th>On-board service</th>
      <th>Leg room service</th>
      <th>Baggage handling</th>
      <th>Checkin service</th>
      <th>Inflight service</th>
      <th>Cleanliness</th>
      <th>Departure Delay in Minutes</th>
      <th>Arrival Delay in Minutes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>103594.000000</td>
      <td>103594.000000</td>
      <td>103594.000000</td>
      <td>103594.000000</td>
      <td>103594.000000</td>
      <td>103594.000000</td>
      <td>103594.000000</td>
      <td>103594.000000</td>
      <td>103594.000000</td>
      <td>103594.000000</td>
      <td>103594.000000</td>
      <td>103594.000000</td>
      <td>103594.000000</td>
      <td>103594.000000</td>
      <td>103594.000000</td>
      <td>103594.000000</td>
      <td>103594.000000</td>
      <td>103594.000000</td>
      <td>103594.000000</td>
      <td>103594.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>51950.102274</td>
      <td>64942.428625</td>
      <td>39.380466</td>
      <td>1189.325202</td>
      <td>2.729753</td>
      <td>3.060081</td>
      <td>2.756984</td>
      <td>2.977026</td>
      <td>3.202126</td>
      <td>3.250497</td>
      <td>3.439765</td>
      <td>3.358341</td>
      <td>3.382609</td>
      <td>3.351401</td>
      <td>3.631687</td>
      <td>3.304323</td>
      <td>3.640761</td>
      <td>3.286397</td>
      <td>14.747939</td>
      <td>15.178678</td>
    </tr>
    <tr>
      <th>std</th>
      <td>29997.914016</td>
      <td>37460.816597</td>
      <td>15.113125</td>
      <td>997.297235</td>
      <td>1.327866</td>
      <td>1.525233</td>
      <td>1.398934</td>
      <td>1.277723</td>
      <td>1.329401</td>
      <td>1.349433</td>
      <td>1.318896</td>
      <td>1.333030</td>
      <td>1.288284</td>
      <td>1.315409</td>
      <td>1.181051</td>
      <td>1.265396</td>
      <td>1.175603</td>
      <td>1.312194</td>
      <td>38.116737</td>
      <td>38.698682</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>7.000000</td>
      <td>31.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>25960.250000</td>
      <td>32562.250000</td>
      <td>27.000000</td>
      <td>414.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>51955.500000</td>
      <td>64890.000000</td>
      <td>40.000000</td>
      <td>842.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>77924.750000</td>
      <td>97370.500000</td>
      <td>51.000000</td>
      <td>1743.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>4.000000</td>
      <td>12.000000</td>
      <td>13.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>103903.000000</td>
      <td>129880.000000</td>
      <td>85.000000</td>
      <td>4983.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>1592.000000</td>
      <td>1584.000000</td>
    </tr>
  </tbody>
</table>
</div>



 

"Customer Type", "Type of Travle", "Class", "Gender", and "satisfaction" are categorical types and contain text values. For best use of data for machine learning models, I will turn those values into numerical type using **OrdinalEnconder()**.

 


```python
from sklearn.preprocessing import OrdinalEncoder
ordinal = OrdinalEncoder()
df['Customer Type'] = ordinal.fit_transform(df[['Customer Type']])
df['Type of Travel'] = ordinal.fit_transform(df[['Type of Travel']])
df['Class'] = ordinal.fit_transform(df[['Class']])
df['satisfaction'] = ordinal.fit_transform(df[['satisfaction']])
df['Gender'] = ordinal.fit_transform(df[['Gender']])
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>id</th>
      <th>Gender</th>
      <th>Customer Type</th>
      <th>Age</th>
      <th>Type of Travel</th>
      <th>Class</th>
      <th>Flight Distance</th>
      <th>Inflight wifi service</th>
      <th>Departure/Arrival time convenient</th>
      <th>...</th>
      <th>Inflight entertainment</th>
      <th>On-board service</th>
      <th>Leg room service</th>
      <th>Baggage handling</th>
      <th>Checkin service</th>
      <th>Inflight service</th>
      <th>Cleanliness</th>
      <th>Departure Delay in Minutes</th>
      <th>Arrival Delay in Minutes</th>
      <th>satisfaction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>70172</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>13</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>460</td>
      <td>3</td>
      <td>4</td>
      <td>...</td>
      <td>5</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>25</td>
      <td>18.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>5047</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>25</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>235</td>
      <td>3</td>
      <td>2</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>6.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>110028</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>26</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1142</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>5</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>24026</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>25</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>562</td>
      <td>2</td>
      <td>5</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>11</td>
      <td>9.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>119299</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>61</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>214</td>
      <td>3</td>
      <td>3</td>
      <td>...</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>



----

### **Step 3: Visualize data for insights**


```python
corr_matrix = df.corr()
corr_matrix["satisfaction"].sort_values(ascending=False)
```




    satisfaction                         1.000000
    Online boarding                      0.503447
    Inflight entertainment               0.398203
    Seat comfort                         0.349112
    On-board service                     0.322450
    Leg room service                     0.313182
    Cleanliness                          0.305050
    Flight Distance                      0.298915
    Inflight wifi service                0.284163
    Baggage handling                     0.247819
    Inflight service                     0.244852
    Checkin service                      0.235914
    Food and drink                       0.209659
    Ease of Online booking               0.171507
    Age                                  0.137040
    id                                   0.013680
    Gender                               0.012356
    Gate location                        0.000449
    Unnamed: 0                          -0.004552
    Departure Delay in Minutes          -0.050515
    Departure/Arrival time convenient   -0.051718
    Arrival Delay in Minutes            -0.057582
    Customer Type                       -0.187558
    Type of Travel                      -0.448995
    Class                               -0.449466
    Name: satisfaction, dtype: float64



 


```python
corr_matrix.style.background_gradient(cmap='coolwarm')
```




<style type="text/css">
#T_38a4a_row0_col0, #T_38a4a_row1_col1, #T_38a4a_row2_col2, #T_38a4a_row3_col3, #T_38a4a_row4_col4, #T_38a4a_row5_col5, #T_38a4a_row6_col6, #T_38a4a_row7_col7, #T_38a4a_row8_col8, #T_38a4a_row9_col9, #T_38a4a_row10_col10, #T_38a4a_row11_col11, #T_38a4a_row12_col12, #T_38a4a_row13_col13, #T_38a4a_row14_col14, #T_38a4a_row15_col15, #T_38a4a_row16_col16, #T_38a4a_row17_col17, #T_38a4a_row18_col18, #T_38a4a_row19_col19, #T_38a4a_row20_col20, #T_38a4a_row21_col21, #T_38a4a_row22_col22, #T_38a4a_row23_col23, #T_38a4a_row24_col24 {
  background-color: #b40426;
  color: #f1f1f1;
}
#T_38a4a_row0_col1, #T_38a4a_row2_col1, #T_38a4a_row3_col1, #T_38a4a_row3_col5, #T_38a4a_row5_col1, #T_38a4a_row11_col1, #T_38a4a_row12_col1, #T_38a4a_row15_col1 {
  background-color: #5875e1;
  color: #f1f1f1;
}
#T_38a4a_row0_col2, #T_38a4a_row4_col23, #T_38a4a_row5_col22, #T_38a4a_row7_col2, #T_38a4a_row10_col22, #T_38a4a_row12_col2, #T_38a4a_row15_col2, #T_38a4a_row18_col22, #T_38a4a_row21_col2, #T_38a4a_row22_col12 {
  background-color: #485fd1;
  color: #f1f1f1;
}
#T_38a4a_row0_col3, #T_38a4a_row8_col4, #T_38a4a_row8_col20, #T_38a4a_row18_col9, #T_38a4a_row20_col9, #T_38a4a_row20_col14 {
  background-color: #86a9fc;
  color: #f1f1f1;
}
#T_38a4a_row0_col4, #T_38a4a_row14_col5, #T_38a4a_row21_col6, #T_38a4a_row21_col18 {
  background-color: #84a7fc;
  color: #f1f1f1;
}
#T_38a4a_row0_col5, #T_38a4a_row0_col6, #T_38a4a_row1_col5, #T_38a4a_row11_col24, #T_38a4a_row13_col19 {
  background-color: #a2c1ff;
  color: #000000;
}
#T_38a4a_row0_col7, #T_38a4a_row2_col6, #T_38a4a_row2_col7, #T_38a4a_row7_col14, #T_38a4a_row8_col7, #T_38a4a_row11_col7, #T_38a4a_row14_col19, #T_38a4a_row22_col7 {
  background-color: #9fbfff;
  color: #000000;
}
#T_38a4a_row0_col8, #T_38a4a_row4_col12, #T_38a4a_row4_col20, #T_38a4a_row9_col1, #T_38a4a_row13_col3, #T_38a4a_row24_col3 {
  background-color: #5673e0;
  color: #f1f1f1;
}
#T_38a4a_row0_col9, #T_38a4a_row0_col14, #T_38a4a_row0_col16, #T_38a4a_row2_col18, #T_38a4a_row7_col10, #T_38a4a_row8_col19, #T_38a4a_row9_col17, #T_38a4a_row10_col18, #T_38a4a_row17_col6, #T_38a4a_row22_col9, #T_38a4a_row23_col17 {
  background-color: #7295f4;
  color: #f1f1f1;
}
#T_38a4a_row0_col10, #T_38a4a_row2_col21, #T_38a4a_row5_col20, #T_38a4a_row21_col1, #T_38a4a_row22_col19, #T_38a4a_row23_col19 {
  background-color: #5f7fe8;
  color: #f1f1f1;
}
#T_38a4a_row0_col11, #T_38a4a_row1_col2, #T_38a4a_row3_col11, #T_38a4a_row3_col14, #T_38a4a_row4_col22, #T_38a4a_row5_col21, #T_38a4a_row11_col2, #T_38a4a_row21_col23, #T_38a4a_row22_col2, #T_38a4a_row22_col11, #T_38a4a_row23_col2, #T_38a4a_row23_col11, #T_38a4a_row23_col12 {
  background-color: #465ecf;
  color: #f1f1f1;
}
#T_38a4a_row0_col12, #T_38a4a_row6_col23, #T_38a4a_row11_col12, #T_38a4a_row17_col23 {
  background-color: #4f69d9;
  color: #f1f1f1;
}
#T_38a4a_row0_col13, #T_38a4a_row2_col4, #T_38a4a_row4_col6, #T_38a4a_row8_col3, #T_38a4a_row8_col21, #T_38a4a_row9_col16, #T_38a4a_row11_col13, #T_38a4a_row13_col9, #T_38a4a_row16_col9 {
  background-color: #85a8fc;
  color: #f1f1f1;
}
#T_38a4a_row0_col15, #T_38a4a_row1_col15, #T_38a4a_row7_col9, #T_38a4a_row11_col15, #T_38a4a_row13_col5 {
  background-color: #6b8df0;
  color: #f1f1f1;
}
#T_38a4a_row0_col17, #T_38a4a_row1_col9, #T_38a4a_row2_col20, #T_38a4a_row12_col18, #T_38a4a_row21_col3, #T_38a4a_row23_col9 {
  background-color: #7093f3;
  color: #f1f1f1;
}
#T_38a4a_row0_col18, #T_38a4a_row0_col20, #T_38a4a_row4_col10, #T_38a4a_row11_col20 {
  background-color: #6687ed;
  color: #f1f1f1;
}
#T_38a4a_row0_col19, #T_38a4a_row1_col10, #T_38a4a_row21_col10 {
  background-color: #6384eb;
  color: #f1f1f1;
}
#T_38a4a_row0_col21, #T_38a4a_row4_col8, #T_38a4a_row5_col7, #T_38a4a_row22_col10, #T_38a4a_row23_col10 {
  background-color: #5d7ce6;
  color: #f1f1f1;
}
#T_38a4a_row0_col22, #T_38a4a_row2_col22, #T_38a4a_row7_col22, #T_38a4a_row7_col23, #T_38a4a_row9_col22, #T_38a4a_row24_col2 {
  background-color: #4a63d3;
  color: #f1f1f1;
}
#T_38a4a_row0_col23, #T_38a4a_row2_col23, #T_38a4a_row3_col15, #T_38a4a_row3_col22, #T_38a4a_row5_col13, #T_38a4a_row9_col23, #T_38a4a_row11_col22 {
  background-color: #4b64d5;
  color: #f1f1f1;
}
#T_38a4a_row0_col24, #T_38a4a_row4_col14, #T_38a4a_row11_col6, #T_38a4a_row22_col5, #T_38a4a_row23_col5 {
  background-color: #a1c0ff;
  color: #000000;
}
#T_38a4a_row1_col0, #T_38a4a_row5_col0, #T_38a4a_row5_col11, #T_38a4a_row7_col0, #T_38a4a_row9_col0, #T_38a4a_row10_col0, #T_38a4a_row13_col0, #T_38a4a_row14_col0, #T_38a4a_row15_col0, #T_38a4a_row16_col0, #T_38a4a_row16_col11, #T_38a4a_row18_col0, #T_38a4a_row20_col0, #T_38a4a_row23_col0 {
  background-color: #3c4ec2;
  color: #f1f1f1;
}
#T_38a4a_row1_col3, #T_38a4a_row1_col4, #T_38a4a_row1_col6, #T_38a4a_row8_col5, #T_38a4a_row12_col4, #T_38a4a_row18_col14 {
  background-color: #88abfd;
  color: #000000;
}
#T_38a4a_row1_col7, #T_38a4a_row21_col7 {
  background-color: #b5cdfa;
  color: #000000;
}
#T_38a4a_row1_col8, #T_38a4a_row2_col12, #T_38a4a_row8_col1, #T_38a4a_row9_col3, #T_38a4a_row9_col12, #T_38a4a_row18_col2, #T_38a4a_row20_col2, #T_38a4a_row22_col1, #T_38a4a_row23_col8 {
  background-color: #516ddb;
  color: #f1f1f1;
}
#T_38a4a_row1_col11, #T_38a4a_row1_col22, #T_38a4a_row2_col11, #T_38a4a_row4_col11, #T_38a4a_row6_col10, #T_38a4a_row8_col22, #T_38a4a_row12_col11, #T_38a4a_row13_col22, #T_38a4a_row19_col22, #T_38a4a_row24_col11 {
  background-color: #445acc;
  color: #f1f1f1;
}
#T_38a4a_row1_col12, #T_38a4a_row5_col14, #T_38a4a_row6_col8, #T_38a4a_row17_col2 {
  background-color: #506bda;
  color: #f1f1f1;
}
#T_38a4a_row1_col13, #T_38a4a_row8_col16, #T_38a4a_row16_col5, #T_38a4a_row19_col21, #T_38a4a_row23_col24 {
  background-color: #94b6ff;
  color: #000000;
}
#T_38a4a_row1_col14, #T_38a4a_row1_col16, #T_38a4a_row4_col15, #T_38a4a_row9_col19, #T_38a4a_row10_col5, #T_38a4a_row11_col4, #T_38a4a_row12_col8, #T_38a4a_row21_col8 {
  background-color: #81a4fb;
  color: #f1f1f1;
}
#T_38a4a_row1_col17, #T_38a4a_row4_col9, #T_38a4a_row9_col18, #T_38a4a_row9_col20, #T_38a4a_row13_col20, #T_38a4a_row19_col6, #T_38a4a_row20_col6 {
  background-color: #7b9ff9;
  color: #f1f1f1;
}
#T_38a4a_row1_col18, #T_38a4a_row1_col19, #T_38a4a_row1_col20, #T_38a4a_row10_col16, #T_38a4a_row14_col8, #T_38a4a_row14_col18, #T_38a4a_row16_col8, #T_38a4a_row18_col8 {
  background-color: #7da0f9;
  color: #f1f1f1;
}
#T_38a4a_row1_col21, #T_38a4a_row17_col1, #T_38a4a_row19_col8, #T_38a4a_row22_col18, #T_38a4a_row23_col18 {
  background-color: #6485ec;
  color: #f1f1f1;
}
#T_38a4a_row1_col23, #T_38a4a_row4_col3, #T_38a4a_row12_col22, #T_38a4a_row16_col22, #T_38a4a_row16_col23 {
  background-color: #4055c8;
  color: #f1f1f1;
}
#T_38a4a_row1_col24, #T_38a4a_row2_col24, #T_38a4a_row22_col6, #T_38a4a_row23_col6 {
  background-color: #a5c3fe;
  color: #000000;
}
#T_38a4a_row2_col0, #T_38a4a_row3_col2, #T_38a4a_row4_col0, #T_38a4a_row11_col0, #T_38a4a_row17_col0 {
  background-color: #3d50c3;
  color: #f1f1f1;
}
#T_38a4a_row2_col3, #T_38a4a_row12_col19, #T_38a4a_row13_col18, #T_38a4a_row17_col5, #T_38a4a_row17_col10, #T_38a4a_row19_col3, #T_38a4a_row21_col20, #T_38a4a_row22_col4, #T_38a4a_row22_col13, #T_38a4a_row23_col13 {
  background-color: #80a3fa;
  color: #f1f1f1;
}
#T_38a4a_row2_col5 {
  background-color: #a3c2fe;
  color: #000000;
}
#T_38a4a_row2_col8, #T_38a4a_row11_col19, #T_38a4a_row13_col6, #T_38a4a_row18_col12, #T_38a4a_row20_col12 {
  background-color: #5a78e4;
  color: #f1f1f1;
}
#T_38a4a_row2_col9, #T_38a4a_row2_col16, #T_38a4a_row11_col14, #T_38a4a_row12_col9, #T_38a4a_row22_col17 {
  background-color: #7396f5;
  color: #f1f1f1;
}
#T_38a4a_row2_col10, #T_38a4a_row3_col17, #T_38a4a_row7_col12, #T_38a4a_row9_col21 {
  background-color: #6180e9;
  color: #f1f1f1;
}
#T_38a4a_row2_col13, #T_38a4a_row4_col17, #T_38a4a_row7_col19, #T_38a4a_row8_col12, #T_38a4a_row10_col14, #T_38a4a_row14_col20, #T_38a4a_row17_col3, #T_38a4a_row17_col21, #T_38a4a_row18_col6, #T_38a4a_row18_col21 {
  background-color: #7a9df8;
  color: #f1f1f1;
}
#T_38a4a_row2_col14, #T_38a4a_row5_col19, #T_38a4a_row11_col16, #T_38a4a_row15_col3, #T_38a4a_row16_col10, #T_38a4a_row18_col10, #T_38a4a_row19_col12, #T_38a4a_row20_col10, #T_38a4a_row22_col14, #T_38a4a_row23_col14 {
  background-color: #6a8bef;
  color: #f1f1f1;
}
#T_38a4a_row2_col15, #T_38a4a_row3_col20, #T_38a4a_row4_col21, #T_38a4a_row11_col17, #T_38a4a_row15_col10 {
  background-color: #6c8ff1;
  color: #f1f1f1;
}
#T_38a4a_row2_col17, #T_38a4a_row12_col3, #T_38a4a_row20_col21 {
  background-color: #779af7;
  color: #f1f1f1;
}
#T_38a4a_row2_col19, #T_38a4a_row9_col15, #T_38a4a_row10_col19, #T_38a4a_row12_col10, #T_38a4a_row13_col1, #T_38a4a_row22_col16 {
  background-color: #688aef;
  color: #f1f1f1;
}
#T_38a4a_row3_col0, #T_38a4a_row3_col4, #T_38a4a_row3_col9, #T_38a4a_row5_col3, #T_38a4a_row5_col8, #T_38a4a_row5_col10, #T_38a4a_row5_col24, #T_38a4a_row6_col0, #T_38a4a_row6_col1, #T_38a4a_row6_col7, #T_38a4a_row6_col12, #T_38a4a_row6_col13, #T_38a4a_row6_col14, #T_38a4a_row6_col15, #T_38a4a_row6_col16, #T_38a4a_row6_col17, #T_38a4a_row6_col18, #T_38a4a_row6_col19, #T_38a4a_row6_col20, #T_38a4a_row6_col21, #T_38a4a_row6_col24, #T_38a4a_row8_col0, #T_38a4a_row12_col0, #T_38a4a_row13_col2, #T_38a4a_row19_col0, #T_38a4a_row19_col11, #T_38a4a_row20_col22, #T_38a4a_row20_col23, #T_38a4a_row21_col0, #T_38a4a_row22_col0, #T_38a4a_row24_col0, #T_38a4a_row24_col5, #T_38a4a_row24_col6, #T_38a4a_row24_col22, #T_38a4a_row24_col23 {
  background-color: #3b4cc0;
  color: #f1f1f1;
}
#T_38a4a_row3_col6, #T_38a4a_row12_col7, #T_38a4a_row18_col19, #T_38a4a_row20_col7, #T_38a4a_row24_col19 {
  background-color: #abc8fd;
  color: #000000;
}
#T_38a4a_row3_col7, #T_38a4a_row11_col18, #T_38a4a_row14_col1, #T_38a4a_row14_col10, #T_38a4a_row16_col1, #T_38a4a_row23_col16 {
  background-color: #6788ee;
  color: #f1f1f1;
}
#T_38a4a_row3_col8, #T_38a4a_row22_col20 {
  background-color: #5572df;
  color: #f1f1f1;
}
#T_38a4a_row3_col10, #T_38a4a_row4_col18, #T_38a4a_row7_col8, #T_38a4a_row10_col12, #T_38a4a_row17_col12, #T_38a4a_row22_col21, #T_38a4a_row23_col21 {
  background-color: #5977e3;
  color: #f1f1f1;
}
#T_38a4a_row3_col12 {
  background-color: #3f53c6;
  color: #f1f1f1;
}
#T_38a4a_row3_col13, #T_38a4a_row23_col20 {
  background-color: #5470de;
  color: #f1f1f1;
}
#T_38a4a_row3_col16, #T_38a4a_row5_col16, #T_38a4a_row6_col4, #T_38a4a_row7_col5, #T_38a4a_row10_col21, #T_38a4a_row16_col12, #T_38a4a_row19_col10, #T_38a4a_row22_col15, #T_38a4a_row23_col15, #T_38a4a_row24_col9 {
  background-color: #6282ea;
  color: #f1f1f1;
}
#T_38a4a_row3_col18, #T_38a4a_row18_col1 {
  background-color: #6e90f2;
  color: #f1f1f1;
}
#T_38a4a_row3_col19, #T_38a4a_row10_col1, #T_38a4a_row11_col21, #T_38a4a_row24_col1 {
  background-color: #5b7ae5;
  color: #f1f1f1;
}
#T_38a4a_row3_col21, #T_38a4a_row7_col11, #T_38a4a_row8_col23, #T_38a4a_row13_col11, #T_38a4a_row13_col23, #T_38a4a_row14_col11, #T_38a4a_row15_col11, #T_38a4a_row18_col11, #T_38a4a_row19_col23, #T_38a4a_row20_col11, #T_38a4a_row21_col22 {
  background-color: #455cce;
  color: #f1f1f1;
}
#T_38a4a_row3_col23, #T_38a4a_row6_col22, #T_38a4a_row11_col23, #T_38a4a_row23_col1 {
  background-color: #4c66d6;
  color: #f1f1f1;
}
#T_38a4a_row3_col24, #T_38a4a_row5_col4, #T_38a4a_row7_col1, #T_38a4a_row9_col14, #T_38a4a_row14_col9, #T_38a4a_row17_col9, #T_38a4a_row18_col4, #T_38a4a_row20_col4, #T_38a4a_row21_col9 {
  background-color: #7597f6;
  color: #f1f1f1;
}
#T_38a4a_row4_col1, #T_38a4a_row5_col18, #T_38a4a_row14_col3 {
  background-color: #5e7de7;
  color: #f1f1f1;
}
#T_38a4a_row4_col2, #T_38a4a_row5_col2, #T_38a4a_row5_col17, #T_38a4a_row5_col23, #T_38a4a_row8_col2, #T_38a4a_row9_col2, #T_38a4a_row10_col2, #T_38a4a_row10_col23, #T_38a4a_row16_col2, #T_38a4a_row18_col23, #T_38a4a_row19_col2 {
  background-color: #4961d2;
  color: #f1f1f1;
}
#T_38a4a_row4_col5, #T_38a4a_row7_col17, #T_38a4a_row8_col14, #T_38a4a_row9_col24, #T_38a4a_row21_col16, #T_38a4a_row22_col24 {
  background-color: #96b7ff;
  color: #000000;
}
#T_38a4a_row4_col7 {
  background-color: #b6cefa;
  color: #000000;
}
#T_38a4a_row4_col13, #T_38a4a_row5_col9, #T_38a4a_row15_col7 {
  background-color: #bcd2f7;
  color: #000000;
}
#T_38a4a_row4_col16, #T_38a4a_row10_col3, #T_38a4a_row12_col16, #T_38a4a_row16_col21 {
  background-color: #82a6fb;
  color: #f1f1f1;
}
#T_38a4a_row4_col19, #T_38a4a_row10_col20, #T_38a4a_row12_col20, #T_38a4a_row14_col6, #T_38a4a_row15_col9, #T_38a4a_row16_col6, #T_38a4a_row19_col1, #T_38a4a_row20_col1 {
  background-color: #6f92f3;
  color: #f1f1f1;
}
#T_38a4a_row4_col24, #T_38a4a_row9_col8, #T_38a4a_row17_col15 {
  background-color: #c1d4f4;
  color: #000000;
}
#T_38a4a_row5_col6, #T_38a4a_row6_col5, #T_38a4a_row12_col14 {
  background-color: #f6bea4;
  color: #000000;
}
#T_38a4a_row5_col12, #T_38a4a_row7_col6, #T_38a4a_row14_col2 {
  background-color: #3e51c5;
  color: #f1f1f1;
}
#T_38a4a_row5_col15, #T_38a4a_row12_col23, #T_38a4a_row14_col22, #T_38a4a_row15_col22, #T_38a4a_row15_col23 {
  background-color: #4257c9;
  color: #f1f1f1;
}
#T_38a4a_row6_col2, #T_38a4a_row6_col11, #T_38a4a_row14_col23, #T_38a4a_row17_col11, #T_38a4a_row21_col11 {
  background-color: #4358cb;
  color: #f1f1f1;
}
#T_38a4a_row6_col3, #T_38a4a_row12_col5, #T_38a4a_row13_col17, #T_38a4a_row17_col19, #T_38a4a_row24_col10, #T_38a4a_row24_col12 {
  background-color: #93b5fe;
  color: #000000;
}
#T_38a4a_row6_col9, #T_38a4a_row8_col18, #T_38a4a_row10_col6, #T_38a4a_row21_col17 {
  background-color: #8badfd;
  color: #000000;
}
#T_38a4a_row7_col3, #T_38a4a_row17_col22 {
  background-color: #4e68d8;
  color: #f1f1f1;
}
#T_38a4a_row7_col4, #T_38a4a_row8_col17, #T_38a4a_row13_col16, #T_38a4a_row23_col7 {
  background-color: #9ebeff;
  color: #000000;
}
#T_38a4a_row7_col13 {
  background-color: #bfd3f6;
  color: #000000;
}
#T_38a4a_row7_col15, #T_38a4a_row17_col14, #T_38a4a_row21_col4 {
  background-color: #90b2fe;
  color: #000000;
}
#T_38a4a_row7_col16, #T_38a4a_row16_col4 {
  background-color: #92b4fe;
  color: #000000;
}
#T_38a4a_row7_col18, #T_38a4a_row7_col21, #T_38a4a_row10_col15, #T_38a4a_row12_col17, #T_38a4a_row16_col3, #T_38a4a_row20_col8 {
  background-color: #799cf8;
  color: #f1f1f1;
}
#T_38a4a_row7_col20, #T_38a4a_row15_col6 {
  background-color: #7699f6;
  color: #f1f1f1;
}
#T_38a4a_row7_col24 {
  background-color: #e1dad6;
  color: #000000;
}
#T_38a4a_row8_col6, #T_38a4a_row18_col13, #T_38a4a_row20_col5 {
  background-color: #9dbdff;
  color: #000000;
}
#T_38a4a_row8_col9, #T_38a4a_row12_col24, #T_38a4a_row17_col20 {
  background-color: #d1dae9;
  color: #000000;
}
#T_38a4a_row8_col10 {
  background-color: #f49a7b;
  color: #000000;
}
#T_38a4a_row8_col11 {
  background-color: #b2ccfb;
  color: #000000;
}
#T_38a4a_row8_col13 {
  background-color: #efcfbf;
  color: #000000;
}
#T_38a4a_row8_col15, #T_38a4a_row19_col14 {
  background-color: #a9c6fd;
  color: #000000;
}
#T_38a4a_row8_col24 {
  background-color: #dedcdb;
  color: #000000;
}
#T_38a4a_row9_col4, #T_38a4a_row19_col4, #T_38a4a_row19_col9 {
  background-color: #8caffe;
  color: #000000;
}
#T_38a4a_row9_col5, #T_38a4a_row24_col15 {
  background-color: #dadce0;
  color: #000000;
}
#T_38a4a_row9_col6, #T_38a4a_row19_col16 {
  background-color: #b7cff9;
  color: #000000;
}
#T_38a4a_row9_col7, #T_38a4a_row11_col5, #T_38a4a_row18_col5, #T_38a4a_row20_col13 {
  background-color: #9abbff;
  color: #000000;
}
#T_38a4a_row9_col10 {
  background-color: #dddcdc;
  color: #000000;
}
#T_38a4a_row9_col11, #T_38a4a_row16_col17, #T_38a4a_row24_col14 {
  background-color: #d3dbe7;
  color: #000000;
}
#T_38a4a_row9_col13, #T_38a4a_row14_col16, #T_38a4a_row15_col8, #T_38a4a_row16_col14 {
  background-color: #98b9ff;
  color: #000000;
}
#T_38a4a_row10_col4, #T_38a4a_row11_col3, #T_38a4a_row15_col19, #T_38a4a_row17_col8, #T_38a4a_row22_col3, #T_38a4a_row23_col3 {
  background-color: #89acfd;
  color: #000000;
}
#T_38a4a_row10_col7, #T_38a4a_row14_col4, #T_38a4a_row16_col19, #T_38a4a_row18_col7 {
  background-color: #aec9fc;
  color: #000000;
}
#T_38a4a_row10_col8 {
  background-color: #f59c7d;
  color: #000000;
}
#T_38a4a_row10_col9, #T_38a4a_row16_col24 {
  background-color: #e5d8d1;
  color: #000000;
}
#T_38a4a_row10_col11, #T_38a4a_row20_col24 {
  background-color: #d7dce3;
  color: #000000;
}
#T_38a4a_row10_col13, #T_38a4a_row11_col9 {
  background-color: #e7d7ce;
  color: #000000;
}
#T_38a4a_row10_col17, #T_38a4a_row12_col6, #T_38a4a_row18_col3, #T_38a4a_row19_col15, #T_38a4a_row21_col5 {
  background-color: #8fb1fe;
  color: #000000;
}
#T_38a4a_row10_col24, #T_38a4a_row24_col17 {
  background-color: #c9d7f0;
  color: #000000;
}
#T_38a4a_row11_col8 {
  background-color: #c0d4f5;
  color: #000000;
}
#T_38a4a_row11_col10, #T_38a4a_row13_col14, #T_38a4a_row15_col16, #T_38a4a_row21_col24 {
  background-color: #e2dad5;
  color: #000000;
}
#T_38a4a_row12_col13, #T_38a4a_row13_col21, #T_38a4a_row14_col7 {
  background-color: #c3d5f4;
  color: #000000;
}
#T_38a4a_row12_col15, #T_38a4a_row15_col14 {
  background-color: #f7b497;
  color: #000000;
}
#T_38a4a_row12_col21 {
  background-color: #f7ad90;
  color: #000000;
}
#T_38a4a_row13_col4, #T_38a4a_row24_col21 {
  background-color: #bad0f8;
  color: #000000;
}
#T_38a4a_row13_col7 {
  background-color: #cfdaea;
  color: #000000;
}
#T_38a4a_row13_col8, #T_38a4a_row16_col15, #T_38a4a_row24_col7 {
  background-color: #dfdbd9;
  color: #000000;
}
#T_38a4a_row13_col10, #T_38a4a_row18_col15, #T_38a4a_row18_col17, #T_38a4a_row19_col24, #T_38a4a_row20_col17 {
  background-color: #d6dce4;
  color: #000000;
}
#T_38a4a_row13_col12, #T_38a4a_row19_col17, #T_38a4a_row21_col19 {
  background-color: #9bbcff;
  color: #000000;
}
#T_38a4a_row13_col15, #T_38a4a_row17_col7 {
  background-color: #bed2f6;
  color: #000000;
}
#T_38a4a_row13_col24 {
  background-color: #f7ba9f;
  color: #000000;
}
#T_38a4a_row14_col12, #T_38a4a_row18_col16 {
  background-color: #f2cab5;
  color: #000000;
}
#T_38a4a_row14_col13, #T_38a4a_row14_col24 {
  background-color: #ead5c9;
  color: #000000;
}
#T_38a4a_row14_col15 {
  background-color: #f7b79b;
  color: #000000;
}
#T_38a4a_row14_col17, #T_38a4a_row17_col4, #T_38a4a_row20_col3 {
  background-color: #8db0fe;
  color: #000000;
}
#T_38a4a_row14_col21 {
  background-color: #f7a889;
  color: #000000;
}
#T_38a4a_row15_col4 {
  background-color: #97b8ff;
  color: #000000;
}
#T_38a4a_row15_col5, #T_38a4a_row23_col4 {
  background-color: #7ea1fa;
  color: #f1f1f1;
}
#T_38a4a_row15_col12 {
  background-color: #f6bda2;
  color: #000000;
}
#T_38a4a_row15_col13 {
  background-color: #cedaeb;
  color: #000000;
}
#T_38a4a_row15_col17 {
  background-color: #c5d6f2;
  color: #000000;
}
#T_38a4a_row15_col18, #T_38a4a_row17_col16 {
  background-color: #d4dbe6;
  color: #000000;
}
#T_38a4a_row15_col20, #T_38a4a_row21_col13 {
  background-color: #d9dce1;
  color: #000000;
}
#T_38a4a_row15_col21 {
  background-color: #f6a385;
  color: #000000;
}
#T_38a4a_row15_col24 {
  background-color: #efcebd;
  color: #000000;
}
#T_38a4a_row16_col7 {
  background-color: #b9d0f9;
  color: #000000;
}
#T_38a4a_row16_col13, #T_38a4a_row19_col7, #T_38a4a_row24_col20 {
  background-color: #afcafc;
  color: #000000;
}
#T_38a4a_row16_col18 {
  background-color: #f0cdbb;
  color: #000000;
}
#T_38a4a_row16_col20 {
  background-color: #f3c8b2;
  color: #000000;
}
#T_38a4a_row17_col13, #T_38a4a_row24_col4 {
  background-color: #a7c5fe;
  color: #000000;
}
#T_38a4a_row17_col18 {
  background-color: #d2dbe8;
  color: #000000;
}
#T_38a4a_row17_col24 {
  background-color: #e3d9d3;
  color: #000000;
}
#T_38a4a_row18_col20, #T_38a4a_row20_col18, #T_38a4a_row21_col12 {
  background-color: #f7b396;
  color: #000000;
}
#T_38a4a_row18_col24 {
  background-color: #d8dce2;
  color: #000000;
}
#T_38a4a_row19_col5 {
  background-color: #a6c4fe;
  color: #000000;
}
#T_38a4a_row19_col13 {
  background-color: #bbd1f8;
  color: #000000;
}
#T_38a4a_row19_col18, #T_38a4a_row19_col20, #T_38a4a_row20_col19 {
  background-color: #adc9fd;
  color: #000000;
}
#T_38a4a_row20_col15 {
  background-color: #dbdcde;
  color: #000000;
}
#T_38a4a_row20_col16 {
  background-color: #f5c4ac;
  color: #000000;
}
#T_38a4a_row21_col14 {
  background-color: #f5a081;
  color: #000000;
}
#T_38a4a_row21_col15 {
  background-color: #f59d7e;
  color: #000000;
}
#T_38a4a_row22_col8 {
  background-color: #536edd;
  color: #f1f1f1;
}
#T_38a4a_row22_col23, #T_38a4a_row23_col22 {
  background-color: #c0282f;
  color: #f1f1f1;
}
#T_38a4a_row24_col8, #T_38a4a_row24_col18 {
  background-color: #b1cbfc;
  color: #000000;
}
#T_38a4a_row24_col13 {
  background-color: #f3c7b1;
  color: #000000;
}
#T_38a4a_row24_col16 {
  background-color: #ccd9ed;
  color: #000000;
}
</style>
<table id="T_38a4a">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_38a4a_level0_col0" class="col_heading level0 col0" >Unnamed: 0</th>
      <th id="T_38a4a_level0_col1" class="col_heading level0 col1" >id</th>
      <th id="T_38a4a_level0_col2" class="col_heading level0 col2" >Gender</th>
      <th id="T_38a4a_level0_col3" class="col_heading level0 col3" >Customer Type</th>
      <th id="T_38a4a_level0_col4" class="col_heading level0 col4" >Age</th>
      <th id="T_38a4a_level0_col5" class="col_heading level0 col5" >Type of Travel</th>
      <th id="T_38a4a_level0_col6" class="col_heading level0 col6" >Class</th>
      <th id="T_38a4a_level0_col7" class="col_heading level0 col7" >Flight Distance</th>
      <th id="T_38a4a_level0_col8" class="col_heading level0 col8" >Inflight wifi service</th>
      <th id="T_38a4a_level0_col9" class="col_heading level0 col9" >Departure/Arrival time convenient</th>
      <th id="T_38a4a_level0_col10" class="col_heading level0 col10" >Ease of Online booking</th>
      <th id="T_38a4a_level0_col11" class="col_heading level0 col11" >Gate location</th>
      <th id="T_38a4a_level0_col12" class="col_heading level0 col12" >Food and drink</th>
      <th id="T_38a4a_level0_col13" class="col_heading level0 col13" >Online boarding</th>
      <th id="T_38a4a_level0_col14" class="col_heading level0 col14" >Seat comfort</th>
      <th id="T_38a4a_level0_col15" class="col_heading level0 col15" >Inflight entertainment</th>
      <th id="T_38a4a_level0_col16" class="col_heading level0 col16" >On-board service</th>
      <th id="T_38a4a_level0_col17" class="col_heading level0 col17" >Leg room service</th>
      <th id="T_38a4a_level0_col18" class="col_heading level0 col18" >Baggage handling</th>
      <th id="T_38a4a_level0_col19" class="col_heading level0 col19" >Checkin service</th>
      <th id="T_38a4a_level0_col20" class="col_heading level0 col20" >Inflight service</th>
      <th id="T_38a4a_level0_col21" class="col_heading level0 col21" >Cleanliness</th>
      <th id="T_38a4a_level0_col22" class="col_heading level0 col22" >Departure Delay in Minutes</th>
      <th id="T_38a4a_level0_col23" class="col_heading level0 col23" >Arrival Delay in Minutes</th>
      <th id="T_38a4a_level0_col24" class="col_heading level0 col24" >satisfaction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_38a4a_level0_row0" class="row_heading level0 row0" >Unnamed: 0</th>
      <td id="T_38a4a_row0_col0" class="data row0 col0" >1.000000</td>
      <td id="T_38a4a_row0_col1" class="data row0 col1" >0.003146</td>
      <td id="T_38a4a_row0_col2" class="data row0 col2" >0.003991</td>
      <td id="T_38a4a_row0_col3" class="data row0 col3" >-0.002983</td>
      <td id="T_38a4a_row0_col4" class="data row0 col4" >0.004806</td>
      <td id="T_38a4a_row0_col5" class="data row0 col5" >0.000707</td>
      <td id="T_38a4a_row0_col6" class="data row0 col6" >-0.001449</td>
      <td id="T_38a4a_row0_col7" class="data row0 col7" >0.002973</td>
      <td id="T_38a4a_row0_col8" class="data row0 col8" >-0.002523</td>
      <td id="T_38a4a_row0_col9" class="data row0 col9" >0.000750</td>
      <td id="T_38a4a_row0_col10" class="data row0 col10" >0.001941</td>
      <td id="T_38a4a_row0_col11" class="data row0 col11" >0.005007</td>
      <td id="T_38a4a_row0_col12" class="data row0 col12" >-0.001974</td>
      <td id="T_38a4a_row0_col13" class="data row0 col13" >0.001126</td>
      <td id="T_38a4a_row0_col14" class="data row0 col14" >0.000343</td>
      <td id="T_38a4a_row0_col15" class="data row0 col15" >0.001474</td>
      <td id="T_38a4a_row0_col16" class="data row0 col16" >0.001046</td>
      <td id="T_38a4a_row0_col17" class="data row0 col17" >0.004061</td>
      <td id="T_38a4a_row0_col18" class="data row0 col18" >-0.000328</td>
      <td id="T_38a4a_row0_col19" class="data row0 col19" >-0.004205</td>
      <td id="T_38a4a_row0_col20" class="data row0 col20" >-0.000011</td>
      <td id="T_38a4a_row0_col21" class="data row0 col21" >-0.000978</td>
      <td id="T_38a4a_row0_col22" class="data row0 col22" >-0.000694</td>
      <td id="T_38a4a_row0_col23" class="data row0 col23" >-0.000045</td>
      <td id="T_38a4a_row0_col24" class="data row0 col24" >-0.004552</td>
    </tr>
    <tr>
      <th id="T_38a4a_level0_row1" class="row_heading level0 row1" >id</th>
      <td id="T_38a4a_row1_col0" class="data row1 col0" >0.003146</td>
      <td id="T_38a4a_row1_col1" class="data row1 col1" >1.000000</td>
      <td id="T_38a4a_row1_col2" class="data row1 col2" >-0.000301</td>
      <td id="T_38a4a_row1_col3" class="data row1 col3" >0.000031</td>
      <td id="T_38a4a_row1_col4" class="data row1 col4" >0.022929</td>
      <td id="T_38a4a_row1_col5" class="data row1 col5" >0.000576</td>
      <td id="T_38a4a_row1_col6" class="data row1 col6" >-0.104340</td>
      <td id="T_38a4a_row1_col7" class="data row1 col7" >0.095184</td>
      <td id="T_38a4a_row1_col8" class="data row1 col8" >-0.021338</td>
      <td id="T_38a4a_row1_col9" class="data row1 col9" >-0.001714</td>
      <td id="T_38a4a_row1_col10" class="data row1 col10" >0.014119</td>
      <td id="T_38a4a_row1_col11" class="data row1 col11" >-0.000427</td>
      <td id="T_38a4a_row1_col12" class="data row1 col12" >0.001254</td>
      <td id="T_38a4a_row1_col13" class="data row1 col13" >0.055394</td>
      <td id="T_38a4a_row1_col14" class="data row1 col14" >0.053091</td>
      <td id="T_38a4a_row1_col15" class="data row1 col15" >0.002592</td>
      <td id="T_38a4a_row1_col16" class="data row1 col16" >0.055255</td>
      <td id="T_38a4a_row1_col17" class="data row1 col17" >0.044459</td>
      <td id="T_38a4a_row1_col18" class="data row1 col18" >0.075134</td>
      <td id="T_38a4a_row1_col19" class="data row1 col19" >0.079346</td>
      <td id="T_38a4a_row1_col20" class="data row1 col20" >0.079468</td>
      <td id="T_38a4a_row1_col21" class="data row1 col21" >0.025313</td>
      <td id="T_38a4a_row1_col22" class="data row1 col22" >-0.019321</td>
      <td id="T_38a4a_row1_col23" class="data row1 col23" >-0.037254</td>
      <td id="T_38a4a_row1_col24" class="data row1 col24" >0.013680</td>
    </tr>
    <tr>
      <th id="T_38a4a_level0_row2" class="row_heading level0 row2" >Gender</th>
      <td id="T_38a4a_row2_col0" class="data row2 col0" >0.003991</td>
      <td id="T_38a4a_row2_col1" class="data row2 col1" >-0.000301</td>
      <td id="T_38a4a_row2_col2" class="data row2 col2" >1.000000</td>
      <td id="T_38a4a_row2_col3" class="data row2 col3" >-0.031558</td>
      <td id="T_38a4a_row2_col4" class="data row2 col4" >0.008921</td>
      <td id="T_38a4a_row2_col5" class="data row2 col5" >0.006808</td>
      <td id="T_38a4a_row2_col6" class="data row2 col6" >-0.012840</td>
      <td id="T_38a4a_row2_col7" class="data row2 col7" >0.006079</td>
      <td id="T_38a4a_row2_col8" class="data row2 col8" >0.008964</td>
      <td id="T_38a4a_row2_col9" class="data row2 col9" >0.008846</td>
      <td id="T_38a4a_row2_col10" class="data row2 col10" >0.007166</td>
      <td id="T_38a4a_row2_col11" class="data row2 col11" >0.000213</td>
      <td id="T_38a4a_row2_col12" class="data row2 col12" >0.005707</td>
      <td id="T_38a4a_row2_col13" class="data row2 col13" >-0.042151</td>
      <td id="T_38a4a_row2_col14" class="data row2 col14" >-0.026643</td>
      <td id="T_38a4a_row2_col15" class="data row2 col15" >0.006071</td>
      <td id="T_38a4a_row2_col16" class="data row2 col16" >0.008019</td>
      <td id="T_38a4a_row2_col17" class="data row2 col17" >0.031842</td>
      <td id="T_38a4a_row2_col18" class="data row2 col18" >0.037333</td>
      <td id="T_38a4a_row2_col19" class="data row2 col19" >0.010438</td>
      <td id="T_38a4a_row2_col20" class="data row2 col20" >0.038936</td>
      <td id="T_38a4a_row2_col21" class="data row2 col21" >0.006439</td>
      <td id="T_38a4a_row2_col22" class="data row2 col22" >0.002534</td>
      <td id="T_38a4a_row2_col23" class="data row2 col23" >0.000396</td>
      <td id="T_38a4a_row2_col24" class="data row2 col24" >0.012356</td>
    </tr>
    <tr>
      <th id="T_38a4a_level0_row3" class="row_heading level0 row3" >Customer Type</th>
      <td id="T_38a4a_row3_col0" class="data row3 col0" >-0.002983</td>
      <td id="T_38a4a_row3_col1" class="data row3 col1" >0.000031</td>
      <td id="T_38a4a_row3_col2" class="data row3 col2" >-0.031558</td>
      <td id="T_38a4a_row3_col3" class="data row3 col3" >1.000000</td>
      <td id="T_38a4a_row3_col4" class="data row3 col4" >-0.281821</td>
      <td id="T_38a4a_row3_col5" class="data row3 col5" >-0.308268</td>
      <td id="T_38a4a_row3_col6" class="data row3 col6" >0.042589</td>
      <td id="T_38a4a_row3_col7" class="data row3 col7" >-0.225363</td>
      <td id="T_38a4a_row3_col8" class="data row3 col8" >-0.007706</td>
      <td id="T_38a4a_row3_col9" class="data row3 col9" >-0.207007</td>
      <td id="T_38a4a_row3_col10" class="data row3 col10" >-0.019627</td>
      <td id="T_38a4a_row3_col11" class="data row3 col11" >0.006294</td>
      <td id="T_38a4a_row3_col12" class="data row3 col12" >-0.059554</td>
      <td id="T_38a4a_row3_col13" class="data row3 col13" >-0.189477</td>
      <td id="T_38a4a_row3_col14" class="data row3 col14" >-0.159722</td>
      <td id="T_38a4a_row3_col15" class="data row3 col15" >-0.110106</td>
      <td id="T_38a4a_row3_col16" class="data row3 col16" >-0.056374</td>
      <td id="T_38a4a_row3_col17" class="data row3 col17" >-0.047809</td>
      <td id="T_38a4a_row3_col18" class="data row3 col18" >0.024890</td>
      <td id="T_38a4a_row3_col19" class="data row3 col19" >-0.032065</td>
      <td id="T_38a4a_row3_col20" class="data row3 col20" >0.023055</td>
      <td id="T_38a4a_row3_col21" class="data row3 col21" >-0.083757</td>
      <td id="T_38a4a_row3_col22" class="data row3 col22" >0.004329</td>
      <td id="T_38a4a_row3_col23" class="data row3 col23" >0.004747</td>
      <td id="T_38a4a_row3_col24" class="data row3 col24" >-0.187558</td>
    </tr>
    <tr>
      <th id="T_38a4a_level0_row4" class="row_heading level0 row4" >Age</th>
      <td id="T_38a4a_row4_col0" class="data row4 col0" >0.004806</td>
      <td id="T_38a4a_row4_col1" class="data row4 col1" >0.022929</td>
      <td id="T_38a4a_row4_col2" class="data row4 col2" >0.008921</td>
      <td id="T_38a4a_row4_col3" class="data row4 col3" >-0.281821</td>
      <td id="T_38a4a_row4_col4" class="data row4 col4" >1.000000</td>
      <td id="T_38a4a_row4_col5" class="data row4 col5" >-0.048593</td>
      <td id="T_38a4a_row4_col6" class="data row4 col6" >-0.117423</td>
      <td id="T_38a4a_row4_col7" class="data row4 col7" >0.099838</td>
      <td id="T_38a4a_row4_col8" class="data row4 col8" >0.017470</td>
      <td id="T_38a4a_row4_col9" class="data row4 col9" >0.038038</td>
      <td id="T_38a4a_row4_col10" class="data row4 col10" >0.024461</td>
      <td id="T_38a4a_row4_col11" class="data row4 col11" >-0.001558</td>
      <td id="T_38a4a_row4_col12" class="data row4 col12" >0.022920</td>
      <td id="T_38a4a_row4_col13" class="data row4 col13" >0.208681</td>
      <td id="T_38a4a_row4_col14" class="data row4 col14" >0.160302</td>
      <td id="T_38a4a_row4_col15" class="data row4 col15" >0.076380</td>
      <td id="T_38a4a_row4_col16" class="data row4 col16" >0.057123</td>
      <td id="T_38a4a_row4_col17" class="data row4 col17" >0.040498</td>
      <td id="T_38a4a_row4_col18" class="data row4 col18" >-0.047619</td>
      <td id="T_38a4a_row4_col19" class="data row4 col19" >0.035003</td>
      <td id="T_38a4a_row4_col20" class="data row4 col20" >-0.049899</td>
      <td id="T_38a4a_row4_col21" class="data row4 col21" >0.053493</td>
      <td id="T_38a4a_row4_col22" class="data row4 col22" >-0.010150</td>
      <td id="T_38a4a_row4_col23" class="data row4 col23" >-0.012147</td>
      <td id="T_38a4a_row4_col24" class="data row4 col24" >0.137040</td>
    </tr>
    <tr>
      <th id="T_38a4a_level0_row5" class="row_heading level0 row5" >Type of Travel</th>
      <td id="T_38a4a_row5_col0" class="data row5 col0" >0.000707</td>
      <td id="T_38a4a_row5_col1" class="data row5 col1" >0.000576</td>
      <td id="T_38a4a_row5_col2" class="data row5 col2" >0.006808</td>
      <td id="T_38a4a_row5_col3" class="data row5 col3" >-0.308268</td>
      <td id="T_38a4a_row5_col4" class="data row5 col4" >-0.048593</td>
      <td id="T_38a4a_row5_col5" class="data row5 col5" >1.000000</td>
      <td id="T_38a4a_row5_col6" class="data row5 col6" >0.487001</td>
      <td id="T_38a4a_row5_col7" class="data row5 col7" >-0.267642</td>
      <td id="T_38a4a_row5_col8" class="data row5 col8" >-0.104879</td>
      <td id="T_38a4a_row5_col9" class="data row5 col9" >0.259829</td>
      <td id="T_38a4a_row5_col10" class="data row5 col10" >-0.133399</td>
      <td id="T_38a4a_row5_col11" class="data row5 col11" >-0.030802</td>
      <td id="T_38a4a_row5_col12" class="data row5 col12" >-0.063124</td>
      <td id="T_38a4a_row5_col13" class="data row5 col13" >-0.224620</td>
      <td id="T_38a4a_row5_col14" class="data row5 col14" >-0.123994</td>
      <td id="T_38a4a_row5_col15" class="data row5 col15" >-0.147978</td>
      <td id="T_38a4a_row5_col16" class="data row5 col16" >-0.056468</td>
      <td id="T_38a4a_row5_col17" class="data row5 col17" >-0.138680</td>
      <td id="T_38a4a_row5_col18" class="data row5 col18" >-0.031355</td>
      <td id="T_38a4a_row5_col19" class="data row5 col19" >0.017043</td>
      <td id="T_38a4a_row5_col20" class="data row5 col20" >-0.022492</td>
      <td id="T_38a4a_row5_col21" class="data row5 col21" >-0.078767</td>
      <td id="T_38a4a_row5_col22" class="data row5 col22" >-0.006046</td>
      <td id="T_38a4a_row5_col23" class="data row5 col23" >-0.005683</td>
      <td id="T_38a4a_row5_col24" class="data row5 col24" >-0.448995</td>
    </tr>
    <tr>
      <th id="T_38a4a_level0_row6" class="row_heading level0 row6" >Class</th>
      <td id="T_38a4a_row6_col0" class="data row6 col0" >-0.001449</td>
      <td id="T_38a4a_row6_col1" class="data row6 col1" >-0.104340</td>
      <td id="T_38a4a_row6_col2" class="data row6 col2" >-0.012840</td>
      <td id="T_38a4a_row6_col3" class="data row6 col3" >0.042589</td>
      <td id="T_38a4a_row6_col4" class="data row6 col4" >-0.117423</td>
      <td id="T_38a4a_row6_col5" class="data row6 col5" >0.487001</td>
      <td id="T_38a4a_row6_col6" class="data row6 col6" >1.000000</td>
      <td id="T_38a4a_row6_col7" class="data row6 col7" >-0.427509</td>
      <td id="T_38a4a_row6_col8" class="data row6 col8" >-0.023046</td>
      <td id="T_38a4a_row6_col9" class="data row6 col9" >0.089793</td>
      <td id="T_38a4a_row6_col10" class="data row6 col10" >-0.094323</td>
      <td id="T_38a4a_row6_col11" class="data row6 col11" >-0.004532</td>
      <td id="T_38a4a_row6_col12" class="data row6 col12" >-0.076834</td>
      <td id="T_38a4a_row6_col13" class="data row6 col13" >-0.296949</td>
      <td id="T_38a4a_row6_col14" class="data row6 col14" >-0.209955</td>
      <td id="T_38a4a_row6_col15" class="data row6 col15" >-0.178928</td>
      <td id="T_38a4a_row6_col16" class="data row6 col16" >-0.207922</td>
      <td id="T_38a4a_row6_col17" class="data row6 col17" >-0.197331</td>
      <td id="T_38a4a_row6_col18" class="data row6 col18" >-0.164016</td>
      <td id="T_38a4a_row6_col19" class="data row6 col19" >-0.157084</td>
      <td id="T_38a4a_row6_col20" class="data row6 col20" >-0.158457</td>
      <td id="T_38a4a_row6_col21" class="data row6 col21" >-0.125933</td>
      <td id="T_38a4a_row6_col22" class="data row6 col22" >0.010105</td>
      <td id="T_38a4a_row6_col23" class="data row6 col23" >0.014701</td>
      <td id="T_38a4a_row6_col24" class="data row6 col24" >-0.449466</td>
    </tr>
    <tr>
      <th id="T_38a4a_level0_row7" class="row_heading level0 row7" >Flight Distance</th>
      <td id="T_38a4a_row7_col0" class="data row7 col0" >0.002973</td>
      <td id="T_38a4a_row7_col1" class="data row7 col1" >0.095184</td>
      <td id="T_38a4a_row7_col2" class="data row7 col2" >0.006079</td>
      <td id="T_38a4a_row7_col3" class="data row7 col3" >-0.225363</td>
      <td id="T_38a4a_row7_col4" class="data row7 col4" >0.099838</td>
      <td id="T_38a4a_row7_col5" class="data row7 col5" >-0.267642</td>
      <td id="T_38a4a_row7_col6" class="data row7 col6" >-0.427509</td>
      <td id="T_38a4a_row7_col7" class="data row7 col7" >1.000000</td>
      <td id="T_38a4a_row7_col8" class="data row7 col8" >0.007050</td>
      <td id="T_38a4a_row7_col9" class="data row7 col9" >-0.019908</td>
      <td id="T_38a4a_row7_col10" class="data row7 col10" >0.065697</td>
      <td id="T_38a4a_row7_col11" class="data row7 col11" >0.004732</td>
      <td id="T_38a4a_row7_col12" class="data row7 col12" >0.056957</td>
      <td id="T_38a4a_row7_col13" class="data row7 col13" >0.215191</td>
      <td id="T_38a4a_row7_col14" class="data row7 col14" >0.157517</td>
      <td id="T_38a4a_row7_col15" class="data row7 col15" >0.128645</td>
      <td id="T_38a4a_row7_col16" class="data row7 col16" >0.109540</td>
      <td id="T_38a4a_row7_col17" class="data row7 col17" >0.133839</td>
      <td id="T_38a4a_row7_col18" class="data row7 col18" >0.063222</td>
      <td id="T_38a4a_row7_col19" class="data row7 col19" >0.073224</td>
      <td id="T_38a4a_row7_col20" class="data row7 col20" >0.057430</td>
      <td id="T_38a4a_row7_col21" class="data row7 col21" >0.093121</td>
      <td id="T_38a4a_row7_col22" class="data row7 col22" >0.001906</td>
      <td id="T_38a4a_row7_col23" class="data row7 col23" >-0.002426</td>
      <td id="T_38a4a_row7_col24" class="data row7 col24" >0.298915</td>
    </tr>
    <tr>
      <th id="T_38a4a_level0_row8" class="row_heading level0 row8" >Inflight wifi service</th>
      <td id="T_38a4a_row8_col0" class="data row8 col0" >-0.002523</td>
      <td id="T_38a4a_row8_col1" class="data row8 col1" >-0.021338</td>
      <td id="T_38a4a_row8_col2" class="data row8 col2" >0.008964</td>
      <td id="T_38a4a_row8_col3" class="data row8 col3" >-0.007706</td>
      <td id="T_38a4a_row8_col4" class="data row8 col4" >0.017470</td>
      <td id="T_38a4a_row8_col5" class="data row8 col5" >-0.104879</td>
      <td id="T_38a4a_row8_col6" class="data row8 col6" >-0.023046</td>
      <td id="T_38a4a_row8_col7" class="data row8 col7" >0.007050</td>
      <td id="T_38a4a_row8_col8" class="data row8 col8" >1.000000</td>
      <td id="T_38a4a_row8_col9" class="data row8 col9" >0.343758</td>
      <td id="T_38a4a_row8_col10" class="data row8 col10" >0.715848</td>
      <td id="T_38a4a_row8_col11" class="data row8 col11" >0.336127</td>
      <td id="T_38a4a_row8_col12" class="data row8 col12" >0.134603</td>
      <td id="T_38a4a_row8_col13" class="data row8 col13" >0.457002</td>
      <td id="T_38a4a_row8_col14" class="data row8 col14" >0.122617</td>
      <td id="T_38a4a_row8_col15" class="data row8 col15" >0.209513</td>
      <td id="T_38a4a_row8_col16" class="data row8 col16" >0.121484</td>
      <td id="T_38a4a_row8_col17" class="data row8 col17" >0.160485</td>
      <td id="T_38a4a_row8_col18" class="data row8 col18" >0.121060</td>
      <td id="T_38a4a_row8_col19" class="data row8 col19" >0.043178</td>
      <td id="T_38a4a_row8_col20" class="data row8 col20" >0.110626</td>
      <td id="T_38a4a_row8_col21" class="data row8 col21" >0.132652</td>
      <td id="T_38a4a_row8_col22" class="data row8 col22" >-0.017451</td>
      <td id="T_38a4a_row8_col23" class="data row8 col23" >-0.019095</td>
      <td id="T_38a4a_row8_col24" class="data row8 col24" >0.284163</td>
    </tr>
    <tr>
      <th id="T_38a4a_level0_row9" class="row_heading level0 row9" >Departure/Arrival time convenient</th>
      <td id="T_38a4a_row9_col0" class="data row9 col0" >0.000750</td>
      <td id="T_38a4a_row9_col1" class="data row9 col1" >-0.001714</td>
      <td id="T_38a4a_row9_col2" class="data row9 col2" >0.008846</td>
      <td id="T_38a4a_row9_col3" class="data row9 col3" >-0.207007</td>
      <td id="T_38a4a_row9_col4" class="data row9 col4" >0.038038</td>
      <td id="T_38a4a_row9_col5" class="data row9 col5" >0.259829</td>
      <td id="T_38a4a_row9_col6" class="data row9 col6" >0.089793</td>
      <td id="T_38a4a_row9_col7" class="data row9 col7" >-0.019908</td>
      <td id="T_38a4a_row9_col8" class="data row9 col8" >0.343758</td>
      <td id="T_38a4a_row9_col9" class="data row9 col9" >1.000000</td>
      <td id="T_38a4a_row9_col10" class="data row9 col10" >0.437021</td>
      <td id="T_38a4a_row9_col11" class="data row9 col11" >0.444601</td>
      <td id="T_38a4a_row9_col12" class="data row9 col12" >0.005189</td>
      <td id="T_38a4a_row9_col13" class="data row9 col13" >0.069990</td>
      <td id="T_38a4a_row9_col14" class="data row9 col14" >0.011416</td>
      <td id="T_38a4a_row9_col15" class="data row9 col15" >-0.004683</td>
      <td id="T_38a4a_row9_col16" class="data row9 col16" >0.068604</td>
      <td id="T_38a4a_row9_col17" class="data row9 col17" >0.012461</td>
      <td id="T_38a4a_row9_col18" class="data row9 col18" >0.071901</td>
      <td id="T_38a4a_row9_col19" class="data row9 col19" >0.093329</td>
      <td id="T_38a4a_row9_col20" class="data row9 col20" >0.073227</td>
      <td id="T_38a4a_row9_col21" class="data row9 col21" >0.014337</td>
      <td id="T_38a4a_row9_col22" class="data row9 col22" >0.000791</td>
      <td id="T_38a4a_row9_col23" class="data row9 col23" >-0.000864</td>
      <td id="T_38a4a_row9_col24" class="data row9 col24" >-0.051718</td>
    </tr>
    <tr>
      <th id="T_38a4a_level0_row10" class="row_heading level0 row10" >Ease of Online booking</th>
      <td id="T_38a4a_row10_col0" class="data row10 col0" >0.001941</td>
      <td id="T_38a4a_row10_col1" class="data row10 col1" >0.014119</td>
      <td id="T_38a4a_row10_col2" class="data row10 col2" >0.007166</td>
      <td id="T_38a4a_row10_col3" class="data row10 col3" >-0.019627</td>
      <td id="T_38a4a_row10_col4" class="data row10 col4" >0.024461</td>
      <td id="T_38a4a_row10_col5" class="data row10 col5" >-0.133399</td>
      <td id="T_38a4a_row10_col6" class="data row10 col6" >-0.094323</td>
      <td id="T_38a4a_row10_col7" class="data row10 col7" >0.065697</td>
      <td id="T_38a4a_row10_col8" class="data row10 col8" >0.715848</td>
      <td id="T_38a4a_row10_col9" class="data row10 col9" >0.437021</td>
      <td id="T_38a4a_row10_col10" class="data row10 col10" >1.000000</td>
      <td id="T_38a4a_row10_col11" class="data row10 col11" >0.458746</td>
      <td id="T_38a4a_row10_col12" class="data row10 col12" >0.031940</td>
      <td id="T_38a4a_row10_col13" class="data row10 col13" >0.404093</td>
      <td id="T_38a4a_row10_col14" class="data row10 col14" >0.030021</td>
      <td id="T_38a4a_row10_col15" class="data row10 col15" >0.047185</td>
      <td id="T_38a4a_row10_col16" class="data row10 col16" >0.038759</td>
      <td id="T_38a4a_row10_col17" class="data row10 col17" >0.107431</td>
      <td id="T_38a4a_row10_col18" class="data row10 col18" >0.038851</td>
      <td id="T_38a4a_row10_col19" class="data row10 col19" >0.010957</td>
      <td id="T_38a4a_row10_col20" class="data row10 col20" >0.035330</td>
      <td id="T_38a4a_row10_col21" class="data row10 col21" >0.016192</td>
      <td id="T_38a4a_row10_col22" class="data row10 col22" >-0.006292</td>
      <td id="T_38a4a_row10_col23" class="data row10 col23" >-0.007984</td>
      <td id="T_38a4a_row10_col24" class="data row10 col24" >0.171507</td>
    </tr>
    <tr>
      <th id="T_38a4a_level0_row11" class="row_heading level0 row11" >Gate location</th>
      <td id="T_38a4a_row11_col0" class="data row11 col0" >0.005007</td>
      <td id="T_38a4a_row11_col1" class="data row11 col1" >-0.000427</td>
      <td id="T_38a4a_row11_col2" class="data row11 col2" >0.000213</td>
      <td id="T_38a4a_row11_col3" class="data row11 col3" >0.006294</td>
      <td id="T_38a4a_row11_col4" class="data row11 col4" >-0.001558</td>
      <td id="T_38a4a_row11_col5" class="data row11 col5" >-0.030802</td>
      <td id="T_38a4a_row11_col6" class="data row11 col6" >-0.004532</td>
      <td id="T_38a4a_row11_col7" class="data row11 col7" >0.004732</td>
      <td id="T_38a4a_row11_col8" class="data row11 col8" >0.336127</td>
      <td id="T_38a4a_row11_col9" class="data row11 col9" >0.444601</td>
      <td id="T_38a4a_row11_col10" class="data row11 col10" >0.458746</td>
      <td id="T_38a4a_row11_col11" class="data row11 col11" >1.000000</td>
      <td id="T_38a4a_row11_col12" class="data row11 col12" >-0.001170</td>
      <td id="T_38a4a_row11_col13" class="data row11 col13" >0.001451</td>
      <td id="T_38a4a_row11_col14" class="data row11 col14" >0.003383</td>
      <td id="T_38a4a_row11_col15" class="data row11 col15" >0.003564</td>
      <td id="T_38a4a_row11_col16" class="data row11 col16" >-0.028532</td>
      <td id="T_38a4a_row11_col17" class="data row11 col17" >-0.005868</td>
      <td id="T_38a4a_row11_col18" class="data row11 col18" >0.002421</td>
      <td id="T_38a4a_row11_col19" class="data row11 col19" >-0.035451</td>
      <td id="T_38a4a_row11_col20" class="data row11 col20" >0.001742</td>
      <td id="T_38a4a_row11_col21" class="data row11 col21" >-0.004015</td>
      <td id="T_38a4a_row11_col22" class="data row11 col22" >0.005533</td>
      <td id="T_38a4a_row11_col23" class="data row11 col23" >0.005143</td>
      <td id="T_38a4a_row11_col24" class="data row11 col24" >0.000449</td>
    </tr>
    <tr>
      <th id="T_38a4a_level0_row12" class="row_heading level0 row12" >Food and drink</th>
      <td id="T_38a4a_row12_col0" class="data row12 col0" >-0.001974</td>
      <td id="T_38a4a_row12_col1" class="data row12 col1" >0.001254</td>
      <td id="T_38a4a_row12_col2" class="data row12 col2" >0.005707</td>
      <td id="T_38a4a_row12_col3" class="data row12 col3" >-0.059554</td>
      <td id="T_38a4a_row12_col4" class="data row12 col4" >0.022920</td>
      <td id="T_38a4a_row12_col5" class="data row12 col5" >-0.063124</td>
      <td id="T_38a4a_row12_col6" class="data row12 col6" >-0.076834</td>
      <td id="T_38a4a_row12_col7" class="data row12 col7" >0.056957</td>
      <td id="T_38a4a_row12_col8" class="data row12 col8" >0.134603</td>
      <td id="T_38a4a_row12_col9" class="data row12 col9" >0.005189</td>
      <td id="T_38a4a_row12_col10" class="data row12 col10" >0.031940</td>
      <td id="T_38a4a_row12_col11" class="data row12 col11" >-0.001170</td>
      <td id="T_38a4a_row12_col12" class="data row12 col12" >1.000000</td>
      <td id="T_38a4a_row12_col13" class="data row12 col13" >0.234492</td>
      <td id="T_38a4a_row12_col14" class="data row12 col14" >0.574561</td>
      <td id="T_38a4a_row12_col15" class="data row12 col15" >0.622374</td>
      <td id="T_38a4a_row12_col16" class="data row12 col16" >0.058999</td>
      <td id="T_38a4a_row12_col17" class="data row12 col17" >0.032415</td>
      <td id="T_38a4a_row12_col18" class="data row12 col18" >0.034811</td>
      <td id="T_38a4a_row12_col19" class="data row12 col19" >0.087055</td>
      <td id="T_38a4a_row12_col20" class="data row12 col20" >0.034077</td>
      <td id="T_38a4a_row12_col21" class="data row12 col21" >0.657648</td>
      <td id="T_38a4a_row12_col22" class="data row12 col22" >-0.029983</td>
      <td id="T_38a4a_row12_col23" class="data row12 col23" >-0.032524</td>
      <td id="T_38a4a_row12_col24" class="data row12 col24" >0.209659</td>
    </tr>
    <tr>
      <th id="T_38a4a_level0_row13" class="row_heading level0 row13" >Online boarding</th>
      <td id="T_38a4a_row13_col0" class="data row13 col0" >0.001126</td>
      <td id="T_38a4a_row13_col1" class="data row13 col1" >0.055394</td>
      <td id="T_38a4a_row13_col2" class="data row13 col2" >-0.042151</td>
      <td id="T_38a4a_row13_col3" class="data row13 col3" >-0.189477</td>
      <td id="T_38a4a_row13_col4" class="data row13 col4" >0.208681</td>
      <td id="T_38a4a_row13_col5" class="data row13 col5" >-0.224620</td>
      <td id="T_38a4a_row13_col6" class="data row13 col6" >-0.296949</td>
      <td id="T_38a4a_row13_col7" class="data row13 col7" >0.215191</td>
      <td id="T_38a4a_row13_col8" class="data row13 col8" >0.457002</td>
      <td id="T_38a4a_row13_col9" class="data row13 col9" >0.069990</td>
      <td id="T_38a4a_row13_col10" class="data row13 col10" >0.404093</td>
      <td id="T_38a4a_row13_col11" class="data row13 col11" >0.001451</td>
      <td id="T_38a4a_row13_col12" class="data row13 col12" >0.234492</td>
      <td id="T_38a4a_row13_col13" class="data row13 col13" >1.000000</td>
      <td id="T_38a4a_row13_col14" class="data row13 col14" >0.420067</td>
      <td id="T_38a4a_row13_col15" class="data row13 col15" >0.285194</td>
      <td id="T_38a4a_row13_col16" class="data row13 col16" >0.155345</td>
      <td id="T_38a4a_row13_col17" class="data row13 col17" >0.123780</td>
      <td id="T_38a4a_row13_col18" class="data row13 col18" >0.083299</td>
      <td id="T_38a4a_row13_col19" class="data row13 col19" >0.204208</td>
      <td id="T_38a4a_row13_col20" class="data row13 col20" >0.074390</td>
      <td id="T_38a4a_row13_col21" class="data row13 col21" >0.331498</td>
      <td id="T_38a4a_row13_col22" class="data row13 col22" >-0.018515</td>
      <td id="T_38a4a_row13_col23" class="data row13 col23" >-0.021949</td>
      <td id="T_38a4a_row13_col24" class="data row13 col24" >0.503447</td>
    </tr>
    <tr>
      <th id="T_38a4a_level0_row14" class="row_heading level0 row14" >Seat comfort</th>
      <td id="T_38a4a_row14_col0" class="data row14 col0" >0.000343</td>
      <td id="T_38a4a_row14_col1" class="data row14 col1" >0.053091</td>
      <td id="T_38a4a_row14_col2" class="data row14 col2" >-0.026643</td>
      <td id="T_38a4a_row14_col3" class="data row14 col3" >-0.159722</td>
      <td id="T_38a4a_row14_col4" class="data row14 col4" >0.160302</td>
      <td id="T_38a4a_row14_col5" class="data row14 col5" >-0.123994</td>
      <td id="T_38a4a_row14_col6" class="data row14 col6" >-0.209955</td>
      <td id="T_38a4a_row14_col7" class="data row14 col7" >0.157517</td>
      <td id="T_38a4a_row14_col8" class="data row14 col8" >0.122617</td>
      <td id="T_38a4a_row14_col9" class="data row14 col9" >0.011416</td>
      <td id="T_38a4a_row14_col10" class="data row14 col10" >0.030021</td>
      <td id="T_38a4a_row14_col11" class="data row14 col11" >0.003383</td>
      <td id="T_38a4a_row14_col12" class="data row14 col12" >0.574561</td>
      <td id="T_38a4a_row14_col13" class="data row14 col13" >0.420067</td>
      <td id="T_38a4a_row14_col14" class="data row14 col14" >1.000000</td>
      <td id="T_38a4a_row14_col15" class="data row14 col15" >0.610614</td>
      <td id="T_38a4a_row14_col16" class="data row14 col16" >0.132030</td>
      <td id="T_38a4a_row14_col17" class="data row14 col17" >0.105447</td>
      <td id="T_38a4a_row14_col18" class="data row14 col18" >0.074553</td>
      <td id="T_38a4a_row14_col19" class="data row14 col19" >0.191545</td>
      <td id="T_38a4a_row14_col20" class="data row14 col20" >0.069193</td>
      <td id="T_38a4a_row14_col21" class="data row14 col21" >0.678478</td>
      <td id="T_38a4a_row14_col22" class="data row14 col22" >-0.027323</td>
      <td id="T_38a4a_row14_col23" class="data row14 col23" >-0.029900</td>
      <td id="T_38a4a_row14_col24" class="data row14 col24" >0.349112</td>
    </tr>
    <tr>
      <th id="T_38a4a_level0_row15" class="row_heading level0 row15" >Inflight entertainment</th>
      <td id="T_38a4a_row15_col0" class="data row15 col0" >0.001474</td>
      <td id="T_38a4a_row15_col1" class="data row15 col1" >0.002592</td>
      <td id="T_38a4a_row15_col2" class="data row15 col2" >0.006071</td>
      <td id="T_38a4a_row15_col3" class="data row15 col3" >-0.110106</td>
      <td id="T_38a4a_row15_col4" class="data row15 col4" >0.076380</td>
      <td id="T_38a4a_row15_col5" class="data row15 col5" >-0.147978</td>
      <td id="T_38a4a_row15_col6" class="data row15 col6" >-0.178928</td>
      <td id="T_38a4a_row15_col7" class="data row15 col7" >0.128645</td>
      <td id="T_38a4a_row15_col8" class="data row15 col8" >0.209513</td>
      <td id="T_38a4a_row15_col9" class="data row15 col9" >-0.004683</td>
      <td id="T_38a4a_row15_col10" class="data row15 col10" >0.047185</td>
      <td id="T_38a4a_row15_col11" class="data row15 col11" >0.003564</td>
      <td id="T_38a4a_row15_col12" class="data row15 col12" >0.622374</td>
      <td id="T_38a4a_row15_col13" class="data row15 col13" >0.285194</td>
      <td id="T_38a4a_row15_col14" class="data row15 col14" >0.610614</td>
      <td id="T_38a4a_row15_col15" class="data row15 col15" >1.000000</td>
      <td id="T_38a4a_row15_col16" class="data row15 col16" >0.420352</td>
      <td id="T_38a4a_row15_col17" class="data row15 col17" >0.299850</td>
      <td id="T_38a4a_row15_col18" class="data row15 col18" >0.378361</td>
      <td id="T_38a4a_row15_col19" class="data row15 col19" >0.120812</td>
      <td id="T_38a4a_row15_col20" class="data row15 col20" >0.405247</td>
      <td id="T_38a4a_row15_col21" class="data row15 col21" >0.691735</td>
      <td id="T_38a4a_row15_col22" class="data row15 col22" >-0.027691</td>
      <td id="T_38a4a_row15_col23" class="data row15 col23" >-0.030703</td>
      <td id="T_38a4a_row15_col24" class="data row15 col24" >0.398203</td>
    </tr>
    <tr>
      <th id="T_38a4a_level0_row16" class="row_heading level0 row16" >On-board service</th>
      <td id="T_38a4a_row16_col0" class="data row16 col0" >0.001046</td>
      <td id="T_38a4a_row16_col1" class="data row16 col1" >0.055255</td>
      <td id="T_38a4a_row16_col2" class="data row16 col2" >0.008019</td>
      <td id="T_38a4a_row16_col3" class="data row16 col3" >-0.056374</td>
      <td id="T_38a4a_row16_col4" class="data row16 col4" >0.057123</td>
      <td id="T_38a4a_row16_col5" class="data row16 col5" >-0.056468</td>
      <td id="T_38a4a_row16_col6" class="data row16 col6" >-0.207922</td>
      <td id="T_38a4a_row16_col7" class="data row16 col7" >0.109540</td>
      <td id="T_38a4a_row16_col8" class="data row16 col8" >0.121484</td>
      <td id="T_38a4a_row16_col9" class="data row16 col9" >0.068604</td>
      <td id="T_38a4a_row16_col10" class="data row16 col10" >0.038759</td>
      <td id="T_38a4a_row16_col11" class="data row16 col11" >-0.028532</td>
      <td id="T_38a4a_row16_col12" class="data row16 col12" >0.058999</td>
      <td id="T_38a4a_row16_col13" class="data row16 col13" >0.155345</td>
      <td id="T_38a4a_row16_col14" class="data row16 col14" >0.132030</td>
      <td id="T_38a4a_row16_col15" class="data row16 col15" >0.420352</td>
      <td id="T_38a4a_row16_col16" class="data row16 col16" >1.000000</td>
      <td id="T_38a4a_row16_col17" class="data row16 col17" >0.355657</td>
      <td id="T_38a4a_row16_col18" class="data row16 col18" >0.519252</td>
      <td id="T_38a4a_row16_col19" class="data row16 col19" >0.243852</td>
      <td id="T_38a4a_row16_col20" class="data row16 col20" >0.550725</td>
      <td id="T_38a4a_row16_col21" class="data row16 col21" >0.123236</td>
      <td id="T_38a4a_row16_col22" class="data row16 col22" >-0.031474</td>
      <td id="T_38a4a_row16_col23" class="data row16 col23" >-0.035227</td>
      <td id="T_38a4a_row16_col24" class="data row16 col24" >0.322450</td>
    </tr>
    <tr>
      <th id="T_38a4a_level0_row17" class="row_heading level0 row17" >Leg room service</th>
      <td id="T_38a4a_row17_col0" class="data row17 col0" >0.004061</td>
      <td id="T_38a4a_row17_col1" class="data row17 col1" >0.044459</td>
      <td id="T_38a4a_row17_col2" class="data row17 col2" >0.031842</td>
      <td id="T_38a4a_row17_col3" class="data row17 col3" >-0.047809</td>
      <td id="T_38a4a_row17_col4" class="data row17 col4" >0.040498</td>
      <td id="T_38a4a_row17_col5" class="data row17 col5" >-0.138680</td>
      <td id="T_38a4a_row17_col6" class="data row17 col6" >-0.197331</td>
      <td id="T_38a4a_row17_col7" class="data row17 col7" >0.133839</td>
      <td id="T_38a4a_row17_col8" class="data row17 col8" >0.160485</td>
      <td id="T_38a4a_row17_col9" class="data row17 col9" >0.012461</td>
      <td id="T_38a4a_row17_col10" class="data row17 col10" >0.107431</td>
      <td id="T_38a4a_row17_col11" class="data row17 col11" >-0.005868</td>
      <td id="T_38a4a_row17_col12" class="data row17 col12" >0.032415</td>
      <td id="T_38a4a_row17_col13" class="data row17 col13" >0.123780</td>
      <td id="T_38a4a_row17_col14" class="data row17 col14" >0.105447</td>
      <td id="T_38a4a_row17_col15" class="data row17 col15" >0.299850</td>
      <td id="T_38a4a_row17_col16" class="data row17 col16" >0.355657</td>
      <td id="T_38a4a_row17_col17" class="data row17 col17" >1.000000</td>
      <td id="T_38a4a_row17_col18" class="data row17 col18" >0.369674</td>
      <td id="T_38a4a_row17_col19" class="data row17 col19" >0.153079</td>
      <td id="T_38a4a_row17_col20" class="data row17 col20" >0.368925</td>
      <td id="T_38a4a_row17_col21" class="data row17 col21" >0.096401</td>
      <td id="T_38a4a_row17_col22" class="data row17 col22" >0.014336</td>
      <td id="T_38a4a_row17_col23" class="data row17 col23" >0.011843</td>
      <td id="T_38a4a_row17_col24" class="data row17 col24" >0.313182</td>
    </tr>
    <tr>
      <th id="T_38a4a_level0_row18" class="row_heading level0 row18" >Baggage handling</th>
      <td id="T_38a4a_row18_col0" class="data row18 col0" >-0.000328</td>
      <td id="T_38a4a_row18_col1" class="data row18 col1" >0.075134</td>
      <td id="T_38a4a_row18_col2" class="data row18 col2" >0.037333</td>
      <td id="T_38a4a_row18_col3" class="data row18 col3" >0.024890</td>
      <td id="T_38a4a_row18_col4" class="data row18 col4" >-0.047619</td>
      <td id="T_38a4a_row18_col5" class="data row18 col5" >-0.031355</td>
      <td id="T_38a4a_row18_col6" class="data row18 col6" >-0.164016</td>
      <td id="T_38a4a_row18_col7" class="data row18 col7" >0.063222</td>
      <td id="T_38a4a_row18_col8" class="data row18 col8" >0.121060</td>
      <td id="T_38a4a_row18_col9" class="data row18 col9" >0.071901</td>
      <td id="T_38a4a_row18_col10" class="data row18 col10" >0.038851</td>
      <td id="T_38a4a_row18_col11" class="data row18 col11" >0.002421</td>
      <td id="T_38a4a_row18_col12" class="data row18 col12" >0.034811</td>
      <td id="T_38a4a_row18_col13" class="data row18 col13" >0.083299</td>
      <td id="T_38a4a_row18_col14" class="data row18 col14" >0.074553</td>
      <td id="T_38a4a_row18_col15" class="data row18 col15" >0.378361</td>
      <td id="T_38a4a_row18_col16" class="data row18 col16" >0.519252</td>
      <td id="T_38a4a_row18_col17" class="data row18 col17" >0.369674</td>
      <td id="T_38a4a_row18_col18" class="data row18 col18" >1.000000</td>
      <td id="T_38a4a_row18_col19" class="data row18 col19" >0.233326</td>
      <td id="T_38a4a_row18_col20" class="data row18 col20" >0.628944</td>
      <td id="T_38a4a_row18_col21" class="data row18 col21" >0.095783</td>
      <td id="T_38a4a_row18_col22" class="data row18 col22" >-0.005683</td>
      <td id="T_38a4a_row18_col23" class="data row18 col23" >-0.008542</td>
      <td id="T_38a4a_row18_col24" class="data row18 col24" >0.247819</td>
    </tr>
    <tr>
      <th id="T_38a4a_level0_row19" class="row_heading level0 row19" >Checkin service</th>
      <td id="T_38a4a_row19_col0" class="data row19 col0" >-0.004205</td>
      <td id="T_38a4a_row19_col1" class="data row19 col1" >0.079346</td>
      <td id="T_38a4a_row19_col2" class="data row19 col2" >0.010438</td>
      <td id="T_38a4a_row19_col3" class="data row19 col3" >-0.032065</td>
      <td id="T_38a4a_row19_col4" class="data row19 col4" >0.035003</td>
      <td id="T_38a4a_row19_col5" class="data row19 col5" >0.017043</td>
      <td id="T_38a4a_row19_col6" class="data row19 col6" >-0.157084</td>
      <td id="T_38a4a_row19_col7" class="data row19 col7" >0.073224</td>
      <td id="T_38a4a_row19_col8" class="data row19 col8" >0.043178</td>
      <td id="T_38a4a_row19_col9" class="data row19 col9" >0.093329</td>
      <td id="T_38a4a_row19_col10" class="data row19 col10" >0.010957</td>
      <td id="T_38a4a_row19_col11" class="data row19 col11" >-0.035451</td>
      <td id="T_38a4a_row19_col12" class="data row19 col12" >0.087055</td>
      <td id="T_38a4a_row19_col13" class="data row19 col13" >0.204208</td>
      <td id="T_38a4a_row19_col14" class="data row19 col14" >0.191545</td>
      <td id="T_38a4a_row19_col15" class="data row19 col15" >0.120812</td>
      <td id="T_38a4a_row19_col16" class="data row19 col16" >0.243852</td>
      <td id="T_38a4a_row19_col17" class="data row19 col17" >0.153079</td>
      <td id="T_38a4a_row19_col18" class="data row19 col18" >0.233326</td>
      <td id="T_38a4a_row19_col19" class="data row19 col19" >1.000000</td>
      <td id="T_38a4a_row19_col20" class="data row19 col20" >0.237256</td>
      <td id="T_38a4a_row19_col21" class="data row19 col21" >0.179431</td>
      <td id="T_38a4a_row19_col22" class="data row19 col22" >-0.018065</td>
      <td id="T_38a4a_row19_col23" class="data row19 col23" >-0.020369</td>
      <td id="T_38a4a_row19_col24" class="data row19 col24" >0.235914</td>
    </tr>
    <tr>
      <th id="T_38a4a_level0_row20" class="row_heading level0 row20" >Inflight service</th>
      <td id="T_38a4a_row20_col0" class="data row20 col0" >-0.000011</td>
      <td id="T_38a4a_row20_col1" class="data row20 col1" >0.079468</td>
      <td id="T_38a4a_row20_col2" class="data row20 col2" >0.038936</td>
      <td id="T_38a4a_row20_col3" class="data row20 col3" >0.023055</td>
      <td id="T_38a4a_row20_col4" class="data row20 col4" >-0.049899</td>
      <td id="T_38a4a_row20_col5" class="data row20 col5" >-0.022492</td>
      <td id="T_38a4a_row20_col6" class="data row20 col6" >-0.158457</td>
      <td id="T_38a4a_row20_col7" class="data row20 col7" >0.057430</td>
      <td id="T_38a4a_row20_col8" class="data row20 col8" >0.110626</td>
      <td id="T_38a4a_row20_col9" class="data row20 col9" >0.073227</td>
      <td id="T_38a4a_row20_col10" class="data row20 col10" >0.035330</td>
      <td id="T_38a4a_row20_col11" class="data row20 col11" >0.001742</td>
      <td id="T_38a4a_row20_col12" class="data row20 col12" >0.034077</td>
      <td id="T_38a4a_row20_col13" class="data row20 col13" >0.074390</td>
      <td id="T_38a4a_row20_col14" class="data row20 col14" >0.069193</td>
      <td id="T_38a4a_row20_col15" class="data row20 col15" >0.405247</td>
      <td id="T_38a4a_row20_col16" class="data row20 col16" >0.550725</td>
      <td id="T_38a4a_row20_col17" class="data row20 col17" >0.368925</td>
      <td id="T_38a4a_row20_col18" class="data row20 col18" >0.628944</td>
      <td id="T_38a4a_row20_col19" class="data row20 col19" >0.237256</td>
      <td id="T_38a4a_row20_col20" class="data row20 col20" >1.000000</td>
      <td id="T_38a4a_row20_col21" class="data row20 col21" >0.088891</td>
      <td id="T_38a4a_row20_col22" class="data row20 col22" >-0.054452</td>
      <td id="T_38a4a_row20_col23" class="data row20 col23" >-0.059196</td>
      <td id="T_38a4a_row20_col24" class="data row20 col24" >0.244852</td>
    </tr>
    <tr>
      <th id="T_38a4a_level0_row21" class="row_heading level0 row21" >Cleanliness</th>
      <td id="T_38a4a_row21_col0" class="data row21 col0" >-0.000978</td>
      <td id="T_38a4a_row21_col1" class="data row21 col1" >0.025313</td>
      <td id="T_38a4a_row21_col2" class="data row21 col2" >0.006439</td>
      <td id="T_38a4a_row21_col3" class="data row21 col3" >-0.083757</td>
      <td id="T_38a4a_row21_col4" class="data row21 col4" >0.053493</td>
      <td id="T_38a4a_row21_col5" class="data row21 col5" >-0.078767</td>
      <td id="T_38a4a_row21_col6" class="data row21 col6" >-0.125933</td>
      <td id="T_38a4a_row21_col7" class="data row21 col7" >0.093121</td>
      <td id="T_38a4a_row21_col8" class="data row21 col8" >0.132652</td>
      <td id="T_38a4a_row21_col9" class="data row21 col9" >0.014337</td>
      <td id="T_38a4a_row21_col10" class="data row21 col10" >0.016192</td>
      <td id="T_38a4a_row21_col11" class="data row21 col11" >-0.004015</td>
      <td id="T_38a4a_row21_col12" class="data row21 col12" >0.657648</td>
      <td id="T_38a4a_row21_col13" class="data row21 col13" >0.331498</td>
      <td id="T_38a4a_row21_col14" class="data row21 col14" >0.678478</td>
      <td id="T_38a4a_row21_col15" class="data row21 col15" >0.691735</td>
      <td id="T_38a4a_row21_col16" class="data row21 col16" >0.123236</td>
      <td id="T_38a4a_row21_col17" class="data row21 col17" >0.096401</td>
      <td id="T_38a4a_row21_col18" class="data row21 col18" >0.095783</td>
      <td id="T_38a4a_row21_col19" class="data row21 col19" >0.179431</td>
      <td id="T_38a4a_row21_col20" class="data row21 col20" >0.088891</td>
      <td id="T_38a4a_row21_col21" class="data row21 col21" >1.000000</td>
      <td id="T_38a4a_row21_col22" class="data row21 col22" >-0.013835</td>
      <td id="T_38a4a_row21_col23" class="data row21 col23" >-0.015774</td>
      <td id="T_38a4a_row21_col24" class="data row21 col24" >0.305050</td>
    </tr>
    <tr>
      <th id="T_38a4a_level0_row22" class="row_heading level0 row22" >Departure Delay in Minutes</th>
      <td id="T_38a4a_row22_col0" class="data row22 col0" >-0.000694</td>
      <td id="T_38a4a_row22_col1" class="data row22 col1" >-0.019321</td>
      <td id="T_38a4a_row22_col2" class="data row22 col2" >0.002534</td>
      <td id="T_38a4a_row22_col3" class="data row22 col3" >0.004329</td>
      <td id="T_38a4a_row22_col4" class="data row22 col4" >-0.010150</td>
      <td id="T_38a4a_row22_col5" class="data row22 col5" >-0.006046</td>
      <td id="T_38a4a_row22_col6" class="data row22 col6" >0.010105</td>
      <td id="T_38a4a_row22_col7" class="data row22 col7" >0.001906</td>
      <td id="T_38a4a_row22_col8" class="data row22 col8" >-0.017451</td>
      <td id="T_38a4a_row22_col9" class="data row22 col9" >0.000791</td>
      <td id="T_38a4a_row22_col10" class="data row22 col10" >-0.006292</td>
      <td id="T_38a4a_row22_col11" class="data row22 col11" >0.005533</td>
      <td id="T_38a4a_row22_col12" class="data row22 col12" >-0.029983</td>
      <td id="T_38a4a_row22_col13" class="data row22 col13" >-0.018515</td>
      <td id="T_38a4a_row22_col14" class="data row22 col14" >-0.027323</td>
      <td id="T_38a4a_row22_col15" class="data row22 col15" >-0.027691</td>
      <td id="T_38a4a_row22_col16" class="data row22 col16" >-0.031474</td>
      <td id="T_38a4a_row22_col17" class="data row22 col17" >0.014336</td>
      <td id="T_38a4a_row22_col18" class="data row22 col18" >-0.005683</td>
      <td id="T_38a4a_row22_col19" class="data row22 col19" >-0.018065</td>
      <td id="T_38a4a_row22_col20" class="data row22 col20" >-0.054452</td>
      <td id="T_38a4a_row22_col21" class="data row22 col21" >-0.013835</td>
      <td id="T_38a4a_row22_col22" class="data row22 col22" >1.000000</td>
      <td id="T_38a4a_row22_col23" class="data row22 col23" >0.965481</td>
      <td id="T_38a4a_row22_col24" class="data row22 col24" >-0.050515</td>
    </tr>
    <tr>
      <th id="T_38a4a_level0_row23" class="row_heading level0 row23" >Arrival Delay in Minutes</th>
      <td id="T_38a4a_row23_col0" class="data row23 col0" >-0.000045</td>
      <td id="T_38a4a_row23_col1" class="data row23 col1" >-0.037254</td>
      <td id="T_38a4a_row23_col2" class="data row23 col2" >0.000396</td>
      <td id="T_38a4a_row23_col3" class="data row23 col3" >0.004747</td>
      <td id="T_38a4a_row23_col4" class="data row23 col4" >-0.012147</td>
      <td id="T_38a4a_row23_col5" class="data row23 col5" >-0.005683</td>
      <td id="T_38a4a_row23_col6" class="data row23 col6" >0.014701</td>
      <td id="T_38a4a_row23_col7" class="data row23 col7" >-0.002426</td>
      <td id="T_38a4a_row23_col8" class="data row23 col8" >-0.019095</td>
      <td id="T_38a4a_row23_col9" class="data row23 col9" >-0.000864</td>
      <td id="T_38a4a_row23_col10" class="data row23 col10" >-0.007984</td>
      <td id="T_38a4a_row23_col11" class="data row23 col11" >0.005143</td>
      <td id="T_38a4a_row23_col12" class="data row23 col12" >-0.032524</td>
      <td id="T_38a4a_row23_col13" class="data row23 col13" >-0.021949</td>
      <td id="T_38a4a_row23_col14" class="data row23 col14" >-0.029900</td>
      <td id="T_38a4a_row23_col15" class="data row23 col15" >-0.030703</td>
      <td id="T_38a4a_row23_col16" class="data row23 col16" >-0.035227</td>
      <td id="T_38a4a_row23_col17" class="data row23 col17" >0.011843</td>
      <td id="T_38a4a_row23_col18" class="data row23 col18" >-0.008542</td>
      <td id="T_38a4a_row23_col19" class="data row23 col19" >-0.020369</td>
      <td id="T_38a4a_row23_col20" class="data row23 col20" >-0.059196</td>
      <td id="T_38a4a_row23_col21" class="data row23 col21" >-0.015774</td>
      <td id="T_38a4a_row23_col22" class="data row23 col22" >0.965481</td>
      <td id="T_38a4a_row23_col23" class="data row23 col23" >1.000000</td>
      <td id="T_38a4a_row23_col24" class="data row23 col24" >-0.057582</td>
    </tr>
    <tr>
      <th id="T_38a4a_level0_row24" class="row_heading level0 row24" >satisfaction</th>
      <td id="T_38a4a_row24_col0" class="data row24 col0" >-0.004552</td>
      <td id="T_38a4a_row24_col1" class="data row24 col1" >0.013680</td>
      <td id="T_38a4a_row24_col2" class="data row24 col2" >0.012356</td>
      <td id="T_38a4a_row24_col3" class="data row24 col3" >-0.187558</td>
      <td id="T_38a4a_row24_col4" class="data row24 col4" >0.137040</td>
      <td id="T_38a4a_row24_col5" class="data row24 col5" >-0.448995</td>
      <td id="T_38a4a_row24_col6" class="data row24 col6" >-0.449466</td>
      <td id="T_38a4a_row24_col7" class="data row24 col7" >0.298915</td>
      <td id="T_38a4a_row24_col8" class="data row24 col8" >0.284163</td>
      <td id="T_38a4a_row24_col9" class="data row24 col9" >-0.051718</td>
      <td id="T_38a4a_row24_col10" class="data row24 col10" >0.171507</td>
      <td id="T_38a4a_row24_col11" class="data row24 col11" >0.000449</td>
      <td id="T_38a4a_row24_col12" class="data row24 col12" >0.209659</td>
      <td id="T_38a4a_row24_col13" class="data row24 col13" >0.503447</td>
      <td id="T_38a4a_row24_col14" class="data row24 col14" >0.349112</td>
      <td id="T_38a4a_row24_col15" class="data row24 col15" >0.398203</td>
      <td id="T_38a4a_row24_col16" class="data row24 col16" >0.322450</td>
      <td id="T_38a4a_row24_col17" class="data row24 col17" >0.313182</td>
      <td id="T_38a4a_row24_col18" class="data row24 col18" >0.247819</td>
      <td id="T_38a4a_row24_col19" class="data row24 col19" >0.235914</td>
      <td id="T_38a4a_row24_col20" class="data row24 col20" >0.244852</td>
      <td id="T_38a4a_row24_col21" class="data row24 col21" >0.305050</td>
      <td id="T_38a4a_row24_col22" class="data row24 col22" >-0.050515</td>
      <td id="T_38a4a_row24_col23" class="data row24 col23" >-0.057582</td>
      <td id="T_38a4a_row24_col24" class="data row24 col24" >1.000000</td>
    </tr>
  </tbody>
</table>




---

### **Step 3: Split data into training set and test set**


```python
# Randomize the dataset
data_randomized = df.sample(frac=1, random_state=1)

# Calculate index for split - take first 80% of the data for test set
training_test_index = round(len(data_randomized) * 0.8)

# Split into training and test sets
training_set = data_randomized[:training_test_index].reset_index(drop=True)
test_set = data_randomized[training_test_index:].reset_index(drop=True)

print(training_set.shape)
print(test_set.shape)
```

    (82875, 25)
    (20719, 25)


Test to see if training set and test set proportions are approximately similar to original data set.


```python
df['satisfaction'].value_counts(normalize=True)
```




    satisfaction
    0.0    0.566606
    1.0    0.433394
    Name: proportion, dtype: float64




```python
training_set['satisfaction'].value_counts(normalize=True)
```




    satisfaction
    0.0    0.566287
    1.0    0.433713
    Name: proportion, dtype: float64




```python
test_set['satisfaction'].value_counts(normalize=True)
```




    satisfaction
    0.0    0.567885
    1.0    0.432115
    Name: proportion, dtype: float64



---

### **Step 4: Apply machine learning methods and performance metrics**

- **Naive Bayes**


```python
trainX = training_set.iloc[:,:-1]
trainy = training_set['satisfaction']

testX = test_set.iloc[:,:-1]

testy = test_set['satisfaction']
NB_model = GaussianNB()
NB_model.fit(trainX, trainy)
y_pred = NB_model.predict(testX)

NB_accuracy = accuracy_score(testy, y_pred)
print(NB_accuracy)
```

    0.8041893913798929



```python
confusion_matrix(testy, y_pred)
```




    array([[9564, 2202],
           [1855, 7098]])



 

Even though the model has pretty good accuracy score, it seems the model is struggle with Fasle Negative.

 

- **Logistic Regression**


```python
from sklearn.linear_model import LogisticRegression
log_reg_model = LogisticRegression(random_state=42)
log_reg_model.fit(trainX, trainy)
y_pred = log_reg_model.predict(testX)
log_reg_accuracy = accuracy_score(testy, y_pred)
print(log_reg_accuracy)
```

    0.6843959650562286


    /opt/anaconda3/lib/python3.12/site-packages/sklearn/linear_model/_logistic.py:469: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(


- **Decision Tree**


```python
from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_model.fit(trainX, trainy)
y_pred = dt_model.predict(testX)
dt_accuracy = accuracy_score(testy, y_pred)
print(dt_accuracy)
```

    0.90873111636662


  

It looks like I could improve the accuracy score by remove some less important feature columns. Based on correlation coefficients, I will drop the least correlated columns to "satisfaction" column: "id", "Gender", "Unnamed: 0", "Gate location", "Departure Delay in Minutes", "Departure/Arrival time convenient", "Arrivael Delay in Minutes"

  


```python
training_set = training_set.drop(columns=['id', 'Unnamed: 0', 'Gender', 'Gate location', 'Departure Delay in Minutes', 
                      'Departure/Arrival time convenient', 'Arrival Delay in Minutes'])
```


```python
test_set = test_set.drop(columns=['id', 'Unnamed: 0', 'Gender', 'Gate location', 'Departure Delay in Minutes', 
                      'Departure/Arrival time convenient', 'Arrival Delay in Minutes'])
```

I will peform all three models again to see if the accuracy score improves.


```python
trainX = training_set.iloc[:,:-1]
trainy = training_set['satisfaction']
testX = test_set.iloc[:,:-1]
testy = test_set['satisfaction']
NB_model = GaussianNB()
NB_model.fit(trainX, trainy)
NB_y_pred = NB_model.predict(testX)
after_NB_accuracy = accuracy_score(testy, NB_y_pred)

log_reg_model = LogisticRegression(random_state=42)
log_reg_model.fit(trainX, trainy)
log_y_pred = log_reg_model.predict(testX)
after_log_reg_accuracy = accuracy_score(testy, log_y_pred)

dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_model.fit(trainX, trainy)
dt_y_pred = dt_model.predict(testX)
after_dt_accuracy = accuracy_score(testy, dt_y_pred)

NB_cf = confusion_matrix(testy, NB_y_pred)
log_cf = confusion_matrix(testy, log_y_pred)
dt_cf = confusion_matrix(testy, dt_y_pred)
```

    /opt/anaconda3/lib/python3.12/site-packages/sklearn/linear_model/_logistic.py:469: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(



```python
print("Naive Bayes accuracy score:", NB_accuracy,
     "\nLogistic Regression accuracy score", log_reg_accuracy,
     "\nDecision Tree accuracy score", dt_accuracy,
      "\n","\nNaive Bayes accuracy score AFTER:", after_NB_accuracy,
     "\nLogistics Regression accuracy score AFTER:", after_log_reg_accuracy,
     "\nDecision Tree accuracy score AFTER:", after_dt_accuracy)
```

    Naive Bayes accuracy score: 0.8041893913798929 
    Logistic Regression accuracy score 0.6843959650562286 
    Decision Tree accuracy score 0.90873111636662 
     
    Naive Bayes accuracy score AFTER: 0.8651479318499927 
    Logistics Regression accuracy score AFTER: 0.8375886867126792 
    Decision Tree accuracy score AFTER: 0.908007143201892



```python
NB_cf
```




    array([[10672,  1094],
           [ 1700,  7253]])




```python
log_cf
```




    array([[10099,  1667],
           [ 1698,  7255]])




```python
dt_cf
```




    array([[10905,   861],
           [ 1045,  7908]])



 

Whether before or after remove less important feature variables, Decision Tree still performs the best out of 3 models. To further improve the performance of Decision Tree model, I will perform bagging to have a better performance for test set.

  


```python
from sklearn.ensemble import BaggingClassifier
bagg = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, max_samples=100, n_jobs= -1, random_state=42)
bagg.fit(trainX, trainy)
y_pred = bagg.predict(testX)
accuracy_score(testy, y_pred)
```




    0.9171292050774651



 ---

### **Conclusion**

- High false negative or false positive in this case (airline industry) would not be as harmful as in medical field, where it could be dangerous and fatal. After removing less important features, the false positive is now higher than the false negative, even though the accuracy score is pretty high.

--> This could still greatly impact the airline business in many way: not addressing the dissatifaction of the customer in time could lead to the loss of loyalty, bad public reputation as customer might file complaint or address the issue to social media when they feel ignored. 

- Specifically, Naives Bayes model changes from false negative > false positive to false poitive > false negative could indicate that even though the less important (less correlated) features to "Satisfaction level" are removed, the model might lack some context to distinguish between satisfy and dissatisfy classes. 


```python

```
