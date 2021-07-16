# Spotify Data Case Study
I listen to a lot of music on Spotify, as do many other people. I promised a friend a list of my top 50 albums but found this very challenging. I was able to list ~25 of them, but after that I had a lot of trouble deciding. This case study is a demonstration that supervised machine learning can solve pressing real-world issues. 

For this challenge I decided on a neural network since my music taste is quite sophistocated and therefore contains many complex patterns. Specifically, it will address the problem of varying feature length better than other methods (since I listen to some albums much more than others). I won't be working with the Spotify API or any song metadata, nor will I reference album, artist, or track ID as features in the learning. These would be very useful inclusions in a deeper analysis, but require much more training data to avoid overfitting. 

The results were more positive than I expected. The neural network and logistic regression each performed noticeably better than random assignment. However the measurement is admittedly a little subjective: I counted how many of their album picks I would even consider for my top 50. 

**Process Overview**

- Contact spotify support to download full listening history. You can only download the last year from the account page.
- Create a small .csv of training data (~50 album-artist pairs with a rating of 1 or 0)
- Clean the dataset. Drop unnecessary columns and rows, address NA values, and create new features.
- Exploratory data analysis to get a general feeling of the dataset.
- Extract key features into matrices.
- Create neural network and logistic regression models in tensorflow. Train them, then predict on all albums in the dataset.
- Combine predicted values with original dataset, arrange by confidence, and evaluate the prediected top 50.

## Get To Know The Data


```python
import json
import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import time
import datetime
```


```python
# read json files to df
for i in range(0,5):
    if i == 0:
        df = pd.read_json('MyData/endsong_' + str(i) + '.json')
    else:
        # ignore_index makes sure they fully combine
        df = df.append(pd.read_json('MyData/endsong_' + str(i) + '.json'), ignore_index=True)
y_train = pd.read_csv('y_train.csv')
```

**Clean**


```python
%%capture
# drop podcasts
df.drop(df[~df['episode_show_name'].isnull()].index) 
# drop empty observations
df.drop(df[df['spotify_track_uri'].isnull()].index)
# drop useless columns
renames = {"master_metadata_track_name":"track","master_metadata_album_artist_name":"artist","master_metadata_album_album_name":"album"}
drops = ["username","conn_country","ip_addr_decrypted","city","region","metro_code","longitude","latitude","offline","offline_timestamp","incognito_mode","user_agent_decrypted","platform","episode_name","episode_show_name","spotify_episode_uri","spotify_track_uri"]
df.rename(columns = renames, inplace = True)
df = df.drop(drops, axis = 1)
# drop Sleepy John
df = df.drop(df[df['artist'].str.match("Sleepy John",case=False,na=False)].index)
# if I do not have over 20 observations of an album I assume it can't be one of my favorites
df = df.groupby(['artist', 'album']).filter(lambda x: len(x) > 20).reset_index()
# merge df with y_train file
df = df.merge(y_train, how='left', on=['artist', 'album'])
```


```python
# create a date column
df['date'] = df['ts'].str.split("T", expand=True)[0]
# create a timestamp for number of days since unix epoch 
# technically it's the number of seconds as a time delta, but this makes no difference 
df['date_ts'] = (pd.to_datetime(df['date']) - np.datetime64('1970-01-01T00:00:00'))
```

**Preview**


```python
# here's what the data looks like
df.head(6)
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
      <th>index</th>
      <th>ts</th>
      <th>ms_played</th>
      <th>track</th>
      <th>artist</th>
      <th>album</th>
      <th>reason_start</th>
      <th>reason_end</th>
      <th>shuffle</th>
      <th>skipped</th>
      <th>predict</th>
      <th>date</th>
      <th>date_ts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2018-07-15T17:15:45Z</td>
      <td>153351</td>
      <td>4th Dimension</td>
      <td>KIDS SEE GHOSTS</td>
      <td>KIDS SEE GHOSTS</td>
      <td>trackdone</td>
      <td>trackdone</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2018-07-15</td>
      <td>17727 days</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>2019-07-25T23:58:38Z</td>
      <td>44912</td>
      <td>I'm in Love Again</td>
      <td>Tomppabeats</td>
      <td>Harbor</td>
      <td>trackdone</td>
      <td>trackdone</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2019-07-25</td>
      <td>18102 days</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>2018-11-08T18:54:12Z</td>
      <td>46359</td>
      <td>Shimmy</td>
      <td>System Of A Down</td>
      <td>Toxicity</td>
      <td>trackdone</td>
      <td>endplay</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2018-11-08</td>
      <td>17843 days</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8</td>
      <td>2019-02-15T06:12:55Z</td>
      <td>245098</td>
      <td>Kids See Ghosts</td>
      <td>KIDS SEE GHOSTS</td>
      <td>KIDS SEE GHOSTS</td>
      <td>trackdone</td>
      <td>trackdone</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2019-02-15</td>
      <td>17942 days</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9</td>
      <td>2017-04-17T15:17:46Z</td>
      <td>8904</td>
      <td>Sing About Me, I'm Dying Of Thirst</td>
      <td>Kendrick Lamar</td>
      <td>good kid, m.A.A.d city</td>
      <td>clickrow</td>
      <td>endplay</td>
      <td>False</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>2017-04-17</td>
      <td>17273 days</td>
    </tr>
    <tr>
      <th>5</th>
      <td>11</td>
      <td>2017-03-24T19:19:26Z</td>
      <td>493400</td>
      <td>Holiday / Boulevard of Broken Dreams</td>
      <td>Green Day</td>
      <td>American Idiot</td>
      <td>trackdone</td>
      <td>trackdone</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017-03-24</td>
      <td>17249 days</td>
    </tr>
  </tbody>
</table>
</div>




```python
%%capture
# describe all features
df.describe(include="all")
```


```python
%%capture
# see what the frequency of trackdone and trackstart messages are
display(df["reason_end"].value_counts())
display(df["reason_start"].value_counts())
```


```python
%%capture
# Observe the album I have listened to most: Blonde
album_grp = df.groupby(['album'])
album_grp.get_group('Blonde').head(4)
```


```python
# 20 albums that I have the most observations for
df.groupby(['artist', 'album']).size().sort_values(ascending=False).head(20)
```




    artist                 album                 
    Frank Ocean            Blonde                    498
    Chon                   Homey                     442
    Flying Lotus           You're Dead!              388
    Radiohead              Kid A                     324
    Porter Robinson        Worlds                    317
    Nujabes                Modal Soul                315
    Jon Bellion            The Human Condition       301
    Kanye West             The Life Of Pablo         299
    Chon                   Grow                      282
    Tokyo Police Club      A Lesson In Crime         270
    Taylor Swift           Lover                     258
    BROCKHAMPTON           SATURATION II             258
    Red Hot Chili Peppers  Stadium Arcadium          254
    Taylor Swift           1989                      253
    Sufjan Stevens         Carrie & Lowell           244
    Red Hot Chili Peppers  Californication           243
                           By the Way                242
    Kendrick Lamar         good kid, m.A.A.d city    229
                           Section.80                227
    Nujabes                Spiritual State           226
    dtype: int64




```python
%%capture
# string matching for artist
df[df['artist'].str.match("ecco",case=False,na=False)].head(3)
```

## Feature and Neural Net Construction


```python
# group by artist-album pair; each group will be one observation
dfgrouped = df.groupby(['artist','album'])
n_obs_max = dfgrouped.size().max()
n_groups = dfgrouped.ngroups

A = []
Y = []
# construct features, zero padding to deal with input length variation.
for name, group in dfgrouped:
    # timestamp (in days)
    a = np.array(group['date_ts'])
    a.resize(n_obs_max)
    # length of play
    b = np.array(group['ms_played'])
    b.resize(n_obs_max)
    # whether shuffle was on
    c = np.array(group['shuffle'])
    c.resize(n_obs_max)
    # standard deviation of song played. 
    # this feature distinguishes between spamming a single song and listening to many songs on the album
    e = np.array(pd.Series(group.track, dtype='category').index).std()
    A.append(a + b)
    Y.append(group['predict'].max())

A = np.vstack(A).T
A = A.astype(np.int64)
# training X
X_train = A[:,(~np.isnan(Y)).T].T
# training Y
Y = np.array(Y)[~np.isnan(Y)]
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    c:\users\owend\appdata\local\programs\python\python39\lib\site-packages\pandas\core\indexes\base.py in get_loc(self, key, method, tolerance)
       3360             try:
    -> 3361                 return self._engine.get_loc(casted_key)
       3362             except KeyError as err:
    

    c:\users\owend\appdata\local\programs\python\python39\lib\site-packages\pandas\_libs\index.pyx in pandas._libs.index.IndexEngine.get_loc()
    

    c:\users\owend\appdata\local\programs\python\python39\lib\site-packages\pandas\_libs\index.pyx in pandas._libs.index.IndexEngine.get_loc()
    

    pandas\_libs\hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()
    

    pandas\_libs\hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()
    

    KeyError: 'predict'

    
    The above exception was the direct cause of the following exception:
    

    KeyError                                  Traceback (most recent call last)

    ~\AppData\Local\Temp/ipykernel_19444/3073811285.py in <module>
         21     e = np.array(pd.Series(group.track, dtype='category').index).std()
         22     A.append(a + b)
    ---> 23     Y.append(group['predict'].max())
         24 
         25 A = np.vstack(A).T
    

    c:\users\owend\appdata\local\programs\python\python39\lib\site-packages\pandas\core\frame.py in __getitem__(self, key)
       3453             if self.columns.nlevels > 1:
       3454                 return self._getitem_multilevel(key)
    -> 3455             indexer = self.columns.get_loc(key)
       3456             if is_integer(indexer):
       3457                 indexer = [indexer]
    

    c:\users\owend\appdata\local\programs\python\python39\lib\site-packages\pandas\core\indexes\base.py in get_loc(self, key, method, tolerance)
       3361                 return self._engine.get_loc(casted_key)
       3362             except KeyError as err:
    -> 3363                 raise KeyError(key) from err
       3364 
       3365         if is_scalar(key) and isna(key) and not self.hasnans:
    

    KeyError: 'predict'



```python
# the structure of this NN is extremely arbitrary. It's more of a proof of concept. 
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(64, activation='tanh', kernel_regularizer=tf.keras.regularizers.L2(0.1)),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])
# binary crossentropy is fine for sigmoid activation
loss_fn = tf.keras.losses.BinaryCrossentropy()

# run model, didn't seem to get much after 20 epochs
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
model.fit(X_train, Y, epochs=20)

# statistics about the prediction
# pd.DataFrame(model.predict(A.T).T[0]).describe()
```


```python
# create grouped df to add predicted Y values. Inappropriately named 'test_y'
dfgrouped_merge = pd.DataFrame(dfgrouped.size())
dfgrouped_merge.columns = ['test_y']
dfgrouped_merge['test_y'] = model.predict(A.T).T[0]
dfgrouped_merge = df.merge(dfgrouped_merge, how='left', on=['artist', 'album'])
```


```python
#df[df.test_y == 1].groupby(['artist','album','test_y']).size().head(55)
def f(x):
    k = f['test_y'].mean()
    return pd.Series(k, index='test_y')

df_predictions = pd.DataFrame(dfgrouped_merge.groupby(['artist','album']).apply(
    lambda x: pd.Series([x.test_y.mean(),x.predict.mean()], index=['test_y','predict']))).sort_values('test_y',ascending=0)
```

## Neural Net Results
Here I ran through a couple different ideas. Theres some variation due to the random starting parameters with gradient descent, so it's good to test out a couple things. I decided scrambling the predicted values was a good way to check the neural net against randomly chosen songs. It's a start at seeing how overfit the model is. I was also curious how a simple logistic regression would perform versus the "deep" neural net. Honestly the 'logistic regression' (single feature sigmoid) works pretty well, and it seems to catch a slightly different trend in the data, pulling up a couple good picks that the neural net missed. 


```python
df_predictions[df_predictions.predict!=1].head(25)
# 1. ) NN set: 12/25 are considerations in my mind; Random set: 9/50 I barely recognized ~20 of them
# 2. ) ADDED "SONG SPAMMING FEATURE":    NN set: 10/25; Random set: 11/47  not sure what to make of the difference if any
# 3. )^ again:    NN set: 19/25. There are albums it consistently likes, that I like too. 
# The top 5-10 are consistently some of my considerations
# Random set: 10/44; Seems worse than the NN, but its pretty helpful in its own way. Randomly listing albums jogs my memory.
# 4. ) "logistic" regression (1 feature sigmoid NN). Performs exceptional on train set. 14/25 on test set !!!
# Random Set: 7/47; again some good picks that don't show up on the NN or LR.
# 5. ) Logistic regression again. 7/25   Random assignment: 10/45
# 6. ) Logistic 13/25, NN: 12/25, Random: not much different from other times
# 7. ) Added much more training data. 
```

## Random Assignment Results


```python
# create a column of the same predictions, but permuted randomly
df_predictions['test_y_scramble'] = list(df_predictions.test_y.sample(frac=1).reset_index(drop=1))
# look at scrambled prediction values to see if the neural network is doing anything productive
# so far looks more effective than randomness... but there's a good chance of confirmation bias
df_predictions.sort_values('test_y_scramble',ascending=0).head(50)
```


```python
%%capture
# plot data to look for what features correlate with the model's predictions
df2 = df.merge(df_predictions, on=['artist','album'])
sns.scatterplot(data=df2, x='shuffle', y='test_y')
```

## Logistic Regression Results


```python
# lets try a logistic regression. Seems to work as well as the NN


model_logitstic_regression = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid'),
])
loss_fn = tf.keras.losses.BinaryCrossentropy()
model_logitstic_regression.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

model_logitstic_regression.fit(X_train, Y, epochs=200)

# statistics about the prediction
# pd.DataFrame(model.predict(A.T).T[0]).describe()

dfgrouped_merge_LR = pd.DataFrame(dfgrouped.size())
dfgrouped_merge_LR.columns = ['test_y']
dfgrouped_merge_LR['test_y'] = model_logitstic_regression.predict(A.T).T[0]
dfgrouped_merge_LR = df.merge(dfgrouped_merge_LR, how='left', on=['artist', 'album'])

#df[df.test_y == 1].groupby(['artist','album','test_y']).size().head(55)
def f(x):
    k = f['test_y'].mean()
    return pd.Series(k, index='test_y')

df_predictions_LR = pd.DataFrame(dfgrouped_merge.groupby(['artist','album']).apply(
    lambda x: pd.Series([x.test_y.mean(),x.predict.mean()], index=['test_y','predict']))).sort_values('test_y',ascending=0)

df_predictions_LR[df_predictions_LR.predict!=1].head(25)


# create a column of the same predictions, but permuted randomly
df_predictions_LR['test_y_scramble'] = list(df_predictions_LR.test_y.sample(frac=1).reset_index(drop=1))
# look at scrambled prediction values to see if the neural network is doing anything productive
# so far looks more effective than randomness... but there's a good chance of confirmation bias

```


```python
df_predictions_LR[df_predictions.predict!=1].head(25)
```


```python
df.columns
```




    Index(['ts', 'username', 'platform', 'ms_played', 'conn_country',
           'ip_addr_decrypted', 'user_agent_decrypted',
           'master_metadata_track_name', 'master_metadata_album_artist_name',
           'master_metadata_album_album_name', 'spotify_track_uri', 'episode_name',
           'episode_show_name', 'spotify_episode_uri', 'reason_start',
           'reason_end', 'shuffle', 'skipped', 'offline', 'offline_timestamp',
           'incognito_mode', 'city', 'region', 'metro_code', 'longitude',
           'latitude'],
          dtype='object')




```python

```
