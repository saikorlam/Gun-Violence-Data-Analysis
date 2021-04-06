################################################################################
#          US Gun violence Data Exploration and Visualization
################################################################################
#
#
#This was an interesting project to get started on as it is one of the most hotly debated topic in United States
#with no obvious solution in sight. Here we take a look at the gun violence incidents from 2013-2018
#to understand how serious this problem has been.
#
#
# Group -11
# Developers     : Seshaiah Erugu
#                  Joshitha Gunreddy
#                  Rajesh Korlam
#
# Date  : 6th December - 2019

# All test cases are atatched in report with screenshots.


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np   # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

#Read the incidents data from csv file to  Dataframe 
dataframe = pd.read_csv('gun-violence-data_01-2013_03-2018.csv')

print("Total Rows:", dataframe.shape[0])
print("Total Columns", dataframe.shape[1])
#As we can see there are almost 240,000 rows of data which is a lot of gun violence incidents in the span of six years.

dataframe.head()

#Next up, we take a look at our columns in the dataframe and perform various analysis based on the time of the incident.
#For that reason we convert our date feature into a datetime object.

dataframe['date'] = pd.to_datetime(dataframe['date'])
dataframe.head(5)

dataframe.describe()

dataframe['date'] = pd.DatetimeIndex(dataframe['date'])


year = pd.DatetimeIndex(dataframe['date']).year.value_counts()
year = year.rename_axis("Year")
year.sort_index().plot(kind='bar', title = "Number of Gun violence by Year", figsize=(10,10))


#This results in an interesting observation that the Gun violence was lowest for 2013
#while it peaks off massively in 2014 and increases every year since then. 2018 however has seen the decline in gun violence.


MonthDF = pd.DataFrame({'Count':pd.DatetimeIndex(dataframe['date']).month_name().value_counts()})
# monthDF = MonthDF.set_index('Months')
MonthDF.index.name= "Month"
MonthDF.dtypes
# MonthDF.set_index =['Jan','Feb', 'March', 'Apr','May', 'Jun','Jul','Aug','Sep','Oct','Nov','Dec']
MonthDF.plot(kind='bar', title = "Number of Gun violence by Month", figsize=(10,10))


#Here we tried to visualize which month was more likely to see gun incidents and
#to no surprise January and July are high on the list due to obvious two big celebrations in New Year and 4th of July.


year = pd.DatetimeIndex(dataframe['date']).day.value_counts()
year = year.rename_axis("Day")
year.sort_index().plot(kind='line', title = "Number of Gun violence by Day of Month", figsize=(18,7))


#Based on this figure we can see, the incidents are really common during the start of the month
#as people are more likely to go out partying, drinking at the start of the month (Fresh Paychecks)
#compared to the end of the month with the incident almost crashing down.

dataframe.head()

dataframe.groupby('state').agg('sum')['n_killed'].sort_values().plot(kind='barh', figsize = (10,10), title="Number of Deaths in every states")

#This graph however shows an interesting information as California comes across as
#the state with the most number of Gun related deaths closely followed by Texas and Florida.
#Kudos to Vermont, Rhode Island and Hawaii.

df = dataframe.groupby('state').agg('sum')[['n_killed','n_injured']]
df.plot(kind='bar', rot=0, figsize = (40,14), title="Gun Casualty by state")

#Here we plot the deaths along with the incidents that didn't result in death and it gives us a completely different picture.
#State of Illionois which had 4th highest number of Gun deaths,
#surpasses every single state massively in terms of incidents which didn't result in death.

#We then decided to remove some of the features from our intial exploration with incident_id, source_url,
#incident_url which aren't really important for our exploration.

dataframe = dataframe.drop(['source_url','incident_url','incident_url_fields_missing', 'incident_id'],axis=1)
dataframe.head()

dataframe['year'] = pd.DatetimeIndex(dataframe['date']).year
dataframe['month'] = pd.DatetimeIndex(dataframe['date']).month_name()
dataframe['day'] = pd.DatetimeIndex(dataframe['date']).day
dataframe.head()


#On our next visualization we try to separate our casualties in deaths and only injured categories based on year and we can see,
#the deaths number increase on a smaller scale compared to the injuries which has significant injuries every year.

df = dataframe.groupby('year')[['n_killed','n_injured']].agg('sum')
df = df.rename(index=str, columns={"n_killed":"People Killed",'n_injured':'People Injured'})
# df.head()
df.plot(kind='bar', rot=0, title="Number of Casaulties by Year", figsize=(10,10))

#We then follow the same visualization in terms of month


df = dataframe.groupby('month')[['n_killed','n_injured']].agg('sum')
df = df.rename(index=str, columns={"n_killed":"People Killed",'n_injured':'People Injured'})
# df.head()
df.sort_values(by='People Killed').plot(kind='bar', rot=0, title="Number of Casaulties by Month", figsize=(10,10))


df = dataframe.groupby('day')[['n_killed']].agg('sum')
df = df.rename(index=str, columns={"n_killed":"People Killed"})
# df.sort_index()
df.sort_values(by='People Killed', ascending = True).plot(kind='barh', title="Number of deaths by day of the month", figsize=(10,8))


#This is up for interpretation as you can see more deaths occur on the 5th of a month but the constant thing has been
#that the end of the month are fairly quiet in terms of gun violence.

dataframe['monthandday'] = dataframe['month'].map(str) + " "+ dataframe['day'].map(str)
dataframe.head()

baddaydf = pd.DataFrame(dataframe['monthandday'].value_counts())
baddaydf = baddaydf.rename(index=str, columns={'monthandday' :'Date'})
# baddaydf = baddaydf.sort_values()
baddaydf[0:10].plot(kind ='bar', title="Top 10 Days with maximum number of Gun Violence")


#It comes as no surprise that two of the most severe days are 1st of January and 4th of July.
#The reason could be a large number of public events and people being intoxicated and prone to being violent early.

df = dataframe.groupby('monthandday')[['n_killed']].agg('sum')
df = df.rename(index=str, columns={"n_killed":"People Killed"})
df = df.sort_values('People Killed', ascending = False)
# df.head()
df[0:10].plot(kind='barh', title="Top 10 days of month by Number of deaths", figsize=(15,10))


dataframe = dataframe.drop(['year','month','day','monthandday'], axis=1)

dataframe['day of the week'] = pd.DatetimeIndex(dataframe['date']).day_name()
dataframe.head()

df = dataframe.groupby('day of the week').agg('sum')[['n_killed','n_injured']]
df = df.rename(index = str, columns={"n_killed":"People Killed", "n_injured":"People Injured"})
df.plot(kind='barh',rot=0, title="Incidents by day of the week", figsize=(15,10))



