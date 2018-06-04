# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 11:45:31 2018

@author: Fabien Gueret 4  iwoka

Data Samurai Challenge

Call Centre

Data
1.leads.csv. This is a list of fictitious company directors, with some basic data about them and their company.
2.calls.csv. This is a list of fictitious calls made by an outbound call centre. The call centre consists of several agents, 
  who make calls one after the other. They don’t get to choose who to call, the system does. The objective of the call is to
  get the lead to signup on the website. When they finish a call, they mark down the outcome, from a fixed list of possible outcomes. 
  Note that a single lead may be called multiple times.
3.signups.csv. This is a list of leads who signed up after being called by someone from the call centre.
  Each signup was risk assessed and either approved or rejected for a loan.

"""
# Dependencies

#Data management library
import pandas as pd
#Time management library
import datetime as dt
# Database Library
import numpy as np
# Statistics
from scipy import stats


#### Open csv files and save data in Dataframes ####
# paths
leadfile = 'leads.csv'
callfile = 'calls.csv'
signupfile = 'signups.csv'
# inflow of data
leads = pd.read_csv(leadfile,header = 0,index_col=0, converters ={'Age':int})
print(leads.head())
calls = pd.read_csv(callfile,header = 0,index_col=3)
print(calls.head())
signups = pd.read_csv(signupfile,header = 0,index_col=0)
print(signups.head())


#### 1.Which agent made the most calls? ####

agents_activity = calls.groupby('Agent').count().drop(['Call Outcome'],axis=1)
#print(agents_activity)
prolific_agent = agents_activity.sort_values(['Phone Number'], ascending=False).head(1)
print(prolific_agent)


#### 2.For the leads that received one or more calls, how many calls were received on average?  ####

lead_contacts = calls.groupby('Phone Number').count()
avg_calls_number = lead_contacts['Call Outcome'].mean()
print(avg_calls_number)


#### 3.For the leads that signed up, how many calls were received, on average? 

signed_ups_phone_number= pd.merge(signups, leads , how='left', left_index= True, right_index=True)
signed_up_calls = pd.merge(signed_ups_phone_number, calls, how='left', left_on= 'Phone Number', right_on= 'Phone Number')
signed_up_call_counts = signed_up_calls.groupby('Phone Number').count()
signed_up_call_avg = signed_up_call_counts['Call Outcome'].mean()
print(signed_up_call_avg)


#### 4.Which agent had the most signups? Which assumptions did you make? (note that there is a many-to-one relationship between calls and leads) ####

success_calls = calls[calls['Call Outcome']=='INTERESTED']
agents_success = success_calls.groupby('Agent').count().drop(['Call Outcome'],axis=1)
best_agent = agents_success.sort_values(['Phone Number'], ascending=False).head(1)
print('Best Agent ', best_agent)


#### 5.Which agent had the most signups per call? ####

agent_effort_to_signup = pd.merge(agents_activity,agents_success,how='left', left_index= True, right_index=True)
agent_effort_to_signup['Success Rate']= agent_effort_to_signup['Phone Number_y']/agent_effort_to_signup['Phone Number_x']
efficient_agent = agent_effort_to_signup.sort_values(['Success Rate'], ascending=False).drop(['Phone Number_y','Phone Number_x'],axis=1).head(1)
print('Efficient Agent', efficient_agent)


#### 6.Was the variation between the agents’ signups-per-call statistically significant? Why? 

# Ho : p1 = avg Ha p1!= avg

avg_success = agent_effort_to_signup['Phone Number_y'].sum()/ agent_effort_to_signup['Phone Number_x'].sum()
agent_effort_to_signup['p_value']=[stats.binom_test(row[1],row[0],avg_success) for index , row in agent_effort_to_signup.iterrows()]
print(agent_effort_to_signup)


#### 7.A lead from which region is most likely to be “interested” in the product?  ####

lead_regions= pd.merge(leads, calls, how='right', left_on= 'Phone Number', right_on='Phone Number')
# calculate the number of interest number by region
lead_interested_by_regions = lead_regions.loc[lead_regions['Call Outcome']=='INTERESTED'].groupby('Region').count()
# calculate the number of unique phone numbers called by region (many calls for one number!)  
lead_phone_number = calls.groupby('Phone Number').count()
all_phone_numbers_regions=pd.merge(lead_phone_number,leads,how='left', left_index= True, right_on='Phone Number')
all_leads_called_by_regions = all_phone_numbers_regions.groupby('Region').count()
interested_region_data=pd.merge(lead_interested_by_regions,all_leads_called_by_regions,how='inner', left_index= True, right_index=True)
interested_region_data['InterestedvsAll']=interested_region_data['Age_x']/interested_region_data['Age_y']
most_interested_region = interested_region_data.sort_values(['InterestedvsAll'], ascending=False).head(1)
print('Interested region : ',most_interested_region['InterestedvsAll'])


#### 8.A lead from which sector is most likely to be “interested” in the product? ####

# calculate the number of interest number by sectors
lead_interested_by_sectors = lead_regions.loc[lead_regions['Call Outcome']=='INTERESTED'].groupby('Sector').count()
all_leads_called_by_sectors = all_phone_numbers_regions.groupby('Sector').count()
interested_sector_data=pd.merge(lead_interested_by_sectors,all_leads_called_by_sectors,how='inner', left_index= True, right_index=True)
interested_sector_data['InterestedvsAll']=interested_sector_data['Age_x']/interested_sector_data['Age_y']
most_interested_sector = interested_sector_data.sort_values(['InterestedvsAll'], ascending=False).head(1)
print('Interested sector : ', most_interested_sector['InterestedvsAll'])


#### 9.Given a lead has already expressed interest and signed up, ####
#### 9.a.signups from which region are most likely to be approved? ####

signups_info = pd.merge(signups , leads , how ='left', left_index = True, right_index = True)
signups_region = signups_info.groupby('Region').count()
approved_signups_region = signups_info.loc[signups_info['Approval Decision']=='APPROVED'].groupby('Region').count()
approved_region_data=pd.merge(signups_region, approved_signups_region,how='inner', left_index= True, right_index=True)
approved_region_data['ApprovedvsAll']=approved_region_data['Age_y']/approved_region_data['Age_x']
most_approved_region = approved_region_data.sort_values(['ApprovedvsAll'], ascending=False).head(1)
print('Approved region : ', most_approved_region['ApprovedvsAll'])

#### 9.b.Is this statistically significant? Why? ####
# Ho : p1 = avg Ha p1!= avg

avg_approved = signups.loc[signups['Approval Decision']=='APPROVED'].count()/signups.count()
avg =avg_approved.sum()
approved_region_data['average']=avg
approved_region_data['p_value']=[stats.binom_test(row[4],row[0],avg) for index , row in approved_region_data.iterrows()]
print(approved_region_data)


#### 10 Suppose you wanted to pick the 1000 leads most likely to sign up (who have not been called so far), based only on age, sector and region.####
#### 10.a.What criteria would you use to pick those leads? ####

all_called = pd.merge(calls,leads, how ='left', left_on = 'Phone Number', right_on = 'Phone Number') 
conditions = [
    (all_called['Call Outcome'] == 'INTERESTED'),
    (all_called['Call Outcome'] == 'NOT INTERESTED')]
choices = [1,0]
all_called['Signup']=np.select(conditions, choices, default='rid')
all_called = all_called.loc[all_called['Signup']!='rid']
# Bin the Age
all_called['Age Bucket']= pd.cut(all_called['Age'],range(0,125,25))
#clean up before binarisation
all_called=all_called.drop(['Phone Number','Age', 'Agent','Call Outcome'],axis=1)
# Binaries predictors and target
training_data = pd.get_dummies(all_called)
training_data= training_data.drop(['Signup_0'],axis=1)
#print(training_data.columns.values)

#Import Library
from sklearn.linear_model import LogisticRegression

# Create logistic regression object
model = LogisticRegression()
# X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
X = training_data.drop(['Signup_1'],axis=1).values
X_data=training_data.drop(['Signup_1'],axis=1)
y= training_data['Signup_1'].values
# Train the model using the training sets and check score
model.fit(X, y)
model.score(X, y)
#Equation coefficient and Intercept
coeff =model.coef_[0]
features =X_data.columns.values
print('Intercept: \n', model.intercept_)

Results = pd.DataFrame(list(zip(features,coeff)),columns= ['features', 'estimated_Coefficients'])
print(Results)
#### 10.b.In what sense are those an optimal criteria set?


#### 10.c.How many signups would you expect to get based on those called leads, assuming they were being called by random agents? 

all_leads = leads[['Region','Sector','Age']]
#print(all_leads) 

# Bin the Age
all_leads['Age Bucket']= pd.cut(all_leads['Age'],range(0,125,25))
#clean up before binarisation
all_leads=all_leads.drop(['Age'],axis=1)
# Binaries predictors and target
testing_data = pd.get_dummies(all_leads)
#print(testing_data.columns.values)

#Predict Output
x_test = testing_data.values

predicted= model.predict(x_test)

print('Predicted signps',sum(predicted),' on ', len(leads))

#### 10.d.If you could choose the agents to make those calls, who would you choose? Why? 

all_called = pd.merge(calls,leads, how ='left', left_on = 'Phone Number', right_on = 'Phone Number') 
conditions = [
    (all_called['Call Outcome'] == 'INTERESTED'),
    (all_called['Call Outcome'] == 'NOT INTERESTED')]
choices = [1,0]
all_called['Signup']=np.select(conditions, choices, default='rid')
all_called = all_called.loc[all_called['Signup']!='rid']
# Bin the Age
all_called['Age Bucket']= pd.cut(all_called['Age'],range(0,125,25))
#clean up before binarisation
all_called=all_called.drop(['Phone Number','Age','Call Outcome'],axis=1)
# Binaries predictors and target
training_data = pd.get_dummies(all_called)
training_data= training_data.drop(['Signup_0'],axis=1)
#print(training_data.columns.values)


# Create 2nd logistic regression object
model2 = LogisticRegression()
# X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
X = training_data.drop(['Signup_1'],axis=1).values
X_data=training_data.drop(['Signup_1'],axis=1)
y= training_data['Signup_1'].values
# Train the model using the training sets and check score
model2.fit(X, y)
model2.score(X, y)
#Equation coefficient and Intercept
#print('Coefficient: \n', model2.coef_)
print('Intercept: \n', model2.intercept_)
 
coeff =model2.coef_[0]
features =X_data.columns.values

Results = pd.DataFrame(list(zip(features,coeff)),columns= ['features', 'estimated_Coefficients'])
print(Results)














