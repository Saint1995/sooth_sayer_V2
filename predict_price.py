from __future__ import absolute_import, division, print_function
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

#this project is aimed at building a Model that helps Users
#by predicting the price of buying a Fairly used Gadget(Tablet, Phone, Nigeria)
#this project uses the Following as Features:
#- Price of New Item
#- Number of Years Item has been used
#- Region of Purchase
#- Type of Gadget &
#- Brand of Gadget


column_names = ['type','location','current_price','years','brand','resell_price']
raw_dataset = pd.read_csv('C:\\Users\Sam\\PycharmProjects\\ML101\\data.csv', names=column_names,na_values = "?", comment='\t', skipinitialspace=True)
dataset = raw_dataset.copy()

print(dataset)

dataset.tail()
#modify data for train data for location
origin = dataset.pop('location')
dataset['Abuja'] = (origin == 1)*1.0
dataset['Taraba'] = (origin == 2)*1.0
dataset['Kogi'] = (origin == 3)*1.0
dataset['Gombe'] = (origin == 4)*1.0
dataset['Bauchi'] = (origin == 5)*1.0
dataset['Adamawa'] = (origin == 6)*1.0
dataset['Kano'] = (origin == 7)*1.0
dataset['Lagos'] = (origin == 8)*1.0
dataset['Other'] = (origin == 9)*1.0
dataset.tail()

#modify data for train for brands
origin = dataset.pop('brand')
dataset['Samsung'] = (origin == 1)*1.0
dataset['Blackberry'] = (origin == 2)*1.0
dataset['HTC'] = (origin == 3)*1.0
dataset['Dell'] = (origin == 4)*1.0
dataset['HP'] = (origin == 5)*1.0
dataset['Gionee'] = (origin == 6)*1.0
dataset['Techno'] = (origin == 6)*1.0
dataset['Apple'] = (origin == 7)*1.0
dataset['Huawei'] = (origin == 8)*1.0
dataset['Lenovo'] = (origin == 9)*1.0
dataset.tail()

#modify data for train for type
origin = dataset.pop('type')
dataset['Tablet'] = (origin == 1)*1.0
dataset['Phone'] = (origin == 2)*1.0
dataset['Laptop'] = (origin == 3)*1.0
dataset.tail()

print(dataset)
#train dataset
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

#seperate labels
train_labels = train_dataset.pop('resell_price')
test_labels = test_dataset.pop('resell_price')

#Still a WIP
model=Sequential([Dense(12,input_dim=24,kernel_initializer='normal',activation='relu'),
                  Dense(8, activation='relu'),
                  Dense(1, activation='linear'),])

model.compile(optimizer='adam',loss='mse',metrics=['mse','mae'],)

model.fit(train_dataset,train_labels,epochs=1500,batch_size=5)

#model.evaluate(test_dataset,test_labels)
