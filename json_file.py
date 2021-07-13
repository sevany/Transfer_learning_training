import json
import os 
train_dir = os.path.join('Training/')
valid_dir = os.path.join('Test/')

train_class_name = [x[1] for x in os.walk(train_dir)]
train_class_name = train_class_name[0]

valid_class_name = [x[1] for x in os.walk(valid_dir)]
valid_class_name = valid_class_name[0]

train_json = {}
valid_json = {}
 
for tcn in train_class_name:
  
  address =  'Training/' + tcn
  for x in os.walk(address):

    train_json[tcn] = x[2] 
for vcn in valid_class_name:
  
  address =  'Test/' + vcn
  for x in os.walk(address):
    
    valid_json[vcn] = x[2] 
dataset_split = {
    'training':train_json,
    'validation':valid_json
}
with open('dataset_split.json', 'w') as fp:
  json.dump(dataset_split, fp)