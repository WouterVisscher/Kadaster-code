import os

model_name = "stackedhourglass" #model_name should be the same as the name of the json file in the Data/JSON_files folder, and the same as the name of the folder in Data/Roadnetwork and Data/Labels, need to fix that
model_path = "".join([os.getcwd(),"/checkpoints/",model_name,"/"])
dataname = "combined"

#json_path = "".join([os.getcwd(),"/Data/JSON_files/",model_name,".json"])
input_folder = "".join([os.getcwd(),"/data/roadnetwork/",dataname])
output_folder = "".join([os.getcwd(),"/data/labels/",dataname])
results_folder = "".join([os.getcwd(),"/predictions/"])

input_image_height = 360
input_image_width = 640
number_of_data_pairs = 1200
data_split_proportion = 0.2

#Hyperparamers (general, overwritten based on hyperparameter tuning results)
learning_rate = 0.00001
batch_size = 32 #16 might be needed for some models, depending on memory
epochs = 100
early_stopping_patience = 5
early_stopping_delta = 0.001 