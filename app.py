# ----- Load base libraries and packages
import gradio as gr
import numpy as np
import pandas as pd
import re
import os
import pickle


# -----Helper Functions
# Function to load ML toolkit
def Load_ml_items(relative_path):
    "Load ML items to reuse them"
    with open(relative_path, 'rb' ) as file:
        loaded_object = pickle.load(file)
    return loaded_object
Loaded_object = Load_ml_items('src/assets/app_toolkit.pkl')

# -----Initializing ml items
model = Loaded_object['model']
scaler = Loaded_object['scaler']
encoder = Loaded_object['encoder']
data = Loaded_object['data']
numericals = Loaded_object['numerical']
categoricals = Loaded_object['categoricals']

# -----Useful lists
expected_inputs = ['work_year',
                   'experience_level',
                   'employment_type',
                   'job_title',
                   'employee_residence',
                   'remote_ratio',
                   'company_location',
                   'company_size']


# -----Function to process inputs and return prediction
def process_and_predict(*args, encoder=encoder, scaler=scaler, model=model):
    
    
    # ---Convert inputs into a DataFrame
    input_data = pd.DataFrame([args], columns=expected_inputs)

    # ---Encode the categorical columns
    encoded_categoricals = encoder.transform(input_data[categoricals])
    encoded_categoricals = pd.DataFrame(encoded_categoricals, columns=encoder.get_feature_names_out().tolist())
    df_processed = input_data.join(encoded_categoricals)
    df_processed.drop(columns=categoricals, inplace=True)

    # ---Scale the numeric columns
    df_processed[numericals] = scaler.transform(df_processed[numericals])

    # ---Restrict column name characters to alphanumerics
    df_processed.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x), inplace=True)

    # ---Making the prediction
    model_output = model.predict(df_processed)
    return {"Predicted salary is:": model_output}

# -----User Inputs
work_year = gr.Dropdown(label="year", choices=list(data['work_year'].unique()))
experience_level = gr.Radio(label="experience level",choices=list(data['experience_level'].unique()))  
employment_type = gr.Radio(label="employment type", choices=list(data['employment_type'].unique()))
job_title = gr.Dropdown(label="Job title", choices=list(data['job_title'].unique()))
employee_residence = gr.Dropdown(label="Employee Residence", choices=list(data['employee_residence'].unique()))
remote_ratio = gr.Radio(label="Remote Ratio", choices=list(data['remote_ratio'].unique()))
company_location = gr.Dropdown(label="Company Location", choices=list(data['company_location'].unique())) 
company_size = gr.Dropdown(label="Company size", choices=list(data['company_size'].unique())) 

# -----Output
gr.Interface(inputs=[work_year,experience_level,employment_type,job_title,employee_residence,remote_ratio,company_location,company_size],
             outputs = gr.Label("Submit forms to view prediction"),
            fn=process_and_predict, 
            title= "Salary prediction($)", 
            description= """This is App is a deployment to predict employee salary """
            ).launch(inbrowser= True,
                     show_error= True,
                     share=True)