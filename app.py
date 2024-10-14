import pandas as pd
import torch
from model_engine import Model
import pickle
import streamlit as st

model=Model()
model.load_state_dict(torch.load("Model_state.pth"))

with open("label_encoder.pkl","rb") as f:
    label_encoder=pickle.load(f) 

with open("one_hot_encoder.pkl","rb") as f:
    OHE=pickle.load(f) 

with open("Standard_Scalar.pkl","rb") as f:
    standard_scalar=pickle.load(f) 

st.title("Customer Churn Prediction")
names=lambda x:x.split("_")[1]
geography=st.selectbox(label="Geography",options=list(map(names,OHE.get_feature_names_out())))
gender=st.selectbox(label="Gender",options=["Male","Female"])
age=st.slider("Age",18,100)
balance=st.number_input("Balance")
credit_score=st.number_input("Credit Score")
estimated_salary=st.number_input("Estimated Salary")
tenure=st.slider("Tenure",0,10)
number_of_products=st.slider("Number of Products",1,4)
has_cr_card=st.selectbox("Has Credit Card",[0,1])
is_active_member=st.selectbox("Is Active Member",[0,1])

input_data={
    "CreditScore":credit_score,"Geography":geography,
    "Gender":gender,"Age":age,"Tenure":tenure,"Balance":balance,
    "NumOfProducts":number_of_products,"HasCrCard":has_cr_card,
    "IsActiveMember":is_active_member,
    "EstimatedSalary":estimated_salary
}


df=pd.DataFrame(input_data,index=[0])
df["Gender"]=label_encoder.transform(df["Gender"])
new_df=OHE.transform(df[["Geography"]])
new_df=pd.DataFrame(new_df,columns=OHE.get_feature_names_out())
concat_df=pd.concat([df,new_df],axis=1)
concat_df=concat_df.drop(["Geography"],axis=1)

st.dataframe(concat_df)

x=standard_scalar.transform(concat_df)

x=torch.Tensor(x)

logits=model(x)

prediction=torch.sigmoid(logits)

if prediction.item()>0.5:
    st.write("Customer is more likely to stay")
else:
    st.write("Customer is more likely to exit")