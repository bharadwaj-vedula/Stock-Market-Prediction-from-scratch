import pandas as pd
import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error

from tensorflow.keras.models import Model,load_model



df_10= pd.read_csv('nse_10yrs.csv')

dates=[]
for i in range(len(df_10)):
    dates.append(datetime.strptime(df_10['Date'][i],"%d-%m-%Y"))

dates=pd.DataFrame({'dates':dates})

df_10= pd.concat([df_10,dates],axis='columns',copy='False')

df_10['dates']=pd.to_datetime(df_10['dates'])
df_10=df_10.set_index('dates')

df_10=df_10.dropna(axis="rows")
df_close=df_10.iloc[:,4]


#dataset creation

sc=StandardScaler()


df_close_scaled= sc.fit_transform(df_close.values.reshape(-1, 1))



df_train=df_close_scaled[0:2000]
df_test= df_close_scaled[2000:]

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

X_train,y_train=create_dataset(df_train,df_train,time_steps=50)
X_test,y_test=create_dataset(df_test,df_test,time_steps=50)


#get model predicitons

#lstm 
lstm_model= load_model("lstm_model.h5")
lstm_pred = lstm_model.predict(X_test)
lstm_mse,lstm_mae=np.sqrt(mean_squared_error(sc.inverse_transform(y_test),sc.inverse_transform(lstm_pred))),mean_absolute_error(sc.inverse_transform(y_test),sc.inverse_transform(lstm_pred))

#gru
gru_model= load_model("gru_model.h5")
gru_pred = gru_model.predict(X_test)
gru_mse,gru_mae=np.sqrt(mean_squared_error(sc.inverse_transform(y_test),sc.inverse_transform(gru_pred))),mean_absolute_error(sc.inverse_transform(y_test),sc.inverse_transform(gru_pred))



#stream lit app

st.title('Stock Market Charts')

st.write('    ')

st.markdown('** DATASET  NIFTY 50 **')

st.write(df_10)

st.write("ONLY CHOOSING THE CLOSING PRICE")
st.write(df_close)

st.write('PLOTTING THE CLOSING PRICE')
fig, ax = plt.subplots()
ax.plot(df_close)
st.plotly_chart(fig)


#training plot gru

st.markdown('** GRU TRAINING **')
gru_hist_loss= np.load('gru_hist_loss.npy',allow_pickle='TRUE')
gru_hist_mae= np.load('gru_hist_mae.npy',allow_pickle='TRUE')

fig, ax = plt.subplots()
ax.plot(gru_hist_loss,label="Loss")
ax.plot(gru_hist_mae, label ="MAE")
st.plotly_chart(fig)

#training plot lstm

st.markdown('** LSTM TRAINING **')
lstm_hist_loss= np.load('lstm_hist_loss.npy',allow_pickle='TRUE')
lstm_hist_mae= np.load('lstm_hist_mae.npy',allow_pickle='TRUE')

fig, ax = plt.subplots()
ax.plot(lstm_hist_loss,label="Loss")
ax.plot(lstm_hist_mae, label ="MAE")
st.plotly_chart(fig)



#plot lstm

st.markdown('** LSTM PREDICTIONS **')
st.write("LSTM MAE:",lstm_mae,"LSTM MSE",lstm_mse)

lstm_pred_scaled= sc.inverse_transform(lstm_pred)
preds=[]
for i in range(len(lstm_pred_scaled)):
    preds.append(lstm_pred_scaled[i])

fig, ax = plt.subplots()
index = dates[-399:].values
index= index.reshape(399)
df_preds= pd.DataFrame({'preds':preds})
index_df= pd.DataFrame({'dates':index})

df_pred=pd.concat([df_preds,index_df],copy=False,axis='columns')

df_pred=df_pred.set_index('dates')


ax.plot(df_close,label="Original values")
ax.plot(df_pred, label ="Predicted values")
st.plotly_chart(fig)

#plot gru 

st.markdown('** GRU PREDICTIONS **')
st.write("GRU MAE:",gru_mae,"GRU MSE",gru_mse)

gru_pred_scaled= sc.inverse_transform(gru_pred)
preds=[]
for i in range(len(gru_pred_scaled)):
    preds.append(gru_pred_scaled[i])

fig, ax = plt.subplots()
index = dates[-399:].values
index= index.reshape(399)
df_preds= pd.DataFrame({'preds':preds})
index_df= pd.DataFrame({'dates':index})

df_pred=pd.concat([df_preds,index_df],copy=False,axis='columns')

df_pred=df_pred.set_index('dates')


ax.plot(df_close,label="Original values")
ax.plot(df_pred, label ="Predicted values")
st.plotly_chart(fig)

