import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

st.title("📈 Stock Price Prediction App")

file = st.file_uploader("Upload Stock Dataset (CSV)")

if file is not None:
    df = pd.read_csv(file, encoding='latin1')
    df["Date"] = pd.to_datetime(df["Date"])
    df.index = df["Date"]
    
    st.write("### Uploaded Data")
    st.write(df.head())
    
    st.subheader("📊 Original Closing Price")
    st.line_chart(df['Close'])

    st.subheader("📊 Original Closing Price")
    plt.figure(figsize=(10,5))
    plt.plot(df["Close"])
    st.pyplot(plt)
   
     # Data preparation
    data = df.sort_index(ascending=True, axis=0)
    new_dataset = pd.DataFrame(index=range(0,len(df)), columns=['Date','Close'])

    for i in range(len(data)):
         new_dataset.loc[i,"Date"] = data.iloc[i]["Date"]
         new_dataset.loc[i,"Close"] = data.iloc[i]["Close"]

    new_dataset.index = new_dataset["Date"]
    new_dataset.drop("Date", axis=1, inplace=True)

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(new_dataset)

     # Train-Test split
    train_data = scaled_data[0:987,:]
    valid_data = scaled_data[987:,:]

    x_train, y_train = [], []

    for i in range(60,len(train_data)):
         x_train.append(train_data[i-60:i,0])
         y_train.append(train_data[i,0])

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

     # Model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
    model.add(LSTM(50))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')

    with st.spinner("Training model..."):
         model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=0)

     # Testing
    inputs = new_dataset[len(new_dataset)-len(valid_data)-60:].values
    inputs = scaler.transform(inputs)

    X_test = []
    for i in range(60,inputs.shape[0]):
         X_test.append(inputs[i-60:i,0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    # Plot results
    train = new_dataset[:987]
    valid = new_dataset[987:]
    valid["Predictions"] = predictions

    st.subheader("📉 Predicted vs Actual")
    plt.figure(figsize=(10,5))
    plt.plot(train["Close"], label="Train")
    plt.plot(valid["Close"], label="Actual")
    plt.plot(valid["Predictions"], label="Predicted")
    plt.legend()
    st.pyplot(plt)

    st.success("✅ Prediction Completed!")
   # Accuracy Calculation
    actual = valid["Close"].values
    predicted = valid["Predictions"].values

    mape = abs((actual - predicted) / actual).mean() * 100
    accuracy = 100 - mape

    st.write("---")
    st.subheader("📊 Model Accuracy")
    st.success(f"✅ Accuracy: {accuracy:.2f} %")
    st.info(f"MAPE Error: {mape:.2f} %")