import yfinance as yf
import pandas as pd
from urllib.request import urlopen,Request
try:
    from bs4 import BeautifulSoup
except:
    from BeautifulSoup import BeautifulSoup
import streamlit as st
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import numpy as np 
import tensorflow as tf
import math
from sklearn.metrics import mean_squared_error
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler

def create_dataset(dataset,time_step=1):
    dataX,dataY=[],[]
    for i in range(len(dataset)-time_step-1):
        a=dataset[i:(i+time_step),0]
        dataX.append(a)
        dataY.append(dataset[i+time_step,0])
    return np.array(dataX),np.array(dataY)

def DataPreProcessing(df):
    df1=df.reset_index()['Close']
    scaler=MinMaxScaler(feature_range=(0,1))
    df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
    return df1,scaler

def SplittingDataSet(df1,scaler):
    train_size=int(len(df1)*0.65)
    train_data,test_data=df1[0:train_size,:],df1[train_size:len(df1),:1]
    time_step=100
    X_train, Y_train=create_dataset(train_data, time_step)
    X_test, Y_test=create_dataset(test_data, time_step)
    X_train =X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))
    model=Sequential()
    model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
    model.add(LSTM(50,return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')
    with st.spinner("Please wait while processing...It may take upto 5 minutes.."):
        model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=100,batch_size=64,verbose=1)
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)
    train_predict=scaler.inverse_transform(train_predict)
    test_predict=scaler.inverse_transform(test_predict)
    st.success("Processing completed!")
    return train_predict,test_predict,model,df1,test_data,X_test

def PlotGraph(train_predict,test_predict,df1,scaler):
    ### Plotting 
    # shift train predictions for plotting
    look_back=100
    trainPredictPlot = np.empty_like(df1)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(df1)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(scaler.inverse_transform(df1), label='Original Data')
    ax.plot(trainPredictPlot, label='Train Predictions')
    ax.plot(testPredictPlot, label='Test Predictions')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend()
    
def newGraph(model,df1,scaler,test_data,X_test):
    x_input=test_data[X_test.shape[0]+1:].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()    
    lst_output=[]
    n_steps=100
    i=0
    mainVal=0
    while(i<30):
        
        if(len(temp_input)>100):
            #print(temp_input)
            x_input=np.array(temp_input[1:])
            print("{} day input {}".format(i,x_input))
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            #print(x_input)
            yhat = model.predict(x_input, verbose=0)
            print("{} day output {}".format(i,yhat))
            inverse_data = scaler.inverse_transform(yhat)
            if(i==29):
                mainVal=inverse_data
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            #print(temp_input)
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            inverse_data = scaler.inverse_transform(yhat)
            if(i==29):
                mainVal=inverse_data
            print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i=i+1
        
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0E1117')
    ax.plot(day_new, scaler.inverse_transform(df1[len(df1)-100:]), label='Original Data',linewidth=5)
    ax.plot(day_pred, scaler.inverse_transform(lst_output), label='Predictions',linewidth=5)
    ax.set_facecolor("black")
    ax.set_xlabel('Time', color='white')
    ax.set_ylabel('Value', color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.legend()
    col1,col2=st.columns(2)
    col1.title("Prediction :")
    col1.write(f"<span style=' font-size: 20px;'>The price after 30 days will be around:</span> <span style='color: green; font-size: 30px; font-weight: bold; font-style: italic;'>{mainVal[0][0]:.2f}</span>", unsafe_allow_html=True)
    col2.pyplot(fig)
    
    
def filter_companies(search_term,company_dict):
    filtered_companies = [code for code in company_dict.keys() if search_term.upper() in code or search_term.upper() in company_dict[code].upper()]
    return filtered_companies


def ScrappingToGetSentiments(finviz_url,company):
    url=finviz_url+company
    news_tables = {}
    req=Request(url=url,headers={'user-agent':'my-app'})
    response = urlopen(req)
    html = BeautifulSoup(response, 'html')
    news_table = html.find(id='news-table')
    news_tables[company] = news_table

    parsed_data = []
    for ticker, news_table in news_tables.items():
        for row in news_table.findAll('tr'):
            if row.a is not None:
                title = row.a.text
                date_data = row.td.text.split(' ')
                if len(date_data) == 21:
                    time = date_data[12]
                else:
                    date = date_data[12]
                    time = date_data[13]
                parsed_data.append([ticker, date, time, title])

    df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])
    return df


def SentimentAnalyser(df):
    vader = SentimentIntensityAnalyzer()
    f = lambda title: vader.polarity_scores(title)['compound']
    df['compound'] = df['title'].apply(f)
    average_score = df['compound'].mean()
    return average_score



def create_circular_meter(score):
    percentage = (score + 1) / 2 * 100
    angle = 90 + (percentage / 100) * 360
    theta = np.linspace(0, 2 * np.pi, 100)
    r = 0.5

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    plt.figure(facecolor='#0E1117')
    fig, ax = plt.subplots(figsize=(1, 1),facecolor="#0E1117")
    ax.set_facecolor("#0E1117")
    if percentage >= 55:
        ax.plot(x, y, color='lime', linewidth=3)
    elif percentage<=55 and percentage>=50:
        ax.plot(x, y, color='orange', linewidth=3)
    else :
        ax.plot(x, y, color='tomato', linewidth=3)
    ax.fill_between(x, y,color='#0E1117',alpha=1)

    ax.text(0, 0, f"{int(percentage)}%", ha='center', va='center', fontsize=10,color="white")
    ax.axis('equal')
    ax.axis('off')
    plt.tight_layout()
    return fig


def main():  
    st.set_page_config(
        page_title="Stock Price Predictor",
        page_icon="chart_with_upwards_trend",
        layout="wide",
    )
    finviz_url='https://finviz.com/quote.ashx?t='
    company_codes = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'JPM', 'WMT', 'NVDA', 'FB', 'V', 'MA', 'BAC', 'RIL', 'TCS', 'TMUS', 'JNJ', 'PG', 'BABA', 'NFLX', 'VZ', 'INTC', 'DIS', 'CSCO', 'PFE', 'HD', 'KO', 'PEP', 'NKE', 'ADBE', 'MCD', 'PYPL', 'ABT', 'CRM', 'ORCL', 'NVO', 'CVX', 'XOM', 'CMCSA', 'ASML', 'TM', 'ABBV', 'NVS', 'AMGN', 'COST', 'AVGO', 'TMO', 'MRK', 'UNH', 'LIN', 'BHP', 'SBUX', 'BMY', 'DHR', 'HDB', 'QCOM', 'TXN', 'NEE', 'ACN', 'LLY', 'LMT', 'NOW', 'LOW', 'AMT', 'NOC', 'SNE', 'UNP', 'UPS', 'CHTR', 'RTX', 'PDD','RIL', 'TCS', 'HDB', 'HINDUNILVR', 'INFY', 'HDFCBANK', 'ICICIBANK', 'SBIN', 'AXISBANK', 'KOTAKBANK', 'ITC', 'LT', 'M&M', 'BHARTIARTL', 'HCLTECH', 'POWERGRID', 'BAJAJFINSV', 'BAJFINANCE', 'HINDALCO', 'GRASIM', 'JSWSTEEL', 'INDUSINDBK', 'ADANIPORTS', 'DRREDDY', 'BAJAJ-AUTO', 'ULTRACEMCO', 'UPL', 'TATAMOTORS', 'HDFCLIFE', 'HEROMOTOCO', 'TECHM', 'ONGC', 'IOC', 'POWERINDIA', 'BPCL', 'SIEMENS', 'HAVELLS', 'GAIL', 'MRF', 'ADANIGREEN', 'AUROPHARMA', 'TVSMOTOR', 'JUBLFOOD', 'SBILIFE', 'CHOLAFIN', 'NAM-INDIA', 'DLF', 'JINDALSTEL', 'LUPIN', 'NMDC', 'SRF', 'VOLTAS', 'PNB', 'SUNTV', 'PIDILITIND', 'MOTHERSUMI', 'BANKBARODA', 'INDIGO', 'AMBUJACEM', 'LALPATHLAB', 'IRCTC', 'BERGEPAINT', 'CADILAHC', 'COLPAL', 'ICICIGI', 'APOLLOHOSP', 'CHOLAHLDNG', 'MCDOWELL-N', 'BEL']
    company_names = [
        'Apple Inc.', 'Microsoft Corporation', 'Amazon.com Inc.', 'Alphabet Inc. (Google)', 'Tesla Inc.', 'JPMorgan Chase & Co.',
        'Walmart Inc.', 'NVIDIA Corporation', 'Facebook Inc.', 'Visa Inc.', 'Mastercard Incorporated', 'Bank of America Corporation',
        'Reliance Industries Limited', 'Tata Consultancy Services Limited', 'T-Mobile US Inc.', 'Johnson & Johnson', 'Procter & Gamble Company',
        'Alibaba Group Holding Limited', 'Netflix Inc.', 'Verizon Communications Inc.', 'Intel Corporation', 'The Walt Disney Company',
        'Cisco Systems Inc.', 'Pfizer Inc.', 'The Home Depot Inc.', 'The Coca-Cola Company', 'PepsiCo Inc.', 'Nike Inc.', 'Adobe Inc.',
        'McDonald\'s Corporation', 'PayPal Holdings Inc.', 'Abbott Laboratories', 'Salesforce.com Inc.', 'Oracle Corporation',
        'Novo Nordisk A/S', 'Chevron Corporation', 'Exxon Mobil Corporation', 'Comcast Corporation', 'ASML Holding N.V.', 'Toyota Motor Corporation',
        'AbbVie Inc.', 'Novartis AG', 'Amgen Inc.', 'Costco Wholesale Corporation', 'Broadcom Inc.', 'Thermo Fisher Scientific Inc.',
        'Merck & Co. Inc.', 'UnitedHealth Group Incorporated', 'Linde plc', 'BHP Group', 'Starbucks Corporation', 'Bristol-Myers Squibb Company',
        'Danaher Corporation', 'HDFC Bank Limited', 'Qualcomm Incorporated', 'Texas Instruments Incorporated', 'NextEra Energy Inc.',
        'Accenture plc', 'Eli Lilly and Company', 'Lockheed Martin Corporation', 'ServiceNow Inc.', 'Lowe\'s Companies Inc.', 'American Tower Corporation',
        'Northrop Grumman Corporation', 'Sony Corporation', 'Union Pacific Corporation', 'United Parcel Service Inc.', 'Charter Communications Inc.',
        'Raytheon Technologies Corporation', 'Pinduoduo Inc.', 'Reliance Industries Limited', 'Tata Consultancy Services Limited', 'HDFC Bank Limited',
        'Hindustan Unilever Limited', 'Infosys Limited', 'Housing Development Finance Corporation Limited', 'ICICI Bank Limited',
        'State Bank of India', 'Axis Bank Limited', 'Kotak Mahindra Bank Limited', 'ITC Limited', 'Larsen & Toubro Limited', 'Mahindra & Mahindra Limited',
        'Bharti Airtel Limited', 'HCL Technologies Limited', 'Power Grid Corporation of India Limited', 'Bajaj Finserv Limited', 'Bajaj Finance Limited',
        'Hindalco Industries Limited', 'Grasim Industries Limited', 'JSW Steel Limited', 'IndusInd Bank Limited', 'Adani Ports and Special Economic Zone Limited',
        'Dr. Reddy\'s Laboratories Limited', 'Bajaj Auto Limited', 'UltraTech Cement Limited', 'UPL Limited', 'Tata Motors Limited', 'HDFC Life Insurance Company Limited',
        'Hero MotoCorp Limited', 'Tech Mahindra Limited', 'Oil and Natural Gas Corporation Limited', 'Indian Oil Corporation Limited', 'Power Grid Corporation of India Limited',
        'Bharat Petroleum Corporation Limited', 'Siemens Limited', 'Havells India Limited', 'GAIL (India) Limited', 'MRF Limited', 'Adani Green Energy Limited',
        'Aurobindo Pharma Limited', 'TVS Motor Company Limited', 'Jubilant Foodworks Limited', 'SBI Life Insurance Company Limited', 'Cholamandalam Investment and Finance Company Limited',
        'Nippon Life India Asset Management Limited', 'DLF Limited', 'Jindal Steel & Power Limited', 'Lupin Limited', 'NMDC Limited', 'SRF Limited',
        'Voltas Limited', 'Punjab National Bank', 'Sun TV Network Limited', 'Pidilite Industries Limited', 'Motherson Sumi Systems Limited', 'Bank of Baroda',
        'InterGlobe Aviation Limited', 'Ambuja Cements Limited', 'Dr. Lal PathLabs Limited', 'Indian Railway Catering and Tourism Corporation Limited',
        'Berger Paints India Limited', 'Cadila Healthcare Limited', 'Colgate-Palmolive (India) Limited', 'ICICI Lombard General Insurance Company Limited',
        'Apollo Hospitals Enterprise Limited', 'Cholamandalam Financial Holdings Limited', 'United Spirits Limited', 'Bharat Electronics Limited'
    ]
    
    company_dict = dict(zip(company_codes, company_names))
    st.title("Real-Time Stock Market Prediction")
    search_term = st.text_input("Search by Company Name", "").strip()
    
    filterBox = filter_companies(search_term,company_dict)
    if not filterBox:
        st.warning("No matching companies found.")
        st.stop()    
    company=st.selectbox("Select Company Code",filterBox)
    with st.form("my-form"):
        color = st.select_slider(
        'Select year range for prediction',
        options=['5', '6', '7', '8', '9', '10'])
        submitted = st.form_submit_button("Submit")
        if submitted:
            df=ScrappingToGetSentiments(finviz_url=finviz_url,company=company)
            sentiment_score=SentimentAnalyser(df=df)
            fig = create_circular_meter(sentiment_score)
            col1,col2,col3,col4=st.columns(4)
            col1.title(company_dict[company])
            col2.title("")
            col3.pyplot(fig)
            col3.write("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Sentiment Score (Source: **_FinViz_**)")
            col4.write("")
            stock = yf.Ticker(company)
            hist = stock.history(period=(color+'y'))

            # Assign the historical stock data to the DataFrame 'df'
            df = hist.copy()

            # Handling missing values using forward fill
            df.fillna(method='ffill', inplace=True)

            # Display the real-time stock data in Streamlit
            df1,scaler=DataPreProcessing(df)
            train_predict,test_predict,model,df1,test_data,X_test=SplittingDataSet(df1,scaler)
            PlotGraph(train_predict,test_predict,df1,scaler)
            newGraph(model,df1,scaler,test_data,X_test)
        

if __name__ == "__main__":
    main()
