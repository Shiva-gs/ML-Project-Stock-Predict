# Import dependencies






import numpy as np
from flask import (
    Flask,
    render_template,
    jsonify)
from joblib import load
import sklearn.datasets 
from io import BytesIO

import pandas as pd
from flask import Flask, send_file

# Set up Flask
app = Flask(__name__)


# home route
@app.route("/")
def index():
    """Renders homepage"""
    return render_template("index.html")

@app.route("/industry")
def industry():
    """Renders requirements page"""
    return render_template("industry.html")

@app.route("/linear")
def linear():
    """Renders requirements page"""
    return render_template("linear.html")

@app.route('/plot')
def plot():
    import numpy as np 
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVR
    from sklearn.model_selection import train_test_split
    import pandas_datareader.data as web
    from datetime import datetime, timedelta
    from sklearn.metrics import mean_squared_error,r2_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler

    now_time = datetime.now()
    start_time = datetime(now_time.year - 5, now_time.month , now_time.day)
    today = str(datetime.now())
    next_30_days = datetime.now() + timedelta(days=30)
    next_30_days = str(next_30_days)
    end_time = str(next_30_days)
    end_time

    def choosestock(stock_name):
        global stock_df, name
        validInput = False
        while not validInput:
            # Get stock data
            try:
                stock_name = input("Choose a stock by entering the ticker symbol: ")
                stock_df = web.DataReader(stock_name,'yahoo', start_time, now_time)
                validInput = True
                print("validInput = ", validInput)
            except: 
                KeyError
                print("The database could not return a valid result.  Please try again")
        name = stock_name

    choosestock(input)
    print(f"Firstday: {stock_df.head(1)}")  
    print(f"Last Day: {stock_df.tail(1)}")

    df = stock_df[['Open','Adj Close']]
    forecast_out = 30
    X = np.array(df.drop(['Adj Close'],1))
    # Get all of the x and y values except the last 'n' rows
    X = X[:-forecast_out]
    print(X)
    y = np.array(df['Adj Close'])
    y = y[:-forecast_out]
    print(y[:-forecast_out])

    # Split the data into 80% training and 20% testing
    # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # Create and train the Support Vector Machine (Regressor)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_rbf.fit(x_train, y_train)

    # Testing Model: Score returns the coefficient of determination R^2 of the prediction. 
    svm_confidence = svr_rbf.score(x_test, y_test)
    print("svm confidence: ", svm_confidence)

    # Create and train the Linear Regression  Model
    lr = LinearRegression()
    # Train the model
    lr.fit(x_train, y_train)

    # Testing Model: Score returns the coefficient of determination R^2 of the prediction. 
    # The best possible score is 1.0
    lr_confidence = lr.score(x_test, y_test)
    print("lr confidence: ", lr_confidence)

    # Set x_forecast equal to the last 30 rows of the original data set from Adj. Close column
    # x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]
    x_forecast = np.array(df.drop(['Adj Close'],1))[-forecast_out:]
    print(x_forecast) 

        # Create and train the Linear Regression  Model
    lr = LinearRegression()
    # Train the model
    lr.fit(x_train, y_train)

    # Print linear regression model predictions for the next 'n' days
    lr_prediction = lr.predict(x_forecast)
    print(lr_prediction)
    lr_pdf=pd.DataFrame(lr_prediction)
    lr_pdf

    #Error 
    #df['InitialPrediction'] = df[['Adj Close']].shift()
    Y_orginal= np.array(df.drop(['Open'],1))
    Y_true=Y_orginal[-30:]
    Y_pred=lr_prediction
    #mean_squared_error(Y_orginal[-30:], lr_prediction)
    MSE = np.square(np.subtract(Y_true,Y_pred)).mean() 
    MSE 

    rmse=r2_score(Y_true, Y_pred)
    rmse

    # Print support vector regressor model predictions for the next 'n' days
    svm_prediction = svr_rbf.predict(x_forecast)

    dt=datetime.today().strftime('%Y-%m-%d')

    df.tail(30) 
    new_df= df.tail(30) 
    new_df['Prediction']=lr_prediction

    plt.style.use('seaborn-darkgrid')
    plt.rc('figure', figsize=(20, 10))
    #plt.style.use('seaborn-whitegrid')
    fig = plt.figure()
    ax = plt.axes()
    x = new_df.index
    labels=x.strftime("%b-%d")
    ax.plot(x.strftime("%b-%d"), new_df['Adj Close']);
    ax.plot(x.strftime("%b-%d"), new_df['Prediction']); 
    plt.xticks(x.strftime("%b-%d"), labels, rotation='vertical')

    plot = df.plot()
    stream = BytesIO()
    plot.figure.savefig(stream)
    stream.seek(0)
    return send_file(stream, mimetype='image/png')

    # ignore last 30 days will used for prediction
    X = stock_df[['High', 'Low', 'Open', 'Close', 'Volume']][:-forecast_out]
    y = stock_df["Adj Close"][:-forecast_out].values.reshape(-1, 1)
    print(X.shape, y.shape)

    # Split the data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Create a StandardScater model and fit it to the training data
    X_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)

    # Transform the training and testing data using the X_scaler and y_scaler models
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    y_train_scaled = y_scaler.transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    # Create a LinearRegression model and fit it to the scaled training data
    model = LinearRegression()
    model.fit(X_train_scaled, y_train_scaled)

    # predict last 30 days data
    x_forecast = stock_df[['High', 'Low', 'Open', 'Close', 'Volume']][-forecast_out:]
    model.fit(X_train, y_train)
    lr_scaled_prediction = model.predict(x_forecast)

    #scaled_df=pd.DataFrame()
    #scaled_df['X_test_scaled']=X_test_scaled

    # Open Price
    #X_train_scaled
    #lst=[]
    #for i in X_train_scaled:
    #    lst.append((i[3]))

    #scaled_df['Adj Close']=lst

    #scaled_df['Adj Close']=
    #lr_prediction
    scaled_df=stock_df[['Adj Close']].tail(30)  
    scaled_df['Predictions']=lr_scaled_prediction
    scaled_df.tail(10)

    # Used X_test_scaled, y_test_scaled, and model.predict(X_test_scaled) to calculate MSE and R2
    # Make predictions using the X_test_scaled data
    # Plot y_test_scaled vs y_test_scaled
    # Scatter plot y_test_scaled vs predictions

    predictions = model.predict(X_test_scaled)
    model.fit(X_train_scaled, y_train_scaled) 
    #len(pred1)

    MSE = mean_squared_error(y_test_scaled, predictions)
    r2 = model.score(X_test_scaled, y_test_scaled)
    print(f"MSE: {round(MSE,6)}, R2: {round(r2,6)}")

    predict_df = stock_df
    len(predictions) 
    X_train_scaled[1]
    y_test_scaled[1]
    plt.scatter(model.predict(X_train_scaled), model.predict(X_train_scaled) - y_train_scaled, c="blue", label=f"{name} Training Data")
    plt.scatter(model.predict(X_test_scaled), model.predict(X_test_scaled) - y_test_scaled, c="orange", label=f"{name} Testing Data")
    plt.legend()
    plt.hlines(y=0, xmin=y_test_scaled.min(), xmax=y_test_scaled.max())
    plt.title(f"{name} Residual Plot")
    plt.show()

    # Model using 1 feature
    plt.style.use('seaborn-darkgrid')
    plt.rc('figure', figsize=(20, 10))
    fig = plt.figure()

    ax = plt.axes()
    x = new_df.index
    labels=x.strftime("%b-%d")

    # plt.subplot(2, 1, 1)
    ax.plot(x.strftime("%b-%d"), new_df['Adj Close'], c="blue", label=f"{name} Adj Close")
    ax.plot(x.strftime("%b-%d"), new_df['Prediction'], c="orange", label=f"{name} Prediction")
    plt.xticks(x.strftime("%b-%d"), labels, rotation='vertical')
    plt.ylabel('Open vs Adj Close', fontsize = 16)
    plt.xlabel("Recent 30 days", fontsize = 16)
    plt.title(f"{name} Linear Regression Model using 1 feature on past 30 days", fontsize = 16)
    plt.legend(fontsize = 16)
    plt.text(18, min(new_df['Prediction']), f"Model Score:    MSE: {round(MSE,6)}     |     R2: {round(r2,6)}", fontsize=16)
    plt.savefig(f"outputs/{name} Trained Model past 30days one feature.png")

    # ML Model using all features
    fig = plt.figure()
    ax = plt.axes()
    x = new_df.index
    labels=x.strftime("%b-%d")
    # plt.subplot(2, 1, 2)
    ax.plot(x.strftime("%b-%d"), scaled_df['Adj Close'].tail(30), c="blue", label=f"{name} Adj Close");
    ax.plot(x.strftime("%b-%d"), scaled_df['Predictions'].tail(30), c="orange", label=f"{name} Prediction"); 
    plt.xticks(x.strftime("%b-%d"), labels, rotation='vertical')
    plt.ylabel('All Parameters', fontsize = 16)
    plt.xlabel("Recent 30 days", fontsize = 16)
    plt.title(f"{name} Linear Regression Model using all features on past 30 days", fontsize = 16)
    plt.legend(fontsize = 16)
    plt.text(18, min(scaled_df['Predictions']), f"Model Score:    MSE: {round(MSE,6)}     |     R2: {round(r2,6)}", fontsize=16)
    plt.savefig(f"outputs/{name} Trained Model past 30days all feature.png")

    # Caluculation of Rank
    pct_gain_loss = (scaled_df['Predictions'] -  scaled_df['Adj Close'])
    # Per Day gain loss predicted 
    #prd_pct = ((310.794221-309.618252)/309.618252)*100
    #round(prd_pct ,2)
    pct_2019_df=pd.DataFrame()
    pct_2019_df['gain_loss_pct']=stock_df['Adj Close'].pct_change() * 100

    #get history rank 
    stock_rnk_df = stock_df
    stock_rnk_df['gain_loss_pct']=pct_2019_df

    pct_2019_df=pct_2019_df.dropna()
    #nw_lst=pct_2019_df.sample(5) 
    #nw_lst['gain_loss_pct'] 

    stock_rnk_df=stock_rnk_df.dropna()
    stock_rnk_df.tail()

    # Get next 1 month Business dates  
    new_dts = pd.bdate_range(start=now_time, end=end_time)
    new_dts 
    abj_close=[range(len(new_dts))]
    future_df=pd.DataFrame({"Date":new_dts})
    future_df.set_index('Date')
    #future_df['future_gain_loss']=pct_2019_df.sample(5)
    future_df_rank=pd.DataFrame({"Date":new_dts})
    future_df_rank.set_index('Date')

    # Rank based on the gain loss pct 
    cols = ['gain_loss_pct']
    #['High','Low','Open','Close','Volume','Adj Close']
    tups = stock_rnk_df[cols].sort_values(cols, ascending=False).apply(tuple, 1)
    f, i = pd.factorize(tups)
    factorized = pd.Series(f + 1, tups.index)

    stock_rnk_df=stock_rnk_df.assign(Rank=factorized)

    stock_rnk_df[stock_rnk_df['Rank']>1].sort_values(by=['Rank']).sample(5)
    #stock_rnk_df['Rank'].sample(5)
    #stock_rnk_df[stock_rnk_df['Rank']==1089]

    # get last day actual Adj close
    lst_rnk=stock_rnk_df['Rank'].tail(1)

    nw_lst=pd.DataFrame()
    nw_lst_rank=pd.DataFrame()
    #get the next n random gain_loss_pct 
    nw_lst['gain_loss_pct']=stock_rnk_df.sample(len(new_dts))['gain_loss_pct']
    nw_lst_rank['gain_loss_pct']=stock_rnk_df[stock_rnk_df['Rank']>lst_rnk[0]].sort_values(by=['Rank']).sample(len(new_dts))['gain_loss_pct']
    #nw_lst=pct_2019_df.sample(5) 
    nw_lst['gain_loss_pct'].head()
    #nw_lst.head()
    stock_rnk_df[stock_rnk_df['Rank']>lst_rnk[0]].sort_values(by=['Rank']).sample(len(new_dts))['gain_loss_pct']

    nw_lst['gain_loss_pct'].head()
    nw_lst_rank['gain_loss_pct'].head()

    # Random % from previous 5 years 
    x_fut_gain_loss = [nw_lst['gain_loss_pct'][i] for i in range(len(nw_lst['gain_loss_pct']))]
    #x_fut_gain_loss_rank = [nw_lst_rank['gain_loss_pct'][j] for j in range(len(nw_lst_rank['gain_loss_pct']))]

    future_df['Fut_PCT']=x_fut_gain_loss
    #future_df_rank['Fut_PCT']=x_fut_gain_loss_rank
    future_df.head()

    fut_adj_open=[]
    fut_adj_close=[]

    # get the previous day open and close and calculated the Adj price
    # for opening price just adding .05 with previous close 
    fut_adj_open.append(stock_df['Adj Close'].tail(1)+ .05)
    fut_adj_close.append(stock_df['Adj Close'].tail(1)+ (stock_df['Adj Close'].tail(1) * future_df['Fut_PCT'][0]/100))

    for i in range(len(new_dts)-1):
        fut_adj_close.append(fut_adj_close[i]+ (fut_adj_close[i] * future_df['Fut_PCT'][i+1]/100))
        fut_adj_open.append(fut_adj_close[i]+ .05)

    x_open=[i[0] for i in fut_adj_open]
    x_adj=[i[0] for i in fut_adj_close]

    future_df['Fut_Open']=x_open
    future_df['Fut_Adj_Close']=x_adj
    future_df.head()

    x_fut_gain_loss_rank = [nw_lst_rank['gain_loss_pct'][j] for j in range(len(nw_lst_rank['gain_loss_pct']))]
 
    future_df_rank['Fut_PCT']=x_fut_gain_loss_rank
    future_df_rank.head()

    #x_Adj_feature=future_df.drop(['Date','Fut_PCT'])
    #x_forecast = np.array(df.drop(['Adj Close'],1))[-forecast_out:]
    x_Adj_feature = np.array(future_df.drop(['Date','Fut_PCT','Fut_Open'],1))
    print((x_Adj_feature))

    # Uer the regression model to predict the future predicted price
    lr_fut_prediction = lr.predict(x_Adj_feature)
    print(lr_fut_prediction)

    future_df['Model_Predict']=lr_fut_prediction
    future_df=future_df.set_index('Date')
    future_df.head()

    nw=future_df[['Fut_Adj_Close','Model_Predict']]
    nw.head()

    scaled_df.tail()
    old_vs_future=scaled_df.tail(9) 

    nw=future_df[['Fut_Adj_Close','Model_Predict']]
    #nw=nw.rename({'Fut_Adj_Close':'Adj Close','Model_Predict':'Prediction'})
    nw.columns=['Adj Close','Predictions']
    old_vs_future=old_vs_future.append(nw)
    old_vs_future.head(15)

    plt.style.use('seaborn-darkgrid')
    plt.rc('figure', figsize=(20, 10))
    # create a color palette
    #plt.style.use('seaborn-whitegrid')
    fig = plt.figure()
    ax = plt.axes()
    x = old_vs_future.index
    labels=x.strftime("%b-%d")
    ax.plot(x.strftime("%b-%d"), old_vs_future['Adj Close'], c="blue", label=f"{name} Adj Close");
    ax.plot(x.strftime("%b-%d"), old_vs_future['Predictions'], c="orange", label=f"{name} 30-day Predicted Close");
    plt.xlabel("Future 30 days", fontsize = 16)
    plt.ylabel(f"Predicted {name} Stock Price", fontsize = 16)
    plt.xticks(x.strftime("%b-%d"), labels, rotation='vertical', fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.legend(fontsize = 16)
    plt.title(f"{name} Linear Regression 30-day Predictive Model using all features", fontsize = 20)
    plt.savefig(f"outputs/{name} 30day prediction.png")

    fut_adj_open_rank=[]
    fut_adj_close_rank=[]
    # get the previous day open and close and calculated the Adj price
    # for opening price just adding .05 with previous close 
 
    fut_adj_open_rank.append(stock_df['Adj Close'].tail(1)+ .05)
    fut_adj_close_rank.append(stock_df['Adj Close'].tail(1)+ (stock_df['Adj Close'].tail(1) * future_df_rank['Fut_PCT'][0]/100))
 
    for i in range(len(new_dts)-1):
        fut_adj_close_rank.append(fut_adj_close_rank[i]+ (fut_adj_close_rank[i] * future_df_rank['Fut_PCT'][i+1]/100))
        fut_adj_open_rank.append(fut_adj_close_rank[i])    
 
    x_open_rank=[j[0] for j in fut_adj_open_rank]
    x_adj_rank=[j[0] for j in fut_adj_close_rank]
 
    future_df_rank['Fut_Open']=x_open_rank
    future_df_rank['Fut_Adj_Close']=x_adj_rank
    future_df_rank.head()

    x_Adj_feature_rank = np.array(future_df_rank.drop(['Date','Fut_PCT','Fut_Open'],1))
    print((x_Adj_feature_rank))

    # Uer the regression model to predict the future predicted price
    lr_fut_prediction_rank = lr.predict(x_Adj_feature_rank)
    print(lr_fut_prediction_rank)

    future_df_rank['Model_Predict']=lr_fut_prediction_rank
    future_df_rank=future_df_rank.set_index('Date')
    future_df_rank.head()

    scaled_df.tail()
    old_vs_future_rank=scaled_df.tail(10) 

    nw=future_df_rank[['Fut_Adj_Close','Model_Predict']]
    #nw=nw.rename({'Fut_Adj_Close':'Adj Close','Model_Predict':'Prediction'})
    nw.columns=['Adj Close','Predictions']
    old_vs_future_rank=old_vs_future_rank.append(nw)
    old_vs_future_rank.head(15)

    plt.style.use('seaborn-darkgrid')
    plt.rc('figure', figsize=(20, 10))
    # create a color palette
    #plt.style.use('seaborn-whitegrid')
    fig = plt.figure()
    ax = plt.axes()
    x = old_vs_future_rank.index
    labels=x.strftime("%b-%d")
    ax.plot(x.strftime("%b-%d"), old_vs_future_rank['Adj Close'], c="blue", label=f"{name} Adj Close");
    ax.plot(x.strftime("%b-%d"), old_vs_future_rank['Predictions'], c="orange", label=f"{name} 30-day Predicted Close");
    plt.xlabel("Future 30 days", fontsize = 16)
    plt.ylabel(f"Predicted {name} Stock Price", fontsize = 16)
    plt.xticks(x.strftime("%b-%d"), labels, rotation='vertical', fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.legend(fontsize = 16)
    plt.title(f"{name} Linear Regression 30-day Predictive Model using all features (Ranked)", fontsize = 20)
    plt.savefig(f"outputs/{name} 30day prediction_rank.png")

    plot = df.plot()
    stream = BytesIO()
    plot.figure.savefig(stream)
    stream.seek(0)
    return send_file(stream, mimetype='image/png')

    app.run(debug=True, port=8000)

if __name__ == "__main__":
    app.run(debug=False, port=8000, host="localhost", threaded=True)
