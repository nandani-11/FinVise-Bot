# -*- coding: utf-8 -*-
"""finvisebot_team10.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PvtZ3MXhG9Q9-wsa_vtMhDhU6MIsQU4C
"""

# -*- coding: utf-8 -*-
"""finvisebot_team10.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1n6wVZn0JPje-1_Ch2M42TBdDSkZ9YfDL
"""

# @title
# Mentioned below is the  v1 code for FinVise bot for BMO submitted as a part of INSY-661 course at McGill Unviersity
# Team members - Adrian Alarcon Delgado, Nandani Yadav, Yash Joshi

# Python code to display bank services based on customer type (New/Existing) and service selection
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import datetime
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import load_model, save_model
from keras import backend as K
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')
print("Are you New/Existing customer? 1/New 2/Existing")
def next_day(day):
    one_day = datetime.timedelta(days=1)
    while True:
        day += one_day
        if day.weekday() == 0:  # Lunes
            return day



def main_process(user):
  import yfinance as yf
  import numpy as np
  import matplotlib.pyplot as plt
  from datetime import date
  import datetime
  import pandas as pd
  import re
  from sklearn.preprocessing import StandardScaler
  from keras.models import Sequential
  from keras.layers import Dense
  from keras.layers import LSTM
  from keras.layers import Dropout
  from keras.models import load_model, save_model
  from keras import backend as K
  import pandas as pd
  import pickle
  import warnings
  user = int(user)

  if user==1:
    print("Which service do you want? 1/Overview of services 2/ATM/Branch Locator")

    service = int(input())

    if service==1:
        import numpy as np
        import re
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        # Scenarios and responses
        scenarios = {
        "greeting": "Hello! Thanks for contacting the bank support chatbot. How can I help you today?",

        "balance": "To check your account balance, please log in to our online or mobile banking platform. You can also call our automated line at 1-877-225-5266.",

        "savings interest": "The current interest rate on our regular savings accounts is 1.5% APY. Rates are variable and subject to change.",

        "loan types": "We offer personal loans, auto loans, home equity loans, and business loans. Personal loans are unsecured and have terms from 1-5 years. Auto and home equity loans are secured by the asset being financed.",

        "credit card": "To apply for a credit card, fill out the online application on our website. We offer low-interest cards as well as rewards cards. All cards have a 25 day interest-free period for new purchases.",

        "online banking": "Signing up for online banking allows you to check balances, view statements, pay bills and transfer funds from your computer or mobile device. Registration takes just a few minutes to complete online.",

        "investments": "We provide a range of investment services from high-interest savings accounts to long-term retirement planning options like IRAs and brokerage accounts.",

        "business services": "Our business banking packages include checking and savings accounts, business loans, payroll processing, and merchant card services designed for companies of all sizes.",

        "atm location": "You can locate the nearest ATM by entering your zip code on our bank's website or mobile app. ATM transactions at non-affiliated machines are surcharge-free up to 3 times per month.",

        "contact info": "For general inquiries, please call us at XXX-XXX-XXXX or visit your local branch during business hours. For account-specific questions, please use online or mobile banking secure messaging."
        }

        THRESHOLD = 0.1

        # Text cleaning function
        def clean_text(text):
          text = text.lower()
          text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
          return text

        # Similarity score calculation
        def similarity(user_input, scenario):
          tfidf_vectorizer = TfidfVectorizer().fit([user_input, scenario])
          tfidf_matrix = tfidf_vectorizer.transform([user_input, scenario])
          return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

        # Chatbot logic
        print("Welcome! How can I help you today?")
        while True:
          user_input = input(">>")
          if user_input.lower() == 'exit':
            break

          cleaned_input = clean_text(user_input)

          scores = [similarity(cleaned_input, key) for key in scenarios]

          best_match_index = np.argmax(scores)

          if scores[best_match_index] > THRESHOLD:
            print(scenarios[list(scenarios)[best_match_index]])
          else:
            print("Please contact a representative for assistance.")

          print("Anything else I can help with? 1 for Yes, 2 for No: ")

          if user_input.lower() == '2':
            print("Thank you for contacting us. Have a nice day!")
            break



    elif service==2:
        import requests
        from IPython.display import display, HTML

        # Get location and search type from user
        print("Enter 1 for branch or 2 for ATM: ")
        search_type = input()

        user_location = input("Enter your location: ")

        # API key and other params
        api_key = 'AIzaSyBe7LT1JuNbVAjT7MOOny1iHGC0t-3DOuY'
        radius = 1000

        # Geocode location
        geocode_url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {'address': user_location, 'key': api_key}
        geo_response = requests.get(geocode_url, params=params).json()
        location = geo_response['results'][0]['geometry']['location']

        # Build search URL
        url = 'https://maps.googleapis.com/maps/api/place/nearbysearch/json'

        if search_type == "1":
          # bank search parameters
          params = {
            'location': f"{location['lat']},{location['lng']}",
            'radius': radius,
            'type': 'bank',
            'keyword': 'bmo',
            'key': api_key
          }
        else:
          # atm search parameters
          params = {
            'location': f"{location['lat']},{location['lng']}",
            'radius': radius,
            'type': 'atm',
            'keyword': 'bmo',
            'key': api_key
          }

        # Make request and get results
        results = requests.get(url, params=params).json()

        # Check for results
        if len(results['results']) == 0:
          print("No results found within search radius")
        else:
          nearest = results['results'][0]

          # Display name and address
          print(nearest['name'])
          print(nearest['vicinity'])

          # Get lat/lng for map
          lat = nearest['geometry']['location']['lat']
          lng = nearest['geometry']['location']['lng']

          # Construct HTML for Google Maps static image
          map_url = f"https://maps.google.com/maps?q={lat},{lng}&z=16&output=embed"
          map_html = f"""
          <iframe
            width="600"
            height="450"
            frameborder="0" style="border:0"
            src="{map_url}" allowfullscreen>
          </iframe>
          """

          # Display map
          display(HTML(map_html))

  elif user==2:
    print("Which service do you want? 1/Investment Recommendations 2/Portfolio Analysis 3/Market Insights 4/Loan Repayments 5/ATM/Branch Locator")
    service = int(input())

    if service==1:
          # Investment strategy options
          strategies = [
            "100% Debt",
            "40% Equity, 60% Debt",
            "60% Equity, 40% Debt",
            "80% Equity, 20% Debt",
            "100% Equity"
          ]

          etfs = [
          'BMO Fixed Income ETF Portfolio\nBMO USD Income ETF Portfolio\nBMO USD Conservative ETF Portfolio\nBMO Income ETF Portfolio\nBMO Conservative ETF Portfolio',
          'BMO Balanced ETF Portfolio\nBMO Monthly Dividend Fund Ltd.\nBMO Diversified Income Portfolio\nBMO Monthly Income Fund\nBMO Canadian Income & Growth Fund',
          'BMO Growth ETF Portfolio\nBMO Asset Allocation Fund\nBMO Global Monthly Income Fund\nBMO Sustainable Global Balanced Fund\nBMO Concentrated Global Balanced Fund',
          'BMO Equity Growth ETF Portfolio\nBMO Global Equity Fund\nBMO Global Low Volatility ETF Class\nBMO Dividend Fund\nBMO Asian Growth and Income Fund',
          'BMO Canadian Equity Fund\nBMO Global Equity Fund\nBMO U.S. Equity Growth MFR Fund\nBMO ARK Innovation Fund\nBMO Global Low Volatility ETF Class'
          ]

          # Investment goals
          goals = [
            "Retirement",
            "Education fund",
            "Car Fund",
            "Emergency fund",
            "Savings for a Purchase"
          ]

          # Risk profiles
          risks = [
            "Low",
            "Medium",
            "High",
            "Very high",
            "Aggressive"
          ]

          # Investment horizons
          horizons = [
          "Very Long Term (Over 10 Years)",
          "Long Term (5-10 Years)",
          "Medium Term (1-5 Years)",
          "Short Term (6 Months - 1 Year)",
          "Very Short Term (Under 6 Months)",
          ]

          # Show goal options
          print("Investment Goals:")
          for num, goal in enumerate(goals, 1):
              print(f"{num}. {goal}")

          # Take goal input
          goal_num = int(input("Select goal number: "))

          # Similarly for risk...
          print("Risk Profiles:")
          for num, risk in enumerate(risks, 1):
            print(f"{num}. {risk}")

          risk_num = int(input("Select risk number: "))

          # Similarly for horizon...
          print("Investment Horizons:")
          for num, horizon in enumerate(horizons, 1):
              print(f"{num}. {horizon}")

          horizon_num = int(input("Select horizon number: "))

          # Calculating Total Score

          total_risk_score = goal_num+risk_num+horizon_num

          # Calculating Index - Mapping 3-15 to 0-4

          strategy_index = int((total_risk_score/3)-1)

          # Recommending a strategy based on the total risk score

          recommended_strategy = strategies[strategy_index]

          # Printing the strategy to the end user

          print(f"Based on your goal of {goals[goal_num-1]}, risk profile of {risks[risk_num-1]} and horizon of {horizons[horizon_num-1]},")
          print(f"the recommended investment strategy is: {recommended_strategy}")

          see_funds = int(input(f"See list of funds at your own discretion related to {recommended_strategy}? 1 for Yes, 2 for No: "))

          if see_funds == 1:
            print("List of funds recommended for you - \n \n"+etfs[strategy_index])
          else:
            print("Ok, thanks for using this tool!")

    elif service==2:
          import numpy as np
          import pandas as pd
          import matplotlib.pyplot as plt

          np.random.seed(0)

          #Dictionary to store portfolio data of users
          portfolios = {}

          #Function to generate mock portfolio data for a user
          def generate_portfolio(user_id):
              industries = ['Technology', 'Healthcare', 'Financials', 'Consumer Goods']

              portfolio_composition = {
              industry: np.random.randint(10,30) for industry in industries
              }

              total_value = sum(portfolio_composition.values())

              time_series = np.empty(12)

              last_month = total_value
              time_series[-1] = last_month

              for i in range(11):
                  variation = np.random.uniform(-0.1,0.1)
                  current_month = last_month * (1 + variation)
                  time_series[-2-i] = current_month
                  last_month = current_month

              portfolios[user_id] = {
              'Portfolio Composition': portfolio_composition,
              'Total Value': total_value,
              'Value over Time': time_series
              }

          for i in range(10):
              user_id = f'User{i+1}'
              generate_portfolio(user_id)

          # Ask for user ID
          user_id = input("Enter your user ID: ")

          # Check if user ID exists in portfolios dictionary
          if user_id in portfolios:
              user_data = portfolios[user_id]

              # Display total portfolio value
              print(f"Total portfolio value for {user_id}: {user_data['Total Value']}$ \n")

              # Display portfolio composition by industry
              composition = user_data['Portfolio Composition']
              industry_names = list(composition.keys())
              industry_values = list(composition.values())

              plt.bar(industry_names, industry_values)
              plt.xlabel('Industry')
              plt.ylabel('Value')
              plt.title(f"Portfolio Composition for {user_id}")
              plt.show()

              # Display value over time
              time_series = user_data['Value over Time']
              months = range(1, len(time_series) + 1)

              plt.plot(months, time_series)
              plt.xlabel('Month')
              plt.ylabel('Value')
              plt.title(f"Portfolio Value over Time for {user_id}")
              plt.show()

          else:
              print("User ID not found in portfolios.")


    elif service==3:



        def create_dataset(x,y,time_step = 1):
          xs = []
          ys = []
          for i in range(len(x) - time_step):
            v = x.iloc[i:(i+time_step)].values
            xs.append(v)
            ys.append(y.iloc[i+time_step])
          return np.array(xs), np.array(ys)


        end_date = datetime.date.today()
        print(end_date)
        next_date = next_day(end_date)
        tickers = pd.read_csv('tickers.csv')
        start_date = end_date + datetime.timedelta(days=-700)
        start_date = datetime.datetime.strftime(start_date, "%Y-%m-%d")
        # Set the ticker

        ticker = int(input('Welcome to the market insights analysis. Please, choose one company to see a forecasting for the stock price (1: Google, 2: Tesla, 3: Amazon, 4: Other: '))

        stocks = {1:'GOOG',2:'TSLA',3:'AMZN'}

        if ticker in list(stocks.keys()):
            ticker_lab = stocks[ticker]
        # Get the data
            data = yf.download(ticker_lab, start_date, end_date)
            last_price = data.iloc[-1,:]['Open']
            last_date  = datetime.datetime.strftime(data.iloc[-1].name, "%Y-%m-%d")
            cur_date = datetime.datetime.strptime(last_date, "%Y-%m-%d") + datetime.timedelta(days=1)
            cur_date = datetime.datetime.strftime(cur_date, "%Y-%m-%d")
            next_date_lab = datetime.datetime.strftime(next_date, "%Y-%m-%d")
            if ticker == 1:
                model = load_model('model_google.h5')

                with open('esc_google.pkl' , 'rb') as f:
                    esc = pickle.load(f)
            elif ticker == 2:
                model = load_model('model_tesla.h5')

                with open('/content/esc_tesla.pkl' , 'rb') as f:
                    esc = pickle.load(f)
            elif ticker == 3:
                model = load_model('model_amazon.h5')


                with open('/content/esc_amazon.pkl' , 'rb') as f:
                    esc = pickle.load(f)
            y_value = esc.transform(last_price.reshape(1,-1))
            y_value = y_value.reshape(1,1,1)
            y_predict = model.predict(y_value)
            y_pred_conv = round(esc.inverse_transform(y_predict[0][0][0].reshape(1,-1))[0][0],2)
            print(f'According with our analysis, the stock price of {ticker_lab} for the date {next_date_lab}  will be $ {y_pred_conv :.2f}')

        else:
            pattern = input('Insert the name of the company: ')

            string_list = list(tickers['Name'].values)

            regex_pattern = re.compile(rf'{re.escape(pattern)}', re.IGNORECASE)
            options = []

            for string in string_list:
                if regex_pattern.search(string):
                    options.append(string)


            df_options = tickers[tickers['Name'].isin(options)][['Symbol','Name']].reset_index()
            while df_options.shape[0]==0:

                pattern = input('Sorry, I did not find options. Please try with another words: ')
                string_list = list(tickers['Name'].values)

                regex_pattern = re.compile(rf'{re.escape(pattern)}', re.IGNORECASE)
                options = []

                for string in string_list:
                    if regex_pattern.search(string):
                        options.append(string)
                df_options = tickers[tickers['Name'].isin(options)][['Symbol','Name']].reset_index()

            print('I found the following options.')
            print(df_options)

            ticker_lab = input('Please enter the symbol of the company selected: ')
            while ticker_lab not in df_options['Symbol'].values:
                ticker_lab = input('Remember to enter the symbol. You can see the options below. If you want to exit, write exit: ')
                print(df_options)
                if ticker_lab == 'exit':
                    break
            #try:
            data = yf.download(ticker_lab, start_date, end_date)
            series = data[['Close']]
            test_size = 30
            index = len(series) - test_size
            train = series[:index]
            test = series[index:]
            esc = StandardScaler()
            train_esc = pd.DataFrame(esc.fit_transform(train), columns= ['Close'])
            test_esc = pd.DataFrame(esc.transform(test), columns = ['Close'])
            time_lag = 1
            x_train, y_train = create_dataset(train_esc, train_esc.Close, time_lag)
            x_test, y_test = create_dataset(test_esc, test_esc.Close, time_lag)
            K.clear_session()

            model = Sequential()
            model.add(LSTM(units = 250, return_sequences = True, input_shape = (x_train.shape[1], x_train.shape[2])))
            model.add(Dropout(rate =  0.05))
            model.add(LSTM(units = 250, return_sequences = True))
            model.add(Dropout(rate = 0.05))
            model.add(LSTM(units = 250, return_sequences = True))
            model.add(LSTM(units = 100, return_sequences = True))
            model.add(Dropout(rate = 0.01))
            model.add(Dense(units = 1))
            model.compile(optimizer =  'adam', loss = 'mean_squared_error')
            history = model.fit(x_train, y_train, epochs = 50, shuffle = False, batch_size = 32)
            last_price = data.iloc[-1,:]['Open']
            last_date  = datetime.datetime.strftime(data.iloc[-1].name, "%Y-%m-%d")
            cur_date = datetime.datetime.strptime(last_date, "%Y-%m-%d") + datetime.timedelta(days=1)
            next_date = next_day(cur_date)
            cur_date = datetime.datetime.strftime(cur_date, "%Y-%m-%d")
            next_date_lab = datetime.datetime.strftime(next_date, "%Y-%m-%d")
            y_value = esc.transform(last_price.reshape(1,-1))
            y_predict = model.predict(y_value)
            preds = model.predict(test_esc)

            y_test_conv = esc.inverse_transform(y_test.reshape(-1, 1))
            y_preds_conv = esc.inverse_transform(preds[:,0][:,0].reshape(-1, 1))
            y_train_conv = esc.inverse_transform(y_train.reshape(-1, 1))
            y_pred_conv = round(esc.inverse_transform(y_predict[0][0][0].reshape(1,-1))[0][0],2)
            print(f'According with our analysis, the stock price of {ticker_lab} for the date {next_date_lab} will be $ {y_pred_conv :.2f}')

            # Your existing code for plotting
            plt.figure(figsize=(15, 6))
            plt.plot(np.arange(len(train_esc), len(train_esc) + len(test_esc)-1), y_test_conv, label='True')
            plt.plot(np.arange(len(train_esc), len(train_esc) + len(test_esc)), y_preds_conv, label='Prediction', color='red')
            plt.plot(np.arange(0, len(train_esc)-1), y_train_conv, label='History', color='green')
            plt.legend()
            plt.show()



            ##except:
            #    print('Thank you for using the chatbot service')


    elif service==4:

          import random

          random.seed(42)

          # Generate data for 10 customers
          loan_data = {}
          for i in range(1, 11):
            cust_id = f"User{i}"

            principal = random.randint(5000, 20000)
            interest_rate = random.randint(3, 10)
            tenure = random.randint(12, 84)
            paid_months = random.randint(0, tenure)

            loan_data[cust_id] = {
              "principal": principal,
              "interest_rate": interest_rate,
              "tenure": tenure,
              "paid_months": paid_months
            }

          # Get customer ID input
          cust_id = input("Enter User ID: ")

          # Check if customer ID exists in data
          if cust_id in loan_data:

            # Get data for that customer
            data = loan_data[cust_id]

            principal = data["principal"]
            interest_rate = data["interest_rate"]
            tenure = data["tenure"]
            paid_months = data["paid_months"]

            # Calculate remaining months
            remaining_months = tenure - paid_months

            # Convert interest rate to monthly
            monthly_interest_rate = interest_rate / 12 / 100

            # Calculate interest for remaining months
            interest_remaining = principal * monthly_interest_rate * remaining_months

            # Calculate total interest on original loan
            total_interest = principal * monthly_interest_rate * tenure

            # Calculate total interest already paid
            interest_paid = total_interest - interest_remaining

            # Calculate remaining principal
            principal_remaining = principal - (interest_paid)

            # Calculate remaining payment
            remaining_payment = principal_remaining + interest_remaining

            # Calculate monthly payment
            monthly_payment = remaining_payment / remaining_months

            print("Loan Details:")
            print(f"Principal: {principal}")
            print(f"Interest Rate (yearly): {interest_rate}%")
            print(f"Tenure: {tenure} months")
            print(f"Paid Months: {paid_months} months")

            print("\nRepayment Schedule:")
            print(f"Remaining Months: {remaining_months}")
            print(f"Monthly Payment: {monthly_payment}")

            # Existing code

            import matplotlib.pyplot as plt

            paid_amount = monthly_payment * paid_months
            formatted_paid = '${:.2f}'.format(paid_amount)
            formatted_pending = '${:.2f}'.format(remaining_payment)

            # Create pie chart
            plt.pie([paid_amount, remaining_payment],
                    labels=[f'Paid: {formatted_paid}', f'Pending: {formatted_pending}'],
                    autopct='%.2f%%')

            plt.title('Total Payment Breakdown')
            plt.show()

    elif service==5:
            import requests
            from IPython.display import display, HTML

            # Get location and search type from user
            print("Enter 1 for branch or 2 for ATM: ")
            search_type = input()

            user_location = input("Enter your location: ")

            # API key and other params
            api_key = 'AIzaSyBwAwx4LVNF9CEFF9u89bgUgAN2TeaAp_g'
            radius = 1000

            # Geocode location
            geocode_url = "https://maps.googleapis.com/maps/api/geocode/json"
            params = {'address': user_location, 'key': api_key}
            geo_response = requests.get(geocode_url, params=params).json()
            location = geo_response['results'][0]['geometry']['location']

            # Build search URL
            url = 'https://maps.googleapis.com/maps/api/place/nearbysearch/json'

            if search_type == "1":
              # bank search parameters
              params = {
                'location': f"{location['lat']},{location['lng']}",
                'radius': radius,
                'type': 'bank',
                'keyword': 'bmo',
                'key': api_key
              }
            else:
              # atm search parameters
              params = {
                'location': f"{location['lat']},{location['lng']}",
                'radius': radius,
                'type': 'atm',
                'keyword': 'bmo',
                'key': api_key
              }

            # Make request and get results
            results = requests.get(url, params=params).json()

            # Check for results
            if len(results['results']) == 0:
              print("No results found within search radius")
            else:
              nearest = results['results'][0]

              # Display name and address
              print(nearest['name'])
              print(nearest['vicinity'])

              # Get lat/lng for map
              lat = nearest['geometry']['location']['lat']
              lng = nearest['geometry']['location']['lng']

              # Construct HTML for Google Maps static image
              map_url = f"https://maps.google.com/maps?q={lat},{lng}&z=16&output=embed"
              map_html = f"""
              <iframe
                width="600"
                height="450"
                frameborder="0" style="border:0"
                src="{map_url}" allowfullscreen>
              </iframe>
              """

              # Display map
              display(HTML(map_html))

  print("Do you want more services? 1/Yes 2/No")
  more = int(input())

  if more==1:
    print("Starting services selection again")
    print("Are you New/Existing customer? 1/New 2/Existing")
    message = input()
    main_process(message)
  elif more==2:
    print("Thank you for banking with us. Have a nice day!")


if __name__ == "__main__":
    # Lee la entrada del usuario desde stdin (archivo asociado al proceso subprocess)
    message = input()
    response = main_process(message)

    # Imprime la respuesta para que se pueda capturar en el backend
    print(response)