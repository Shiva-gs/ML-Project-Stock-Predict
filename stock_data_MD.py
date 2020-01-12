from datetime import datetime
from concurrent import futures

import pandas as pd
from pandas import DataFrame
import pandas_datareader.data as web

def download_stock(stock):
	""" try to query the iex for a stock, if failed note with print """
	try:
		print(stock)
		stock_df = web.DataReader(stock,'yahoo', start_time, now_time)
		stock_df['Name'] = stock
		output_name = 'individual_stocks_5yr/' + stock + '_data.csv'
		stock_df.to_csv(output_name)
	except:
		bad_names.append(stock)
		print('bad: %s' % (stock))

if __name__ == '__main__':

	""" set the download window """
	now_time = datetime.now()
	start_time = datetime(now_time.year - 5, now_time.month , now_time.day)

    # Scraping Wikipedia with Pandas. Define urls
url_s_and_p100 = 'https://en.wikipedia.org/wiki/S%26P_100'
url_s_and_p500 = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
url_dow30 = 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average'

# Scraping S&P100
table100 = pd.read_html(url_s_and_p100)
stocks100 = table100[2]
sp100_list = stocks100['Symbol'].to_list()
# Scraping S&P500
table500 = pd.read_html(url_s_and_p500)
stocks500 = table500[0]
sp500_list = stocks500['Symbol'].tolist()
# Scraping DJIA (Dow30)
table30 = pd.read_html(url_dow30)
stocks30 = table30[1]
stocks30 = stocks30['Symbol'].str.strip('NYSE:\xa0')
dow30_list = stocks30.tolist()

# list of s_anp_p companies 
s_and_p = sp500_list
sp100 = sp100_list
dow30 = dow30_list
		
bad_names =[] #to keep track of failed queries

    # here we use the concurrent.futures module's ThreadPoolExecutor
	# to speed up the downloads buy doing them in parallel 		# as opposed to sequentially

	#set the maximum thread number
max_workers = 50

workers = min(max_workers, len(s_and_p)) #in case a smaller number of stocks than threads was passed in
with futures.ThreadPoolExecutor(workers) as executor:
    res = executor.map(download_stock, s_and_p)

	
#  Save failed queries to a text file to retry
if len(bad_names) > 0:
    with open('failed_queries.txt','w') as outfile:
        for name in bad_names:
            outfile.write(name+'\n')

#timing:
finish_time = datetime.now()
duration = finish_time - now_time
minutes, seconds = divmod(duration.seconds, 60)
print('getSandP_threaded.py')
print(f'The threaded script took {minutes} minutes and {seconds} seconds to run the S&P500 list of stocks.')
#The threaded script took 0 minutes and 31 seconds to run.