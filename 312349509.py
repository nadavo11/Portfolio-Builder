# None of these imports are strictly required, but use of at least some is strongly encouraged
# Other imports which don't require installation can be used without consulting with course staff.
# If you feel these aren't sufficient, and you need other modules which require installation,
# you're welcome to consult with the course staff.

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import datetime
from datetime import date
import itertools as it
import math
import yfinance as yfi
from typing import List
import matplotlib as mpl
import matplotlib.pyplot as plt


class PortfolioBuilder:
    def __init__(self):
        self.datafra = []

    def plot_data(self, daily_data, title):

        plt.plot(daily_data['Adj Close'])

        plt.title(title)
        locator = mpl.dates.MonthLocator()
        fmt = mpl.dates.DateFormatter('%b')

        plt.xlabel('date')
        plt.ylabel('Close Price USD [$]')

        plt.xticks(rotation=70)
        plt.legend(daily_data['Adj Close'])

        plt.show()
        # daily_data['Close'].plot.scatter("Date","AAPL")
        return daily_data["Adj Close"]

    def get_daily_data(self, tickers_list: List[str],
                       start_date: date,
                       end_date: date = date.today()
                       ) -> pd.DataFrame:

        """daily_data = web.get_data_yahoo(tickers_list, start="2017-01-01", end="2017-04-30")"""


        tickers = ' '.join(tickers_list)
        try:
            daily_data = yfi.download(tickers, start_date+datetime.timedelta(days =1), end_date +datetime.timedelta(days =1),threads = False)
            self.num_of_tickers = len(tickers_list)
            self.datafra = daily_data['Adj Close']
            self.days = np.ma.size(daily_data, 1)
            title = str(start_date) + '-' + str(end_date) + ' prices'
            return daily_data['Adj Close']
        except Exception as e:
            print("all fine, try again")
            raise ValueError

        #self.plot_data(daily_data,title)



        """
        get stock tickers adj_close price for specified dates.

        :param List[str] tickers_list: stock tickers names as a list of strings.
        :param date start_date: first date for query
        :param date end_date: optional, last date for query, if not used assumes today
        :return: daily adjusted close price data as a pandas DataFrame
        :rtype: pd.DataFrame

        example call: get_daily_data(['GOOG', 'INTC', 'MSFT', ''AAPL'], date(2018, 12, 31), date(2019, 12, 31))
        """


    def fillup_omega(self,a,m):
        possible_vals = np.arange(0, dtype = float, step = 1/a, stop = 1.0001)

        omega = [x for x in it.product(possible_vals, repeat = m) if round(np.sum(x)*1000) == 1000 ]

        return omega
    def get_all_x_vectors(self,data):
        i = 0
        x_list = []
        prev_day = None

        for index, day in data.iterrows():
            day_prices = np.array([x for x in day])
            if i != 0:

                x_list.append(day_prices / prev_day_prices)
            if i == 0:
                pass
            i += 1
            prev_day_prices = day_prices
        #print (x_list)
        return np.array(x_list)

    def calculate_all_wealthes_for_b(self,x,b):
        S_list = np.array([np.dot(x[0],b)])

        for i,day in enumerate(x):
            if i == 0:
                continue
            else:
                S_list =np.append(S_list,[S_list[i-1] * np.dot(x[i], b)], axis=0)

        return S_list

    def calculate_next_b(self, omega, S_lists):
        omega = np.array(omega) #make omega an array

        sigma_S = np.sum(S_lists, axis =0) #sum every colomn of S lists for the"MECHANE"
                                            # to get each days sum of wealthes
        m = self.num_of_tickers

        next_b_list = [[1/m for i in range(m)]]         #start next portfolio list as a "boring" option
        for day_index in range(len(S_lists[0])):         #for each day fromday 0
            mone = 0
            for b_index in range(len(omega)):              #calc for each b
                mone = mone + omega[b_index] * S_lists[b_index][day_index]

                """                                              multiply this b VECTOR with its own wealth up to the 
                                                                 relevant day               """
                                                     #sum al these [3 dim] VECTORS for "MECHANE"

            next_b_list.append(mone/sigma_S[day_index])
        next_b_list = np.array(next_b_list)
        return next_b_list

    def find_universal_portfolio(self, portfolio_quantization: int = 20) -> List[float]:


        no_of_tickers = np.ma.size(self.datafra, 0)

        omega = self.fillup_omega(portfolio_quantization,no_of_tickers)
        x_list = self.get_all_x_vectors(self.datafra)
        S_lists = np.array([])

        for index, b in enumerate(omega):
            S_list = self.calculate_all_wealthes_for_b(x_list,b)
            if index ==0:
                S_lists = np.array([S_list])
            else:
                S_lists = np.append(S_lists, [S_list], axis=0)


        S_lists = np.array(S_lists)


        next_b_list = self.calculate_next_b(omega,S_lists)

        return_val = [1.0]

        for i, x in enumerate(x_list):
            return_val.append( return_val[i]*np.dot(next_b_list[i],x))

        #return_val = np.sum(next_b_list, axis = 1)

        return return_val

        """
        calculates the universal portfolio for the previously requested stocks
        :param int portfolio_quantization: size of discrete steps of between computed portfolios. each step has size 1/portfolio_quantization
        :return: returns a list of floats, representing the growth trading  per day
        """


    def find_exponential_gradient_portfolio(self, learn_rate: float = 0.5) -> List[float]:
        x_list = self.get_all_x_vectors(self.datafra)
        m = len(x_list[0])                      #no. of tickers
        b_list = np.array([[(1 / m) for x in x_list[0]]])
                                                # starts with  a 'naive' approach
        S_list = np.array([1.0])
        for day_index, x in enumerate(x_list):  #this loop goes over each day

            mechane= 0
            next_b = np.array([])
            for i, xi in enumerate(x):
                mechane += b_list.item(day_index, i) * np.exp((learn_rate * xi) / np.dot(x, b_list[day_index]))

            for i, xi in enumerate(x):       # this runs for each entery
                """ exp is exp argument , we build next b
                 """
                exp = np.array([np.exp((learn_rate*xi)/np.dot(b_list[day_index],x))])
                mone = b_list.item((day_index, i)) * exp
                """
                            after this loop we have the MONE 
                """
                next_b = np.append(next_b, mone/mechane)

            b_list = np.append(b_list, [next_b], axis=0)

        S_list = [1.0]
        for t, xt in enumerate(x_list):

            nxt = S_list[t]*np.dot(x_list[t],b_list[t])
            S_list.append(nxt)




        return S_list


        """
        calculates the exponential gradient portfolio for the previously requested stocks

        :param float learn_rate: the learning rate of the algorithm, defaults to 0.5
        :return: returns a list of floats, representing the growth trading  per day
        """
        pass


if __name__ == '__main__':  # You should keep this line for our auto-grading code.
    print('write your tests here')  # don't forget to indent your code here!
    pfb = PortfolioBuilder()

    data = pfb.get_daily_data( ['GOOG',  "MSFT"], date(2018,1,1), date(2020,2,1))

    # print(data)
    uni =pfb.find_universal_portfolio()
    expgrad =pfb.find_exponential_gradient_portfolio()
    print("exp:",expgrad)
    print( "uni:",uni)
    print("goodbye world")
