import sqlite3
import nltk
import numpy as np
from nltk.corpus import wordnet


#category 1
#fetch data saved in database according to captured entity(entities)
def get_balance(req):
    value = 0
    conn = sqlite3.connect('Financial_Mary.db')
    c = conn.cursor()
    if req[0] != '':
        if req[1] != '':
            if req[2] != '':
                c.execute("""SELECT SUM(balance) FROM bank_accounts
                    WHERE bank_name = '{}' AND account_id = '{}'
                    AND account_type = '{}'""".format(req[0],req[1],req[2]))
            else:
                c.execute("""SELECT SUM(balance) FROM bank_accounts
                    WHERE bank_name = '{}' AND account_id = '{}'
                    """.format(req[0],req[1]))
        else:
            if req[2] != '':
                c.execute("""SELECT SUM(balance) FROM bank_accounts
                    WHERE bank_name = '{}'
                    AND account_type = '{}'""".format(req[0],req[2]))
            else:
                c.execute("""SELECT SUM(balance) FROM bank_accounts
                    WHERE bank_name = '{}'""".format(req[0]))
    else:
        if req[1] != '':
            if req[2] != '':
                c.execute("""SELECT SUM(balance) FROM bank_accounts
                    WHERE account_id = '{}'
                    AND account_type = '{}'""".format(req[1],req[2]))
            else:
                c.execute("""SELECT SUM(balance) FROM bank_accounts
                    WHERE account_id = '{}'
                    """.format(req[1]))
        else:
            if req[2] != '':
                c.execute("""SELECT SUM(balance) FROM bank_accounts
                    WHERE account_type = '{}'""".format(req[2]))

            else:
                c.execute("""SELECT SUM(balance) FROM bank_accounts""")

    value = c.fetchall()[0][0]
    print('your balance: {}'.format(value))
    conn.close()



#category 2
#calculate monthly earning minus monthly spending
def get_budget():
    conn = sqlite3.connect('Financial_Mary.db')
    c = conn.cursor()

    c.execute('SELECT SUM(amount_monthly) FROM in_and_out')
    budget = c.fetchall()[0][0]
    conn.close()
    print('your monthly Earning {}'.format(budget))



#category 3
#give Clerkie advice
def is_affordable(price):
    price_rep = {'b':1000000000, 'm':1000000, 'k':1000}
    if price[-1] in price_rep:
        num = float(price[:-1])
        price_f = float(num)*price_rep[price[-1]]
    else:
        price_f = float(price)

    conn = sqlite3.connect('Financial_Mary.db')
    c = conn.cursor()

    c.execute("SELECT * FROM neighbourhoods WHERE area = 'beverly_hills'")
    parameters = c.fetchall()[0][1:]

    c.execute("SELECT * FROM mortgage WHERE over_1500 = true")
    mortgage = c.fetchall()[0][1:]

    c.execute("""SELECT SUM(balance) FROM bank_accounts""")
    balance = c.fetchall()[0][0]

    c.execute("SELECT SUM(amount_monthly) FROM in_and_out WHERE source!= 'rent'")
    budget = c.fetchall()[0][0]

    conn.close()

    #combine house price change rate
    #from long term and short term
    long_term_rate = calc_house_monthly_rate(parameters[0],parameters[1])
    short_term_rate = calc_house_monthly_rate(parameters[2],parameters[3])
    predicted_rate = long_term_rate + short_term_rate

    down_payment = price_f*mortgage[0]

    APR = mortgage[1]


    #Mary can't afford downpayment
    if down_payment > balance:
        print('You are ${} short from 20% down payment'.format(down_payment-balance))

    else:
        monthly_payment = calc_monthly_pay(price_f-down_payment, APR/12, 360)
        print('Your current budget {}'.format(int(budget)))
        print('Your monthly payment will be {}'.format(int(monthly_payment)))


        #Mary can't afford monthly payment
        if  monthly_payment > budget:
            print('You are ${} short from monthly payment'.format(int(monthly_payment-budget)))

        #Mary can buy the house!
        else:
            print("""Congratulations! You can buy this house
            and save {} each month""".format(int(budget - monthly_payment)))


    #Clerkie's prediction of house price in next 6 months
    #based on house price change rate
    print("""\nHouse price now: {}\nClerkie predicted house price after 6 months will be: {}"""
          .format(price,int(price_f*(1+predicted_rate)**6)))

#calculate house price monthly changing rate
def calc_house_monthly_rate(rate,months):
    return np.power(1+rate, 1/months) - 1


#calculate mortgate monthly payment
def calc_monthly_pay(L,c,n):
    P = L*(c*(1 + c)**n)/((1 + c)**n - 1)
    return P

def loan_question():
    print('having a loan question? Visit')
    print('https://www.wellsfargo.com/goals-credit/smarter-credit/credit-101/getting-a-loan/')

def mortgage_FAQ():
    print('having a mortgage question? Visit')
    print('https://www.chase.com/mortgage/home-loans/faqs')

def spending():
    conn = sqlite3.connect('Financial_Mary.db')
    c = conn.cursor()

    c.execute("SELECT SUM(amount_monthly) FROM in_and_out WHERE type != 'income'")
    budget = c.fetchall()[0][0]
    conn.close()
    print('your monthly spending {}'.format(budget))
