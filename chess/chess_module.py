import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

np.set_printoptions(precision=4, suppress=True)

# Set constants
HALF_LIFE = 1. # Decay rate of memory in days
REFRESH = 7. # Amount of time since that I NEED to refresh in days
SPACED_REP = .85 # When I need to retest based on percentage of knowledge

def time_to_know(time_since, half_life):
    """Convert time since values into how well I know the lines [0,1]"""

    decay = np.log(2) / half_life
    output = np.exp(-decay * time_since)
    return output

def time_to_prob(time_since, half_life):
    """Return probability of testing base on `time_since`"""

    BELL_HEIGHT = 4

    decay = np.log(2) / half_life
    std = np.log(SPACED_REP) / (-1.5 * decay)
    mean = std * 3.

    exponent = -(time_since - (REFRESH * half_life)) / half_life
    sigmoid = 1 / (1 + np.exp(exponent))
    bell = np.exp(-((time_since - mean) ** 2) / (2 * std * std)) * BELL_HEIGHT

    output = sigmoid + bell
    return output

def new_half_lives(know, half_life):
    """Turn an old half_life into a new half_life"""

    GROWTH = 1.8

    output = (GROWTH*know*half_life) + HALF_LIFE - (know*HALF_LIFE)
    return output

def new_center(half_life):
    """For testing only, please delete"""

    decay = np.log(2) / half_life
    '''
    std = np.log(SPACED_REP) / (-1.5 * decay)
    mean = std * 3.
    '''

    mean = 2. * np.log(SPACED_REP) / (-decay)
    std = mean / 3.

    return mean

def dates_to_times(dates):
    """Return days since the date"""

    today = datetime.datetime.today()
    output = np.array([(today - d).total_seconds() / (60.*60.*24.)
                       for d in dates])

    return output

def date_to_time(date):
    """Return days since the date"""

    output = (datetime.datetime.today() - date).total_seconds() / (60.*60.*24.)
    return output

def half_life_comp(dates, half_life):
    """Same as `new_half_lives`"""

    time_since = dates_to_times(dates)

    know = time_to_know(time_since, half_life)
    output = new_half_lives(know, half_life)

    return output

def new_half_life(date, life):
    time_since = date_to_time(date)
    know = time_to_know(time_since, life)
    output = new_half_lives(know, life)
    return output

def dates_to_prob(dates, half_life):
    """Convert `dates` into `prob`"""

    time_since = dates_to_times(dates)

    output = time_to_prob(time_since, half_life)
    return output

'''
know_lines = time_to_know(days_since, half_life)
prob_test = time_to_prob(days_since, half_life)

know = SPACED_REP

print half_life, new_center(half_life)

for _ in range(10):
    half_life = new_half_lives(know, half_life)
    print half_life, new_center(half_life)

plt.plot(days_since, know_lines,
         days_since, prob_test,
         )
plt.show()
'''

"""
# Load time since testing
days_since = np.arange(0, 14, .01)
dates = [datetime.datetime.today() - datetime.timedelta(d) for d in days_since]
half_life = np.array([HALF_LIFE for _ in range(days_since.shape[0])])

print half_life_comp(dates, half_life).tolist()

today = datetime.datetime.today()

text = '[%weight 1.0] [%streak 0.0] [%datetime {}]'.format(today)

m = re.search('\[%datetime (.+?)]', text)
if m:
    found = m.group(1)
    print found
    print pd.to_datetime(found).to_pydatetime()
    print pd.to_datetime(found).to_pydatetime() == today

print(re.sub('\[%streak (.+?)]', '', text))

print datetime.datetime.today() - datetime.timedelta(365*2000)

print new_half_life(dates[1], half_life[1])
"""
