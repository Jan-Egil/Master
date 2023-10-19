import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
from tqdm import tqdm

file_string = 'substorm_list.csv'
substorm_df = pd.read_csv(file_string)
num_substorms_tot = len(substorm_df.index)
print(substorm_df)
print(num_substorms_tot)

curr_year = 2000
curr_month = 10

monthyear = []
num_in_monthyear = []

counter = 0

for i in range(num_substorms_tot):
    full_str = substorm_df.Date_UTC[i]
    year = int(full_str[0:4])
    month = int(full_str[5:7])
    
    if month == curr_month:
        counter += 1
    else:
        monthyear.append(date(curr_year, curr_month, 1))
        num_in_monthyear.append(counter)
        curr_month = month
        curr_year = year
        counter = 1




plt.plot(monthyear, num_in_monthyear)
plt.show()
    