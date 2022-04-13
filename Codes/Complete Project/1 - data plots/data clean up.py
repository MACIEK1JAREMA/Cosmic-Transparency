'''
We clean up the datafile
The data in original txt is not evenly spaced
We read into python via numpy, reformat and save into a panads data frame
that is then saved as an excell file ready to read in and use more easily.
'''

# import needed modules
import numpy as np
import pandas as pd

# read in data using numpy, as string to keep column names
npdata = np.loadtxt('data\\SNe data Raw.txt', dtype='str')

# split data into first column, first row, rest and format separately
npdata_names = npdata[1:, 0].astype(str)
npdata_headers = npdata[0, 1:].astype(str).transpose()
npdata_data = npdata[1:, 1:].astype(float)

# stack them into one pandas data frame:
df = pd.DataFrame(data=npdata_data,
                  index=np.arange(1, len(npdata_data[:, 0])+1),
                  columns=list(npdata_headers)
                  )

# add the unaccounted for names of SNe column for completeness
df.insert(0, 'Names', npdata_names)

# save the cleaned file
df.to_excel('data\\SNe data.xlsx', index=False)

# test data read in:
data_read = pd.read_excel('data\\SNe data.xlsx')
