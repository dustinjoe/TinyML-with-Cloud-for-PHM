import MySQLdb

mydb = MySQLdb.connect(
  host="localhost",
  user="admin",
  passwd="xyz123",
  db="esp32_cmapss_grp"
)

print(mydb)

c = mydb.cursor()

seq_len = 5
select_dat_query = "SELECT sensor1,sensor2,sensor3,sensor4 FROM SensorData2 LIMIT "+str(seq_len)
c.execute(select_dat_query)

import numpy as np
#cursor = mydb.cursor()
results = c.fetchall()
#num_row_query = "SELECT COUNT(*) FROM SensorData"
#cursor.execute(num_row_query)
#numrows = cursor.fetchall()
numrows = int(c.rowcount)
#print( type(numrows[0]) )
#curs.fetchall() is the iterator as per Kieth's answer
#count=numrows means advance allocation
#dtype='i4,i4' means two columns, both 4 byte (32 bit) integers
# recast this nested tuple to a python list and flatten it so it's a proper iterable:
x = map(list, list(results))              # change the type
x = sum(x, [])                            # flatten

# D is a 1D NumPy array
D = np.fromiter(iter=x, dtype=float,count=-1)  

# 'restore' the original dimensions of the result set:
D = D.reshape(numrows, -1)

print( D) #output entire array
print(type(D))
print(D.shape)
