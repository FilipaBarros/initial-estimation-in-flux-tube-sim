
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler


df = pd.DataFrame()  



def join(path,column_index,n1,n2):
  #print("----------",path,"----------")
  data = pd.DataFrame()
  files = glob.glob(path+'/*.csv')
  for idx, filename in enumerate(files):
    #print(filename)
    df = pd.read_csv(filename)
    #df = df.iloc[:,column_index]
    #print(filename)
    name = filename[n1:n2]
    #print(name)
    data[name] = df.iloc[:,column_index]
    #print(name)
  return data


real_N = join("real",10,17,33)
real_V = join("real",11,17,33)
real_T = join("real",12,17,33)
#print(real_N)
#print(real_V)
#print(real_T)

predicted_with_out_N = join("outliers",10,31,47)
predicted_with_out_V = join("outliers",11,31,47)
predicted_with_out_T = join("outliers",12,31,47)
#print(predicted_with_out_N)
#print(predicted_with_out_V)
#print(predicted_with_out_T)


predicted_no_out_N = join("No outliers",10,34,50)
predicted_no_out_V = join("No outliers",11,34,50)
predicted_no_out_T = join("No outliers",12,34,50)
#print(predicted_no_out_N)
#print(predicted_no_out_V)
#print(predicted_no_out_T)




#Graficos de N:
#tres subplots. no primeiro todos os reais
#no segundo os previstos com ouliers
#finalmente no terceiro os previstos sem outliers
figdimension = (15,15)
fig, [ax1,ax2,ax3] = plt.subplots(3,1,figsize=figdimension)

plt.title('N models')

ax1.set_title("Real")
ax1.set_yscale("log")
ax2.set_title("Predicted With Outliers")
ax2.set_yscale("log")
ax3.set_title("Predicted With No Outliers")
ax3.set_yscale("log")

for i in range(0,20):
  ax1.plot(real_N.iloc[:,i])
  ax2.plot(predicted_with_out_N.iloc[:,i])
  ax3.plot(predicted_no_out_N.iloc[:,i])

plt.savefig("comparison_Ns")
plt.cla()
plt.clf()
plt.close()



#Gráficos de V: 
figdimension = (15,15)
fig, [ax1,ax2,ax3] = plt.subplots(3,1,figsize=figdimension)

plt.title('V models')

ax1.set_title("Real")
ax1.set_yscale("log")
ax2.set_title("Predicted With Outliers")
ax2.set_yscale("log")
ax3.set_title("Predicted With No Outliers")
ax3.set_yscale("log")

for i in range(0,20):
  ax1.plot(real_V.iloc[:,i])
  ax2.plot(predicted_with_out_V.iloc[:,i])
  ax3.plot(predicted_no_out_V.iloc[:,i])

plt.savefig("comparison_Vs")
plt.cla()
plt.clf()
plt.close()


#Gráficos de T
figdimension = (15,15)
fig, [ax1,ax2,ax3] = plt.subplots(3,1,figsize=figdimension)

plt.title('T models')

ax1.set_title("Real")
ax1.set_yscale("log")
ax2.set_title("Predicted With Outliers")
ax2.set_yscale("log")
ax3.set_title("Predicted With No Outliers")
ax3.set_yscale("log")

for i in range(0,20):
  ax1.plot(real_T.iloc[:,i])
  ax2.plot(predicted_with_out_T.iloc[:,i])
  ax3.plot(predicted_no_out_T.iloc[:,i])

plt.savefig("comparison_Ts")
plt.cla()
plt.clf()
plt.close()