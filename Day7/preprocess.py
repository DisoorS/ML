import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv("data\laptopData.csv")

df.drop(columns=["Unnamed: 0"] , inplace=True)
df.dropna(inplace=True)

companies = list(set(df["Company"]))
companies = [str(i) for i in companies]
companies.sort()
companies = { companies[i] : i for i in range(len(companies))}

encoded_company = list(df["Company"])
for i in range(len(encoded_company)):
    encoded_company[i] = companies[encoded_company[i]]

df["Company"] = encoded_company

typeName = list(set(df["TypeName"]))
typeName.sort()
typeName = {typeName[i] : i for i in range(len(typeName))}
encoded_typename = list(df["TypeName"])
for i in range(len(encoded_typename)) : 
    encoded_typename[i] = typeName[encoded_typename[i]]

df["TypeName"] = encoded_typename


inches = df["Inches"].tolist()
avginche =round(sum([float(i) for  i in inches if i != '?'])/len(inches) , 2)

for i in inches:
    if i == "?":
        inches[inches.index(i)] = avginche
    else : 
        inches[inches.index(i)] = float(i)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(pd.DataFrame(inches))

df["Inches"] = scaled_data


screen = list(set(df["ScreenResolution"]))
screen_dict = {}
for i in range(len(screen)) :
    screen_dict[screen[i]] = i

screen_encoded = df["ScreenResolution"].tolist()
for i in range(len(screen_encoded)) : 
    screen_encoded[i] = screen_dict[screen_encoded[i]]

df["ScreenResolution"] = screen_encoded

cpu = list(set(df["Cpu"]))
cpu_dict = {}
for i in range(len(cpu)):
    cpu_dict[cpu[i]] = i

cpu_encode = df["Cpu"].tolist()
for i in range(len(cpu_encode)):
    cpu_encode[i] = cpu_dict[cpu_encode[i]]
df["Cpu"] = cpu_encode

ram = list(df["Ram"])
for i in ram :
    if str(i)[:-2].isdigit() :
        ram[ram.index(i)] = int(str(i)[:-2])
    else :
        ram[ram.index(i)] = int(str(df["Ram"].mode()[0])[:-2])

scaledram = scaler.fit_transform(pd.DataFrame(ram))
df["Ram"] = scaledram

memory = list(df["Memory"])
for i in memory:
    if str(i)[:-6].isdigit():
        memory[memory.index(i)] = int(str(i)[:-6])
    else:
        memory[memory.index(i)] = int(str(df["Memory"].mode()[0])[:-6])

ppmemory = scaler.fit_transform(pd.DataFrame(memory))
df["Memory"] = ppmemory

gpu = list(set(df["Gpu"]))
gpu_dict = {}
for i in range(len(gpu)):
    gpu_dict[gpu[i]] = i
gpu_encode = list(df["Gpu"])
for i in range(len(gpu_encode)):
    gpu_encode[i] = gpu_dict[gpu_encode[i]]
df["Gpu"] = gpu_encode

oss = list(set(df["OpSys"]))
maxi = df["OpSys"].mode()[0]

ossy = list(set(oss))
oss_dict = {}
for i in range(len(ossy)):
    oss_dict[ossy[i]] = i

os_encode = list(df["OpSys"])
for i in range(len(os_encode)):
    if os_encode[i] == "No OS":
        os_encode[i] = maxi
    os_encode[i] = oss_dict[os_encode[i]]
df["OpSys"] = os_encode

weight = list(df["Weight"])
weightmean = 0
for i in range(len(weight)):
    if weight[i] != "?":
        weightmean += float(weight[i][:-2])
weightmean = round(weightmean / len(weight) , 2)

for i in weight:
    if(i == '?') :
        weight[weight.index(i)] = weightmean
    else :
        weight[weight.index(i)] = float(weight[weight.index(i)][:-2])

ppweight = scaler.fit_transform(pd.DataFrame(weight))
df["Weight"] = ppweight

price = df["Price"]
pp_price = scaler.fit_transform(pd.DataFrame(price))
df["Price"] = pp_price

import random as r
sold = [r.choice([0,1]) for i in range(len(df["Price"]))]
df["sold"] = sold
# df.drop(columns=["Price"] , inplace=True)


df.plot()
plt.show()