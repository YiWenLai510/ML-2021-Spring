import pandas as pd 
from statistics import mode
import csv
import glob
import os
# data = pd.read_csv("7layer_b2048.csv")
# print(data)
# results = data['Class'].tolist()
# print(len(results))
# cnt = 0
# for i in range(len(results)):
#     if i != 0 and i != len(results)-1:
#         # print(i)
#         if results[i+1] == results[i-1]:
#             if results[i] != results[i+1]:
#                 results[i] = results[i+1]
#                 cnt += 1
# print(cnt)
# with open('7layerb2048_postProcess.csv','w') as f:
#     f.write('Id,Class\n')
#     for i, y in enumerate(results):
#         f.write('{},{}\n'.format(i, y))

##### ensemble #####
folder_name = 'best_result' 
file_type = 'csv'
seperator =','
df = pd.concat([pd.read_csv(f, sep=seperator,header=0).iloc[:, 1] for f in glob.glob(folder_name + "/*."+file_type)],ignore_index=True, axis=1)
print(df)
results = []
ids = []
for index, row in df.iterrows():
    results.append(mode(row.to_list()))
    ids.append(index)
print(len(results))
resultdf = pd.DataFrame(list(zip(ids,results)),columns=['Id', 'Class'])
print(resultdf)
resultdf.to_csv('ensemble_results.csv',index=False)#header=['Id','Class']
print('finish predict')
