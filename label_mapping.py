import pandas as pd
from collections import defaultdict

# load from CSV
df = pd.read_csv("NLP.csv")

#new column
df['label_num'] = 0

#disease dictionary
data = set()
disease_dict = defaultdict()

num = 0

for i in range(len(df['focus_area'])):
    disease = df.loc[i, 'focus_area']
    if(disease not in data):
        data.add(disease)
        disease_dict[disease] = num
        df.loc[i, 'label_num'] = num
        num += 1
    else:
        df.loc[i, 'label_num'] = disease_dict[disease]

#print(disease_dict)

df.to_csv("NLP(labelmapping).csv", index = False)


    
