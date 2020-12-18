#%%
import pandas as pd
from scipy.stats import chi2_contingency
import numpy as np
#%%
heart_data = pd.read_csv("/home/yuxuan/kaggle/heart_failure_clinical_records_dataset.csv")
cat_features = ["anaemia","diabetes","high_blood_pressure","sex","smoking","DEATH_EVENT"]
num_features = pd.Series(heart_data.columns)
num_features = num_features[~num_features.isin(cat_features)]
num_features

#%%

for i in cat_features:
    ct = pd.crosstab(columns=heart_data[i],index=heart_data["DEATH_EVENT"])
    stat, p, dof, expected = chi2_contingency(ct)
    print(f"\n{'-'*len(f'CROSSTAB BETWEEN {i.upper()} & DEATH_EVENT')}")
    print(f"CROSSTAB BETWEEN {i.upper()} & DEATH_EVENT")
    print(f"{'-'*len(f'CROSSTAB BETWEEN {i.upper()} & DEATH_EVENT')}")
    print(ct)
    print(f"\nH0: THERE IS NO RELATIONSHIP BETWEEN DEATH_EVENT & {i.upper()}\nH1: THERE IS RELATIONSHIP BETWEEN DEATH_EVENT & {i.upper()}")
    print(f"\nP-VALUE: {np.round(p,2)}")
    print("REJECT H0" if p<0.05 else "FAILED TO REJECT H0")