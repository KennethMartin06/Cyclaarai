import pandas as pd
df = pd.read_excel('data/raw/pet_bottle_10000.xlsx')
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df = df[df['IV (dL/g)'].between(0.60, 0.85)]
df = df[df['Processing_Temp (°C)'].between(250, 290)]
df = df[(df['rPET_Content (%)'] + df['Bio_PET_Content (%)']) <= 100]
df.to_csv('data/processed/pet_clean.csv', index=False)
print(f"Clean rows: {len(df)}")