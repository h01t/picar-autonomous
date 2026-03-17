import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("dataset/labels.csv")
plt.hist(df["steering"], bins=50)
plt.show()