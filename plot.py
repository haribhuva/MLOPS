import matplotlib.pyplot as plt
import pandas as pd

train = pd.read_csv('artifact/11_03_2026_17_21_04/data_ingestion/ingested/test.csv')
sensor_col = [f"sensor_{i}" for i in range(1, 22)]

for sensor in sensor_col:
    plt.figure(figsize=(8,4))
    plt.plot(train['cycle'], train[sensor])
    plt.xlabel('Cycle')
    plt.ylabel(sensor)
    plt.title(f"{sensor} vs Cycle")
    plt.show()