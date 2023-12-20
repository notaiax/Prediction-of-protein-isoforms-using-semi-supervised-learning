import pandas as pd
from DNN.IsoDatasets import GtexDataset

NUM_ISOFORMS = 156958

dataset = GtexDataset()
num_samples = len(dataset)

totals = [0 for _ in range(NUM_ISOFORMS)]

for (j, (_, iso_sample)) in enumerate(dataset):
    print(f"Running on sample {j}")
    for (i, v) in enumerate(iso_sample):
        totals[i] += v

averages = [v/num_samples for v in totals]

dataframe = pd.DataFrame({"totals": totals, "averages": averages})
print(dataframe)
dataframe.to_csv("baseline.csv", index=False)