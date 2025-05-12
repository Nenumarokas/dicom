import matplotlib.pyplot as plt
import json
import os

model = 'train33_calcinate'
selected_folder = f'{os.getcwd()}/models/{model}'
with open(f'{selected_folder}/metrics.json', 'r') as f:
    metrics = json.load(f)

plt.figure(figsize=(10, 8))
for key in metrics:
    plt.plot(metrics[key])

plt.ylim([-0.1, 1.1])
plt.grid(which='both')
plt.legend(metrics.keys())
plt.show()
