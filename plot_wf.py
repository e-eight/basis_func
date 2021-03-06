import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_table('ho_wf.txt', sep=" ", names=("p", "v"))
data.plot(x="p", y="v")
plt.ylim(-1,1)
plt.show()
