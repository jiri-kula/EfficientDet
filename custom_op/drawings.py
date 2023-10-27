# %%
import matplotlib.pyplot as plt
import matplotlib.lines as lines


fig = plt.figure()
fig.add_artist(lines.Line2D([0, 1], [0, 1], color="red"))
fig.add_artist(lines.Line2D([0, 1], [1, 0], color="red"))
plt.show()
