from modeling import *

y_train.shape

df = pd.DataFrame({'Actual': y_train_final, 'Predicted': y_pred_final})
df1 = df.head(25)

df1 = df.head(30)

df1

# Histogram
df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

# Scatter Plot
df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()