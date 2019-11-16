from modeling import *

y_train.shape
y_train_final

df_test

df = pd.DataFrame({'Actual': y_train_final, 'Predicted': y_pred_final})
df1_head = df.head(25)

df1_head = df.head(30)

# df

# Histogram
df1_head.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

# Scatter Plot
df1_head.plot(kind='line',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()