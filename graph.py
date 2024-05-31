import matplotlib.pyplot as plt

# Data for plotting
days = [25, 25, 50, 50, 50, 100, 100, 200, 200, ]
rf_mae = [2.041826050420169, 2.041826050420169 ,1.959590376569037, 2.993038912133889, 2.993038912133889, 3.2117937369519836, 3.2117937369519836, 3.82567236704901, 3.82567236704901]
lstm_mae = [2.1307269309548755, 2.252190922969528 ,2.10334622243458, 3.1263981092125817, 3.096510604634941, 3.540801607701376, 3.430813512463661, 3.8987456234851385, 3.932591473821302]

plt.figure(figsize=(10, 6))
plt.plot(days, rf_mae, marker='o', label='Random Forest MAE')
plt.plot(days, lstm_mae, marker='s', label='LSTM MAE')

plt.xlabel('Days of Data')
plt.ylabel('Mean Absolute Error')
plt.title('MAE Results for Different Days of Data')
plt.legend()
plt.grid(True)
plt.savefig('mae_plot.png')
plt.show()
