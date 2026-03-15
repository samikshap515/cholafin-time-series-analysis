import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

df = pd.read_csv('../dataset/cholafin.csv')

df['DATE'] = pd.to_datetime(df['DATE'])
df['CLOSE'] = df['CLOSE'].astype(str).str.replace(',', '').astype(float)

df = df.sort_values('DATE')
df.set_index('DATE', inplace=True)
df = df.ffill()

plt.figure(figsize=(10,5))
plt.plot(df['CLOSE'])
plt.title("CHOLAFIN Closing Price Trend")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
plt.savefig("../outputs/trend.png", dpi=300)
plt.show()

result = adfuller(df['CLOSE'])
print("ADF Statistic:", result[0])
print("p-value:", result[1])

plot_acf(df['CLOSE'])
plt.savefig("../outputs/acf.png", dpi=300)
plt.show()

plot_pacf(df['CLOSE'])
plt.savefig("../outputs/pacf.png", dpi=300)
plt.show()

model = ARIMA(df['CLOSE'], order=(1,1,1))
model_fit = model.fit()

forecast_result = model_fit.get_forecast(steps=30)
forecast = forecast_result.predicted_mean
conf_int = forecast_result.conf_int()

forecast_df = pd.DataFrame({
    "Day": range(1,31),
    "Forecast_Price": forecast.values
})

forecast_df.to_csv("../outputs/forecast_values.csv", index=False)
print(forecast_df)

future_dates = pd.date_range(start=df.index[-1], periods=31, freq='D')[1:]

plt.figure(figsize=(12,6))
plt.plot(df['CLOSE'], label="Historical Price")
plt.plot(future_dates, forecast, linestyle="dashed", label="Forecast")

plt.fill_between(future_dates,
                 conf_int.iloc[:,0],
                 conf_int.iloc[:,1],
                 alpha=0.3,
                 label="Confidence Interval")

plt.title("CHOLAFIN Stock Price Forecast (Next 30 Days)")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.grid(True)

plt.savefig("../outputs/forecast.png", dpi=300)
plt.show()
