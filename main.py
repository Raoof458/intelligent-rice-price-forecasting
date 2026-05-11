import streamlit as st

predictions = model.predict(X_test)

predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# -----------------------------
# Prediction Graph
# -----------------------------

prediction_fig = go.Figure()

prediction_fig.add_trace(
    go.Scatter(
        y=y_test_actual.flatten(),
        mode='lines',
        name='Actual Price'
    )
)

prediction_fig.add_trace(
    go.Scatter(
        y=predictions.flatten(),
        mode='lines',
        name='Predicted Price'
    )
)

prediction_fig.update_layout(
    title='Actual vs Predicted Rice Prices',
    xaxis_title='Time',
    yaxis_title='Rice Price'
)

st.plotly_chart(prediction_fig, use_container_width=True)

# -----------------------------
# Future Prediction
# -----------------------------

last_60_days = scaled_data[-60:]
X_future = np.array([last_60_days[:, 0]])
X_future = X_future.reshape(1, 60, 1)

future_prediction = model.predict(X_future)
future_prediction = scaler.inverse_transform(future_prediction)

st.subheader("📈 Future Rice Price Prediction")
st.write(f"Predicted Next Rice Price: ${future_prediction[0][0]:.2f}")

# -----------------------------
# Accuracy Calculation
# -----------------------------

mape = np.mean(np.abs((y_test_actual - predictions) / y_test_actual)) * 100
accuracy = 100 - mape

st.subheader("Model Performance")
st.write(f"MAPE: {mape:.2f}%")
st.write(f"Accuracy: {accuracy:.2f}%")

# -----------------------------
# Footer
# -----------------------------

st.markdown("---")
st.write("Developed by Abdul Raoof")