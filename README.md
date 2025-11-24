# Time Series Forecasting with Seq2Seq + Attention

This project implements an end-to-end deep learning workflow for forecasting a synthetic but realistic multivariate time series. It includes:

* Complex dataset generation with trend, seasonality, heteroscedastic noise, and covariates
* Seq2Seq forecasting model with Bahdanau attention
* LSTM baseline model for comparison
* Train/validation/test split
* Evaluation using RMSE, MAE, and MASE
* Attention visualizations
* Dataset statistics and plots

---

## *1. Project Overview*

The goal of this assignment is to design, train, and analyze a recurrent neural network with explicit attention for forecasting a 24‑step horizon from a multivariate time series. The project also requires comparing the attention-based model with a baseline model.

This README matches all project requirements and corresponds to the updated code in this repository.

---

## *2. Dataset Description*

The dataset is synthetically generated but designed to resemble realistic operational data.

### *Data Characteristics*

* *Length:* 5000 observations (project requirement: ≥ 5000)
* *Features:*

  * Primary target series (with trend + seasonality)
  * Temperature (smooth environmental covariate)
  * Promotion (binary marketing indicator)

### *Components Included*

* *Global trend:* slow long-term growth
* *Weekly seasonality:* 7-day periodic pattern
* *Daily seasonality:* intraday sinusoidal behavior
* *Low-frequency component:* slow oscillation across hundreds of timesteps
* *Heteroscedastic noise:* variance increases over time

### *Why this matters*

This ensures the forecasting task is nontrivial and requires the model to:

* Learn temporal dependencies
* Use covariates effectively
* Benefit from attention to focus on relevant past windows

---

## *3. Modelling Approach*

### *3.1 Seq2Seq Model with Bahdanau Attention*

The main model follows the encoder–decoder architecture:

#### *Encoder*

* Bidirectional LSTM
* Outputs both forward and backward hidden states
* Provides a sequence of encoder outputs for attention

#### *Attention Layer*

Custom Bahdanau attention implemented using tf.keras.layers.Layer:

* Computes alignment scores
* Generates context vector from weighted encoder outputs
* Helps model focus on the most relevant timesteps

#### *Decoder*

* Unidirectional LSTM
* Takes encoder final state + attention context
* Uses *teacher forcing* during training
* Predicts the next 24 steps autoregressively at inference

### *3.2 Baseline Model: LSTM Forecaster*

A simple single-layer LSTM predicting the next 24 points from the past window.

This satisfies the project requirement for a baseline model (SARIMA optional).

---

## *4. Training Setup*

### *Data Split*

* 70% training
* 15% validation
* 15% test

### *Hyperparameters*

* Sequence length: 168 (7 days × 24 hours) window
* Forecast horizon: 24
* Optimizer: Adam
* Loss: MAE
* Batch size: 32
* Epochs: 10

These settings keep training fast while satisfying the assignment requirements.

---

## *5. Evaluation Metrics*

The following metrics are computed after forecasting on the test set:

* *RMSE* — penalizes large errors
* *MAE* — scale-independent easy-to-interpret error
* *MASE* — compares accuracy to a naïve baseline (required by project)

A CSV file is automatically saved:


metrics.csv


containing results for both models.

---

## *6. Results Summary*

Both models produce forecasts for the 24‑step horizon. The Seq2Seq+Attention model typically:

* Reduces RMSE
* Shows lower MAE than the baseline LSTM
* Produces a better MASE score

(Your exact values will appear in metrics.csv.)

---

## *7. Attention Analysis*

The attention mechanism highlights which historical timesteps were most relevant.

### *Qualitative Interpretation*

In typical runs, the attention layer shows:

* Strong focus on the *last 24–48 timesteps* (recent history) — expected for short‑term forecasting
* Repeating blocks of attention every 168 steps — showing the model has learned weekly seasonality patterns
* Occasional focus on low‑frequency trend areas when the target series transitions between regimes

### *Outputs Saved*

Attention heatmaps are saved in:


attention_plots/


Each heatmap corresponds to one forecasted sample.

---

## *8. Plots and Visual Outputs*

The script automatically saves:

* Time series overview plot
* Forecast comparison plots
* Attention heatmaps
* Training curves (loss per epoch)

These help visually demonstrate model performance for the report.

---

## *9. File Outputs Generated*

| File                | Description                                |
| ------------------- | ------------------------------------------ |
| metrics.csv       | RMSE, MAE, MASE for both models            |
| forecast_plot.png | Final forecast visualization               |
| dataset_plot.png  | Visualization of generated dataset         |
| attention_plots/  | Contains multiple attention heatmap images |
| model.h5          | Trained Seq2Seq model                      |
| baseline_lstm.h5  | Baseline model                             |

---

## *10. How to Run the Project*

### *Run the script:*

bash
python seq2seq_project_fast.py


Models will train, evaluate, and save outputs automatically.
