# Weather Prediction using LSTM Neural Networks

## Project Overview
This project investigates the application of Long Short-Term Memory (LSTM) neural networks for weather forecasting, capable of making predictions from next-day to one-year in advance. Using historical weather station data from the Global Historical Climatology Network (GHCN), I've built and evaluated various LSTM model configurations to understand the impact of window size and neuron count on prediction accuracy. The project demonstrates how relatively simple neural network architectures can effectively capture weather patterns while providing insights into potential climate change trends.

## Data Source
The data comes from the Global Historical Climatology Network daily (GHCN-daily) dataset, which provides comprehensive weather records from stations worldwide. The primary station used in this analysis is RSM00024507 located in Tura, Russia, selected for its ample data volume and relatively low level of weather uncertainty.

The dataset includes:
- Daily maximum temperatures (TMAX)
- Daily minimum temperatures (TMIN)
- Average temperatures (TAVG)
- Precipitation records (PRCP)

For the climate change analysis portion, data from three Russian weather stations (Tura, Viljujsk, and Tompo) was combined to examine long-term temperature trends.

<img width="1274" height="754" alt="1-day prediction average temp" src="https://github.com/user-attachments/assets/1a26c71d-4cba-43aa-b7cf-0ed0f3ec7326" />
For more images, please refer to the Reference_Images folder

## Methodology

### Data Preprocessing
- Cleaning data by removal of missing values and filtering inconsistent data
- Data transformation and reduction via batches from daily to weekly and monthly average temperature datasets
- Data alignment across multiple weather stations for comparative analysis
- Temporal windowing for sequential prediction

### Model Architecture
The project explores different LSTM architectures:
- A basic bi-layered LSTM model with 48 neurons (7,505 trainable parameters)
- A larger bi-layered LSTM model with 384 neurons (461,441 trainable parameters)
- A three-layered LSTM model with normalization layers for multi-station analysis (150,595 trainable parameters)

All models utilize ReLU and hyperbolic tangent (tanh) activation functions with a linear output layer.

### Key Experiments
1. Comparison of neural network complexity (48 vs. 384 neurons) for next-day predictions
2. Testing boundary conditions with window size impact (1-week vs. 1-year) on one-year forecasting accuracy
3. Multi-variable prediction using a comprehensive data compilation
4. Multi-station analysis to investigate potential global warming trends

## Key Findings

### Model Performance
- The 48-neuron model achieved comparable accuracy to the 384-neuron model (average absolute deviation of 3.157°C vs. 3.148°C) while requiring only 1/5 of the computational time
- Window size significantly impacts prediction accuracy, with larger windows generally producing better results for long-term forecasts
- Extremely large window sizes (10-year) led to exploding gradient problems that required normalization techniques to mitigate
- The model demonstrated high accuracy in predicting temperature patterns (TAVG, TMAX, TMIN) but struggled with precipitation (PRCP) forecasting
- Multi-station analysis revealed a warming trend of approximately 1.99°C over 13 years across the studied Russian weather stations

### Practical Insights
- Simple LSTM architectures can effectively capture seasonal temperature patterns
- Prediction accuracy decreases with forecast horizon, as expected with weather data
- Precipitation forecasting requires more complex models that incorporate additional atmospheric variables
- LSTM models can complement traditional numerical weather prediction (NWP) methods by reducing computational requirements

## How to Use This Repository

### Prerequisites
- Python 3.6+
- TensorFlow 2.x
- NumPy, Pandas, Matplotlib
- Jupyter Notebook

## File Structure
- `weather_prediction.ipynb`: Main Jupyter notebook containing the analysis and models
- `PHAS0056_Mini_project_report__20119017.pdf`: Detailed research report with methodology and findings
- `data/`: Directory containing raw and processed data (not included in repository due to size)
- `models/`: Saved model weights and architectures
- `requirements.txt`: List of required Python packages

## Results and Visualizations
The notebook includes numerous visualizations that demonstrate:
- Comparison between actual and predicted temperature values
- Analysis of absolute deviations in different model configurations
- Loss vs. epoch plots showing model training dynamics
- Linear trend analysis for global warming investigation

## Future Improvements
- Incorporate additional meteorological variables for more accurate precipitation forecasting
- Explore hybrid models combining LSTM with CNN architectures
- Implement gradient clamping and normalization techniques to improve model stability
- Extend the analysis to more weather stations for broader climate trend analysis
- Optimize hyperparameters beyond window size and neuron count

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- The Global Historical Climatology Network for providing the dataset
- UCL for granting access to the GHCN database with appropriate permissions
- The TensorFlow and Keras teams for their excellent deep learning frameworks

---

*This project was completed as part of a Physics mini-project at University College London.*
