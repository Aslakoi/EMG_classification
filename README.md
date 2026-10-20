# Hand gesture classification based on EMG signals

Course project for EE4C12 Machine Learning for Electrical Engineering (TU Delft), by Aslak Iveland and Ivan Fran Stojic.

The goal is to recognise six static hand gestures (rest, fist, wrist flexion, wrist extension, radial deviation and ulnar deviation) from the eight EMG channels of a MYO bracelet, for a person whose data the model has never seen.

## Approach

1. **Data analysis and cleaning:** removal of unmarked data (class 0), of the gesture performed by only two subjects (class 7) and of repeated rows written by the logging software (89% of all rows), which leaves about 105 new samples per second.
2. **Features:** windows of 20 samples with 50% overlap; five common EMG features per channel (MAV, RMS, WL, ZC, MNF), selected with a filter method (mutual information) and a wrapper method (forward selection). The mean absolute value (MAV) of the 8 channels is used.
3. **Validation:** every split is made by subject. 29 subjects are used for development with 5-fold cross-validation by subject, and 7 subjects are kept apart for a single final test.
4. **Models:** logistic regression as a baseline, a multi-layer perceptron (MLP) tuned with a random search and learning curves, deeper MLPs, and a 1D convolutional network on the raw signal.

## Results

Macro F1 for new subjects:

| Model | Cross-validation (29 subjects) | Test (7 subjects) |
|---|---|---|
| Logistic regression | 0.811 | 0.802 |
| MLP, 2 hidden layers (64, 32), final model | 0.835 | 0.780 |

The difference between the two models on the test subjects is not significant (bootstrap interval of the difference: -0.072 to 0.028). The main limitation is the difference between people: the F1 per test subject ranges from 0.25 to 1.00. A short calibration recording per user raised the F1 of logistic regression from 0.815 to 0.875 in cross-validation.

## Repository

- `main.ipynb`: the complete pipeline, from data analysis to the final test, with an explanation of every choice
- `figures/`: all figures produced by the notebook
- `requirements.txt`: the Python packages and versions used

## Running the notebook

1. Extract the data zip, so that `Project_Data_EE4C12_S&S_EMG.csv` is in the project folder.
2. Create a virtual environment and install the requirements:

```bash
python -m venv .venv
.venv\Scripts\activate             # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

3. Open `main.ipynb`, select the `.venv` kernel and run all cells (about 15 minutes on a laptop CPU). All random choices use fixed seeds, so the results are reproducible.
