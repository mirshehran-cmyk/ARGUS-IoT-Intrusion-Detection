# ARGUS — IoT Intrusion Detection System

**A machine-learning-powered network monitoring dashboard for detecting potentially malicious traffic.**

ARGUS is an IoT Intrusion Detection System built with Python, Flask, and XGBoost. It combines the UNSW-NB15 intrusion detection dataset with network traffic captured from the developer's own Windows laptop to help address the gap between benchmark training data and everyday network activity.

## 🚀 Key Features

- **XGBoost Classification:** Machine-learning-based network traffic classification.
- **Live Network Monitoring:** Extracts network flow features and analyzes traffic where supported by the capture environment.
- **Custom Windows Traffic Training:** Incorporates locally captured Windows traffic as assumed-benign examples to help the model recognize ordinary laptop activity.
- **Interactive Dashboard:** Displays traffic statistics, prediction history, and potential threats.
- **Telegram Notifications:** Sends monitoring updates and qualifying attack alerts through a Telegram bot.
- **Model Evaluation:** Includes a Jupyter notebook and supporting scripts for training, testing, and analysis.

## 🧠 Machine Learning Approach

ARGUS combines two sources of data:

1. **UNSW-NB15 Dataset:** Provides network traffic examples containing normal activity and different categories of attacks.
2. **Custom Windows Traffic:** Flow features captured from the developer's Windows laptop are incorporated as assumed-benign examples to better represent the deployment environment.

The workflow includes feature preprocessing, categorical encoding, feature scaling, XGBoost model training, and live prediction.

**Why custom traffic matters:** A model trained only on benchmark data may behave differently on real-world devices. Incorporating Windows traffic helps address this domain gap and may reduce false-positive alerts.

*Note: Windows traffic was treated as benign during collection but was not independently verified as ground truth. Real-world detection performance requires evaluation on separate, independently labeled traffic.*

## 🛠️ Technology Stack

- **Language:** Python
- **Machine Learning:** XGBoost, Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Web Framework:** Flask
- **Development and Analysis:** Jupyter Notebook
- **Notifications:** Telegram Bot API
- **Dataset:** UNSW-NB15

## 📂 Project Structure

```text
ARGUS-IoT-Intrusion-Detection/
├── app.py
├── live_features.py
├── packet_capture.py
├── templates/
│   └── index.html
├── ARGUS_Retrain_V3_Final.ipynb
├── xgb_model.json
├── scaler.pkl
├── label_encoders.pkl
├── model_config_v3.json
├── .gitignore
└── README.md
```

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/mirshehran-cmyk/ARGUS-IoT-Intrusion-Detection.git
cd ARGUS-IoT-Intrusion-Detection
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Install the Python packages required by the project. If a `requirements.txt` file is available, run:

```bash
pip install -r requirements.txt
```

### 4. Configure Telegram (optional)

Create a local `.env` file in the project directory:

```text
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

Never commit your real credentials to GitHub.

### 5. Start ARGUS

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser, or use the local address shown in the terminal.

*Live packet capture may require additional system permissions or dependencies.*

## 🔐 Security and Limitations

- Use only on networks and devices you own or are authorized to monitor.
- Keep API tokens and other secrets out of the repository.
- Predictions are model outputs, not confirmed attack labels.
- Detection performance depends on traffic visibility, feature extraction, and model generalization.
- ARGUS is an academic project and is not a substitute for a production-grade security system.

## 🔮 Future Improvements

- Evaluate the model on independent, labeled network traffic.
- Improve detection of previously unseen attacks.
- Expand testing across different network environments.
- Enhance alert reporting and dashboard analytics.

## 👨‍💻 Author

**Shehran Shakeel**  
B.Tech in Computer Science and Engineering  
Specialization: Artificial Intelligence and Machine Learning

## 📚 Dataset

**UNSW-NB15** — a network intrusion detection dataset used for model development and evaluation.
