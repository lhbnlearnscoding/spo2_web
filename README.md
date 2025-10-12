# 🫁 SpO₂ Exercise Monitor & Analysis

A simple Streamlit web app for **monitoring and analyzing SpO₂ (blood oxygen saturation)** during physical activity.  
The app can **detect desaturation events** (SpO₂ drops below a threshold for a minimum duration), visualize data in real-time, and export reports.

👉 Live demo: [https://your-app-link.streamlit.app](https://your-app-link.streamlit.app)

---

## ✨ Features

- 📈 Real-time **visualization of SpO₂ vs. time**  
- ⚠️ **Desaturation event detection** (threshold + minimum duration)
- 🧮 Automatic calculation of:
  - Minimum SpO₂
  - Average SpO₂
  - Total session duration
  - Number of events
- 🧾 **Export results** as a CSV file
- 🧪 Support for **CSV upload** or **simulated exercise data**

---

## 🧰 Tech Stack

- [Python](https://www.python.org/) 3.10+
- [Streamlit](https://streamlit.io/) — for building the interactive web UI
- [NumPy](https://numpy.org/) & [Pandas](https://pandas.pydata.org/) — for data processing
- [Matplotlib](https://matplotlib.org/) — for plotting SpO₂ curves
- Deployed on [Streamlit Community Cloud](https://share.streamlit.io/)

---

## 🖥️ Installation & Usage (Local)

```bash
# Clone the repository
git clone https://github.com/<your-username>/SpO2_analyse.git
cd SpO2_analyse

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
