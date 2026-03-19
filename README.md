# 🚀 ViralIQ — Instagram Viral Predictor

Predict whether your Instagram Reel or Post will go **Viral 🔥 or Not 📉** using Machine Learning.

## 📸 App Preview

![App Screenshot](screenshot.png)
---

## 📌 Features

* 🔥 Viral / Not Viral Prediction
* 📊 Confidence Score (Probability)
* 📈 Feature Importance Visualization
* 🎨 Premium Dark UI
* ⚡ Fast Flask Backend
* 🔓 No Login Required

---

## 🧠 Model Details

* Model: Gradient Boosting Classifier
* Inputs:

  * Follower Count
  * Caption Length
  * Hashtags Count
  * Post Hour
  * Day of Week
  * Account Type
  * Media Type
  * Content Category
  * Traffic Source
  * Engagement Rate

---

## 📂 Project Structure

```
ViralIQ/
│
├── app.py                # Main Flask app
├── model_v2.pkl         # Trained ML model
├── encoders_v2.pkl      # Label encoders
├── README.md            # Project documentation
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ViralIQ.git
cd ViralIQ
```

---

### 2. Install dependencies

```bash
pip install flask numpy joblib scikit-learn
```

---

### 3. Run the app

```bash
python app.py
```

---

### 4. Open in browser

```
http://localhost:5000
```

---

## 🎯 How It Works

1. User enters Instagram post details
2. Data is encoded using LabelEncoders
3. Model predicts virality
4. UI shows:

   * Result (Viral / Not Viral)
   * Confidence %
   * Feature importance

---

## 🧪 Example Output

* 🚀 Viral (87% confidence)
* 📉 Not Viral (32% confidence)

---

## 💡 Future Improvements

* 🌍 Add location-based prediction
* 📊 Real Instagram API integration
* 🤖 Auto caption & hashtag suggestions
* ☁️ Deploy on cloud (Render / Railway)

## 🌐 Live Demo
Coming Soon...
---

## 👨‍💻 Author

**Sri Ram Koduru**

---

## ⭐ Support

If you like this project:

👉 Star ⭐ the repo
👉 Share with friends
👉 Build your own AI tools

---

## ⚠️ Disclaimer

This model is trained on sample data and provides predictions based on patterns — not guaranteed real-world results.

---
