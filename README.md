#  Smart Flight Finder – Flight Price Prediction System

##  Project Overview

Smart Flight Finder is a **machine learning powered flight recommendation system** that predicts airline ticket prices and suggests suitable flights based on user preferences.

The system uses **Machine Learning models and clustering techniques** to analyze flight data and provide intelligent recommendations through an interactive **Streamlit web application**.

Users can search flights by **source, destination, and travel date**, view predicted prices, categorize flights by price range, and filter by **ticket type and departure time**.

---

##  Features

✔ Flight search based on **source, destination, and travel date**

✔ **Machine Learning price prediction** using Random Forest

✔ **Flight price categorization** using K-Means clustering
(Budget / Standard / Premium)

✔ **Traveler type classification**
(Leisure / Mixed / Business travelers)

✔ Interactive **Streamlit UI**

✔ Airline **logos and flight cards**

✔ Flight filtering by

* Ticket type
* Departure time slot

✔ Smart **personalized flight suggestions**

---

##  Machine Learning Models Used

| Model                    | Purpose                           |
| ------------------------ | --------------------------------- |
| Random Forest Regressor  | Predict flight ticket prices      |
| K-Means Clustering       | Categorize flights by price range |
| Random Forest Classifier | Identify traveler type            |

---

##  Dataset

The project uses a dataset containing flight information such as:

* Airline
* Source
* Destination
* Date & Month
* Duration
* Total Stops
* Departure & Arrival Time
* Price

Additional features like **total duration in minutes** are engineered for better model performance.

---

##  Tech Stack

**Programming Language**

* Python

**Libraries**

* Pandas
* NumPy
* Scikit-learn
* Streamlit
* PIL

**Machine Learning**

* Random Forest Regression
* Random Forest Classification
* K-Means Clustering

---

##  Application Workflow

1️⃣ User enters **source, destination, and travel date**

2️⃣ System filters matching flights from the dataset

3️⃣ **Random Forest model predicts ticket prices**

4️⃣ **K-Means clustering categorizes flights**

* Budget
* Standard
* Premium

5️⃣ **Classifier predicts traveler type**

* Leisure
* Mixed
* Business

6️⃣ User selects:

* Flight
* Ticket type
* Departure time slot

7️⃣ System shows **final recommended flights**

---

##  Results

![Flight Finder Demo](demo.gif)

---

## 📷 User Interface

The application interface includes:

* Flight cards with airline logos
* Predicted ticket prices
* Price category labels
* Traveler recommendations
* Interactive filters

Built using **Streamlit for an intuitive and responsive experience.**

---

##  Project Structure

```
airline_explorer/
│
├── app6.py
├── extended_flight_dataset.csv
├── background.png
│
├── logos/
│   ├── indigo.png
│   ├── air_india.png
│   └── spicejet.png
│
└── README.md
```

---

## ▶ How to Run the Project

### 1️⃣ Clone the repository

```
git clone https://github.com/yourusername/flight-price-prediction.git
```

### 2️⃣ Navigate to the project folder

```
cd flight-price-prediction
```

### 3️⃣ Install required libraries

```
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit app

```
streamlit run app6.py
```

---

##  Future Improvements

* Add **real-time flight APIs**
* Improve price prediction with **XGBoost**
* Deploy on **Streamlit Cloud**
* Add **interactive charts and analytics**
* Add **round-trip flight prediction**

---

##  Author

**Shanmuga Priya**

Machine Learning & Data Analytics Enthusiast
Interested in **AI, Data Science, and Intelligent Systems**

---
