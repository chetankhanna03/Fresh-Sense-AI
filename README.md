#  FreshSense AI: Fruit Freshness Detector

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.x-blue)](https://www.python.org/)
[![Flask Version](https://img.shields.io/badge/Framework-Flask-informational)](https://flask.palletsprojects.com/)

---

##  Overview

**FreshSense AI** is a web-based application designed to quickly determine the freshness (or rottenness) of various fruits using a deep learning model (likely a Convolutional Neural Network or CNN).

The system features a clean, responsive, dark-mode user interface and uses a two-part architecture:
* **Frontend:** HTML, CSS, and JavaScript for media capture/upload and presenting results.
* **Backend:** A Python Flask API for running the prediction model.

### Supported Fruits & Statuses

The model is designed to classify the following fruits:
* **Fruits:** Apple, Banana, Orange, Peach, Pomegranate, Strawberry
* **Statuses:** Fresh / Rotten (or Spoiled)

---

##  Deployment Architecture

To ensure speed and separation of concerns, this project uses a standard dual-deployment approach:

| Component | Technology | Role | Recommended Host |
| :--- | :--- | :--- | :--- |
| **Frontend** | HTML/CSS/JS | Handles UI, Camera, and File Upload | **Netlify** (Static Hosting) |
| **Backend** | Python / Flask / Gunicorn | Hosts the AI prediction model (`/predict` endpoint) | **Render** or **Heroku** (Web Service) |

---

## ⚙️ Setup and Installation

Follow these steps to set up the project locally for development.

### Prerequisites

* Python 3.x
* `pip` (Python package installer)

### 1. Clone the Repository

```bash
git clone [https://github.com/YOUR_USERNAME/fruit-freshness-detector.git](https://github.com/YOUR_USERNAME/fruit-freshness-detector.git)
cd fruit-freshness-detector
