# 🔍 SkillSight AI — Job Market Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58-red?logo=streamlit)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

> An intelligent job market analytics platform that transforms real job posting data into actionable career insights.

---

## 🚀 Live Features

| Page | Description |
|------|-------------|
| 🏠 Executive Dashboard | KPIs, top skills, hiring trends from 1800+ real jobs |
| 📊 Skill Analytics | Demand charts, co-occurrence heatmap |
| 📈 Skill Trends | Emerging vs declining skills detection |
| 💰 Salary Intelligence | Salary benchmarks by role |
| 💡 Salary Predictor | ML model predicts salary from skills + experience |
| 🏢 Company Insights | Top hiring companies and cities |
| 🎯 Career Advisor | Personalised skill gap + learning roadmap |
| 📄 Resume Analyzer | Upload PDF → instant market readiness score |
| 🎯 Job Match Score | Paste any JD → get match % using TF-IDF cosine similarity |
| 🗺️ Geographic Heatmap | Interactive map of hiring demand by city |

---

## 🛠️ Tech Stack

- **Python** — core language
- **Pandas / NumPy** — data processing
- **Scikit-Learn** — salary prediction model (Gradient Boosting)
- **NLP** — regex + pattern matching skill extractor (50+ skills)
- **Plotly** — interactive visualizations
- **Streamlit** — dashboard UI
- **SQLAlchemy** — database ORM
- **pdfplumber** — resume PDF parsing

---

## 📊 ML Models

| Model | Algorithm | Purpose |
|-------|-----------|---------|
| Salary Predictor | Gradient Boosting Regressor | Predict annual salary from skills + experience + location |
| Job Match Scorer | TF-IDF + Cosine Similarity | Match user profile against job descriptions |
| Skill Trend Detector | Time Series Analysis | Identify emerging vs declining skills |

---

## 🗂️ Project Structure
---

## ⚡ Quick Start

```bash
# Clone the repo
git clone https://github.com/thearjuntyagi-ai/skillsight-ai.git
cd skillsight-ai

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
python -m streamlit run src/dashboard/app.py
```

---

## 📁 Data Source

This project uses the [LinkedIn Job Postings dataset](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) from Kaggle — 1800+ real job postings across Data Analyst, Data Scientist, Data Engineer, ML Engineer roles.

---

## 💡 Key Highlights for Interviews

- **End-to-end pipeline** — raw data → cleaning → NLP → ML → dashboard
- **Real data** — 1800+ actual LinkedIn job postings
- **Production code** — modular OOP design, logging, error handling
- **Business impact** — solves a real problem for job seekers

---

Built by **Arjun Tyagi** | [GitHub](https://github.com/thearjuntyagi-ai)