# Model Performance & Results

## Regression (Salary Prediction)

| Model | R² | RMSE (₽) | MAE (₽) |
|-------|-----|----------|---------|
| **Random Forest** | **0.499** | **36,802** | **25,321** |
| Gradient Boosting | 0.485 | 37,330 | 26,420 |
| Ridge Regression | 0.432 | 39,183 | 28,614 |

**Key Findings**
- **Random Forest achieved the best performance** with R² = 0.499, explaining ~50% of salary variance.
- RMSE of ~37K rubles indicates average prediction error of approximately 37,000 rubles.
- Linear model (Ridge) performed worst, suggesting **non-linear relationships** in the data.
- Moderate R² (~0.5) is attributed to:
  - High salary variability even for similar resumes
  - Missing key features: skills, specific technologies, company size, exact location

---

## Classification (Developer Level)

| Model | Accuracy | Macro F1 | Junior F1 | Middle F1 | Senior F1 |
|-------|----------|----------|-----------|-----------|-----------|
| **Gradient Boosting** | **0.747** | **0.627** | **0.519** | **0.491** | **0.871** |
| Random Forest | 0.738 | 0.617 | 0.545 | 0.444 | 0.862 |
| Logistic Regression | 0.708 | 0.537 | 0.373 | 0.376 | 0.864 |

**Class Distribution** (test set):
- Senior: 62.5%
- Middle: 19.5%
- Junior: 18.0%

**Key Findings**
- **Gradient Boosting is the best model** with macro F1 = 0.627 and accuracy = 74.7%.
- **Strong class imbalance**: Senior class dominates (62.5%), resulting in:
  - Excellent performance on Senior (F1 = 0.871)
  - Poor performance on Junior/Middle (F1 = 0.519 / 0.491)
- **Logistic Regression significantly underperforms** (macro F1 = 0.537), indicating **non-linear class boundaries**.
- Main challenges:
  - Fuzzy class boundaries: 2–5 years of experience can be junior, middle, or senior depending on skills
  - Limited features: no information about skills, projects, or education quality
