"""
Travel Insurance Flight Risk Analysis
=====================================
Business Objective: Identify factors contributing to high-risk flights for insurance pricing optimization
Company: Travel Insurance Analytics Team
Date: August 2025

BUSINESS CONTEXT:
- Flight delay claims trigger $800 payout if delay >180 minutes OR cancellation occurs
- Goal: Risk-based pricing to optimize customer acquisition and risk management
- Low-risk flights → Lower premiums → Expand customer base
- High-risk flights → Higher premiums → Natural risk screening + adequate compensation
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve
)
import numpy as np

# =============================================================================
# BUSINESS HYPOTHESES FOR TRAVEL INSURANCE RISK PRICING
# =============================================================================
"""
H1: Certain airlines have systematically higher claim rates
    Business Impact: Airline-specific pricing tiers
    
H2: Peak travel times (morning rush, evening) increase delay risk
    Business Impact: Time-based premium adjustments
    
H3: Specific routes have inherent risk due to weather/infrastructure
    Business Impact: Route-specific risk premiums
    
H4: Weekend/holiday travel patterns affect delay probability
    Business Impact: Calendar-based pricing models
    
H5: Seasonal factors (winter weather, summer storms) increase risk
    Business Impact: Seasonal premium adjustments
    
BUSINESS SUCCESS METRICS:
- Model accuracy >80% for risk classification
- Clear differentiation between high/low risk segments
- Actionable features for pricing algorithm integration
"""

# =============================================================================
# DATA PREPARATION & CLAIM SIMULATION
# =============================================================================

# Load flight operational data
flight_data = pd.read_csv("data/data.csv")
print(f"Analyzing {len(flight_data):,} flight records for insurance risk modeling")

# Data quality assessment
initial_count = len(flight_data)
flight_data.dropna(subset=['Airline'], inplace=True)
print(f"Removed {initial_count - len(flight_data):,} records with missing airline data")

# =============================================================================
# INSURANCE CLAIM LOGIC IMPLEMENTATION
# =============================================================================

# Create cancellation indicator
flight_data['is_cancelled'] = flight_data['delay_time'].apply(
    lambda x: 1 if x == 'Cancelled' else 0
)

# Convert delay time to minutes (handle 'Cancelled' strings)
flight_data['delay_minutes'] = pd.to_numeric(
    flight_data['delay_time'], errors='coerce'
).fillna(0)

# CRITICAL: Define claim trigger based on insurance policy
# Claim occurs if: delay > 180 minutes OR flight cancelled
flight_data['triggers_claim'] = (
    (flight_data['delay_minutes'] > 180) | 
    (flight_data['is_cancelled'] == 1)
).astype(int)

# Calculate expected claim amount per flight
CLAIM_AMOUNT = 800  # $800 per qualifying claim
flight_data['expected_claim'] = flight_data['triggers_claim'] * CLAIM_AMOUNT

print(f"\nINSURANCE CLAIM ANALYSIS:")
claim_rate = flight_data['triggers_claim'].mean()
print(f"Overall claim rate: {claim_rate:.2%}")
print(f"Expected payout per flight: ${flight_data['expected_claim'].mean():.2f}")
print(f"Total potential exposure: ${flight_data['expected_claim'].sum():,.0f}")

# =============================================================================
# RISK FACTOR ENGINEERING
# =============================================================================

# Extract temporal risk factors
flight_data['flight_date'] = pd.to_datetime(flight_data['flight_date'], errors='coerce')
flight_data['day_of_week'] = flight_data['flight_date'].dt.dayofweek  # 0=Monday
flight_data['month'] = flight_data['flight_date'].dt.month
flight_data['year'] = flight_data['flight_date'].dt.year

# Define operational time periods for risk assessment
def categorize_flight_time(hour):
    """
    Categorize flights by operational risk periods
    Based on airport congestion and operational complexity
    """
    if pd.isna(hour):
        return 'unknown'
    elif 5 <= hour < 12:
        return 'morning'           # High traffic, crew transitions
    elif 12 <= hour < 17:
        return 'afternoon'         # Lower congestion period
    elif 17 <= hour < 21:
        return 'evening'           # Peak departure time
    else:
        return 'night'             # Red-eye, maintenance windows

flight_data['operational_period'] = flight_data['std_hour'].apply(categorize_flight_time)

# =============================================================================
# PRELIMINARY RISK SEGMENTATION ANALYSIS
# =============================================================================

print(f"\n" + "="*70)
print("RISK SEGMENTATION FOR PRICING MODEL")
print("="*70)

# H1: Airline risk profiles
print(f"\nAIRLINE RISK ANALYSIS (Top/Bottom 5):")
airline_risk = flight_data.groupby('Airline').agg({
    'triggers_claim': ['count', 'mean', 'sum']
}).round(4)
airline_risk.columns = ['flight_count', 'claim_rate', 'total_claims']
airline_risk = airline_risk[airline_risk['flight_count'] >= 100]  # Minimum sample size
airline_risk = airline_risk.sort_values('claim_rate', ascending=False)

print("Highest Risk Airlines:")
print(airline_risk.head().to_string())
print("\nLowest Risk Airlines:")
print(airline_risk.tail().to_string())

# H2: Time period risk analysis
print(f"\nOPERATIONAL PERIOD RISK:")
period_risk = flight_data.groupby('operational_period')['triggers_claim'].agg(['count', 'mean']).round(4)
period_risk.columns = ['flight_count', 'claim_rate']
print(period_risk.sort_values('claim_rate', ascending=False).to_string())

# H4: Day of week patterns
print(f"\nDAY OF WEEK RISK PATTERNS:")
dow_risk = flight_data.groupby('day_of_week')['triggers_claim'].agg(['count', 'mean']).round(4)
dow_risk.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
dow_risk.columns = ['flight_count', 'claim_rate']
print(dow_risk.to_string())

# =============================================================================
# PREDICTIVE MODEL FOR RISK PRICING
# =============================================================================

# Feature selection for pricing model
pricing_features = ['Airline', 'Departure', 'Arrival', 'day_of_week', 'month', 'operational_period']
X = pd.get_dummies(flight_data[pricing_features], drop_first=True)
y = flight_data['triggers_claim']  # Binary: Will this flight trigger a claim?

print(f"\nMODEL FEATURES:")
print(f"Total features engineered: {X.shape[1]}")
print(f"Training samples: {len(X):,}")
print(f"Positive class rate: {y.mean():.2%}")

# Stratified train-test split to maintain claim rate balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train logistic regression for interpretable risk scoring
risk_model = LogisticRegression(
    max_iter=1000, 
    random_state=42
    # Removed class_weight='balanced' for higher accuracy
)
risk_model.fit(X_train, y_train)

# Generate risk predictions
y_pred = risk_model.predict(X_test)
y_risk_proba = risk_model.predict_proba(X_test)[:, 1]  # Probability of claim

# =============================================================================
# MODEL PERFORMANCE FOR BUSINESS VALIDATION
# =============================================================================

print(f"\n" + "="*70)
print("INSURANCE RISK MODEL PERFORMANCE")
print("="*70)

# Core performance metrics
train_accuracy = risk_model.score(X_train, y_train)
test_accuracy = risk_model.score(X_test, y_test)
auc_score = roc_auc_score(y_test, y_risk_proba)

print(f"Training Accuracy: {train_accuracy:.1%}")
print(f"Testing Accuracy: {test_accuracy:.1%}")
print(f"AUC Score: {auc_score:.3f}")

# Business-relevant performance breakdown
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0

print(f"\nBUSINESS IMPACT METRICS:")
print(f"True Negatives (Correctly ID'd low-risk): {tn:,}")
print(f"False Positives (Overpriced low-risk): {fp:,}")
print(f"False Negatives (Underpriced high-risk): {fn:,}")
print(f"True Positives (Correctly ID'd high-risk): {tp:,}")
print(f"Precision (High-risk accuracy): {precision:.1%}")
print(f"Recall (High-risk coverage): {recall:.1%}")

# =============================================================================
# RISK FACTOR ANALYSIS FOR PRICING ALGORITHM
# =============================================================================

# Extract feature importance for pricing insights
feature_importance = pd.Series(
    risk_model.coef_[0], 
    index=X.columns
).sort_values(key=abs, ascending=False)

print(f"\n" + "="*70)
print("PRICING FACTOR ANALYSIS")
print("="*70)

print(f"\nTOP 10 RISK-INCREASING FACTORS:")
risk_increasing = feature_importance[feature_importance > 0].head(10)
for feature, coef in risk_increasing.items():
    risk_multiplier = np.exp(coef)
    print(f"{feature:<40} | Coef: {coef:+.3f} | Risk Multiplier: {risk_multiplier:.2f}x")

print(f"\nTOP 10 RISK-DECREASING FACTORS:")
risk_decreasing = feature_importance[feature_importance < 0].head(10)
for feature, coef in risk_decreasing.items():
    risk_multiplier = np.exp(coef)
    print(f"{feature:<40} | Coef: {coef:+.3f} | Risk Multiplier: {risk_multiplier:.2f}x")

# =============================================================================
# VISUALIZATION SUITE FOR STAKEHOLDER PRESENTATION
# =============================================================================

# 1. Confusion Matrix for Risk Classification
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low-Risk', 'High-Risk'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Insurance Risk Classification Performance\n(Logistic Regression Model)', 
         fontsize=16, fontweight='bold', pad=20)

# Add business context annotations
plt.text(0.5, -0.15, f'Model Accuracy: {test_accuracy:.1%} | AUC: {auc_score:.3f}', 
         ha='center', transform=plt.gca().transAxes, fontsize=12)
plt.tight_layout()
plt.savefig('insurance_risk_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: insurance_risk_confusion_matrix.png")

# 2. ROC Curve for Risk Discrimination
plt.figure(figsize=(10, 8))
fpr, tpr, thresholds = roc_curve(y_test, y_risk_proba)
plt.plot(fpr, tpr, linewidth=3, label=f'Risk Model (AUC = {auc_score:.3f})', color='darkblue')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Classification')

# Add optimal threshold point
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10, 
         label=f'Optimal Threshold: {optimal_threshold:.3f}')

plt.xlabel('False Positive Rate\n(Low-risk flights incorrectly classified as high-risk)', fontsize=12)
plt.ylabel('True Positive Rate\n(High-risk flights correctly identified)', fontsize=12)
plt.title('Insurance Risk Model Discrimination Ability\n(ROC Curve Analysis)', 
         fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('insurance_risk_roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: insurance_risk_roc_curve.png")

# 3. Feature Importance for Pricing Algorithm
plt.figure(figsize=(14, 10))
top_features = feature_importance.head(15)

# Color coding: red for risk-increasing, blue for risk-decreasing
colors = ['darkred' if coef > 0 else 'darkblue' for coef in top_features.values]
bars = plt.barh(range(len(top_features)), top_features.values, color=colors, alpha=0.8)

plt.yticks(range(len(top_features)), top_features.index, fontsize=11)
plt.xlabel('Risk Coefficient (Log-Odds Scale)', fontsize=12)
plt.title('Top 15 Pricing Factors for Travel Insurance Risk Model\n(Logistic Regression Coefficients)', 
         fontsize=16, fontweight='bold', pad=20)

# Add coefficient values as text
for i, (feature, coef) in enumerate(top_features.items()):
    plt.text(coef + (0.01 if coef > 0 else -0.01), i, f'{coef:.3f}', 
             va='center', ha='left' if coef > 0 else 'right', fontsize=10)

plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
plt.grid(True, alpha=0.3, axis='x')

# Legend
plt.legend(['Risk Increasing (Higher Premiums)', 'Risk Decreasing (Lower Premiums)'], 
          handles=[plt.Rectangle((0,0),1,1, color='darkred', alpha=0.8),
                  plt.Rectangle((0,0),1,1, color='darkblue', alpha=0.8)],
          loc='lower right')

plt.tight_layout()
plt.savefig('insurance_pricing_factors.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: insurance_pricing_factors.png")

# =============================================================================
# HYPOTHESIS VALIDATION & BUSINESS CONCLUSIONS
# =============================================================================

print(f"\n" + "="*70)
print("HYPOTHESIS VALIDATION FOR PRICING STRATEGY")
print("="*70)

# Analyze top features by category
airline_features = [f for f in feature_importance.head(15).index if 'Airline_' in f]
time_features = [f for f in feature_importance.head(15).index if 'operational_period_' in f]
route_features = [f for f in feature_importance.head(15).index if any(x in f for x in ['Departure_', 'Arrival_'])]
temporal_features = [f for f in feature_importance.head(15).index if any(x in f for x in ['day_of_week', 'month_'])]

print(f"\nH1 - AIRLINE RISK DIFFERENTIATION:")
if airline_features:
    print(f"✅ VALIDATED: {len(airline_features)} airline factors in top predictors")
    print(f"   Pricing recommendation: Implement airline-specific risk tiers")
    for af in airline_features[:3]:
        coef = feature_importance[af]
        direction = "premium increase" if coef > 0 else "discount"
        print(f"   • {af}: {direction} of {abs(coef):.1%}")
else:
    print(f"❌ NOT VALIDATED: Airline choice not a significant risk factor")

print(f"\nH2 - OPERATIONAL TIME RISK:")
if time_features:
    print(f"✅ VALIDATED: {len(time_features)} time period factors significant")
    print(f"   Pricing recommendation: Time-of-day premium adjustments")
    for tf in time_features:
        coef = feature_importance[tf]
        direction = "higher premiums" if coef > 0 else "discounts"
        print(f"   • {tf}: {direction} (coefficient: {coef:+.3f})")
else:
    print(f"❌ NOT VALIDATED: Flight timing not a significant risk factor")

print(f"\nH3 - ROUTE-SPECIFIC RISK:")
if route_features:
    print(f"✅ VALIDATED: {len(route_features)} route factors in top predictors")
    print(f"   Pricing recommendation: Route-specific risk premiums")
else:
    print(f"❌ NOT VALIDATED: Route choice not among top risk factors")

print(f"\nH4 & H5 - TEMPORAL PATTERNS:")
if temporal_features:
    print(f"✅ VALIDATED: {len(temporal_features)} temporal factors significant")
    print(f"   Pricing recommendation: Calendar-based dynamic pricing")
else:
    print(f"❌ NOT VALIDATED: Temporal patterns not significant risk factors")

# =============================================================================
# BUSINESS IMPLEMENTATION RECOMMENDATIONS
# =============================================================================

print(f"\n" + "="*70)
print("PRICING MODEL IMPLEMENTATION STRATEGY")
print("="*70)

# Calculate risk-based pricing tiers
risk_percentiles = np.percentile(y_risk_proba, [20, 40, 60, 80, 100])
base_premium = 50  # Base premium amount

print(f"\nRISK-BASED PRICING TIERS:")
tier_names = ['Ultra-Low Risk', 'Low Risk', 'Medium Risk', 'High Risk', 'Ultra-High Risk']
for i, (tier, threshold) in enumerate(zip(tier_names, risk_percentiles)):
    risk_multiplier = 1 + (i * 0.3)  # 30% increase per tier
    suggested_premium = base_premium * risk_multiplier
    print(f"Tier {i+1} - {tier:<15}: Risk ≤ {threshold:.1%} | Premium: ${suggested_premium:.0f}")

print(f"\nKEY IMPLEMENTATION RECOMMENDATIONS:")
print(f"""
1. MODEL DEPLOYMENT:
   • Integrate model into booking platform for real-time pricing
   • Set optimal threshold at {optimal_threshold:.3f} for binary classification
   • Implement A/B testing framework for pricing strategy validation

2. PRICING STRATEGY:
   • Use 5-tier risk-based pricing structure
   • Apply {(feature_importance.abs().mean()*100):.0f}% average premium adjustment range
   • Focus monitoring on top {len(feature_importance[feature_importance.abs() > 0.1])} high-impact features

3. BUSINESS CONTROLS:
   • Monthly model retraining with new flight data
   • Competitor pricing analysis for market positioning
   • Customer elasticity testing for optimal premium levels
   • Regulatory compliance for non-discriminatory pricing

4. RISK MANAGEMENT:
   • Expected claim rate: {y.mean():.2%}
   • Model precision: {precision:.1%} (claim prediction accuracy)
   • Recommended loss ratio target: 60-70%
   • Portfolio diversification across low/high risk segments
""")

print(f"\n" + "="*70)
print("INSURANCE RISK ANALYSIS COMPLETE")
print("="*70)