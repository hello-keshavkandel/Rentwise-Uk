# SHAP Explainability - RentWise UK (Module 2)

## Global importance (average £ impact on predicted rent)
- **being in London**: £194
- **local income**: £182
- **number of bedrooms**: £142
- **being in South East**: £89
- **crime per 1,000 residents**: £77
- **property type/size**: £63
- **being in East of England**: £35
- **being in South West**: £27
- **being in Wales**: £12
- **being in North West**: £8

## Example: explaining one prediction
The model starts from a baseline of about £1293 and adjusts to **£1788** for this property:
- Being in south east increases the estimate by £337.
- Number of bedrooms increases the estimate by £257.
- Local income decreases the estimate by £137.
- Being in london decreases the estimate by £88.
- Property type/size increases the estimate by £67.
- Crime per 1,000 residents increases the estimate by £55.

*SHAP values are in pounds and sum to the model's prediction. Figures from synthetic development data; regenerated on real ONS data.*