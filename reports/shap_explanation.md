# SHAP Explainability - RentWise UK (Module 2)

## Global importance (average £ impact on predicted rent)
- **local income**: £196
- **number of bedrooms**: £121
- **being in South East**: £81
- **being in London**: £58
- **local crime level**: £57
- **property type/size**: £55
- **being in East of England**: £32
- **being in South West**: £18
- **being in North West**: £14
- **being in Wales**: £11

## Example: explaining one prediction
The model starts from a baseline of about £1182 and adjusts to **£1782** for this property:
- Being in south east increases the estimate by £381.
- Number of bedrooms increases the estimate by £259.
- Local income decreases the estimate by £114.
- Property type/size increases the estimate by £49.
- Local crime level increases the estimate by £20.
- Being in london decreases the estimate by £18.

*SHAP values are in pounds and sum to the model's prediction. Figures from synthetic development data; regenerated on real ONS data.*