"""
make_sample_data.py  —  Phase 1 (development stand-in).

I generate a SYNTHETIC dataset with the same columns I expect from the real
ONS data, so I can build and test my whole pipeline today without waiting on
downloads. It is clearly synthetic and I never use it for final results - the real
ONS data replaces it later with no other code change.

I cover ~117 Local Authorities across the ten UK regions I model, so the
development set is broad enough (~1,750 rows) to exercise cross-regional
validation, a realistic affordability spread, and stable SHAP estimates.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config


class SampleDataGenerator:
    """I build a realistic synthetic rent dataset at (local authority x property type x year).

    Why I do this (I explain this in the viva): software is built fastest when I
    always have something to run. Because I build rent from a transparent formula
    (income, transport, crime, bedrooms) plus noise, a good model *should* recover
    those drivers - which is exactly the check I want while testing the pipeline.
    """

    # I list real Local Authority names grouped by the ten regions I model, so the
    # development set has genuine UK breadth. The values attached to them are
    # synthetic; only the names and their region are real.
    LA_BY_REGION = {
        "London": [
            "City of London", "Camden", "Lambeth", "Westminster", "Islington",
            "Hackney", "Tower Hamlets", "Southwark", "Lewisham", "Greenwich",
            "Wandsworth", "Hammersmith and Fulham", "Kensington and Chelsea",
            "Newham", "Barnet", "Brent", "Ealing", "Croydon", "Bromley",
            "Enfield", "Haringey", "Waltham Forest", "Redbridge", "Hounslow",
            "Harrow",
        ],
        "North West": [
            "Manchester", "Liverpool", "Salford", "Bolton", "Stockport", "Wigan",
            "Trafford", "Rochdale", "Oldham", "Bury", "Tameside", "Sefton",
            "Wirral", "Preston",
        ],
        "Yorkshire and The Humber": [
            "Leeds", "Sheffield", "Bradford", "Kirklees", "Wakefield",
            "Calderdale", "Kingston upon Hull", "York", "Rotherham", "Doncaster",
        ],
        "West Midlands": [
            "Birmingham", "Coventry", "Wolverhampton", "Dudley", "Sandwell",
            "Walsall", "Solihull", "Stoke-on-Trent", "Telford and Wrekin",
            "Warwick",
        ],
        "East Midlands": [
            "Derby", "Nottingham", "Leicester", "Lincoln", "Northampton",
            "Chesterfield", "Mansfield", "Charnwood", "Rushcliffe", "Gedling",
        ],
        "South East": [
            "Milton Keynes", "Brighton and Hove", "Reading", "Oxford",
            "Southampton", "Portsmouth", "Slough", "Medway", "Maidstone",
            "Canterbury", "Basingstoke and Deane", "Guildford",
        ],
        "South West": [
            "Bristol", "North Somerset", "Plymouth", "Exeter",
            "Bath and North East Somerset", "Cornwall", "Swindon", "Gloucester",
            "Cheltenham", "Bournemouth, Christchurch and Poole",
        ],
        "East of England": [
            "Bedford", "Norwich", "Ipswich", "Cambridge", "Peterborough",
            "Luton", "Colchester", "Chelmsford", "Watford", "Southend-on-Sea",
        ],
        "North East": [
            "County Durham", "Newcastle upon Tyne", "Sunderland", "Gateshead",
            "North Tyneside", "South Tyneside", "Northumberland", "Darlington",
        ],
        "Wales": [
            "Cardiff", "Swansea", "Newport", "Wrexham", "Rhondda Cynon Taf",
            "Bridgend", "Caerphilly", "Flintshire",
        ],
    }

    # I give each region a plausible ONS-style area-code prefix (synthetic serials).
    REGION_CODE_PREFIX = {
        "London": "E09", "North West": "E08", "Yorkshire and The Humber": "E08",
        "West Midlands": "E08", "East Midlands": "E06", "South East": "E06",
        "South West": "E06", "East of England": "E06", "North East": "E06",
        "Wales": "W06",
    }

    # I shift each region's income and rent level so affordability varies the way it
    # does in reality (London / South East highest burden; North East / Wales lower).
    REGION_INCOME_SHIFT = {
        "London": 0, "South East": 2500, "East of England": 1200,
        "South West": 0, "West Midlands": -1500, "East Midlands": -2200,
        "North West": -1800, "Yorkshire and The Humber": -2200,
        "Wales": -2600, "North East": -3200,
    }
    REGION_RENT_SHIFT = {
        "London": 150, "South East": 70, "East of England": 30,
        "South West": 10, "West Midlands": -20, "East Midlands": -35,
        "North West": -28, "Yorkshire and The Humber": -40,
        "Wales": -50, "North East": -65,
    }

    PROPERTY_TYPES = {"Room": 1, "Flat": 2, "Terraced": 3, "Semi-detached": 4, "Detached": 5}

    def __init__(self, n_years: int = 3):
        # I fix the random generator with my seed so the data is reproducible.
        self.n_years = n_years
        self.rng = np.random.default_rng(config.RANDOM_SEED)

    def _local_authorities(self):
        # I turn the region -> names map into (code, name, region) rows with codes.
        serial = {}
        for region, names in self.LA_BY_REGION.items():
            prefix = self.REGION_CODE_PREFIX[region]
            for name in names:
                serial[prefix] = serial.get(prefix, 0) + 1
                code = f"{prefix}{serial[prefix]:06d}"
                yield code, name, region

    def generate(self) -> pd.DataFrame:
        # I create one row per (local authority x property type x year).
        rows = []
        for (la_code, la_name, region) in self._local_authorities():
            # I give each area a base income (region shift + local noise).
            base_income = self.rng.normal(35500, 5200) + self.REGION_INCOME_SHIFT[region]
            median_annual_income = max(20000, base_income)

            # I set the area's location features once (London busier and higher-crime).
            crime = max(15, self.rng.normal(90 if region == "London" else 58, 18))
            transport = float(np.clip(
                self.rng.normal(0.75 if region == "London" else 0.45, 0.15), 0, 1))

            for ptype, bedrooms in self.PROPERTY_TYPES.items():
                for year in range(2022, 2022 + self.n_years):
                    # This is my transparent "true" rent that the model should learn.
                    rent = (250
                            + bedrooms * 180
                            + (median_annual_income / 1000) * 7
                            + transport * 250
                            - crime * 1.2
                            + self.REGION_RENT_SHIFT[region]
                            + (year - 2022) * 25)
                    rent += self.rng.normal(0, 55)  # I add a little random noise
                    rows.append({
                        "la_code": la_code, "la_name": la_name, "region": region,
                        "year": year, "property_type": ptype, "bedrooms": bedrooms,
                        "median_annual_income": round(median_annual_income, 0),
                        "crime_per_1000": round(crime, 1),
                        "transport_score": round(transport, 3),
                        "median_monthly_rent": round(max(250, rent), 0),
                    })
        return pd.DataFrame(rows)

    def run(self) -> Path:
        # I generate the data and save it to the processed folder.
        config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        df = self.generate()
        out = config.DATA_PROCESSED / "rentwise_sample.csv"
        df.to_csv(out, index=False)
        print(f"[SampleDataGenerator] I wrote {len(df)} rows "
              f"({df['la_name'].nunique()} local authorities x "
              f"{len(self.PROPERTY_TYPES)} property types x {self.n_years} years) to {out}")
        print("[SampleDataGenerator] Columns:", list(df.columns))
        print("NOTE: this is SYNTHETIC development data, not real ONS data.")
        return out


if __name__ == "__main__":
    SampleDataGenerator().run()
