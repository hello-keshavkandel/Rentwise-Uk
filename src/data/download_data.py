"""
download_data.py  —  Phase 1, Step 3: real data download (mechanism).

I download the real open datasets into data/raw/. I wrote this defensively: if a
government URL has moved or my network blocks it, I don't crash - I print a clear
'download this by hand' message and carry on, because ONS file names change with
every release.
"""

import sys
from pathlib import Path
import requests

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config


class DataDownloader:
    """I fetch the six open datasets my project uses into data/raw/."""

    # name -> (url to fetch, local filename)
    SOURCES = {
        "HM Land Registry Price Paid (latest)": (
            "http://prod1.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-monthly-update-new-version.csv",
            "hm_land_registry_price_paid_latest.csv"),
        "ONS IPHRP (rental index) - landing": (
            "https://www.ons.gov.uk/economy/inflationandpriceindices/bulletins/indexofprivatehousingrentalprices/latest",
            "ons_iphrp_landing.html"),
        "ONS PRMS (median rents) - landing": (
            "https://www.ons.gov.uk/peoplepopulationandcommunity/housing/bulletins/privaterentalmarketsummarystatisticsinengland/latest",
            "ons_prms_landing.html"),
        "ONS ASHE (earnings) - landing": (
            "https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/earningsandworkinghours/bulletins/annualsurveyofhoursandearnings/latest",
            "ons_ashe_landing.html"),
    }

    def __init__(self):
        # I make sure the raw-data folder exists before I download anything.
        config.DATA_RAW.mkdir(parents=True, exist_ok=True)

    def _download_one(self, name: str, url: str, filename: str) -> bool:
        # I try to download one file, and I explain clearly if I can't.
        target = config.DATA_RAW / filename
        try:
            print(f"[DataDownloader] I am fetching {name} ...")
            resp = requests.get(url, timeout=30, headers={"User-Agent": "RentWiseUK/1.0"})
            resp.raise_for_status()
            target.write_bytes(resp.content)
            print(f"                I saved -> {target} ({len(resp.content):,} bytes)")
            return True
        except Exception as e:
            print(f"                I could not download automatically: {e}")
            print(f"                Please open {url} and save it into {config.DATA_RAW}")
            return False

    def run(self) -> None:
        # I loop through every source and report how many I got automatically.
        print("RentWise UK - data download\n" + "-" * 40)
        ok = sum(self._download_one(n, u, f) for n, (u, f) in self.SOURCES.items())
        print("-" * 40)
        print(f"I downloaded {ok}/{len(self.SOURCES)} automatically.")
        print("Anything missing, I grab by hand from the printed link.")


if __name__ == "__main__":
    DataDownloader().run()
