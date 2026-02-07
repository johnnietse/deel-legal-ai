import csv
from pathlib import Path

OUTPUT_CSV = Path("real_cases_to_scrape.csv")

LANDMARK_CASES = [
    {
        "Caseid": "2020scc16",
        "URL": "https://www.canlii.org/en/ca/scc/doc/2020/2020scc16/2020scc16.html",
        "Case Name": "Uber Technologies Inc. v. Heller",
        "Outcome": "Independent Contractor (Arbitration Clause Invalid)"
    },
    {
        "Caseid": "2020scc26",
        "URL": "https://www.canlii.org/en/ca/scc/doc/2020/2020scc26/2020scc26.html",
        "Case Name": "Matthews v. Ocean Nutrition Canada Ltd.",
        "Outcome": "Employee (Bonus Entitlement)"
    },
    {
        "Caseid": "2020onca391",
        "URL": "https://www.canlii.org/en/on/onca/doc/2020/2020onca391/2020onca391.html",
        "Case Name": "Waksdale v. Swegon North America Inc.",
        "Outcome": "Employee (Termination Clause Void)"
    },
    {
        "Caseid": "2019onca512",
        "URL": "https://www.canlii.org/en/on/onca/doc/2019/2019onca512/2019onca512.html",
        "Case Name": "Dawe v. The Equitable Life Insurance Company",
        "Outcome": "Employee (Notice Period)"
    },
    {
        "Caseid": "2017onca402",
        "URL": "https://www.canlii.org/en/on/onca/doc/2017/2017onca402/2017onca402.html",
        "Case Name": "Brake v. PJ-M2R Restaurant Inc.",
        "Outcome": "Employee (Mitigation)"
    },
    {
        "Caseid": "2016onca79",
        "URL": "https://www.canlii.org/en/on/onca/doc/2016/2016onca79/2016onca79.html",
        "Case Name": "Keenan v. Canac Kitchens Ltd.",
        "Outcome": "Dependent Contractor"
    },
    {
        "Caseid": "2009onca916",
        "URL": "https://www.canlii.org/en/on/onca/doc/2009/2009onca916/2009onca916.html",
        "Case Name": "McKee v. Reid's Heritage Homes Ltd.",
        "Outcome": "Employee vs Contractor"
    },
    {
        "Caseid": "2001scc59",
        "URL": "https://www.canlii.org/en/ca/scc/doc/2001/2001scc59/2001scc59.html",
        "Case Name": "671122 Ontario Ltd. v. Sagaz Industries Canada Inc.",
        "Outcome": "Independent Contractor (Leading Case)"
    },
    {
        "Caseid": "2024onca55",
        "URL": "https://www.canlii.org/en/on/onca/doc/2024/2024onca55/2024onca55.html",
        "Case Name": "D'Souza v. 123 Corp", # Placeholder for 2024
        "Outcome": "Unknown"
    }
]

def create_csv():
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["Caseid", "URL", "Case Name", "Outcome"])
        writer.writeheader()
        writer.writerows(LANDMARK_CASES)
    print(f"Created {OUTPUT_CSV} with {len(LANDMARK_CASES)} landmark cases")

if __name__ == "__main__":
    create_csv()
