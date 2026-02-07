"""
Generate a comprehensive 500+ case employment law training dataset
Based on Canadian employment law patterns and the Sagaz test factors
"""

import csv
import random
from pathlib import Path

# Output path
OUTPUT_PATH = Path("data/employment_cases_large.csv")

# Sagaz test factor mappings with realistic values
SUPERVISION_LEVELS = {
    "Employee": ["High", "High", "High", "Moderate", "Moderate"],
    "Independent Contractor": ["Minimal", "Minimal", "Low", "Low", "Moderate"]
}

HIRE_ABILITY = {
    "Employee": ["No", "No", "No", "No", "Limited"],
    "Independent Contractor": ["Yes", "Yes", "Yes", "Limited", "Limited"]
}

DELEGATION = {
    "Employee": ["No", "No", "No", "Limited", "Limited"],
    "Independent Contractor": ["Yes", "Yes", "Yes", "Yes", "Limited"]
}

TOOLS_OWNERSHIP = {
    "Employee": ["Employer", "Employer", "Employer", "Mixed", "Employer"],
    "Independent Contractor": ["Worker", "Worker", "Worker", "Mixed", "Worker"]
}

PROFIT_CHANCE = {
    "Employee": ["No", "No", "No", "Limited", "No"],
    "Independent Contractor": ["Yes", "Yes", "Yes", "Yes", "Limited"]
}

LOSS_RISK = {
    "Employee": ["No", "No", "No", "No", "Limited"],
    "Independent Contractor": ["Yes", "Yes", "Yes", "Yes", "Limited"]
}

EXCLUSIVITY = {
    "Employee": ["Yes", "Yes", "Yes", "Yes", "Mostly"],
    "Independent Contractor": ["No", "No", "No", "Limited", "No"]
}

HOURS_CONTROL = {
    "Employee": ["Employer", "Employer", "Employer", "Employer", "Mixed"],
    "Independent Contractor": ["Worker", "Worker", "Worker", "Mixed", "Worker"]
}

WORK_LOCATIONS = {
    "Employee": [
        "Employer premises", "Company office", "Retail location", "Factory floor",
        "Hospital", "Restaurant", "Warehouse", "Call center", "School", "Hotel",
        "Store location", "Branch office", "Corporate HQ", "Regional office"
    ],
    "Independent Contractor": [
        "Remote", "Home office", "Client sites", "Various locations", "Own premises",
        "Coworking space", "On the road", "Multiple client locations", "Hybrid"
    ]
}

UNIFORM_REQUIRED = {
    "Employee": ["Yes", "Yes", "Yes", "Yes", "Sometimes"],
    "Independent Contractor": ["No", "No", "No", "No", "Sometimes"]
}

# Job roles by classification likelihood
EMPLOYEE_ROLES = [
    "Administrative Assistant", "Accountant", "Marketing Coordinator", "Sales Associate",
    "Customer Service Representative", "HR Manager", "Project Manager", "Software Developer",
    "Nurse", "Teacher", "Warehouse Worker", "Production Supervisor", "Quality Control",
    "Operations Manager", "Financial Analyst", "Legal Assistant", "Executive Assistant",
    "IT Support Specialist", "Data Entry Clerk", "Receptionist", "Cashier",
    "Bank Teller", "Security Guard", "Maintenance Technician", "Chef", "Server",
    "Bartender", "Hotel Concierge", "Flight Attendant", "Bus Driver", "Delivery Driver",
    "Retail Manager", "Pharmacy Technician", "Dental Hygienist", "Paramedic",
    "Social Worker", "Librarian", "Research Assistant", "Lab Technician",
    "Graphic Designer (in-house)", "Video Editor (staff)", "Content Writer (staff)",
    "Payroll Administrator", "Benefits Coordinator", "Training Specialist", "Recruiter",
    "Branch Manager", "Regional Director", "VP Operations", "CFO", "General Counsel"
]

CONTRACTOR_ROLES = [
    "Freelance Writer", "Independent Consultant", "Contract Developer", "Plumber",
    "Electrician", "Carpenter", "Photographer", "Videographer", "Translator",
    "Independent Sales Agent", "Real Estate Agent", "Insurance Broker", "Mortgage Agent",
    "Tax Preparer (seasonal)", "Event Planner", "Wedding Photographer", "DJ",
    "Personal Trainer", "Massage Therapist", "Yoga Instructor", "Private Tutor",
    "Graphic Designer (freelance)", "Web Designer", "UX Consultant", "SEO Specialist",
    "Marketing Consultant", "Business Consultant", "Management Consultant", "IT Consultant",
    "Bookkeeper", "Virtual Assistant", "Social Media Manager (contract)", "Content Creator",
    "Truck Owner-Operator", "Rideshare Driver", "Delivery Courier", "Food Delivery Driver",
    "Home Inspector", "Appraiser", "Surveyor", "Landscape Designer", "Interior Designer",
    "Architect (project-based)", "Engineer (consulting)", "Actuary (consulting)",
    "Expert Witness", "Mediator", "Arbitrator", "Court Reporter (freelance)"
]

# Company name patterns
COMPANY_TYPES = ["Inc.", "Ltd.", "Corp.", "Co.", "LLC", "Holdings", "Group", "Services", "Solutions"]
COMPANY_BASES = [
    "ABC", "XYZ", "First", "National", "Premier", "Global", "Dynamic", "Apex", "Summit",
    "Pinnacle", "Horizon", "NextGen", "Innovative", "Strategic", "Elite", "Prime",
    "Pacific", "Atlantic", "Northern", "Southern", "Western", "Eastern", "Central",
    "Metropolitan", "Regional", "Community", "United", "Allied", "Advanced", "Modern"
]
COMPANY_INDUSTRIES = [
    "Manufacturing", "Technology", "Financial", "Healthcare", "Retail", "Logistics",
    "Construction", "Hospitality", "Media", "Insurance", "Legal", "Consulting",
    "Education", "Real Estate", "Transportation", "Energy", "Telecom", "Pharmaceutical"
]

def generate_company_name():
    base = random.choice(COMPANY_BASES)
    industry = random.choice(COMPANY_INDUSTRIES)
    suffix = random.choice(COMPANY_TYPES)
    if random.random() > 0.5:
        return f"{base} {industry} {suffix}"
    return f"{base} {suffix}"

def generate_case_name(idx):
    first_names = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
        "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
        "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
        "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker",
        "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
        "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
        "Carter", "Roberts", "Chen", "Wong", "Kim", "Singh", "Patel", "O'Brien", "Murphy"
    ]
    return f"{random.choice(first_names)} v. {generate_company_name()}"

def generate_case(idx, outcome):
    """Generate a single case with consistent factors for the outcome"""
    
    if outcome == "Employee":
        role = random.choice(EMPLOYEE_ROLES)
    else:
        role = random.choice(CONTRACTOR_ROLES)
    
    case = {
        "Caseid": idx,
        "URL": f"https://canlii.org/en/on/onsc/{2020 + (idx % 5)}/case{idx}",
        "Case Name": generate_case_name(idx),
        "Supervision/review of work": random.choice(SUPERVISION_LEVELS[outcome]),
        "Ability to hire employees": random.choice(HIRE_ABILITY[outcome]),
        "Delegation of tasks": random.choice(DELEGATION[outcome]),
        "Ownership of tools": random.choice(TOOLS_OWNERSHIP[outcome]),
        "Chance of profit": random.choice(PROFIT_CHANCE[outcome]),
        "Risk of loss": random.choice(LOSS_RISK[outcome]),
        "Exclusivity of services": random.choice(EXCLUSIVITY[outcome]),
        "Who sets the work hours": random.choice(HOURS_CONTROL[outcome]),
        "Where the work is performed": random.choice(WORK_LOCATIONS[outcome]),
        "Is the worker required to wear a uniform?": random.choice(UNIFORM_REQUIRED[outcome]),
        "Outcome": outcome
    }
    
    return case

def add_edge_cases(cases):
    """Add realistic edge/borderline cases"""
    edge_cases = [
        # Gig economy cases - often contested
        {
            "Caseid": len(cases) + 1,
            "URL": "https://canlii.org/uber/2024/case1",
            "Case Name": "Heller v. Uber Technologies Inc.",
            "Supervision/review of work": "Moderate",
            "Ability to hire employees": "No",
            "Delegation of tasks": "No",
            "Ownership of tools": "Worker",
            "Chance of profit": "Limited",
            "Risk of loss": "Limited",
            "Exclusivity of services": "No",
            "Who sets the work hours": "Worker",
            "Where the work is performed": "Various locations",
            "Is the worker required to wear a uniform?": "No",
            "Outcome": "Independent Contractor"  # Classification disputed
        },
        # Dependent contractor scenario
        {
            "Caseid": len(cases) + 2,
            "URL": "https://canlii.org/dep/2023/case1",
            "Case Name": "Thurston v. Ontario (Children's Lawyer)",
            "Supervision/review of work": "Low",
            "Ability to hire employees": "No",
            "Delegation of tasks": "Limited",
            "Ownership of tools": "Mixed",
            "Chance of profit": "Limited",
            "Risk of loss": "No",
            "Exclusivity of services": "Yes",
            "Who sets the work hours": "Mixed",
            "Where the work is performed": "Mixed",
            "Is the worker required to wear a uniform?": "No",
            "Outcome": "Employee"  # Dependent contractor treated as employee
        },
        # IT consultant edge case
        {
            "Caseid": len(cases) + 3,
            "URL": "https://canlii.org/it/2024/case1",
            "Case Name": "Peterson v. Tech Solutions Corp.",
            "Supervision/review of work": "Low",
            "Ability to hire employees": "Yes",
            "Delegation of tasks": "Yes",
            "Ownership of tools": "Worker",
            "Chance of profit": "Yes",
            "Risk of loss": "Yes",
            "Exclusivity of services": "Mostly",
            "Who sets the work hours": "Worker",
            "Where the work is performed": "Remote",
            "Is the worker required to wear a uniform?": "No",
            "Outcome": "Independent Contractor"
        },
        # Real estate agent
        {
            "Caseid": len(cases) + 4,
            "URL": "https://canlii.org/re/2023/case1",
            "Case Name": "McKee v. Reid's Heritage Homes Ltd.",
            "Supervision/review of work": "Moderate",
            "Ability to hire employees": "No",
            "Delegation of tasks": "No",
            "Ownership of tools": "Employer",
            "Chance of profit": "Yes",
            "Risk of loss": "No",
            "Exclusivity of services": "Yes",
            "Who sets the work hours": "Employer",
            "Where the work is performed": "Employer premises",
            "Is the worker required to wear a uniform?": "No",
            "Outcome": "Employee"  # Despite commission structure
        },
        # Trucking owner-operator
        {
            "Caseid": len(cases) + 5,
            "URL": "https://canlii.org/truck/2024/case1",
            "Case Name": "Williams v. National Trucking Ltd.",
            "Supervision/review of work": "Minimal",
            "Ability to hire employees": "Yes",
            "Delegation of tasks": "Yes",
            "Ownership of tools": "Worker",
            "Chance of profit": "Yes",
            "Risk of loss": "Yes",
            "Exclusivity of services": "No",
            "Who sets the work hours": "Worker",
            "Where the work is performed": "On the road",
            "Is the worker required to wear a uniform?": "No",
            "Outcome": "Independent Contractor"
        }
    ]
    
    return cases + edge_cases

def main():
    print("=" * 60)
    print("GENERATING LARGE EMPLOYMENT CASES DATASET")
    print("=" * 60)
    
    cases = []
    
    # Generate 600 employee cases
    print("\n📊 Generating 600 Employee cases...")
    for i in range(1, 601):
        cases.append(generate_case(i, "Employee"))
    
    # Generate 600 independent contractor cases
    print("📊 Generating 600 Independent Contractor cases...")
    for i in range(601, 1201):
        cases.append(generate_case(i, "Independent Contractor"))
    
    # Add edge cases
    print("📊 Adding edge cases...")
    cases = add_edge_cases(cases)
    
    # Add more mixed/borderline cases
    print("📊 Generating 50 borderline cases...")
    for i in range(len(cases) + 1, len(cases) + 51):
        # 50/50 random outcome for borderline
        outcome = random.choice(["Employee", "Independent Contractor"])
        case = generate_case(i, outcome)
        # Make some factors "mixed" to represent borderline
        if random.random() > 0.5:
            case["Ownership of tools"] = "Mixed"
        if random.random() > 0.5:
            case["Who sets the work hours"] = "Mixed"
        if random.random() > 0.5:
            case["Exclusivity of services"] = "Mostly"
        cases.append(case)
    
    # Shuffle to mix outcomes
    random.shuffle(cases)
    
    # Reassign case IDs
    for i, case in enumerate(cases, 1):
        case["Caseid"] = i
    
    # Write to CSV
    print(f"\n💾 Writing {len(cases)} cases to {OUTPUT_PATH}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        "Caseid", "URL", "Case Name",
        "Supervision/review of work", "Ability to hire employees",
        "Delegation of tasks", "Ownership of tools",
        "Chance of profit", "Risk of loss",
        "Exclusivity of services", "Who sets the work hours",
        "Where the work is performed", "Is the worker required to wear a uniform?",
        "Outcome"
    ]
    
    with open(OUTPUT_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cases)
    
    # Count outcomes
    employees = sum(1 for c in cases if c["Outcome"] == "Employee")
    contractors = sum(1 for c in cases if c["Outcome"] == "Independent Contractor")
    
    print("\n" + "-" * 40)
    print("DATASET SUMMARY")
    print("-" * 40)
    print(f"✅ Total cases: {len(cases)}")
    print(f"👔 Employee cases: {employees}")
    print(f"📝 Independent Contractor cases: {contractors}")
    print(f"📁 Saved to: {OUTPUT_PATH}")
    
    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
