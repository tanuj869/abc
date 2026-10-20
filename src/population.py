"""
State-wise population projections for 2024 (in Lakhs).
Source: National Commission on Population (MoHFW), Report of the Technical Group
on Population Projections for India and States 2011-2036.
1 Lakh = 100,000 persons.
"""

from src.state_mapping import CANONICAL_STATES

# Projected population in Lakhs (2024)
STATE_POPULATION_2024_LAKHS = {
    "Andhra Pradesh": 533.3,
    "Arunachal Pradesh": 15.9,
    "Assam": 361.0,
    "Bihar": 1285.0,
    "Chhattisgarh": 304.5,
    "Goa": 15.9,
    "Gujarat": 724.8,
    "Haryana": 305.2,
    "Himachal Pradesh": 74.8,
    "Jammu and Kashmir": 137.3,
    "Jharkhand": 401.2,
    "Karnataka": 682.5,
    "Kerala": 358.8,
    "Madhya Pradesh": 873.4,
    "Maharashtra": 1271.6,
    "Manipur": 32.4,
    "Meghalaya": 34.0,
    "Mizoram": 12.5,
    "Nagaland": 22.7,
    "Odisha": 466.8,
    "Punjab": 308.2,
    "Rajasthan": 819.3,
    "Sikkim": 7.0,
    "Tamil Nadu": 769.8,
    "Telangana": 384.7,
    "Tripura": 42.0,
    "Uttar Pradesh": 2382.4,
    "Uttarakhand": 117.5,
    "West Bengal": 997.4,
    "Andaman and Nicobar Islands": 4.1,
    "Chandigarh": 12.5,
    "Dadra and Nagar Haveli and Daman and Diu": 11.8,
    "Ladakh": 3.0,
    "Lakshadweep": 0.7,
    "NCT of Delhi": 217.2,
    "Puducherry": 16.6,
}

def get_population_lakhs(state_name: str) -> float:
    """Return projected 2024 population in lakhs for a canonical state."""
    return STATE_POPULATION_2024_LAKHS.get(state_name, None)
