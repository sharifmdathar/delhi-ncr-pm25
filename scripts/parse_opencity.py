#!/usr/bin/env python3
"""Parse OpenCity wide-format CSVs into long-format station_id,timestamp,pm25.

Handles two formats:
1. Full month name: "January-2017" with empty rows between header and data
2. Abbreviated month: "Jan-17" with data immediately after header
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from datetime import datetime

MONTH_MAP = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6,
    "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sep": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}

def parse_opencity_file(filepath: Path, station_id: str) -> pd.DataFrame:
    """Parse a single OpenCity CSV into long format."""
    
    with open(filepath, 'r') as f:
        lines = [line.rstrip('\n') for line in f]
    
    rows = []
    current_year = None
    current_month = None
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        
        if not line or line == '""':
            continue
        
        # Year header
        if line.startswith("Year,"):
            current_year = int(line.split(",")[1])
            continue
        
        # Month header - handle both "January-2017" and "Jan-17" formats
        month_match = re.match(r"^([A-Za-z]+)-(\d{2,4}),", line)
        if month_match:
            month_str = month_match.group(1).lower()
            year_str = month_match.group(2)
            
            if month_str not in MONTH_MAP:
                continue
            current_month = MONTH_MAP[month_str]
            
            # Handle 2-digit vs 4-digit year
            if len(year_str) == 2:
                current_year = 2000 + int(year_str)
            else:
                current_year = int(year_str)
            continue
        
        # Data row: day, hour1, hour2, ..., hour24
        parts = line.split(",")
        if len(parts) < 2:
            continue
        
        try:
            day = int(parts[0])
        except ValueError:
            continue
        
        if current_year is None or current_month is None:
            continue
        
        # Process 24 hourly values
        for hour in range(24):
            if hour + 1 >= len(parts):
                break
            val = parts[hour + 1].strip()
            if val == "" or val.upper() in ("NA", "NAN", "NULL"):
                continue
            try:
                pm25 = float(val)
                if pm25 < 0 or pm25 > 1000:
                    continue
                
                ts = datetime(current_year, current_month, day, hour)
                rows.append({
                    "station_id": station_id,
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "pm25": pm25
                })
            except (ValueError, IndexError):
                continue
    
    return pd.DataFrame(rows)


# Station name normalization (map OpenCity names to standard IDs)
STATION_NAME_MAP = {
    "alipur_dpcc": "DL001",
    "anand_vihar_dpcc": "DL002",
    "ashok_vihar_dpcc": "DL003",
    "aya_nagar_imd": "DL004",
    "bawana_dpcc": "DL005",
    "burari_crossing_imd": "DL006",
    "chandni_chowk_iitm": "DL007",
    "crri_mathura_road_imd": "DL008",
    "dr_karni_singh_shooting_range_dpcc": "DL009",
    "dtu_cpcb": "DL009",
    "dwarka_sector_8_dpcc": "DL010",
    "igi_airport_t3_imd": "DL011",
    "ihbas_dilshad_garden_cpcb": "DL012",
    "ito_cpcb": "DL013",
    "jahangirpuri_dpcc": "DL014",
    "jawaharlal_nehru_stadium_dpcc": "DL015",
    "lodhi_road_iitm": "DL016",
    "lodhi_road_imd": "DL017",
    "major_dhyan_chand_national_stadium_dpcc": "DL018",
    "mandir_marg_dpcc": "DL019",
    "mundka_dpcc": "DL020",
    "najafgarh_dpcc": "DL021",
    "narela_dpcc": "DL022",
    "nehru_nagar_dpcc": "DL023",
    "north_campus_du_imd": "DL024",
    "nsit_dwarka_cpcb": "DL025",
    "okhla_phase_2_dpcc": "DL026",
    "patparganj_dpcc": "DL027",
    "punjabi_bagh_dpcc": "DL028",
    "pusa_dpcc": "DL029",
    "pusa_imd": "DL030",
    "rohini_dpcc": "DL031",
    "r_k_puram_dpcc": "DL032",
    "shadipur_cpcb": "DL033",
    "sirifort_cpcb": "DL034",
    "sonia_vihar_dpcc": "DL035",
    "sri_auroindo_marg_dpcc": "DL036",
    "sri_aurobindo_marg_dpcc": "DL036",
    "vivek_vihar_dpcc": "DL037",
    "wazirpur_dpcc": "DL038",
    # NCR stations (from Kaggle)
    "new_industrial_town_faridabad_hspcb": "HR006",
    "sector_11_faridabad_hspcb": "HR007",
    "sector_30_faridabad_hspcb": "HR008",
    "sector_16a_faridabad_hspcb": "HR009",
    "indirapuram_ghaziabad_uppcb": "UP004",
    "loni_ghaziabad_uppcb": "UP005",
    "sanjay_nagar_ghaziabad_uppcb": "UP006",
    "vasundhara_ghaziabad_uppcb": "UP007",
    "knowledge_park_iii_greater_noida_uppcb": "UP008",
    "knowledge_park_v_greater_noida_uppcb": "UP009",
    "sector_125_noida_uppcb": "UP022",
    "sector_62_noida_imd": "UP023",
    "sector_1_noida_uppcb": "UP024",
    "sector_116_noida_uppcb": "UP025",
}

def normalize_station_id(filename_stem: str) -> str:
    """Convert filename to standard station ID."""
    # Remove year suffixes like _2017_23, _2017-23, _2024_25, _2024-25
    name = re.sub(r'_\d{4}[-_]\d{2}$', '', filename_stem)
    name = name.lower().replace("-", "_")
    return STATION_NAME_MAP.get(name, name.upper())


def parse_all_opencity(opencity_dir: Path, output_path: Path):
    """Parse all OpenCity station files and combine."""
    all_dfs = []
    
    csv_files = list(opencity_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} station files")
    
    for fp in csv_files:
        station_id = normalize_station_id(fp.stem)
        print(f"  Parsing {station_id} ({fp.name})...")
        try:
            df = parse_opencity_file(fp, station_id)
            if len(df) > 0:
                all_dfs.append(df)
                print(f"    -> {len(df)} rows")
            else:
                print(f"    -> EMPTY")
        except Exception as e:
            print(f"    -> ERROR: {e}")
    
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        # Sort by station_id, timestamp
        combined = combined.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
        combined.to_csv(output_path, index=False)
        print(f"\nCombined: {len(combined)} rows, {combined['station_id'].nunique()} stations -> {output_path}")
    else:
        print("No data parsed!")


if __name__ == "__main__":
    opencity_dir = Path("data/raw/opencity")
    output_path = Path("data/raw/cpcb_hourly_opencity.csv")
    parse_all_opencity(opencity_dir, output_path)