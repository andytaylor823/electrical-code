"""Manual reconstruction of Table 310.4(1) from OCR fragments.

Uses a compact data-driven format: each conductor type defines its
temperature/application entries and AWG/mm/mils triplets. The script
cross-joins them into flattened rows automatically.

Usage:
    python scripts/reconstruct_310_4_1.py
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.resolve()
CACHE_FILE = ROOT / "data" / "intermediate" / "tables" / "table_llm_cache.json"
OUTPUT_FILE = ROOT / "data" / "intermediate" / "tables" / "table_310_4_1_reconstructed.json"
REVIEW_FILE = ROOT / "data" / "intermediate" / "table_310_4_1_review.md"

TITLE = "Table 310.4(1) Conductor Applications and Insulations Rated 600 Volts"

COLUMNS = [
    "Trade Name",
    "Type Letter",
    "Maximum Operating Temperature",
    "Application Provisions",
    "Insulation",
    "AWG or kcmil",
    "mm",
    "mils",
    "Outer Covering",
]

FOOTNOTES = [
    "Note: Conductors in Table 310.4(1) shall be permitted to be rated up to 1000 volts if listed and marked.",
    "1. Outer coverings shall not be required where listed without a covering.",
    "2. Higher temperature rated constructions shall be permitted where design conditions require maximum conductor operating temperatures above 90°C (194°F).",
    "3. Conductor sizes shall be permitted for signaling circuits permitting 300-volt insulation.",
    "4. The ampacity of Type UF cable shall be limited in accordance with 340.80.",
    "5. Type UF insulation thickness shall include the integral jacket.",
    "6. Insulation thickness shall be permitted to be 2.03 mm (80 mils) for listed Type USE conductors that have been subjected to special investigations. The nonmetallic covering over individual rubber-covered conductors of aluminum-sheathed cable and of lead-sheathed or multiconductor cable shall not be required to be flame retardant.",
    "Informational Note: See NFPA 79-2021, Electrical Standard for Industrial Machinery.",
]


# ── Conductor type definitions ────────────────────────────────────────────────
# Each entry in CONDUCTOR_TYPES has:
#   trade_name, type_letter, insulation,
#   entries: list of {temp, application, triplets: [(awg,mm,mils),...], covering}
#
# The script cross-joins each entry's triplets into flat rows.

CONDUCTOR_TYPES = [
    # ── FEP / FEPB ────────────────────────────────────────────────────────
    {
        "trade_name": "Fluorinated ethylene propylene",
        "type_letter": "FEP or FEPB",
        "insulation": "Fluorinated ethylene propylene",
        "entries": [
            {
                "temp": "90°C (194°F)",
                "application": "Dry and damp locations",
                "triplets": [("14-10", "0.51", "20"), ("8-2", "0.76", "30")],
                "covering": "None",
            },
            {
                "temp": "200°C (392°F)",
                "application": "Dry locations - special applications",
                "triplets": [("6-2", "0.36", "14"), ("14-8", "0.36", "14")],
                "covering": "Glass or other suitable braid material",
            },
        ],
    },
    # ── MI ─────────────────────────────────────────────────────────────────
    {
        "trade_name": "Mineral insulation (metal sheathed)",
        "type_letter": "MI",
        "insulation": "Magnesium oxide",
        "entries": [
            {
                "temp": "90°C (194°F)",
                "application": "Dry and wet locations",
                "triplets": [("18-16", "0.58", "23"), ("16-10", "0.91", "36"), ("9-4", "1.27", "50"), ("3-500", "1.40", "55")],
                "covering": "Copper or alloy steel",
            },
            {
                "temp": "250°C (482°F)",
                "application": "For special applications (footnote 2)",
                "triplets": [("18-16", "0.58", "23"), ("16-10", "0.91", "36"), ("9-4", "1.27", "50"), ("3-500", "1.40", "55")],
                "covering": "Copper or alloy steel",
            },
        ],
    },
    # ── MTW (A) - 60°C ────────────────────────────────────────────────────
    {
        "trade_name": "Moisture-, heat-, and oil-resistant thermoplastic",
        "type_letter": "MTW",
        "insulation": "Flame-retardant, moisture-, heat-, and oil-resistant thermoplastic",
        "entries": [
            {
                "temp": "60°C (140°F)",
                "application": "Machine tool wiring in wet locations",
                "triplets": [
                    ("22-12", "0.76", "30"),
                    ("10", "0.76", "30"),
                    ("8", "1.14", "45"),
                    ("6", "1.52", "60"),
                    ("4-2", "1.52", "60"),
                    ("1-4/0", "2.03", "80"),
                    ("213-500", "2.41", "95"),
                    ("501-1000", "2.79", "110"),
                ],
                "covering": "None",
            },
            {
                "temp": "90°C (194°F)",
                "application": "Machine tool wiring in dry locations",
                "triplets": [
                    ("22-12", "0.38", "15"),
                    ("10", "0.51", "20"),
                    ("8", "0.76", "30"),
                    ("6", "0.76", "30"),
                    ("4-2", "1.02", "40"),
                    ("1-4/0", "1.52", "60"),
                    ("213-500", "1.78", "70"),
                ],
                "covering": "Nylon jacket or equivalent",
            },
        ],
    },
    # ── Paper ──────────────────────────────────────────────────────────────
    {
        "trade_name": "Paper",
        "type_letter": "-",
        "insulation": "Paper",
        "entries": [
            {
                "temp": "85°C (185°F)",
                "application": "For underground service conductors, or by special permission",
                "triplets": [("-", "-", "-")],
                "covering": "Lead sheath",
            },
        ],
    },
    # ── PFA ────────────────────────────────────────────────────────────────
    {
        "trade_name": "Perfluoro-alkoxy",
        "type_letter": "PFA",
        "insulation": "Perfluoro-alkoxy",
        "entries": [
            {
                "temp": "90°C (194°F)",
                "application": "Dry and damp locations",
                "triplets": [("14-10", "0.51", "20"), ("8-2", "0.76", "30"), ("1-4/0", "1.14", "45")],
                "covering": "None",
            },
            {
                "temp": "200°C (392°F)",
                "application": "Dry locations - special applications",
                "triplets": [("14-10", "0.51", "20"), ("8-2", "0.76", "30"), ("1-4/0", "1.14", "45")],
                "covering": "None",
            },
        ],
    },
    # ── PFAH ───────────────────────────────────────────────────────────────
    {
        "trade_name": "Perfluoro-alkoxy",
        "type_letter": "PFAH",
        "insulation": "Perfluoro-alkoxy",
        "entries": [
            {
                "temp": "250°C (482°F)",
                "application": "Dry locations only. Only for leads within apparatus or within raceways connected to apparatus (nickel or nickel-coated copper only)",
                "triplets": [("14-10", "0.51", "20"), ("8-2", "0.76", "30"), ("1-4/0", "1.14", "45")],
                "covering": "None",
            },
        ],
    },
    # ── RHH ────────────────────────────────────────────────────────────────
    {
        "trade_name": "Thermoset",
        "type_letter": "RHH",
        "insulation": "Thermoset",
        "entries": [
            {
                "temp": "90°C (194°F)",
                "application": "Dry and damp locations",
                "triplets": [
                    ("14-10", "1.14", "45"),
                    ("8-2", "1.52", "60"),
                    ("1-4/0", "2.03", "80"),
                    ("213-500", "2.41", "95"),
                    ("501-1000", "2.79", "110"),
                    ("1001-2000", "3.18", "125"),
                ],
                "covering": "Moisture-resistant, flame-retardant, nonmetallic covering",
            },
        ],
    },
    # ── RHW ────────────────────────────────────────────────────────────────
    {
        "trade_name": "Moisture-resistant thermoset",
        "type_letter": "RHW",
        "insulation": "Flame-retardant, moisture-resistant thermoset",
        "entries": [
            {
                "temp": "75°C (167°F)",
                "application": "Dry and wet locations",
                "triplets": [
                    ("14-10", "1.14", "45"),
                    ("8-2", "1.52", "60"),
                    ("1-4/0", "2.03", "80"),
                    ("213-500", "2.41", "95"),
                    ("501-1000", "2.79", "110"),
                    ("1001-2000", "3.18", "125"),
                ],
                "covering": "Moisture-resistant, flame-retardant, nonmetallic covering",
            },
        ],
    },
    # ── RHW-2 ──────────────────────────────────────────────────────────────
    {
        "trade_name": "Moisture-resistant thermoset",
        "type_letter": "RHW-2",
        "insulation": "Flame-retardant, moisture-resistant thermoset",
        "entries": [
            {
                "temp": "90°C (194°F)",
                "application": "Dry and wet locations",
                "triplets": [
                    ("14-10", "1.14", "45"),
                    ("8-2", "1.52", "60"),
                    ("1-4/0", "2.03", "80"),
                    ("213-500", "2.41", "95"),
                    ("501-1000", "2.79", "110"),
                    ("1001-2000", "3.18", "125"),
                ],
                "covering": "Moisture-resistant, flame-retardant, nonmetallic covering",
            },
        ],
    },
    # ── SA ─────────────────────────────────────────────────────────────────
    {
        "trade_name": "Silicone",
        "type_letter": "SA",
        "insulation": "Silicone rubber",
        "entries": [
            {
                "temp": "90°C (194°F)",
                "application": "Dry and damp locations",
                "triplets": [
                    ("14-10", "1.14", "45"),
                    ("8-2", "1.52", "60"),
                    ("1-4/0", "2.03", "80"),
                    ("213-500", "2.41", "95"),
                    ("501-1000", "2.79", "110"),
                    ("1001-2000", "3.18", "125"),
                ],
                "covering": "Glass or other suitable braid material",
            },
            {
                "temp": "200°C (392°F)",
                "application": "For special application",
                "triplets": [
                    ("14-10", "1.14", "45"),
                    ("8-2", "1.52", "60"),
                    ("1-4/0", "2.03", "80"),
                    ("213-500", "2.41", "95"),
                    ("501-1000", "2.79", "110"),
                    ("1001-2000", "3.18", "125"),
                ],
                "covering": "Glass or other suitable braid material",
            },
        ],
    },
    # ── SIS ────────────────────────────────────────────────────────────────
    {
        "trade_name": "Thermoset",
        "type_letter": "SIS",
        "insulation": "Flame-retardant thermoset",
        "entries": [
            {
                "temp": "90°C (194°F)",
                "application": "Switchboard and switchgear wiring only",
                "triplets": [("14-10", "0.76", "30"), ("8-2", "1.14", "45"), ("1-4/0", "1.40", "55")],
                "covering": "None",
            },
        ],
    },
    # ── TBS ────────────────────────────────────────────────────────────────
    {
        "trade_name": "Thermoplastic and fibrous outer braid",
        "type_letter": "TBS",
        "insulation": "Thermoplastic",
        "entries": [
            {
                "temp": "90°C (194°F)",
                "application": "Switchboard and switchgear wiring only",
                "triplets": [("14-10", "0.76", "30"), ("8", "1.14", "45"), ("6-2", "1.52", "60"), ("1-4/0", "2.03", "80")],
                "covering": "Fibrous outer braid",
            },
        ],
    },
    # ── TFE ────────────────────────────────────────────────────────────────
    {
        "trade_name": "Extended polytetrafluoroethylene",
        "type_letter": "TFE",
        "insulation": "Extruded polytetrafluoroethylene",
        "entries": [
            {
                "temp": "250°C (482°F)",
                "application": "Dry locations only. Only for leads within apparatus or within raceways connected to apparatus, or as open wiring (nickel or nickel-coated copper only)",
                "triplets": [("14-10", "0.51", "20"), ("8-2", "0.76", "30"), ("1-4/0", "1.14", "45")],
                "covering": "None",
            },
        ],
    },
    # ── THHN ───────────────────────────────────────────────────────────────
    {
        "trade_name": "Heat-resistant thermoplastic",
        "type_letter": "THHN",
        "insulation": "Flame-retardant, heat-resistant thermoplastic",
        "entries": [
            {
                "temp": "90°C (194°F)",
                "application": "Dry and damp locations",
                "triplets": [
                    ("14-12", "0.38", "15"),
                    ("10", "0.51", "20"),
                    ("8-6", "0.76", "30"),
                    ("4-2", "1.02", "40"),
                    ("1-4/0", "1.27", "50"),
                    ("250-500", "1.52", "60"),
                    ("501-1000", "1.78", "70"),
                ],
                "covering": "Nylon jacket or equivalent",
            },
        ],
    },
    # ── THHW ───────────────────────────────────────────────────────────────
    {
        "trade_name": "Moisture- and heat-resistant thermoplastic",
        "type_letter": "THHW",
        "insulation": "Flame-retardant, moisture- and heat-resistant thermoplastic",
        "entries": [
            {
                "temp": "75°C (167°F)",
                "application": "Wet location",
                "triplets": [
                    ("14-10", "0.76", "30"),
                    ("8", "1.14", "45"),
                    ("6-2", "1.52", "60"),
                    ("1-4/0", "2.03", "80"),
                    ("213-500", "2.41", "95"),
                    ("501-1000", "2.79", "110"),
                    ("1001-2000", "3.18", "125"),
                ],
                "covering": "None",
            },
            {
                "temp": "90°C (194°F)",
                "application": "Dry location",
                "triplets": [
                    ("14-10", "0.76", "30"),
                    ("8", "1.14", "45"),
                    ("6-2", "1.52", "60"),
                    ("1-4/0", "2.03", "80"),
                    ("213-500", "2.41", "95"),
                    ("501-1000", "2.79", "110"),
                    ("1001-2000", "3.18", "125"),
                ],
                "covering": "None",
            },
        ],
    },
    # ── THW ────────────────────────────────────────────────────────────────
    {
        "trade_name": "Moisture-resistant thermoplastic",
        "type_letter": "THW",
        "insulation": "Flame-retardant, moisture-resistant thermoplastic",
        "entries": [
            {
                "temp": "75°C (167°F)",
                "application": "Dry and wet locations",
                "triplets": [
                    ("14-10", "0.76", "30"),
                    ("8", "1.14", "45"),
                    ("6-2", "1.52", "60"),
                    ("1-4/0", "2.03", "80"),
                    ("213-500", "2.41", "95"),
                    ("501-1000", "2.79", "110"),
                    ("1001-2000", "3.18", "125"),
                ],
                "covering": "None",
            },
            {
                "temp": "90°C (194°F)",
                "application": "Special applications within electric discharge lighting equipment. Limited to 1000 open-circuit volts or less. (Size 14-8 only as permitted in 410.68.)",
                "triplets": [
                    ("14-10", "0.76", "30"),
                    ("8", "1.14", "45"),
                    ("6-2", "1.52", "60"),
                    ("1-4/0", "2.03", "80"),
                    ("213-500", "2.41", "95"),
                    ("501-1000", "2.79", "110"),
                    ("1001-2000", "3.18", "125"),
                ],
                "covering": "None",
            },
        ],
    },
    # ── THW-2 ──────────────────────────────────────────────────────────────
    {
        "trade_name": "Moisture- and heat-resistant thermoplastic",
        "type_letter": "THW-2",
        "insulation": "Flame-retardant, moisture- and heat-resistant thermoplastic",
        "entries": [
            {
                "temp": "90°C (194°F)",
                "application": "Dry and wet locations",
                "triplets": [
                    ("14-10", "0.76", "30"),
                    ("8", "1.14", "45"),
                    ("6-2", "1.52", "60"),
                    ("1-4/0", "2.03", "80"),
                    ("213-500", "2.41", "95"),
                    ("501-1000", "2.79", "110"),
                    ("1001-2000", "3.18", "125"),
                ],
                "covering": "None",
            },
        ],
    },
    # ── THWN ───────────────────────────────────────────────────────────────
    {
        "trade_name": "Moisture- and heat-resistant thermoplastic",
        "type_letter": "THWN",
        "insulation": "Flame-retardant, moisture- and heat-resistant thermoplastic",
        "entries": [
            {
                "temp": "75°C (167°F)",
                "application": "Dry and wet locations",
                "triplets": [
                    ("14-12", "0.38", "15"),
                    ("10", "0.51", "20"),
                    ("8-6", "0.76", "30"),
                    ("4-2", "1.02", "40"),
                    ("1-4/0", "1.27", "50"),
                    ("250-500", "1.52", "60"),
                    ("501-1000", "1.78", "70"),
                ],
                "covering": "Nylon jacket or equivalent",
            },
        ],
    },
    # ── THWN-2 ─────────────────────────────────────────────────────────────
    {
        "trade_name": "Moisture- and heat-resistant thermoplastic",
        "type_letter": "THWN-2",
        "insulation": "Flame-retardant, moisture-resistant thermoplastic",
        "entries": [
            {
                "temp": "90°C (194°F)",
                "application": "Dry and wet locations",
                "triplets": [
                    ("14-12", "0.38", "15"),
                    ("10", "0.51", "20"),
                    ("8-6", "0.76", "30"),
                    ("4-2", "1.02", "40"),
                    ("1-4/0", "1.27", "50"),
                    ("250-500", "1.52", "60"),
                    ("501-1000", "1.78", "70"),
                ],
                "covering": "Nylon jacket or equivalent",
            },
        ],
    },
    # ── TW (NEW) ───────────────────────────────────────────────────────────
    {
        "trade_name": "Moisture-resistant thermoplastic",
        "type_letter": "TW",
        "insulation": "Flame-retardant, moisture-resistant thermoplastic",
        "entries": [
            {
                "temp": "60°C (140°F)",
                "application": "Dry and wet locations",
                "triplets": [
                    ("14-10", "0.76", "30"),
                    ("8", "1.14", "45"),
                    ("6-2", "1.52", "60"),
                    ("1-4/0", "2.03", "80"),
                    ("213-500", "2.41", "95"),
                    ("501-1000", "2.79", "110"),
                    ("1001-2000", "3.18", "125"),
                ],
                "covering": "None",
            },
        ],
    },
    # ── UF ─────────────────────────────────────────────────────────────────
    {
        "trade_name": "Underground feeder and branch-circuit cable - single conductor",
        "type_letter": "UF",
        "insulation": "Moisture-resistant thermoplastic",
        "entries": [
            {
                "temp": "60°C (140°F)",
                "application": "See Part II of Article 340",
                "triplets": [("14-10", "1.52", "60"), ("8-2", "2.03", "80"), ("1-4/0", "2.41", "95")],
                "covering": "Integral with insulation",
            },
            {
                "temp": "75°C (167°F)",
                "application": "See Part II of Article 340",
                "triplets": [("14-10", "1.52", "60"), ("8-2", "2.03", "80"), ("1-4/0", "2.41", "95")],
                "covering": "Integral with insulation",
            },
        ],
    },
    # ── USE ────────────────────────────────────────────────────────────────
    {
        "trade_name": "Underground service-entrance cable - single conductor",
        "type_letter": "USE",
        "insulation": "Heat- and moisture-resistant",
        "entries": [
            {
                "temp": "75°C (167°F)",
                "application": "See Part II of Article 338",
                "triplets": [
                    ("14-10", "1.14", "45"),
                    ("8-2", "1.52", "60"),
                    ("1-4/0", "2.03", "80"),
                    ("213-500", "2.41", "95"),
                    ("501-1000", "2.79", "110"),
                    ("1001-2000", "3.18", "125"),
                ],
                "covering": "Moisture-resistant nonmetallic covering (See 338.2.)",
            },
        ],
    },
    # ── USE-2 ──────────────────────────────────────────────────────────────
    {
        "trade_name": "Underground service-entrance cable - single conductor",
        "type_letter": "USE-2",
        "insulation": "Heat- and moisture-resistant",
        "entries": [
            {
                "temp": "90°C (194°F)",
                "application": "Dry and wet locations",
                "triplets": [
                    ("14-10", "1.14", "45"),
                    ("8-2", "1.52", "60"),
                    ("1-4/0", "2.03", "80"),
                    ("213-500", "2.41", "95"),
                    ("501-1000", "2.79", "110"),
                    ("1001-2000", "3.18", "125"),
                ],
                "covering": "Moisture-resistant nonmetallic covering (See 338.2.)",
            },
        ],
    },
    # ── XHH ────────────────────────────────────────────────────────────────
    {
        "trade_name": "Thermoset",
        "type_letter": "XHH",
        "insulation": "Flame-retardant thermoset",
        "entries": [
            {
                "temp": "90°C (194°F)",
                "application": "Dry and damp locations",
                "triplets": [
                    ("14-10", "0.76", "30"),
                    ("8-2", "1.14", "45"),
                    ("1-4/0", "1.40", "55"),
                    ("213-500", "1.63", "65"),
                    ("501-1000", "2.03", "80"),
                    ("1001-2000", "2.41", "95"),
                ],
                "covering": "None",
            },
        ],
    },
    # ── XHHN ───────────────────────────────────────────────────────────────
    {
        "trade_name": "Thermoset",
        "type_letter": "XHHN",
        "insulation": "Flame-retardant thermoset",
        "entries": [
            {
                "temp": "90°C (194°F)",
                "application": "Dry and damp locations",
                "triplets": [
                    ("14-12", "0.38", "15"),
                    ("10", "0.51", "20"),
                    ("8-6", "0.76", "30"),
                    ("4-2", "1.02", "40"),
                    ("1-4/0", "1.27", "50"),
                    ("250-500", "1.52", "60"),
                    ("501-1000", "1.78", "70"),
                ],
                "covering": "Nylon jacket or equivalent",
            },
        ],
    },
    # ── XHHW ───────────────────────────────────────────────────────────────
    {
        "trade_name": "Moisture-resistant thermoset",
        "type_letter": "XHHW",
        "insulation": "Flame-retardant, moisture-resistant thermoset",
        "entries": [
            {
                "temp": "90°C (194°F)",
                "application": "Dry and damp locations",
                "triplets": [
                    ("14-10", "0.76", "30"),
                    ("8-2", "1.14", "45"),
                    ("1-4/0", "1.40", "55"),
                    ("213-500", "1.63", "65"),
                    ("501-1000", "2.03", "80"),
                    ("1001-2000", "2.41", "95"),
                ],
                "covering": "None",
            },
            {
                "temp": "75°C (167°F)",
                "application": "Wet locations",
                "triplets": [
                    ("14-10", "0.76", "30"),
                    ("8-2", "1.14", "45"),
                    ("1-4/0", "1.40", "55"),
                    ("213-500", "1.63", "65"),
                    ("501-1000", "2.03", "80"),
                    ("1001-2000", "2.41", "95"),
                ],
                "covering": "None",
            },
        ],
    },
    # ── XHHW-2 (NEW) ──────────────────────────────────────────────────────
    {
        "trade_name": "Moisture-resistant thermoset",
        "type_letter": "XHHW-2",
        "insulation": "Flame-retardant, moisture-resistant thermoset",
        "entries": [
            {
                "temp": "90°C (194°F)",
                "application": "Dry and wet locations",
                "triplets": [
                    ("14-10", "0.76", "30"),
                    ("8-2", "1.14", "45"),
                    ("1-4/0", "1.40", "55"),
                    ("213-500", "1.63", "65"),
                    ("501-1000", "2.03", "80"),
                    ("1001-2000", "2.41", "95"),
                ],
                "covering": "None",
            },
        ],
    },
    # ── XHWN ───────────────────────────────────────────────────────────────
    {
        "trade_name": "Moisture-resistant thermoset",
        "type_letter": "XHWN",
        "insulation": "Flame-retardant, moisture-resistant thermoset",
        "entries": [
            {
                "temp": "75°C (167°F)",
                "application": "Dry and wet locations",
                "triplets": [
                    ("14-12", "0.38", "15"),
                    ("10", "0.51", "20"),
                    ("8-6", "0.76", "30"),
                    ("4-2", "1.02", "40"),
                    ("1-4/0", "1.27", "50"),
                    ("250-500", "1.52", "60"),
                    ("501-1000", "1.78", "70"),
                ],
                "covering": "Nylon jacket or equivalent",
            },
        ],
    },
    # ── XHWN-2 ─────────────────────────────────────────────────────────────
    {
        "trade_name": "Moisture-resistant thermoset",
        "type_letter": "XHWN-2",
        "insulation": "Flame-retardant, moisture-resistant thermoset",
        "entries": [
            {
                "temp": "90°C (194°F)",
                "application": "Dry and wet locations",
                "triplets": [
                    ("14-12", "0.38", "15"),
                    ("10", "0.51", "20"),
                    ("8-6", "0.76", "30"),
                    ("4-2", "1.02", "40"),
                    ("1-4/0", "1.27", "50"),
                    ("250-500", "1.52", "60"),
                    ("501-1000", "1.78", "70"),
                ],
                "covering": "Nylon jacket or equivalent",
            },
        ],
    },
    # ── Z ──────────────────────────────────────────────────────────────────
    {
        "trade_name": "Modified ethylene tetrafluoroethylene",
        "type_letter": "Z",
        "insulation": "Modified ethylene tetrafluoroethylene",
        "entries": [
            {
                "temp": "90°C (194°F)",
                "application": "Dry and damp locations",
                "triplets": [("14-12", "0.38", "15"), ("10", "0.51", "20")],
                "covering": "None",
            },
            {
                "temp": "150°C (302°F)",
                "application": "Dry locations - special applications",
                "triplets": [("8-4", "0.64", "25"), ("3-1", "0.89", "35"), ("1/0-4/0", "1.14", "45")],
                "covering": "None",
            },
        ],
    },
    # ── ZW ─────────────────────────────────────────────────────────────────
    {
        "trade_name": "Modified ethylene tetrafluoroethylene",
        "type_letter": "ZW",
        "insulation": "Modified ethylene tetrafluoroethylene",
        "entries": [
            {
                "temp": "75°C (167°F)",
                "application": "Wet locations",
                "triplets": [("14-10", "0.76", "30"), ("8-2", "1.14", "45")],
                "covering": "None",
            },
            {
                "temp": "90°C (194°F)",
                "application": "Dry and damp locations",
                "triplets": [("14-10", "0.76", "30"), ("8-2", "1.14", "45")],
                "covering": "None",
            },
            {
                "temp": "150°C (302°F)",
                "application": "Dry locations - special applications",
                "triplets": [("14-10", "0.76", "30"), ("8-2", "1.14", "45")],
                "covering": "None",
            },
        ],
    },
    # ── ZW-2 ───────────────────────────────────────────────────────────────
    {
        "trade_name": "Modified ethylene tetrafluoroethylene",
        "type_letter": "ZW-2",
        "insulation": "Modified ethylene tetrafluoroethylene",
        "entries": [
            {
                "temp": "90°C (194°F)",
                "application": "Dry and wet locations",
                "triplets": [("14-10", "0.76", "30"), ("8-2", "1.14", "45")],
                "covering": "None",
            },
        ],
    },
]


# ── Build flattened rows ──────────────────────────────────────────────────────


def build_rows():
    """Cross-join each conductor type's entries × triplets into flat rows."""
    rows = []
    for ctype in CONDUCTOR_TYPES:
        for entry in ctype["entries"]:
            for awg, mm, mils in entry["triplets"]:
                rows.append(
                    [
                        ctype["trade_name"],
                        ctype["type_letter"],
                        entry["temp"],
                        entry["application"],
                        ctype["insulation"],
                        awg,
                        mm,
                        mils,
                        entry["covering"],
                    ]
                )
    return rows


def render_markdown(table_structure):
    """Render the table structure as a markdown string with row indices."""
    headers = table_structure["column_headers"]
    rows = table_structure["data_rows"]
    footnotes = table_structure["footnotes"]

    lines = [f"# {table_structure['title']}", ""]
    lines.append("| Row | " + " | ".join(headers) + " |")
    lines.append("| --- | " + " | ".join(["---"] * len(headers)) + " |")
    for i, row in enumerate(rows):
        lines.append("| " + str(i) + " | " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Footnotes")
    lines.append("")
    for note in footnotes:
        lines.append("- " + note)
    return "\n".join(lines)


def main():
    """Build and save the reconstructed table."""
    rows = build_rows()
    table = {
        "title": TITLE,
        "column_headers": COLUMNS,
        "data_rows": rows,
        "footnotes": FOOTNOTES,
    }

    # Validate row widths
    n_cols = len(COLUMNS)
    for i, row in enumerate(rows):
        if len(row) != n_cols:
            logger.error("Row %d has %d cells, expected %d: %s", i, len(row), n_cols, row)
            return

    logger.info("Table 310.4(1): %d columns, %d rows, %d footnotes", n_cols, len(rows), len(FOOTNOTES))

    # Count unique types
    types_seen = sorted({row[1] for row in rows})
    logger.info("Conductor types: %s", types_seen)

    # Save JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fopen:
        json.dump(table, fopen, indent=2)
    logger.info("Wrote JSON to %s", OUTPUT_FILE)

    # Save markdown for review
    markdown = render_markdown(table)
    with open(REVIEW_FILE, "w", encoding="utf-8") as fopen:
        fopen.write(markdown)
    logger.info("Wrote review markdown to %s", REVIEW_FILE)

    # Update the LLM cache
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r", encoding="utf-8") as fopen:
            cache = json.load(fopen)
        cache["Table310.4(1)"] = table
        with open(CACHE_FILE, "w", encoding="utf-8") as fopen:
            json.dump(cache, fopen, indent=2)
        logger.info("Updated LLM cache at %s", CACHE_FILE)


if __name__ == "__main__":
    main()
