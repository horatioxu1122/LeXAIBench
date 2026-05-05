import csv
from pathlib import Path
from typing import Any

from ..paths import repo_relative_path


RAW_TO_BENCHMARK_HEADERS = {
    "laufkont": "status",
    "laufzeit": "duration",
    "moral": "credit_history",
    "verw": "purpose",
    "hoehe": "amount",
    "sparkont": "savings",
    "beszeit": "employment_duration",
    "rate": "installment_rate",
    "famges": "personal_status_sex",
    "buerge": "other_debtors",
    "wohnzeit": "present_residence",
    "verm": "property",
    "alter": "age",
    "weitkred": "other_installment_plans",
    "wohn": "housing",
    "bishkred": "number_credits",
    "beruf": "job",
    "pers": "people_liable",
    "telef": "telephone",
    "gastarb": "foreign_worker",
    "kredit": "credit_risk",
}

NUMERIC_FIELDS = {"duration", "amount", "age"}

CATEGORICAL_LABELS: dict[str, dict[int, str]] = {
    "status": {
        1: "no checking account",
        2: "... < 0 DM",
        3: "0<= ... < 200 DM",
        4: "... >= 200 DM / salary for at least 1 year",
    },
    "credit_history": {
        0: "delay in paying off in the past",
        1: "critical account/other credits elsewhere",
        2: "no credits taken/all credits paid back duly",
        3: "existing credits paid back duly till now",
        4: "all credits at this bank paid back duly",
    },
    "purpose": {
        0: "others",
        1: "car (new)",
        2: "car (used)",
        3: "furniture/equipment",
        4: "radio/television",
        5: "domestic appliances",
        6: "repairs",
        7: "education",
        8: "vacation",
        9: "retraining",
        10: "business",
    },
    "savings": {
        1: "unknown/no savings account",
        2: "... <  100 DM",
        3: "100 <= ... <  500 DM",
        4: "500 <= ... < 1000 DM",
        5: "... >= 1000 DM",
    },
    "employment_duration": {
        1: "unemployed",
        2: "< 1 yr",
        3: "1 <= ... < 4 yrs",
        4: "4 <= ... < 7 yrs",
        5: ">= 7 yrs",
    },
    "installment_rate": {
        1: ">= 35",
        2: "25 <= ... < 35",
        3: "20 <= ... < 25",
        4: "< 20",
    },
    "personal_status_sex": {
        1: "male : divorced/separated",
        2: "female : non-single or male : single",
        3: "male : married/widowed",
        4: "female : single",
    },
    "other_debtors": {
        1: "none",
        2: "co-applicant",
        3: "guarantor",
    },
    "present_residence": {
        1: "< 1 yr",
        2: "1 <= ... < 4 yrs",
        3: "4 <= ... < 7 yrs",
        4: ">= 7 yrs",
    },
    "property": {
        1: "unknown / no property",
        2: "car or other",
        3: "building soc. savings agr./life insurance",
        4: "real estate",
    },
    "other_installment_plans": {
        1: "bank",
        2: "stores",
        3: "none",
    },
    "housing": {
        1: "for free",
        2: "rent",
        3: "own",
    },
    "number_credits": {
        1: "1",
        2: "2-3",
        3: "4-5",
        4: ">= 6",
    },
    "job": {
        1: "unemployed/unskilled - non-resident",
        2: "unskilled - resident",
        3: "skilled employee/official",
        4: "manager/self-empl./highly qualif. employee",
    },
    "people_liable": {
        1: "3 or more",
        2: "0 to 2",
    },
    "telephone": {
        1: "no",
        2: "yes (under customer name)",
    },
    "foreign_worker": {
        1: "yes",
        2: "no",
    },
    "credit_risk": {
        0: "bad",
        1: "good",
    },
}

DEFAULT_CREDIT_SHEET_NAME = "Data"
DEFAULT_CREDIT_HEADER_ROW_INDEX = 1
DEFAULT_CREDIT_TARGET_RENAME = {
    "default payment next month": "default_payment_next_month",
}
DEFAULT_CREDIT_EXPECTED_COLUMNS = [
    "ID",
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
    "default_payment_next_month",
]

HELOC_OPENML_DATA_ID = 45554
HELOC_EXPECTED_FEATURE_COLUMNS = [
    "ExternalRiskEstimate",
    "MSinceOldestTradeOpen",
    "MSinceMostRecentTradeOpen",
    "AverageMInFile",
    "NumSatisfactoryTrades",
    "NumTrades60Ever2DerogPubRec",
    "NumTrades90Ever2DerogPubRec",
    "PercentTradesNeverDelq",
    "MSinceMostRecentDelq",
    "MaxDelq2PublicRecLast12M",
    "MaxDelqEver",
    "NumTotalTrades",
    "NumTradesOpeninLast12M",
    "PercentInstallTrades",
    "MSinceMostRecentInqexcl7days",
    "NumInqLast6M",
    "NumInqLast6Mexcl7days",
    "NetFractionRevolvingBurden",
    "NetFractionInstallBurden",
    "NumRevolvingTradesWBalance",
    "NumInstallTradesWBalance",
    "NumBank2NatlTradesWHighUtilization",
    "PercentTradesWBalance",
]
HELOC_OUTPUT_COLUMNS = ["RiskPerformance", *HELOC_EXPECTED_FEATURE_COLUMNS]
HELOC_CATEGORICAL_CODE_COLUMNS = {
    "MaxDelq2PublicRecLast12M",
    "MaxDelqEver",
}

GIVE_ME_SOME_CREDIT_OPENML_DATA_ID = 46929
GIVE_ME_SOME_CREDIT_TARGET = "FinancialDistressNextTwoYears"
GIVE_ME_SOME_CREDIT_FEATURE_COLUMNS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]
GIVE_ME_SOME_CREDIT_OUTPUT_COLUMNS = [
    *GIVE_ME_SOME_CREDIT_FEATURE_COLUMNS,
    GIVE_ME_SOME_CREDIT_TARGET,
]

HMDA_TARGET = "mortgage_application_outcome"
HMDA_OUTPUT_COLUMNS = [
    HMDA_TARGET,
    "loan_amount",
    "loan_type",
    "loan_purpose",
    "occupancy_type",
    "derived_dwelling_category",
    "lien_status",
    "total_units",
    "income",
    "debt_to_income_ratio",
    "combined_loan_to_value_ratio",
    "applicant_age",
    "co_applicant_age",
    "derived_race",
    "derived_ethnicity",
    "derived_sex",
    "state_code",
    "county_code",
    "census_tract",
]

HMDA_INPUT_ALIASES: dict[str, list[str]] = {
    "action_taken": ["action_taken"],
    "loan_amount": ["loan_amount"],
    "loan_type": ["loan_type"],
    "loan_purpose": ["loan_purpose"],
    "occupancy_type": ["occupancy_type"],
    "derived_dwelling_category": [
        "derived_dwelling_category",
        "dwelling_category",
        "property_type",
    ],
    "lien_status": ["lien_status"],
    "total_units": ["total_units"],
    "income": ["income", "applicant_income_000s"],
    "debt_to_income_ratio": ["debt_to_income_ratio"],
    "combined_loan_to_value_ratio": ["combined_loan_to_value_ratio"],
    "applicant_age": ["applicant_age"],
    "co_applicant_age": ["co_applicant_age"],
    "derived_race": ["derived_race", "applicant_race"],
    "derived_ethnicity": ["derived_ethnicity", "applicant_ethnicity"],
    "derived_sex": ["derived_sex", "applicant_sex"],
    "state_code": ["state_code"],
    "county_code": ["county_code"],
    "census_tract": ["census_tract"],
}

HMDA_ACTION_TAKEN_LABELS = {
    "1": "approved",
    "2": "approved",
    "3": "denied",
    "7": "denied",
    "8": "approved",
}

HMDA_LOAN_TYPE_LABELS = {
    "1": "conventional",
    "2": "fha",
    "3": "va",
    "4": "rhs_or_fsa",
}

HMDA_LOAN_PURPOSE_LABELS = {
    "1": "home_purchase",
    "2": "home_improvement",
    "4": "other_purpose",
    "31": "refinancing",
    "32": "cash_out_refinancing",
}

HMDA_OCCUPANCY_TYPE_LABELS = {
    "1": "primary_residence",
    "2": "second_residence",
    "3": "investment_property",
}

HMDA_LIEN_STATUS_LABELS = {
    "1": "first_lien",
    "2": "subordinate_lien",
    "3": "not_secured_by_lien",
    "4": "not_applicable",
}

NY_COURTS_PRETRIAL_TARGET = "pretrial_release_outcome"
NY_COURTS_PRETRIAL_OUTPUT_COLUMNS = [
    NY_COURTS_PRETRIAL_TARGET,
    "gender",
    "race",
    "ethnicity",
    "age_at_arrest",
    "county_name",
    "court_type",
    "arrest_type",
    "top_charge_at_arraign",
    "top_severity_at_arraign",
    "top_charge_weight_at_arraign",
    "top_charge_at_arraign_violent_felony_ind",
    "arraign_charge_category",
    "representation_at_securing_order",
    "prior_vfo_cnt",
    "prior_nonvfo_cnt",
    "prior_misd_cnt",
]

NY_COURTS_PRETRIAL_INPUT_ALIASES: dict[str, list[str]] = {
    "release_decision_at_arraign": [
        "Release Decision at Arraign",
        "release_decision_at_arraign",
        "larg_rel_decision",
    ],
    "gender": ["Gender", "gender", "arr_sex"],
    "race": ["Race", "race", "arr_re_combi"],
    "ethnicity": ["Ethnicity", "ethnicity"],
    "age_at_arrest": ["Age at Arrest", "age_at_arrest", "arr_age"],
    "county_name": ["County Name", "county_name", "larg_county_name", "arr_county"],
    "court_type": ["Court Type", "court_type", "larg_court_type"],
    "arrest_type": ["Arrest Type", "arrest_type"],
    "top_charge_at_arraign": [
        "Top Charge at Arraign",
        "top_charge_at_arraign",
        "larg_charge",
    ],
    "top_severity_at_arraign": [
        "Top Severity at Arraign",
        "top_severity_at_arraign",
        "larg_class",
    ],
    "top_charge_weight_at_arraign": [
        "Top Charge Weight at Arraign",
        "top_charge_weight_at_arraign",
        "larg_attempt",
    ],
    "top_charge_at_arraign_violent_felony_ind": [
        "Top Charge at Arraign Violent Felony Ind",
        "top_charge_at_arraign_violent_felony_ind",
        "larg_vfo",
    ],
    "arraign_charge_category": [
        "Arraign Charge Category",
        "arraign_charge_category",
        "larg_class",
    ],
    "representation_at_securing_order": [
        "Representation at Securing Order",
        "representation_at_securing_order",
    ],
    "prior_vfo_cnt": ["prior_vfo_cnt", "Prior VFO Count", "pvfo_cnt"],
    "prior_nonvfo_cnt": ["prior_nonvfo_cnt", "Prior Non-VFO Count", "pnonvfo_cnt"],
    "prior_misd_cnt": ["prior_misd_cnt", "Prior Misd Count", "pmisd_cnt"],
}

NY_COURTS_PRETRIAL_RELEASE_LABELS = {
    "ror": "released",
    "release_on_recognizance": "released",
    "nonmonetary_release": "released",
    "nmr": "released",
    "nmr_rus": "released",
    "remanded": "detained",
    "bail_set": "detained",
    "bail-set": "detained",
}

NY_COURTS_PRETRIAL_OPTIONAL_FIELDS = {
    "ethnicity",
    "representation_at_securing_order",
}

IPUMS_CPS_UI_TARGET = "ui_application_outcome"
IPUMS_CPS_UI_OUTPUT_COLUMNS = [
    IPUMS_CPS_UI_TARGET,
    "age",
    "sex",
    "race",
    "hispan",
    "educ",
    "statefip",
    "empstat",
    "whyunemp",
    "classwkr",
]

IPUMS_CPS_UI_INPUT_ALIASES: dict[str, list[str]] = {
    "ucaplast": ["UCAPLAST", "ucaplast"],
    "ucsupflg": ["UCSUPFLG", "ucsupflg"],
    "ucnomain": ["UCNOMAIN", "ucnomain"],
    "ucreclast": ["UCRECLAST", "ucreclast"],
    "age": ["AGE", "age"],
    "sex": ["SEX", "sex"],
    "race": ["RACE", "race"],
    "hispan": ["HISPAN", "hispan"],
    "educ": ["EDUC", "educ"],
    "statefip": ["STATEFIP", "statefip"],
    "empstat": ["EMPSTAT", "empstat"],
    "whyunemp": ["WHYUNEMP", "whyunemp"],
    "classwkr": ["CLASSWKR", "classwkr"],
}

NHAMCS_TARGET = "triage_priority_outcome"
NHAMCS_OUTPUT_COLUMNS = [
    NHAMCS_TARGET,
    "age",
    "sex",
    "raceun",
    "ethun",
    "paytyper",
    "tempf",
    "pulse",
    "bpsys",
    "bpdias",
    "painscale",
    "arrems",
    "ambtransfer",
]

NHAMCS_INPUT_ALIASES: dict[str, list[str]] = {
    "immedr": ["IMMEDR", "immedr"],
    "age": ["AGE", "age"],
    "sex": ["SEX", "sex"],
    "raceun": ["RACEUN", "raceun"],
    "ethun": ["ETHUN", "ethun"],
    "paytyper": ["PAYTYPER", "paytyper"],
    "tempf": ["TEMPF", "tempf"],
    "pulse": ["PULSE", "pulse"],
    "bpsys": ["BPSYS", "bpsys"],
    "bpdias": ["BPDIAS", "bpdias"],
    "painscale": ["PAINSCALE", "painscale"],
    "arrems": ["ARREMS", "arrems"],
    "ambtransfer": ["AMBTRANSFER", "ambtransfer"],
}

FDA_INSPECTIONS_TARGET = "inspection_classification_outcome"
FDA_INSPECTIONS_OUTPUT_COLUMNS = [
    FDA_INSPECTIONS_TARGET,
    "program_area",
    "project_area",
    "state",
    "country",
    "district",
    "fiscal_year",
]

FDA_INSPECTIONS_INPUT_ALIASES: dict[str, list[str]] = {
    "classification": ["Classification", "classification"],
    "program_area": ["Program Area", "program_area"],
    "project_area": ["Project Area", "project_area"],
    "state": ["State", "state"],
    "country": ["Country", "country"],
    "district": ["District", "district"],
    "end_date": ["Inspection End Date", "End Date", "inspection_end_date", "end_date"],
    "fiscal_year": ["Fiscal Year", "fiscal_year"],
}


def prepare_south_german_credit_dataset(
    input_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    raw_rows = _read_south_german_asc(input_path)
    prepared_rows = [_decode_raw_row(row) for row in raw_rows]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = list(RAW_TO_BENCHMARK_HEADERS.values())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(prepared_rows)

    target_distribution: dict[str, int] = {}
    for row in prepared_rows:
        label = str(row["credit_risk"])
        target_distribution[label] = target_distribution.get(label, 0) + 1

    return {
        "dataset_id": "south_german_credit",
        "input_path": repo_relative_path(input_path),
        "output_path": repo_relative_path(output_path),
        "raw_row_count": len(raw_rows),
        "written_row_count": len(prepared_rows),
        "column_count": len(header),
        "columns": header,
        "target_distribution": target_distribution,
        "status": "ok",
    }


def prepare_default_credit_dataset(
    input_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    prepared_rows = _read_default_credit_xls(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = DEFAULT_CREDIT_EXPECTED_COLUMNS
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(prepared_rows)

    target_distribution: dict[str, int] = {}
    for row in prepared_rows:
        label = str(row["default_payment_next_month"])
        target_distribution[label] = target_distribution.get(label, 0) + 1

    return {
        "dataset_id": "default_of_credit_card_clients",
        "input_path": repo_relative_path(input_path),
        "output_path": repo_relative_path(output_path),
        "raw_row_count": len(prepared_rows),
        "written_row_count": len(prepared_rows),
        "column_count": len(header),
        "columns": header,
        "target_distribution": target_distribution,
        "status": "ok",
    }


def prepare_heloc_dataset_from_openml(
    output_path: Path,
    openml_data_id: int = HELOC_OPENML_DATA_ID,
) -> dict[str, Any]:
    try:
        import pandas as pd
        from sklearn.datasets import fetch_openml
    except ImportError as exc:
        raise ImportError(
            "Preparing the HELOC dataset requires pandas and scikit-learn."
        ) from exc

    bundle = fetch_openml(data_id=openml_data_id, as_frame=True, parser="auto")
    features = bundle.data.copy()
    target = bundle.target.copy()

    actual_columns = list(features.columns)
    if actual_columns != HELOC_EXPECTED_FEATURE_COLUMNS:
        raise ValueError(
            "Unexpected HELOC feature header from OpenML. "
            f"Expected {HELOC_EXPECTED_FEATURE_COLUMNS}, got {actual_columns}."
        )

    dataframe = pd.concat(
        [target.rename("RiskPerformance"), features],
        axis=1,
    )

    prepared_rows: list[dict[str, Any]] = []
    missing_counts = {column: 0 for column in HELOC_OUTPUT_COLUMNS}
    for record in dataframe.to_dict(orient="records"):
        prepared_row: dict[str, Any] = {}
        for column in HELOC_OUTPUT_COLUMNS:
            value = _normalize_heloc_value(column, record[column])
            if value is None:
                missing_counts[column] += 1
            prepared_row[column] = value
        prepared_rows.append(prepared_row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HELOC_OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(prepared_rows)

    target_distribution: dict[str, int] = {}
    for row in prepared_rows:
        label = str(row["RiskPerformance"])
        target_distribution[label] = target_distribution.get(label, 0) + 1

    return {
        "dataset_id": "heloc",
        "source": {
            "kind": "openml",
            "data_id": openml_data_id,
        },
        "output_path": repo_relative_path(output_path),
        "raw_row_count": len(prepared_rows),
        "written_row_count": len(prepared_rows),
        "column_count": len(HELOC_OUTPUT_COLUMNS),
        "columns": HELOC_OUTPUT_COLUMNS,
        "target_distribution": target_distribution,
        "missing_value_counts": {
            key: value
            for key, value in missing_counts.items()
            if value > 0
        },
        "status": "ok",
    }


def prepare_give_me_some_credit_dataset_from_openml(
    output_path: Path,
    openml_data_id: int = GIVE_ME_SOME_CREDIT_OPENML_DATA_ID,
) -> dict[str, Any]:
    try:
        import pandas as pd
        from sklearn.datasets import fetch_openml
    except ImportError as exc:
        raise ImportError(
            "Preparing the Give Me Some Credit dataset requires pandas and scikit-learn."
        ) from exc

    bundle = fetch_openml(data_id=openml_data_id, as_frame=True, parser="auto")
    features = bundle.data.copy()
    target = bundle.target.copy()

    actual_columns = list(features.columns)
    if actual_columns != GIVE_ME_SOME_CREDIT_FEATURE_COLUMNS:
        raise ValueError(
            "Unexpected Give Me Some Credit feature header from OpenML. "
            f"Expected {GIVE_ME_SOME_CREDIT_FEATURE_COLUMNS}, got {actual_columns}."
        )

    target_name = getattr(target, "name", None) or GIVE_ME_SOME_CREDIT_TARGET
    if target_name != GIVE_ME_SOME_CREDIT_TARGET:
        raise ValueError(
            "Unexpected Give Me Some Credit target name from OpenML. "
            f"Expected {GIVE_ME_SOME_CREDIT_TARGET}, got {target_name}."
        )

    dataframe = pd.concat(
        [features, target.rename(GIVE_ME_SOME_CREDIT_TARGET)],
        axis=1,
    )

    prepared_rows: list[dict[str, Any]] = []
    missing_counts = {column: 0 for column in GIVE_ME_SOME_CREDIT_OUTPUT_COLUMNS}
    for record in dataframe.to_dict(orient="records"):
        prepared_row: dict[str, Any] = {}
        for column in GIVE_ME_SOME_CREDIT_OUTPUT_COLUMNS:
            value = _normalize_give_me_some_credit_value(column, record[column])
            if value is None:
                missing_counts[column] += 1
            prepared_row[column] = value
        prepared_rows.append(prepared_row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GIVE_ME_SOME_CREDIT_OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(prepared_rows)

    target_distribution: dict[str, int] = {}
    for row in prepared_rows:
        label = str(row[GIVE_ME_SOME_CREDIT_TARGET])
        target_distribution[label] = target_distribution.get(label, 0) + 1

    return {
        "dataset_id": "give_me_some_credit",
        "source": {
            "kind": "openml",
            "data_id": openml_data_id,
        },
        "output_path": repo_relative_path(output_path),
        "raw_row_count": len(prepared_rows),
        "written_row_count": len(prepared_rows),
        "column_count": len(GIVE_ME_SOME_CREDIT_OUTPUT_COLUMNS),
        "columns": GIVE_ME_SOME_CREDIT_OUTPUT_COLUMNS,
        "target_distribution": target_distribution,
        "missing_value_counts": {
            key: value
            for key, value in missing_counts.items()
            if value > 0
        },
        "status": "ok",
    }


def prepare_hmda_dataset(
    input_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Input file has no header row: {input_path}")
        header = list(reader.fieldnames)
        resolved_inputs = _resolve_required_headers(header, HMDA_INPUT_ALIASES)

        prepared_rows: list[dict[str, Any]] = []
        skipped_action_taken: dict[str, int] = {}
        for record in reader:
            outcome = _normalize_hmda_action_taken(record[resolved_inputs["action_taken"]])
            if outcome is None:
                skipped_key = _normalize_hmda_text(record[resolved_inputs["action_taken"]]) or "missing"
                skipped_action_taken[skipped_key] = skipped_action_taken.get(skipped_key, 0) + 1
                continue

            prepared_row = {
                HMDA_TARGET: outcome,
                "loan_amount": _normalize_hmda_numeric_or_text(record[resolved_inputs["loan_amount"]]),
                "loan_type": _normalize_hmda_category(
                    record[resolved_inputs["loan_type"]],
                    HMDA_LOAN_TYPE_LABELS,
                ),
                "loan_purpose": _normalize_hmda_category(
                    record[resolved_inputs["loan_purpose"]],
                    HMDA_LOAN_PURPOSE_LABELS,
                ),
                "occupancy_type": _normalize_hmda_category(
                    record[resolved_inputs["occupancy_type"]],
                    HMDA_OCCUPANCY_TYPE_LABELS,
                ),
                "derived_dwelling_category": _normalize_hmda_text(
                    record[resolved_inputs["derived_dwelling_category"]]
                ),
                "lien_status": _normalize_hmda_category(
                    record[resolved_inputs["lien_status"]],
                    HMDA_LIEN_STATUS_LABELS,
                ),
                "total_units": _normalize_hmda_numeric_or_text(
                    record[resolved_inputs["total_units"]]
                ),
                "income": _normalize_hmda_numeric_or_text(record[resolved_inputs["income"]]),
                "debt_to_income_ratio": _normalize_hmda_numeric_or_text(
                    record[resolved_inputs["debt_to_income_ratio"]]
                ),
                "combined_loan_to_value_ratio": _normalize_hmda_numeric_or_text(
                    record[resolved_inputs["combined_loan_to_value_ratio"]]
                ),
                "applicant_age": _normalize_hmda_age(record[resolved_inputs["applicant_age"]]),
                "co_applicant_age": _normalize_hmda_age(
                    record[resolved_inputs["co_applicant_age"]]
                ),
                "derived_race": _normalize_hmda_text(record[resolved_inputs["derived_race"]]),
                "derived_ethnicity": _normalize_hmda_text(
                    record[resolved_inputs["derived_ethnicity"]]
                ),
                "derived_sex": _normalize_hmda_text(record[resolved_inputs["derived_sex"]]),
                "state_code": _normalize_hmda_geo_code(
                    record[resolved_inputs["state_code"]],
                    width=2,
                ),
                "county_code": _normalize_hmda_geo_code(
                    record[resolved_inputs["county_code"]],
                    width=3,
                ),
                "census_tract": _normalize_hmda_geo_code(
                    record[resolved_inputs["census_tract"]],
                ),
            }
            prepared_rows.append(prepared_row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HMDA_OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(prepared_rows)

    target_distribution: dict[str, int] = {}
    for row in prepared_rows:
        label = str(row[HMDA_TARGET])
        target_distribution[label] = target_distribution.get(label, 0) + 1

    return {
        "dataset_id": "hmda",
        "input_path": repo_relative_path(input_path),
        "output_path": repo_relative_path(output_path),
        "raw_row_count": len(prepared_rows) + sum(skipped_action_taken.values()),
        "written_row_count": len(prepared_rows),
        "column_count": len(HMDA_OUTPUT_COLUMNS),
        "columns": HMDA_OUTPUT_COLUMNS,
        "target_distribution": target_distribution,
        "skipped_action_taken_counts": skipped_action_taken,
        "status": "ok",
    }


def prepare_ny_courts_pretrial_dataset(
    input_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Input file has no header row: {input_path}")
        header = list(reader.fieldnames)
        resolved_inputs = _resolve_required_headers(
            header,
            NY_COURTS_PRETRIAL_INPUT_ALIASES,
            dataset_label="NY Courts pretrial",
            optional_fields=NY_COURTS_PRETRIAL_OPTIONAL_FIELDS,
        )

        prepared_rows: list[dict[str, Any]] = []
        skipped_release_decisions: dict[str, int] = {}
        for record in reader:
            outcome = _normalize_ny_courts_release_decision(
                record[resolved_inputs["release_decision_at_arraign"]]
            )
            if outcome is None:
                skipped_key = (
                    _normalize_hmda_text(
                        record[resolved_inputs["release_decision_at_arraign"]]
                    )
                    or "missing"
                )
                skipped_release_decisions[skipped_key] = (
                    skipped_release_decisions.get(skipped_key, 0) + 1
                )
                continue

            prepared_row = {
                NY_COURTS_PRETRIAL_TARGET: outcome,
                "gender": _normalize_hmda_text(record[resolved_inputs["gender"]]),
                "race": _normalize_hmda_text(record[resolved_inputs["race"]]),
                "ethnicity": _normalize_hmda_text(record[resolved_inputs["ethnicity"]])
                if "ethnicity" in resolved_inputs
                else None,
                "age_at_arrest": _normalize_hmda_numeric_or_text(
                    record[resolved_inputs["age_at_arrest"]]
                ),
                "county_name": _normalize_hmda_text(record[resolved_inputs["county_name"]]),
                "court_type": _normalize_hmda_text(record[resolved_inputs["court_type"]]),
                "arrest_type": _normalize_hmda_text(record[resolved_inputs["arrest_type"]]),
                "top_charge_at_arraign": _normalize_hmda_text(
                    record[resolved_inputs["top_charge_at_arraign"]]
                ),
                "top_severity_at_arraign": _normalize_hmda_text(
                    record[resolved_inputs["top_severity_at_arraign"]]
                ),
                "top_charge_weight_at_arraign": _normalize_hmda_text(
                    record[resolved_inputs["top_charge_weight_at_arraign"]]
                ),
                "top_charge_at_arraign_violent_felony_ind": _normalize_hmda_text(
                    record[resolved_inputs["top_charge_at_arraign_violent_felony_ind"]]
                ),
                "arraign_charge_category": _normalize_hmda_text(
                    record[resolved_inputs["arraign_charge_category"]]
                ),
                "representation_at_securing_order": _normalize_hmda_text(
                    record[resolved_inputs["representation_at_securing_order"]]
                )
                if "representation_at_securing_order" in resolved_inputs
                else None,
                "prior_vfo_cnt": _normalize_hmda_numeric_or_text(
                    record[resolved_inputs["prior_vfo_cnt"]]
                ),
                "prior_nonvfo_cnt": _normalize_hmda_numeric_or_text(
                    record[resolved_inputs["prior_nonvfo_cnt"]]
                ),
                "prior_misd_cnt": _normalize_hmda_numeric_or_text(
                    record[resolved_inputs["prior_misd_cnt"]]
                ),
            }
            prepared_rows.append(prepared_row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=NY_COURTS_PRETRIAL_OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(prepared_rows)

    target_distribution: dict[str, int] = {}
    for row in prepared_rows:
        label = str(row[NY_COURTS_PRETRIAL_TARGET])
        target_distribution[label] = target_distribution.get(label, 0) + 1

    return {
        "dataset_id": "ny_courts_pretrial",
        "input_path": repo_relative_path(input_path),
        "output_path": repo_relative_path(output_path),
        "raw_row_count": len(prepared_rows) + sum(skipped_release_decisions.values()),
        "written_row_count": len(prepared_rows),
        "column_count": len(NY_COURTS_PRETRIAL_OUTPUT_COLUMNS),
        "columns": NY_COURTS_PRETRIAL_OUTPUT_COLUMNS,
        "target_distribution": target_distribution,
        "skipped_release_decision_counts": skipped_release_decisions,
        "status": "ok",
    }


def prepare_ipums_cps_ui_dataset(
    input_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Input file has no header row: {input_path}")
        header = list(reader.fieldnames)
        resolved_inputs = _resolve_required_headers(
            header,
            {
                key: value
                for key, value in IPUMS_CPS_UI_INPUT_ALIASES.items()
                if key not in {"ucaplast", "ucsupflg", "ucnomain", "hispan"}
            },
            dataset_label="IPUMS CPS UI",
        )
        header_lookup = {name.strip(): name for name in header}
        ucaplast_column = next(
            (
                header_lookup[alias]
                for alias in IPUMS_CPS_UI_INPUT_ALIASES["ucaplast"]
                if alias in header_lookup
            ),
            None,
        )
        ucsupflg_column = next(
            (
                header_lookup[alias]
                for alias in IPUMS_CPS_UI_INPUT_ALIASES["ucsupflg"]
                if alias in header_lookup
            ),
            None,
        )
        ucnomain_column = next(
            (
                header_lookup[alias]
                for alias in IPUMS_CPS_UI_INPUT_ALIASES["ucnomain"]
                if alias in header_lookup
            ),
            None,
        )
        hispan_column = next(
            (
                header_lookup[alias]
                for alias in IPUMS_CPS_UI_INPUT_ALIASES["hispan"]
                if alias in header_lookup
            ),
            None,
        )
        if ucaplast_column is None and (ucsupflg_column is None or ucnomain_column is None):
            raise ValueError(
                "IPUMS CPS UI input is missing both UCAPLAST and the UCSUPFLG+UCNOMAIN pair needed "
                "to derive application status."
            )

        prepared_rows: list[dict[str, Any]] = []
        skipped_application_status: dict[str, int] = {}
        skipped_receipt_status: dict[str, int] = {}
        for record in reader:
            if ucaplast_column is not None:
                applied = _normalize_ipums_ui_yes_no(record[ucaplast_column])
                applied_raw_value = record[ucaplast_column]
            else:
                applied = _derive_ipums_ui_application_status(
                    ucsupflg_value=record[ucsupflg_column],
                    ucnomain_value=record[ucnomain_column],
                )
                applied_raw_value = record[ucnomain_column]
            if applied != "yes":
                skipped_key = _normalize_hmda_text(applied_raw_value) or "missing"
                skipped_application_status[skipped_key] = (
                    skipped_application_status.get(skipped_key, 0) + 1
                )
                continue

            outcome = _normalize_ipums_ui_receipt(record[resolved_inputs["ucreclast"]])
            if outcome is None:
                skipped_key = (
                    _normalize_hmda_text(record[resolved_inputs["ucreclast"]]) or "missing"
                )
                skipped_receipt_status[skipped_key] = (
                    skipped_receipt_status.get(skipped_key, 0) + 1
                )
                continue

            prepared_rows.append(
                {
                    IPUMS_CPS_UI_TARGET: outcome,
                    "age": _normalize_hmda_numeric_or_text(record[resolved_inputs["age"]]),
                    "sex": _normalize_ipums_sex(record[resolved_inputs["sex"]]),
                    "race": _normalize_hmda_text(record[resolved_inputs["race"]]),
                    "hispan": (
                        _normalize_hmda_text(record[hispan_column])
                        if hispan_column is not None
                        else "not_in_extract"
                    ),
                    "educ": _normalize_hmda_text(record[resolved_inputs["educ"]]),
                    "statefip": _normalize_hmda_numeric_or_text(
                        record[resolved_inputs["statefip"]]
                    ),
                    "empstat": _normalize_hmda_text(record[resolved_inputs["empstat"]]),
                    "whyunemp": _normalize_hmda_text(record[resolved_inputs["whyunemp"]]),
                    "classwkr": _normalize_hmda_text(record[resolved_inputs["classwkr"]]),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=IPUMS_CPS_UI_OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(prepared_rows)

    target_distribution: dict[str, int] = {}
    for row in prepared_rows:
        label = str(row[IPUMS_CPS_UI_TARGET])
        target_distribution[label] = target_distribution.get(label, 0) + 1

    return {
        "dataset_id": "ipums_cps_ui",
        "input_path": repo_relative_path(input_path),
        "output_path": repo_relative_path(output_path),
        "raw_row_count": (
            len(prepared_rows)
            + sum(skipped_application_status.values())
            + sum(skipped_receipt_status.values())
        ),
        "written_row_count": len(prepared_rows),
        "column_count": len(IPUMS_CPS_UI_OUTPUT_COLUMNS),
        "columns": IPUMS_CPS_UI_OUTPUT_COLUMNS,
        "target_distribution": target_distribution,
        "skipped_application_status_counts": skipped_application_status,
        "skipped_receipt_status_counts": skipped_receipt_status,
        "status": "ok",
    }


def prepare_nhamcs_dataset(
    input_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Input file has no header row: {input_path}")
        header = list(reader.fieldnames)
        resolved_inputs = _resolve_required_headers(
            header,
            NHAMCS_INPUT_ALIASES,
            dataset_label="NHAMCS",
        )

        prepared_rows: list[dict[str, Any]] = []
        skipped_triage_counts: dict[str, int] = {}
        for record in reader:
            outcome = _normalize_nhamcs_triage(record[resolved_inputs["immedr"]])
            if outcome is None:
                skipped_key = _normalize_hmda_text(record[resolved_inputs["immedr"]]) or "missing"
                skipped_triage_counts[skipped_key] = skipped_triage_counts.get(skipped_key, 0) + 1
                continue

            prepared_rows.append(
                {
                    NHAMCS_TARGET: outcome,
                    "age": _normalize_hmda_numeric_or_text(record[resolved_inputs["age"]]),
                    "sex": _normalize_nhamcs_sex(record[resolved_inputs["sex"]]),
                    "raceun": _normalize_nhamcs_race(record[resolved_inputs["raceun"]]),
                    "ethun": _normalize_nhamcs_ethnicity(record[resolved_inputs["ethun"]]),
                    "paytyper": _normalize_nhamcs_paytype(record[resolved_inputs["paytyper"]]),
                    "tempf": _normalize_nhamcs_tempf(record[resolved_inputs["tempf"]]),
                    "pulse": _normalize_nhamcs_numeric(record[resolved_inputs["pulse"]]),
                    "bpsys": _normalize_nhamcs_numeric(record[resolved_inputs["bpsys"]]),
                    "bpdias": _normalize_nhamcs_numeric(record[resolved_inputs["bpdias"]]),
                    "painscale": _normalize_nhamcs_numeric(
                        record[resolved_inputs["painscale"]]
                    ),
                    "arrems": _normalize_nhamcs_yes_no(record[resolved_inputs["arrems"]]),
                    "ambtransfer": _normalize_nhamcs_yes_no(
                        record[resolved_inputs["ambtransfer"]]
                    ),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=NHAMCS_OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(prepared_rows)

    target_distribution: dict[str, int] = {}
    for row in prepared_rows:
        label = str(row[NHAMCS_TARGET])
        target_distribution[label] = target_distribution.get(label, 0) + 1

    return {
        "dataset_id": "nhamcs",
        "input_path": repo_relative_path(input_path),
        "output_path": repo_relative_path(output_path),
        "raw_row_count": len(prepared_rows) + sum(skipped_triage_counts.values()),
        "written_row_count": len(prepared_rows),
        "column_count": len(NHAMCS_OUTPUT_COLUMNS),
        "columns": NHAMCS_OUTPUT_COLUMNS,
        "target_distribution": target_distribution,
        "skipped_triage_counts": skipped_triage_counts,
        "status": "ok",
    }


def prepare_fda_inspections_dataset(
    input_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Input file has no header row: {input_path}")
        header = list(reader.fieldnames)
        header_lookup = {name.strip(): name for name in header}
        resolved_inputs = _resolve_required_headers(
            header,
            {
                key: value
                for key, value in FDA_INSPECTIONS_INPUT_ALIASES.items()
                if key not in {"end_date", "fiscal_year"}
            },
            dataset_label="FDA inspections",
        )
        end_date_column = next(
            (
                header_lookup[alias]
                for alias in FDA_INSPECTIONS_INPUT_ALIASES["end_date"]
                if alias in header_lookup
            ),
            None,
        )
        fiscal_year_column = next(
            (
                header_lookup[alias]
                for alias in FDA_INSPECTIONS_INPUT_ALIASES["fiscal_year"]
                if alias in header_lookup
            ),
            None,
        )
        if end_date_column is None and fiscal_year_column is None:
            raise ValueError(
                "FDA inspections input is missing both a fiscal-year column and an "
                "inspection end date column."
            )
        if end_date_column is not None:
            resolved_inputs["end_date"] = end_date_column
        if fiscal_year_column is not None:
            resolved_inputs["fiscal_year"] = fiscal_year_column

        prepared_rows: list[dict[str, Any]] = []
        skipped_classification_counts: dict[str, int] = {}
        for record in reader:
            outcome = _normalize_fda_classification(record[resolved_inputs["classification"]])
            if outcome is None:
                skipped_key = (
                    _normalize_hmda_text(record[resolved_inputs["classification"]]) or "missing"
                )
                skipped_classification_counts[skipped_key] = (
                    skipped_classification_counts.get(skipped_key, 0) + 1
                )
                continue

            fiscal_year_value = (
                _normalize_hmda_numeric_or_text(record[resolved_inputs["fiscal_year"]])
                if "fiscal_year" in resolved_inputs
                else None
            )
            if fiscal_year_value is None:
                fiscal_year_value = _extract_year_from_text(record[resolved_inputs["end_date"]])

            prepared_rows.append(
                {
                    FDA_INSPECTIONS_TARGET: outcome,
                    "program_area": _normalize_hmda_text(record[resolved_inputs["program_area"]]),
                    "project_area": _normalize_hmda_text(record[resolved_inputs["project_area"]]),
                    "state": _normalize_hmda_text(record[resolved_inputs["state"]]),
                    "country": _normalize_hmda_text(record[resolved_inputs["country"]]),
                    "district": _normalize_hmda_text(record[resolved_inputs["district"]]),
                    "fiscal_year": fiscal_year_value,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FDA_INSPECTIONS_OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(prepared_rows)

    target_distribution: dict[str, int] = {}
    for row in prepared_rows:
        label = str(row[FDA_INSPECTIONS_TARGET])
        target_distribution[label] = target_distribution.get(label, 0) + 1

    return {
        "dataset_id": "fda_inspections",
        "input_path": repo_relative_path(input_path),
        "output_path": repo_relative_path(output_path),
        "raw_row_count": len(prepared_rows) + sum(skipped_classification_counts.values()),
        "written_row_count": len(prepared_rows),
        "column_count": len(FDA_INSPECTIONS_OUTPUT_COLUMNS),
        "columns": FDA_INSPECTIONS_OUTPUT_COLUMNS,
        "target_distribution": target_distribution,
        "skipped_classification_counts": skipped_classification_counts,
        "status": "ok",
    }


def _read_south_german_asc(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    if not lines:
        raise ValueError(f"Input file is empty: {path}")

    raw_header = lines[0].split()
    expected_header = list(RAW_TO_BENCHMARK_HEADERS.keys())
    if raw_header != expected_header:
        raise ValueError(
            "Unexpected South German raw header. "
            f"Expected {expected_header}, got {raw_header}."
        )

    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        values = line.split()
        if len(values) != len(raw_header):
            raise ValueError(
                f"Malformed South German row with {len(values)} fields; "
                f"expected {len(raw_header)}."
            )
        rows.append(dict(zip(raw_header, values, strict=True)))
    return rows


def _read_default_credit_xls(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "Preparing the Default of Credit Card Clients dataset requires pandas "
            "with an .xls reader. Install the benchmark empirical extras and xlrd."
        ) from exc

    dataframe = pd.read_excel(
        path,
        sheet_name=DEFAULT_CREDIT_SHEET_NAME,
        header=DEFAULT_CREDIT_HEADER_ROW_INDEX,
        engine="xlrd",
    )
    normalized_columns = []
    for column in dataframe.columns:
        label = str(column).strip()
        normalized_columns.append(DEFAULT_CREDIT_TARGET_RENAME.get(label, label))
    dataframe.columns = normalized_columns

    actual_columns = list(dataframe.columns)
    if actual_columns != DEFAULT_CREDIT_EXPECTED_COLUMNS:
        raise ValueError(
            "Unexpected Default Credit header. "
            f"Expected {DEFAULT_CREDIT_EXPECTED_COLUMNS}, got {actual_columns}."
        )

    prepared_rows: list[dict[str, Any]] = []
    for record in dataframe.to_dict(orient="records"):
        prepared_row: dict[str, Any] = {}
        for column in DEFAULT_CREDIT_EXPECTED_COLUMNS:
            prepared_row[column] = _coerce_integral_value(record[column], column_name=column)
        prepared_rows.append(prepared_row)
    return prepared_rows


def _decode_raw_row(raw_row: dict[str, str]) -> dict[str, Any]:
    decoded: dict[str, Any] = {}
    for raw_name, benchmark_name in RAW_TO_BENCHMARK_HEADERS.items():
        raw_value = raw_row[raw_name]
        if benchmark_name in NUMERIC_FIELDS:
            decoded[benchmark_name] = int(raw_value)
            continue

        if benchmark_name not in CATEGORICAL_LABELS:
            raise KeyError(f"Missing categorical mapping for '{benchmark_name}'.")

        code = int(raw_value)
        labels = CATEGORICAL_LABELS[benchmark_name]
        if code not in labels:
            raise KeyError(
                f"Unknown categorical code {code} for feature '{benchmark_name}'."
            )
        decoded[benchmark_name] = labels[code]
    return decoded


def _coerce_integral_value(value: Any, column_name: str) -> int:
    if value is None:
        raise ValueError(f"Encountered missing value in column '{column_name}'.")
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != value:
            raise ValueError(f"Encountered NaN value in column '{column_name}'.")
        if not value.is_integer():
            raise ValueError(
                f"Expected integral value in column '{column_name}', got {value}."
            )
        return int(value)

    text = str(value).strip()
    if text == "":
        raise ValueError(f"Encountered empty value in column '{column_name}'.")
    return int(text)


def _normalize_heloc_value(column_name: str, value: Any) -> Any:
    if _is_missing_like(value):
        return None
    if column_name == "RiskPerformance":
        return str(value).strip()
    if column_name in HELOC_CATEGORICAL_CODE_COLUMNS:
        return f"code_{int(float(value))}"
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return float(value)

    text = str(value).strip()
    if text == "":
        return None
    try:
        numeric = float(text)
    except ValueError:
        return text
    if numeric.is_integer():
        return int(numeric)
    return numeric


def _normalize_give_me_some_credit_value(column_name: str, value: Any) -> Any:
    if _is_missing_like(value):
        return None
    if column_name == GIVE_ME_SOME_CREDIT_TARGET:
        return str(value).strip()
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return float(value)

    text = str(value).strip()
    if text == "":
        return None
    try:
        numeric = float(text)
    except ValueError:
        return text
    if numeric.is_integer():
        return int(numeric)
    return numeric


def _resolve_required_headers(
    header: list[str],
    aliases: dict[str, list[str]],
    dataset_label: str = "Input",
    optional_fields: set[str] | None = None,
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    header_lookup = {name.strip(): name for name in header}
    missing: list[str] = []
    optional_fields = optional_fields or set()
    for canonical_name, alias_options in aliases.items():
        actual_name = next(
            (header_lookup[alias] for alias in alias_options if alias in header_lookup),
            None,
        )
        if actual_name is None:
            if canonical_name in optional_fields:
                continue
            missing.append(f"{canonical_name} ({', '.join(alias_options)})")
        else:
            resolved[canonical_name] = actual_name
    if missing:
        raise ValueError(
            f"{dataset_label} input is missing required columns: " + "; ".join(missing)
        )
    return resolved


def _normalize_ny_courts_release_decision(value: Any) -> str | None:
    normalized = _normalize_hmda_text(value)
    if normalized is None:
        return None
    return NY_COURTS_PRETRIAL_RELEASE_LABELS.get(normalized)


def _normalize_ipums_ui_yes_no(value: Any) -> str | None:
    normalized = _normalize_hmda_text(value)
    if normalized is None:
        return None
    if normalized in {"yes", "applied", "received", "2"}:
        return "yes"
    if normalized in {"no", "did_not_apply", "did_not_receive", "1"}:
        return "no"
    if "yes" in normalized:
        return "yes"
    if "no" in normalized:
        return "no"
    return None


def _normalize_ipums_ucsupp_interview_flag(value: Any) -> str | None:
    normalized = _normalize_hmda_text(value)
    if normalized is None:
        return None
    if normalized in {"2", "02", "interviewed"}:
        return "interviewed"
    if normalized in {"1", "01", "not_interviewed"}:
        return "not_interviewed"
    if normalized in {"99", "niu", "not_in_universe"}:
        return "niu"
    return normalized


def _derive_ipums_ui_application_status(
    ucsupflg_value: Any,
    ucnomain_value: Any,
) -> str | None:
    interview_status = _normalize_ipums_ucsupp_interview_flag(ucsupflg_value)
    if interview_status != "interviewed":
        return None
    no_apply_reason = _normalize_hmda_text(ucnomain_value)
    if no_apply_reason in {None, "", "niu", "not_in_universe"}:
        return "yes"
    return "no"


def _normalize_ipums_ui_receipt(value: Any) -> str | None:
    normalized = _normalize_ipums_ui_yes_no(value)
    if normalized == "yes":
        return "received"
    if normalized == "no":
        return "not_received"
    return None


def _normalize_ipums_sex(value: Any) -> str | None:
    normalized = _normalize_hmda_text(value)
    if normalized is None:
        return None
    if normalized == "1":
        return "male"
    if normalized == "2":
        return "female"
    return normalized


def _normalize_nhamcs_triage(value: Any) -> str | None:
    normalized = _normalize_hmda_text(value)
    if normalized is None:
        return None
    if normalized in {"1", "2", "3", "immediate", "emergent", "urgent"}:
        return "high_priority"
    if normalized in {"4", "5", "semi_urgent", "nonurgent"}:
        return "low_priority"
    return None


def _normalize_nhamcs_sex(value: Any) -> str | None:
    normalized = _normalize_hmda_text(value)
    if normalized is None:
        return None
    if normalized in {"_8", "_9"}:
        return None
    if normalized == "1":
        return "female"
    if normalized == "2":
        return "male"
    return normalized


def _normalize_nhamcs_race(value: Any) -> str | None:
    normalized = _normalize_hmda_text(value)
    mapping = {
        "1": "white",
        "2": "black_african_american",
        "3": "asian",
        "4": "native_hawaiian_or_other_pacific_islander",
        "5": "american_indian_or_alaska_native",
        "6": "multiple_races",
    }
    if normalized is None:
        return None
    if normalized in {"_8", "_9"}:
        return None
    return mapping.get(normalized, normalized)


def _normalize_nhamcs_ethnicity(value: Any) -> str | None:
    normalized = _normalize_hmda_text(value)
    mapping = {
        "1": "hispanic_or_latino",
        "2": "not_hispanic_or_latino",
    }
    if normalized is None:
        return None
    if normalized in {"_8", "_9"}:
        return None
    return mapping.get(normalized, normalized)


def _normalize_nhamcs_paytype(value: Any) -> str | None:
    normalized = _normalize_hmda_text(value)
    mapping = {
        "1": "private_insurance",
        "2": "medicare",
        "3": "medicaid_or_chip_or_state_program",
        "4": "workers_compensation",
        "5": "self_pay",
        "6": "no_charge_or_charity",
        "7": "other",
    }
    if normalized is None:
        return None
    if normalized in {"_8", "_9"}:
        return None
    return mapping.get(normalized, normalized)


def _normalize_nhamcs_yes_no(value: Any) -> str | None:
    normalized = _normalize_hmda_text(value)
    if normalized is None:
        return None
    if normalized in {"1", "yes"}:
        return "yes"
    if normalized in {"2", "no"}:
        return "no"
    if normalized in {"_7", "not_applicable"}:
        return "not_applicable"
    return None


def _normalize_nhamcs_numeric(value: Any) -> Any:
    normalized = _normalize_hmda_numeric_or_text(value)
    if isinstance(normalized, (int, float)) and normalized < 0:
        return None
    return normalized


def _normalize_nhamcs_tempf(value: Any) -> Any:
    normalized = _normalize_nhamcs_numeric(value)
    if isinstance(normalized, int) and 896 <= normalized <= 1056:
        return normalized / 10
    return normalized


def _normalize_fda_classification(value: Any) -> str | None:
    normalized = _normalize_hmda_text(value)
    if normalized is None:
        return None
    if normalized in {
        "nai",
        "no_action_indicated",
        "no_action_indicated_(nai)",
    }:
        return "no_action_indicated"
    if normalized in {
        "vai",
        "oai",
        "voluntary_action_indicated",
        "official_action_indicated",
        "voluntary_action_indicated_(vai)",
        "official_action_indicated_(oai)",
    }:
        return "action_indicated"
    return None


def _extract_year_from_text(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value)
    for token in text.replace("/", "-").split("-"):
        token = token.strip()
        if len(token) == 4 and token.isdigit():
            return int(token)
    return None


def _normalize_hmda_action_taken(value: Any) -> str | None:
    normalized = _normalize_hmda_text(value)
    if normalized is None:
        return None
    return HMDA_ACTION_TAKEN_LABELS.get(normalized)


def _normalize_hmda_category(value: Any, labels: dict[str, str]) -> Any:
    normalized = _normalize_hmda_text(value)
    if normalized is None:
        return None
    return labels.get(normalized, normalized)


def _normalize_hmda_numeric_or_text(value: Any) -> Any:
    if _is_missing_like(value):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return float(value)

    text = str(value).strip()
    if text == "":
        return None
    collapsed = text.replace(",", "")
    try:
        numeric = float(collapsed)
    except ValueError:
        return _normalize_hmda_text(text)
    if numeric.is_integer():
        return int(numeric)
    return numeric


def _normalize_hmda_age(value: Any) -> Any:
    normalized = _normalize_hmda_text(value)
    if normalized is None:
        return None
    age_map = {
        "8888": "not_applicable",
        "9999": "no_co_applicant",
    }
    return age_map.get(normalized, normalized)


def _normalize_hmda_geo_code(value: Any, width: int | None = None) -> Any:
    normalized = _normalize_hmda_text(value)
    if normalized is None:
        return None
    compact = normalized.replace(".", "")
    if compact.isdigit() and width is not None:
        return compact.zfill(width)
    return compact


def _normalize_hmda_text(value: Any) -> str | None:
    if _is_missing_like(value):
        return None
    text = str(value).strip()
    if text == "":
        return None
    lowered = text.lower()
    lowered = lowered.replace("%", " percent")
    lowered = lowered.replace("/", " ")
    lowered = lowered.replace("-", "_")
    lowered = " ".join(lowered.split())
    return lowered.replace(" ", "_")


def _is_missing_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return value != value
    text = str(value).strip().lower()
    return text in {"", "nan", "na", "n/a", "null", "none", "?"}
