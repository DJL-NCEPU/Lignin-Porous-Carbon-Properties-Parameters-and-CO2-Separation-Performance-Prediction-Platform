# -*- coding: utf-8 -*-
"""
Streamlit app for two-stage prediction of N-doped porous carbons.

Workflow:
4 user-entered preparation parameters
    -> each preparation-property model automatically uses its own saved input subset/order
    -> preparation-property models
    -> predicted physicochemical properties
    -> conversion of N/O fractions to absolute at.%
    -> adsorption/separation models
    -> CO2/H2O/N2 isotherms + CO2/H2O and CO2/N2 separation curves

Run locally:
    streamlit run streamlit_app.py
"""

import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# 0. PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AI-assisted Design of N-Doped Porous Carbons",
    page_icon="🧪",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* ===== Global layout ===== */
    .stApp {
        background: #eef8ef;
    }

    .block-container {
        padding-top: 0.28rem;
        padding-bottom: 0.35rem;
        padding-left: 0.55rem;
        padding-right: 0.55rem;
        max-width: 100%;
    }

    [data-testid="stVerticalBlock"] {
        gap: 0.20rem;
    }

    [data-testid="stHorizontalBlock"] {
        gap: 0.45rem;
    }

    h1, h2, h3, p {
        margin-top: 0 !important;
    }


    /* ========================================================
       Tabs
       ======================================================== */
    div[data-baseweb="tab-list"] {
        gap: 0.45rem;
        margin-bottom: 0.10rem;
    }

    button[data-baseweb="tab"] {
        padding: 0.12rem 0.30rem 0.20rem 0.30rem;
        font-size: 0.72rem;
    }


    /* ========================================================
       Main title
       ======================================================== */
    .platform-title {
        width: 100%;

        background: linear-gradient(
            90deg,
            #079638 0%,
            #55bf78 100%
        );

        color: white;

        font-weight: 800;

        text-align: center;

        font-size: 1.24rem;

        line-height: 1.18;

        padding: 0.48rem 0.70rem;

        border-radius: 0.95rem;

        margin: 0.05rem 0 0.35rem 0;

        letter-spacing: 0.01em;
    }


    .step-bar {
        width: 100%;

        background: rgba(
            255,
            255,
            255,
            0.80
        );

        border: 1px solid rgba(
            80,
            140,
            95,
            0.08
        );

        color: #165b3d;

        text-align: center;

        font-size: 0.90rem;

        padding: 0.42rem 0.55rem;

        border-radius: 1rem;

        margin-bottom: 0.62rem;
    }


    .section-chip {
        display: inline-block;

        background: #ededed;

        color: #111;

        border-radius: 0.45rem;

        padding: 0.34rem 0.72rem;

        font-weight: 750;

        font-size: 0.92rem;

        margin-bottom: 0.22rem;
    }


    .section-heading {
        display: block;

        position: relative;

        z-index: 10;

        color: #111;

        font-size: 0.90rem;

        font-weight: 750;

        text-align: center;

        line-height: 1.20;

        margin: 0.16rem 0 0.34rem 0;

        padding-top: 0.04rem;

        min-height: 1.15rem;
    }


    /* ========================================================
       Inputs
       ======================================================== */
    [data-testid="stNumberInput"] label {
        font-size: 0.76rem !important;

        font-weight: 700 !important;

        color: #171717 !important;
    }


    [data-testid="stNumberInput"] input {
        min-height: 2.0rem !important;

        font-size: 0.83rem !important;

        background: #f3f5f7 !important;
    }


    /* ========================================================
       Buttons
       ======================================================== */
    .stButton > button {
        width: 100%;

        border: none !important;

        border-radius: 0.32rem !important;

        background: #344b63 !important;

        color: white !important;

        font-size: 0.78rem !important;

        font-weight: 700 !important;

        min-height: 2.15rem !important;

        box-shadow: none !important;
    }


    .stButton > button:hover {
        background: #2a3f54 !important;

        color: white !important;
    }


    /* ========================================================
       Applicability domain
       ======================================================== */
    .good-box,
    .warn-box {
        padding: 0.20rem 0.40rem;

        border-radius: 0.35rem;

        font-size: 0.64rem;

        line-height: 1.05;

        margin-top: 0.08rem;
    }


    .good-box {
        background: rgba(
            255,
            255,
            255,
            0.65
        );

        border: 1px solid rgba(
            30,
            160,
            90,
            0.23
        );

        color: #245c3c;
    }


    .warn-box {
        background: rgba(
            255,
            245,
            220,
            0.72
        );

        border: 1px solid rgba(
            230,
            150,
            20,
            0.28
        );
    }


    /* ========================================================
       Property tables
       ======================================================== */
    .property-card {
        display: block;

        width: 100%;

        overflow: hidden;

        border: none;

        border-radius: 0.85rem;

        background: rgba(
            255,
            255,
            255,
            0.88
        );

        min-height: 0;

        height: auto;

        margin: 0;

        padding: 0;
    }


    .compact-table {
        width: 100%;

        border-collapse: collapse;

        table-layout: fixed;

        margin: 0;

        padding: 0;

        height: auto;
    }


    .compact-table th {
        background: #edf1f2;

        font-weight: 500;

        color: #747b82;

        text-align: center;
    }


    .compact-table th,
    .compact-table td {
        padding: 4px 7px;

        border-bottom: 1px solid rgba(
            140,
            150,
            155,
            0.22
        );

        border-right: 1px solid rgba(
            140,
            150,
            155,
            0.14
        );

        font-size: 10.5px;

        line-height: 1.05;

        white-space: normal;

        vertical-align: middle;
    }


    .compact-table td:first-child {
        text-align: left;
    }


    .compact-table td:last-child {
        text-align: center;
    }


    .compact-table th:last-child,
    .compact-table td:last-child {
        border-right: none;
    }


    .compact-table tr:last-child td {
        border-bottom: none;
    }


    .property-card table,
    .property-card tbody {
        margin-bottom: 0 !important;
    }


    /* ========================================================
       Separator
       ======================================================== */
    .soft-rule {
        border: 0;

        border-top: 1px solid rgba(
            72,
            114,
            84,
            0.22
        );

        margin: 0.42rem 0 0.35rem 0;
    }


    /* ========================================================
       Expander
       ======================================================== */
    [data-testid="stExpander"] {
        border-radius: 0.30rem !important;

        background: rgba(
            255,
            255,
            255,
            0.40
        );
    }


    /* ========================================================
       Plotly - prevent titles from being clipped
       ======================================================== */
    [data-testid="stPlotlyChart"] {
        margin-top: 0.10rem;

        overflow: visible !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 1. MODEL ROOT
# ============================================================
# Deployment version:
# streamlit_app.py and models/ must be in the same folder.
MODEL_ROOT = Path("models")


# ============================================================
# 2. MODEL FILE MAP
# ============================================================
MODEL_FILES = {

    # ========================================================
    # Material properties -> adsorption / separation
    # ========================================================
    "CO2": {
        "model": Path("CO2/CO2.pkl"),
        "scaler": Path("CO2/scaler-CO2.pkl"),
    },

    "H2O": {
        "model": Path("H2O/H2O.pkl"),
        "scaler": Path("H2O/scaler-H2O.pkl"),
    },

    "N2": {
        "model": Path("N2/N2.pkl"),
        "scaler": Path("N2/scaler-N2.pkl"),
    },

    "CO2_H2O": {
        "model": Path("CO2-H2O/CO2-H2O.pkl"),
        "scaler": Path("CO2-H2O/scaler-CO2-H2O.pkl"),
    },

    "CO2_N2": {
        "model": Path("CO2-N2/CO2-N2.pkl"),
        "scaler": Path("CO2-N2/scaler-CO2-N2.pkl"),
    },


    # ========================================================
    # Preparation parameters -> physicochemical properties
    # ========================================================
    "SBET": {
        "model": Path(
            "制备/SBET/model-SBET.pkl"
        ),
        "scaler": Path(
            "制备/SBET/scaler-SBET.pkl"
        ),
    },

    "Vt": {
        "model": Path(
            "制备/Vt/model-Vt.pkl"
        ),
        "scaler": Path(
            "制备/Vt/scaler-Vt.pkl"
        ),
    },

    "Vmes": {
        "model": Path(
            "制备/Vmes/model-Vmes.pkl"
        ),
        "scaler": Path(
            "制备/Vmes/scaler-Vmes.pkl"
        ),
    },

    "Vm": {
        "model": Path(
            "制备/Vm/model-Vm.pkl"
        ),
        "scaler": Path(
            "制备/Vm/scaler--Vm.pkl"
        ),
    },

    "Vn": {
        "model": Path(
            "制备/Vn/model-Vn.pkl"
        ),
        "scaler": Path(
            "制备/Vn/scaler-Vn.pkl"
        ),
    },

    "Total N": {
        "model": Path(
            "制备/Total N/model-Total N.pkl"
        ),
        "scaler": Path(
            "制备/Total N/scaler-Total N.pkl"
        ),
    },

    "Pyridinic-N fraction": {
        "model": Path(
            "制备/Pyridinic-N/model-Pyridinic-N.pkl"
        ),
        "scaler": Path(
            "制备/Pyridinic-N/scaler-Pyridinic-N.pkl"
        ),
    },

    "Pyrrolic-N/Pyridonic-N fraction": {
        "model": Path(
            "制备/Pyrrolic-N_Pyridonic-N/"
            "model-Pyrrolic-N _ Pyridonic-N.pkl"
        ),
        "scaler": Path(
            "制备/Pyrrolic-N_Pyridonic-N/"
            "scaler-Pyrrolic-N _ Pyridonic-N.pkl"
        ),
    },

    "Graphitic-N fraction": {
        "model": Path(
            "制备/Graphitic-N/model-Graphitic-N.pkl"
        ),
        "scaler": Path(
            "制备/Graphitic-N/scaler-Graphitic-N.pkl"
        ),
    },

    "Oxidized-N fraction": {
        "model": Path(
            "制备/Oxidized-N/model-Oxidized-N.pkl"
        ),
        "scaler": Path(
            "制备/Oxidized-N/scaler-Oxidized-N.pkl"
        ),
    },

    "Total O": {
        "model": Path(
            "制备/Total O/model-Total O.pkl"
        ),
        "scaler": Path(
            "制备/Total O/scaler-Total O.pkl"
        ),
    },

    "C=O fraction": {
        "model": Path(
            "制备/C=O/model-C=O.pkl"
        ),
        "scaler": Path(
            "制备/C=O/scaler-C=O.pkl"
        ),
    },

    "C-O/C-OH fraction": {
        "model": Path(
            "制备/C-O_C-OH/model-C-O-C-OH.pkl"
        ),
        "scaler": Path(
            "制备/C-O_C-OH/scaler-C-O-C-OH.pkl"
        ),
    },

    "O-C=O/-COOH fraction": {
        "model": Path(
            "制备/O-C=O_-COOH/"
            "model-O-C=O_-COOH.pkl"
        ),
        "scaler": Path(
            "制备/O-C=O_-COOH/"
            "scaler-O-C=O_-COOH.pkl"
        ),
    },

    "WCA": {
        "model": Path(
            "制备/WCA/model-WCA.pkl"
        ),
        "scaler": Path(
            "制备/WCA/scaler-WCA.pkl"
        ),
    },
}


# ============================================================
# 3. FEATURE DEFINITIONS
# ============================================================
PREP_FEATURES = [
    "TKOH",
    "RKOH/L",
    "TN",
    "RN/AC",
]


PREP_TARGETS = [
    "SBET",
    "Vt",
    "Vmes",
    "Vm",
    "Vn",
    "Total N",
    "Pyridinic-N fraction",
    "Pyrrolic-N/Pyridonic-N fraction",
    "Graphitic-N fraction",
    "Oxidized-N fraction",
    "Total O",
    "C=O fraction",
    "C-O/C-OH fraction",
    "O-C=O/-COOH fraction",
    "WCA",
]


# Only use this if a scaler has no feature_names_in_.
PREP_FEATURE_OVERRIDE = {
}


PERFORMANCE_BASE_FEATURES = [
    "SBET",
    "Vt",
    "Vmes",
    "Vm",
    "Vn",
    "Pyridinic-N",
    "Pyrrolic-N/Pyridonic-N",
    "Graphitic-N",
    "Oxidized-N",
    "C=O",
    "C-O/C-OH",
    "O-C=O/-COOH",
    "WCA",
]


PREP_DOMAIN = {
    "TKOH": (500.0, 800.0),
    "RKOH/L": (0.5, 2.0),
    "TN": (500.0, 900.0),
    "RN/AC": (0.0, 3.0),
}


PRESSURE_GRID = np.arange(
    0.0,
    101.0,
    10.0,
)

RH_GRID = np.arange(
    0.0,
    101.0,
    10.0,
)

SEP_RH_GRID = np.arange(
    10.0,
    101.0,
    10.0,
)

CO2_PARTIAL_GRID = np.arange(
    10.0,
    100.0,
    10.0,
)


# ============================================================
# 4. LOADING / PREDICTION HELPERS
# ============================================================
def _load_pickle_or_joblib(path: Path):

    try:
        return joblib.load(path)

    except Exception as e_joblib:

        try:
            with open(path, "rb") as f:
                return pickle.load(f)

        except Exception as e_pickle:
            raise RuntimeError(
                f"Cannot load: {path}\n"
                f"joblib error: {e_joblib}\n"
                f"pickle error: {e_pickle}"
            )


def _extract_scaler(obj):

    if hasattr(obj, "transform"):
        return obj

    if isinstance(obj, dict):

        for key in (
            "scaler",
            "x_scaler",
            "X_scaler",
            "input_scaler",
        ):

            candidate = obj.get(key)

            if (
                candidate is not None
                and hasattr(candidate, "transform")
            ):
                return candidate

    raise TypeError(
        "Loaded scaler object has no .transform()."
    )


def _normalize_feature_name(name):

    s = str(name).strip().lower()

    replacements = {
        "₀": "0",
        "₁": "1",
        "₂": "2",
        "₃": "3",
        "₄": "4",
        "₅": "5",
        "₆": "6",
        "₇": "7",
        "₈": "8",
        "₉": "9",
        "－": "-",
        "–": "-",
        "—": "-",
        "／": "/",
        "：": ":",
    }

    for a, b in replacements.items():
        s = s.replace(a, b)

    return "".join(
        ch
        for ch in s
        if ch.isalnum()
    )


_PREP_ALIASES = {

    "TKOH": {
        "tkoh",
        "kohactivationtemperature",
        "activationtemperature",
        "kohactivatingtemperature",
    },

    "RKOH/L": {
        "rkohl",
        "kohl",
        "kohlignin",
        "kohligininratio",
        "kohligninratio",
        "kohbiomassratio",
        "kohprecursorratio",
    },

    "TN": {
        "tn",
        "ndopingtemperature",
        "nitrogendopingtemperature",
        "melaminetreatmenttemperature",
        "dopingtemperature",
    },

    "RN/AC": {
        "rnac",
        "melamineac",
        "melamineactivatedcarbon",
        "melaminecarbon",
        "melamineactivatedcarbonratio",
        "melaminecarbonratio",
        "nitrogensourceacratio",
    },
}


def _canonical_prep_feature(name):

    token = _normalize_feature_name(name)

    for canonical, aliases in _PREP_ALIASES.items():

        if token in aliases:
            return canonical

    return None


def _prep_feature_names_from_scaler(
    pair,
    target_name,
):

    scaler = pair["scaler"]

    feature_names = getattr(
        scaler,
        "feature_names_in_",
        None,
    )

    if feature_names is not None:

        exact_names = [
            str(x)
            for x in feature_names
        ]

        canonical_names = []
        unresolved = []

        for raw_name in exact_names:

            canonical = _canonical_prep_feature(
                raw_name
            )

            if canonical is None:
                unresolved.append(raw_name)

            else:
                canonical_names.append(
                    canonical
                )

        if unresolved:

            override = PREP_FEATURE_OVERRIDE.get(
                target_name
            )

            if override:

                if len(override) != len(exact_names):

                    raise ValueError(
                        f"{target_name}: override has "
                        f"{len(override)} features, "
                        f"but scaler expects "
                        f"{len(exact_names)}."
                    )

                return (
                    exact_names,
                    list(override),
                )

            raise ValueError(
                f"{target_name}: could not map saved "
                f"scaler feature name(s): "
                f"{unresolved}. "
                f"Saved names are {exact_names}. "
                f"Add an explicit entry to "
                f"PREP_FEATURE_OVERRIDE."
            )

        return (
            exact_names,
            canonical_names,
        )

    override = PREP_FEATURE_OVERRIDE.get(
        target_name
    )

    if override:

        n_expected = getattr(
            scaler,
            "n_features_in_",
            None,
        )

        if (
            n_expected is not None
            and int(n_expected) != len(override)
        ):

            raise ValueError(
                f"{target_name}: override has "
                f"{len(override)} features, "
                f"but scaler expects "
                f"{int(n_expected)}."
            )

        return (
            None,
            list(override),
        )

    n_expected = getattr(
        scaler,
        "n_features_in_",
        None,
    )

    if n_expected is None:

        raise ValueError(
            f"{target_name}: scaler has neither "
            f"feature_names_in_ nor "
            f"n_features_in_."
        )

    n_expected = int(n_expected)

    if n_expected == 4:

        return (
            None,
            list(PREP_FEATURES),
        )

    raise ValueError(
        f"{target_name}: scaler expects "
        f"{n_expected} preparation features "
        f"but stores no feature names. "
        f"Add PREP_FEATURE_OVERRIDE."
    )


def _build_prep_input(
    pair,
    target_name,
    prep_values,
):

    exact_names, canonical_names = (
        _prep_feature_names_from_scaler(
            pair,
            target_name,
        )
    )

    values = []

    for canonical in canonical_names:

        if canonical not in prep_values:

            raise KeyError(
                f"{target_name}: required "
                f"preparation feature "
                f"'{canonical}' is unavailable."
            )

        values.append(
            float(
                prep_values[canonical]
            )
        )

    if exact_names is not None:

        return pd.DataFrame(
            [values],
            columns=exact_names,
        )

    return np.asarray(
        [values],
        dtype=float,
    )


def _predict_core(
    model,
    X_scaled,
    want_std=False,
):

    X_scaled = np.asarray(
        X_scaled,
        dtype=float,
    )

    if (
        not hasattr(model, "predict")
        and isinstance(model, dict)
    ):

        for key in (
            "model",
            "estimator",
            "final_model",
        ):

            if (
                key in model
                and hasattr(model[key], "predict")
            ):

                model = model[key]
                break

    if not hasattr(
        model,
        "predict",
    ):

        raise TypeError(
            f"Loaded object of type "
            f"{type(model).__name__} "
            f"has no .predict() method."
        )

    if want_std:

        try:

            mean, std = model.predict(
                X_scaled,
                return_std=True,
            )

            return (
                np.asarray(
                    mean,
                    dtype=float,
                ).reshape(-1),

                np.asarray(
                    std,
                    dtype=float,
                ).reshape(-1),
            )

        except (
            TypeError,
            ValueError,
        ):
            pass

    pred = model.predict(
        X_scaled
    )

    return (
        np.asarray(
            pred,
            dtype=float,
        ).reshape(-1),
        None,
    )


def scalar_prep_prediction(
    pair,
    target_name,
    prep_values,
):

    X = _build_prep_input(
        pair,
        target_name,
        prep_values,
    )

    X_scaled = (
        pair["scaler"]
        .transform(X)
    )

    y, _ = _predict_core(
        pair["model"],
        X_scaled,
        want_std=False,
    )

    return float(
        y[0]
    )


@st.cache_resource(
    show_spinner=False
)
def load_all_models(
    model_root_str: str
):

    root = Path(
        model_root_str
    )

    loaded = {}
    missing = []

    for name, info in MODEL_FILES.items():

        model_path = (
            root
            / info["model"]
        )

        scaler_path = (
            root
            / info["scaler"]
        )

        if not model_path.exists():
            missing.append(
                str(model_path)
            )

        if not scaler_path.exists():
            missing.append(
                str(scaler_path)
            )

        if (
            model_path.exists()
            and scaler_path.exists()
        ):

            loaded[name] = {

                "model":
                    _load_pickle_or_joblib(
                        model_path
                    ),

                "scaler":
                    _extract_scaler(
                        _load_pickle_or_joblib(
                            scaler_path
                        )
                    ),

                "model_path":
                    model_path,

                "scaler_path":
                    scaler_path,
            }

    return (
        loaded,
        missing,
    )


def predict_from_pair(
    pair,
    X,
    want_std=False,
):

    X = np.asarray(
        X,
        dtype=float,
    )

    X_scaled = (
        pair["scaler"]
        .transform(X)
    )

    return _predict_core(
        pair["model"],
        X_scaled,
        want_std=want_std,
    )


def fraction_to_percent(value):

    value = float(value)

    if abs(value) <= 1.2:

        return (
            100.0
            * value
        )

    return value


def optional_normalize_to_100(
    values: dict,
    enabled: bool,
):

    out = dict(values)

    if not enabled:
        return out

    total = sum(
        max(float(v), 0.0)
        for v in out.values()
    )

    if total <= 0:
        return out

    return {
        k:
            100.0
            * max(float(v), 0.0)
            / total

        for k, v
        in out.items()
    }


# ============================================================
# 5. FIRST-STAGE MODEL
# ============================================================
def predict_properties(
    models,
    tkoh,
    rkoh_l,
    tn,
    rn_ac,
    simple_closure=False,
):

    prep_values = {

        "TKOH":
            float(tkoh),

        "RKOH/L":
            float(rkoh_l),

        "TN":
            float(tn),

        "RN/AC":
            float(rn_ac),
    }

    raw = {

        target:
            scalar_prep_prediction(
                models[target],
                target,
                prep_values,
            )

        for target
        in PREP_TARGETS
    }

    n_frac = {

        "Pyridinic-N":
            fraction_to_percent(
                raw[
                    "Pyridinic-N fraction"
                ]
            ),

        "Pyrrolic-N/Pyridonic-N":
            fraction_to_percent(
                raw[
                    "Pyrrolic-N/Pyridonic-N fraction"
                ]
            ),

        "Graphitic-N":
            fraction_to_percent(
                raw[
                    "Graphitic-N fraction"
                ]
            ),

        "Oxidized-N":
            fraction_to_percent(
                raw[
                    "Oxidized-N fraction"
                ]
            ),
    }

    o_frac = {

        "C=O":
            fraction_to_percent(
                raw[
                    "C=O fraction"
                ]
            ),

        "C-O/C-OH":
            fraction_to_percent(
                raw[
                    "C-O/C-OH fraction"
                ]
            ),

        "O-C=O/-COOH":
            fraction_to_percent(
                raw[
                    "O-C=O/-COOH fraction"
                ]
            ),
    }

    n_frac = optional_normalize_to_100(
        n_frac,
        simple_closure,
    )

    o_frac = optional_normalize_to_100(
        o_frac,
        simple_closure,
    )

    total_n = float(
        raw["Total N"]
    )

    total_o = float(
        raw["Total O"]
    )

    n_abs = {

        k:
            total_n
            * v
            / 100.0

        for k, v
        in n_frac.items()
    }

    o_abs = {

        k:
            total_o
            * v
            / 100.0

        for k, v
        in o_frac.items()
    }

    return {

        "SBET":
            float(raw["SBET"]),

        "Vt":
            float(raw["Vt"]),

        "Vmes":
            float(raw["Vmes"]),

        "Vm":
            float(raw["Vm"]),

        "Vn":
            float(raw["Vn"]),

        "Total N":
            total_n,

        "Total O":
            total_o,

        "WCA":
            float(raw["WCA"]),

        **{
            f"{k} fraction": v
            for k, v
            in n_frac.items()
        },

        **{
            f"{k} fraction": v
            for k, v
            in o_frac.items()
        },

        **n_abs,

        **o_abs,
    }


# ============================================================
# 6. SECOND-STAGE MODEL
# ============================================================
def make_performance_base_row(p):

    return [

        p["SBET"],
        p["Vt"],
        p["Vmes"],
        p["Vm"],
        p["Vn"],

        p["Pyridinic-N"],
        p["Pyrrolic-N/Pyridonic-N"],
        p["Graphitic-N"],
        p["Oxidized-N"],

        p["C=O"],
        p["C-O/C-OH"],
        p["O-C=O/-COOH"],

        p["WCA"],
    ]


def make_curve_input(
    properties,
    operating_values,
):

    base = (
        make_performance_base_row(
            properties
        )
    )

    return np.array(
        [
            base
            + [float(v)]

            for v
            in operating_values
        ],
        dtype=float,
    )


def predict_all_curves(
    models,
    properties,
):

    return {

        "CO2":
            predict_from_pair(
                models["CO2"],
                make_curve_input(
                    properties,
                    PRESSURE_GRID,
                ),
                want_std=True,
            ),

        "H2O":
            predict_from_pair(
                models["H2O"],
                make_curve_input(
                    properties,
                    RH_GRID,
                ),
                want_std=True,
            ),

        "N2":
            predict_from_pair(
                models["N2"],
                make_curve_input(
                    properties,
                    PRESSURE_GRID,
                ),
                want_std=True,
            ),

        "CO2_H2O":
            predict_from_pair(
                models["CO2_H2O"],
                make_curve_input(
                    properties,
                    SEP_RH_GRID,
                ),
                want_std=True,
            ),

        "CO2_N2":
            predict_from_pair(
                models["CO2_N2"],
                make_curve_input(
                    properties,
                    CO2_PARTIAL_GRID,
                ),
                want_std=True,
            ),
    }


# ============================================================
# 7. PLOTTING
# ============================================================
def make_curve_figure(
    x,
    mean,
    std,
    x_title,
    y_title,
    name,
    clip_nonnegative=False,
    line_color="#2b78c5",
    title_text=None,
):

    x = np.asarray(
        x,
        dtype=float,
    )

    mean = np.asarray(
        mean,
        dtype=float,
    )

    if clip_nonnegative:

        mean = np.maximum(
            mean,
            0.0,
        )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=mean,

            mode="lines+markers",

            name=name,

            line=dict(
                width=1.35,
                color=line_color,
            ),

            marker=dict(
                size=5.6,
                color=line_color,
            ),

            hovertemplate=(
                "%{x:.2f}<br>"
                "%{y:.3f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(

        height=255,

        margin=dict(
            l=10,
            r=6,
            t=58,
            b=10,
        ),

        title=dict(
            text=(
                title_text
                or name
            ),

            x=0.5,

            xanchor="center",

            y=0.965,

            yanchor="top",

            font=dict(
                size=12,
                color="#111",
            ),
        ),

        xaxis_title=x_title,

        yaxis_title=y_title,

        hovermode="closest",

        showlegend=False,

        paper_bgcolor=(
            "rgba(0,0,0,0)"
        ),

        plot_bgcolor=(
            "rgba(255,255,255,0.06)"
        ),

        font=dict(
            size=9,
            color="#222",
        ),
    )

    fig.update_xaxes(

        tickmode="array",

        tickvals=x,

        tickfont=dict(
            size=8
        ),

        title_font=dict(
            size=9
        ),

        showgrid=True,

        gridcolor=(
            "rgba(120,150,125,0.13)"
        ),

        zeroline=False,

        automargin=True,

        linecolor=(
            "rgba(80,90,85,0.35)"
        ),

        mirror=False,
    )

    fig.update_yaxes(

        tickfont=dict(
            size=8
        ),

        title_font=dict(
            size=9
        ),

        showgrid=True,

        gridcolor=(
            "rgba(120,150,125,0.13)"
        ),

        zeroline=False,

        automargin=True,

        linecolor=(
            "rgba(80,90,85,0.35)"
        ),
    )

    return fig


def render_compact_table(df):

    html = df.to_html(
        index=False,
        classes="compact-table",
        border=0,
        escape=False,
    )

    st.markdown(
        (
            '<div class="property-card">'
            f"{html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


# ============================================================
# 8. DIAGNOSTICS
# ============================================================
def model_diagnostics_table(
    models
):

    rows = []

    for name, pair in models.items():

        model = pair["model"]
        scaler = pair["scaler"]

        model_n = getattr(
            model,
            "n_features_in_",
            None,
        )

        scaler_n = getattr(
            scaler,
            "n_features_in_",
            None,
        )

        feature_names = getattr(
            scaler,
            "feature_names_in_",
            None,
        )

        feature_names_text = ""

        if feature_names is not None:

            feature_names_text = ", ".join(
                map(
                    str,
                    feature_names,
                )
            )

        resolved_prep = ""

        if name in PREP_TARGETS:

            try:

                _, canonical = (
                    _prep_feature_names_from_scaler(
                        pair,
                        name,
                    )
                )

                resolved_prep = " → ".join(
                    canonical
                )

            except Exception as exc:

                resolved_prep = (
                    f"ERROR: {exc}"
                )

        rows.append(
            {
                "Model":
                    name,

                "Scaler n_features":
                    scaler_n,

                "Model n_features":
                    model_n,

                "Scaler feature_names":
                    feature_names_text,

                "Resolved prep inputs":
                    resolved_prep,

                "Model file":
                    str(
                        pair["model_path"]
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 9. MODEL LOADING
# ============================================================
try:

    models, missing_files = (
        load_all_models(
            str(
                MODEL_ROOT
            )
        )
    )

except Exception as exc:

    st.error(
        "Model loading failed."
    )

    st.exception(exc)

    st.stop()


if missing_files:

    st.error(
        f"{len(missing_files)} "
        f"model/scaler files were not found. "
        f"Please check MODEL_ROOT and MODEL_FILES."
    )

    with st.expander(
        "Show missing files"
    ):

        for path in missing_files:

            st.write(path)

    st.stop()


required_keys = set(
    MODEL_FILES
)

loaded_keys = set(
    models
)


if required_keys != loaded_keys:

    st.error(
        "Not all model pairs were loaded."
    )

    st.write(
        "Missing model keys:",
        sorted(
            required_keys
            - loaded_keys
        ),
    )

    st.stop()


# ============================================================
# 10. MAIN UI
# ============================================================
tab_predict, tab_diag = st.tabs(
    [
        "Forward prediction",
        "Model diagnostics",
    ]
)


with tab_predict:

    # ========================================================
    # Main title
    # ========================================================
    st.markdown(
        (
            '<div class="platform-title">'

            "Lignin Porous Carbon Properties Parameters "
            "and CO₂ Separation Performance Prediction Platform"

            "</div>"
        ),

        unsafe_allow_html=True,
    )


    # ========================================================
    # Step bar
    # ========================================================
    st.markdown(
        (
            '<div class="step-bar">'

            "Step1: Enter Preparation Parameters "

            "→ "

            "Step2: Predict Properties Parameters "

            "→ "

            "Step3: Predict CO₂ Separation Performance"

            "</div>"
        ),

        unsafe_allow_html=True,
    )


    # ========================================================
    # Top panel
    # ========================================================
    left, pore_col, chem_col = st.columns(
        [
            1.05,
            0.72,
            1.38,
        ],
        gap="small",
    )


    # ========================================================
    # Preparation parameters
    # ========================================================
    with left:

        st.markdown(
            (
                '<div class="section-chip">'
                "⚗ Preparation Parameters"
                "</div>"
            ),
            unsafe_allow_html=True,
        )


        in_l, in_r = st.columns(
            2,
            gap="small",
        )


        with in_l:

            tkoh = st.number_input(
                "Tₖₒₕ (°C)",
                min_value=300.0,
                max_value=1200.0,
                value=700.0,
                step=1.0,
                help=(
                    "KOH activation temperature; "
                    "experimental domain: "
                    "500–800 °C."
                ),
            )


            tn = st.number_input(
                "Tₙ (°C)",
                min_value=300.0,
                max_value=1200.0,
                value=800.0,
                step=1.0,
                help=(
                    "N-doping temperature; "
                    "experimental domain: "
                    "500–900 °C."
                ),
            )


        with in_r:

            rkoh_l = st.number_input(
                "Rₖₒₕ/L",
                min_value=0.0,
                max_value=5.0,
                value=1.20,
                step=0.01,
                help=(
                    "KOH/lignin mass ratio; "
                    "experimental domain: "
                    "0.5–2.0."
                ),
            )


            rn_ac = st.number_input(
                "Rₙ/AC",
                min_value=0.0,
                max_value=5.0,
                value=1.80,
                step=0.01,
                help=(
                    "Melamine/activated-carbon "
                    "mass ratio; "
                    "experimental domain: "
                    "0–3."
                ),
            )


        inputs = {
            "TKOH": tkoh,
            "RKOH/L": rkoh_l,
            "TN": tn,
            "RN/AC": rn_ac,
        }


        out_of_domain = []


        for key, value in inputs.items():

            lo, hi = PREP_DOMAIN[key]

            if not (
                lo
                <= value
                <= hi
            ):

                out_of_domain.append(
                    f"{key}: "
                    f"{value:g} "
                    f"({lo:g}–{hi:g})"
                )


        if out_of_domain:

            st.markdown(
                (
                    '<div class="warn-box">'
                    "<b>⚠ Outside applicability domain:</b> "
                    + "; ".join(
                        out_of_domain
                    )
                    + "</div>"
                ),
                unsafe_allow_html=True,
            )


        else:

            st.markdown(
                (
                    '<div class="good-box">'
                    "✓ Within experimental applicability domain"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )


        predict_properties_clicked = st.button(
            "Predict Properties Parameters",
            key="predict_properties_only",
            use_container_width=True,
        )


        predict_performance_clicked = st.button(
            "Predict CO₂ Separation Performance",
            key="predict_all",
            use_container_width=True,
        )


        with st.expander(
            "Advanced",
            expanded=False,
        ):

            simple_closure = st.checkbox(
                "Simple 100% N/O closure",
                value=False,
                help=(
                    "Keep OFF unless "
                    "you intentionally want "
                    "simple normalization."
                ),
            )


            clip_adsorption = st.checkbox(
                "Clip negative adsorption to 0",
                value=True,
            )


    simple_closure = locals().get(
        "simple_closure",
        False,
    )


    clip_adsorption = locals().get(
        "clip_adsorption",
        True,
    )


    # ========================================================
    # First stage only
    # ========================================================
    if predict_properties_clicked:

        try:

            properties = predict_properties(
                models,
                tkoh,
                rkoh_l,
                tn,
                rn_ac,
                simple_closure=simple_closure,
            )


            st.session_state[
                "last_prediction"
            ] = {
                "inputs":
                    inputs,

                "properties":
                    properties,

                "curves":
                    None,
            }


        except Exception as exc:

            st.error(
                "Property prediction failed."
            )

            st.exception(exc)


    # ========================================================
    # Full two-stage prediction
    # ========================================================
    if predict_performance_clicked:

        try:

            properties = predict_properties(
                models,
                tkoh,
                rkoh_l,
                tn,
                rn_ac,
                simple_closure=simple_closure,
            )


            curves = predict_all_curves(
                models,
                properties,
            )


            st.session_state[
                "last_prediction"
            ] = {
                "inputs":
                    inputs,

                "properties":
                    properties,

                "curves":
                    curves,
            }


        except Exception as exc:

            st.error(
                "Prediction failed."
            )

            st.exception(exc)


    result = (
        st.session_state
        .get(
            "last_prediction"
        )
    )


    p = (
        result["properties"]
        if result is not None
        else None
    )


    curves = (
        result["curves"]
        if result is not None
        else None
    )


    def pv(
        key,
        fmt,
        unit="",
    ):

        if p is None:
            return "—"

        return (
            f"{format(p[key], fmt)}"
            f"{unit}"
        )


    # ========================================================
    # Pore structure
    # ========================================================
    with pore_col:

        st.markdown(
            (
                '<div class="section-heading">'
                "Pore structure"
                "</div>"
            ),
            unsafe_allow_html=True,
        )


        pore_df = pd.DataFrame(
            {
                "Property": [
                    "S<sub>BET</sub>",
                    "V<sub>t</sub>",
                    "V<sub>mes</sub>",
                    "V<sub>m</sub>",
                    "V<sub>n</sub>",
                ],

                "Predicted value": [
                    pv(
                        "SBET",
                        ".1f",
                        " m²/g",
                    ),

                    pv(
                        "Vt",
                        ".3f",
                        " cm³/g",
                    ),

                    pv(
                        "Vmes",
                        ".3f",
                        " cm³/g",
                    ),

                    pv(
                        "Vm",
                        ".3f",
                        " cm³/g",
                    ),

                    pv(
                        "Vn",
                        ".3f",
                        " cm³/g",
                    ),
                ],
            }
        )


        render_compact_table(
            pore_df
        )


    # ========================================================
    # Chemical properties + wettability
    # ========================================================
    with chem_col:

        st.markdown(
            (
                '<div class="section-heading">'
                "Chemical Properties, Wettability"
                "</div>"
            ),
            unsafe_allow_html=True,
        )


        n_side, o_side = st.columns(
            2,
            gap="small",
        )


        with n_side:

            n_df = pd.DataFrame(
                {
                    "Property": [
                        "Total N",
                        "Pyridinic-N",
                        (
                            "Pyrrolic-N/"
                            "<br>"
                            "Pyridonic-N"
                        ),
                        "Graphitic-N",
                        "Oxidized-N",
                    ],

                    "Predicted value": [
                        pv(
                            "Total N",
                            ".2f",
                            " at.%",
                        ),

                        pv(
                            "Pyridinic-N fraction",
                            ".2f",
                            "%",
                        ),

                        pv(
                            "Pyrrolic-N/Pyridonic-N fraction",
                            ".2f",
                            "%",
                        ),

                        pv(
                            "Graphitic-N fraction",
                            ".2f",
                            "%",
                        ),

                        pv(
                            "Oxidized-N fraction",
                            ".2f",
                            "%",
                        ),
                    ],
                }
            )


            render_compact_table(
                n_df
            )


        with o_side:

            o_df = pd.DataFrame(
                {
                    "Property": [
                        "Total O",
                        "C=O",
                        "C-O/C-OH",
                        "O-C=O/-COOH",
                        "WCA",
                    ],

                    "Predicted value": [
                        pv(
                            "Total O",
                            ".2f",
                            " at.%",
                        ),

                        pv(
                            "C=O fraction",
                            ".2f",
                            "%",
                        ),

                        pv(
                            "C-O/C-OH fraction",
                            ".2f",
                            "%",
                        ),

                        pv(
                            "O-C=O/-COOH fraction",
                            ".2f",
                            "%",
                        ),

                        pv(
                            "WCA",
                            ".0f",
                            "°",
                        ),
                    ],
                }
            )


            render_compact_table(
                o_df
            )


    st.markdown(
        '<hr class="soft-rule">',
        unsafe_allow_html=True,
    )


    # ========================================================
    # Five curves
    # ========================================================
    if curves is None:

        st.info(
            "Click **Predict CO₂ Separation Performance** "
            "to generate the five prediction curves."
        )


    else:

        co2_mean, co2_std = curves[
            "CO2"
        ]

        h2o_mean, h2o_std = curves[
            "H2O"
        ]

        n2_mean, n2_std = curves[
            "N2"
        ]

        sep_h2o_mean, sep_h2o_std = curves[
            "CO2_H2O"
        ]

        sep_n2_mean, sep_n2_std = curves[
            "CO2_N2"
        ]


        g1, g2, g3, g4, g5 = st.columns(
            5,
            gap="small",
        )


        with g1:

            st.plotly_chart(
                make_curve_figure(
                    PRESSURE_GRID,
                    co2_mean,
                    co2_std,
                    "Pressure (kPa)",
                    "CO₂ adsorption capacity (mmol/g)",
                    "CO₂",
                    clip_adsorption,
                    "#e34b4b",
                    "323K CO₂<br>Adsorption Isotherm",
                ),

                use_container_width=True,

                config={
                    "displayModeBar": False
                },
            )


        with g2:

            st.plotly_chart(
                make_curve_figure(
                    RH_GRID,
                    h2o_mean,
                    h2o_std,
                    "RH (%)",
                    "H₂O adsorption capacity (mmol/g)",
                    "H₂O",
                    clip_adsorption,
                    "#31a354",
                    "323K H₂O<br>Adsorption Isotherm",
                ),

                use_container_width=True,

                config={
                    "displayModeBar": False
                },
            )


        with g3:

            st.plotly_chart(
                make_curve_figure(
                    PRESSURE_GRID,
                    n2_mean,
                    n2_std,
                    "Pressure (kPa)",
                    "N₂ adsorption capacity (mmol/g)",
                    "N₂",
                    clip_adsorption,
                    "#9a73c8",
                    "323K N₂<br>Adsorption Isotherm",
                ),

                use_container_width=True,

                config={
                    "displayModeBar": False
                },
            )


        with g4:

            st.plotly_chart(
                make_curve_figure(
                    SEP_RH_GRID,
                    sep_h2o_mean,
                    sep_h2o_std,
                    "RH (%)",
                    "CO₂/H₂O separation capacity (mmol/g)",
                    "CO₂/H₂O",
                    False,
                    "#f28e1c",
                    "CO₂/H₂O<br>Separation Capacity",
                ),

                use_container_width=True,

                config={
                    "displayModeBar": False
                },
            )


        with g5:

            st.plotly_chart(
                make_curve_figure(
                    CO2_PARTIAL_GRID,
                    sep_n2_mean,
                    sep_n2_std,
                    "CO₂ partial pressure (kPa)",
                    "CO₂/N₂ separation capacity (mmol/g)",
                    "CO₂/N₂",
                    False,
                    "#169bd5",
                    "CO₂/N₂<br>Separation Capacity",
                ),

                use_container_width=True,

                config={
                    "displayModeBar": False
                },
            )


        # ====================================================
        # Export results
        # ====================================================
        export_rows = []


        for key, value in result[
            "inputs"
        ].items():

            export_rows.append(
                [
                    "Preparation input",
                    key,
                    value,
                    "",
                ]
            )


        for key, value in p.items():

            if key == "SBET":

                unit = "m2/g"


            elif key in (
                "Vt",
                "Vmes",
                "Vm",
                "Vn",
            ):

                unit = "cm3/g"


            elif key == "WCA":

                unit = "degree"


            elif "fraction" in key:

                unit = "%"


            else:

                unit = "at.%"


            export_rows.append(
                [
                    "Predicted property",
                    key,
                    value,
                    unit,
                ]
            )


        export_df = pd.DataFrame(
            export_rows,
            columns=[
                "Category",
                "Variable",
                "Value",
                "Unit",
            ],
        )


        adsorption_df = pd.DataFrame(
            {
                "Pressure_kPa":
                    PRESSURE_GRID,

                "CO2_uptake_mmol_g":
                    co2_mean,

                "N2_uptake_mmol_g":
                    n2_mean,

                "RH_percent":
                    RH_GRID,

                "H2O_uptake_mmol_g":
                    h2o_mean,
            }
        )


        co2h2o_df = pd.DataFrame(
            {
                "RH_percent":
                    SEP_RH_GRID,

                "CO2_H2O_separation_mmol_g":
                    sep_h2o_mean,
            }
        )


        co2n2_df = pd.DataFrame(
            {
                "CO2_partial_pressure_kPa":
                    CO2_PARTIAL_GRID,

                "N2_partial_pressure_kPa":
                    (
                        100.0
                        - CO2_PARTIAL_GRID
                    ),

                "CO2_N2_separation_mmol_g":
                    sep_n2_mean,
            }
        )


        with st.expander(
            "Export results",
            expanded=False,
        ):

            d1, d2, d3, d4 = st.columns(
                4
            )


            with d1:

                st.download_button(
                    "Properties (.csv)",

                    (
                        export_df
                        .to_csv(
                            index=False
                        )
                        .encode(
                            "utf-8-sig"
                        )
                    ),

                    file_name=(
                        "predicted_properties.csv"
                    ),

                    mime="text/csv",

                    use_container_width=True,
                )


            with d2:

                st.download_button(
                    "Adsorption (.csv)",

                    (
                        adsorption_df
                        .to_csv(
                            index=False
                        )
                        .encode(
                            "utf-8-sig"
                        )
                    ),

                    file_name=(
                        "adsorption_curves.csv"
                    ),

                    mime="text/csv",

                    use_container_width=True,
                )


            with d3:

                st.download_button(
                    "CO₂-H₂O (.csv)",

                    (
                        co2h2o_df
                        .to_csv(
                            index=False
                        )
                        .encode(
                            "utf-8-sig"
                        )
                    ),

                    file_name=(
                        "CO2_H2O_separation_curve.csv"
                    ),

                    mime="text/csv",

                    use_container_width=True,
                )


            with d4:

                st.download_button(
                    "CO₂-N₂ (.csv)",

                    (
                        co2n2_df
                        .to_csv(
                            index=False
                        )
                        .encode(
                            "utf-8-sig"
                        )
                    ),

                    file_name=(
                        "CO2_N2_separation_curve.csv"
                    ),

                    mime="text/csv",

                    use_container_width=True,
                )


# ============================================================
# MODEL DIAGNOSTICS
# ============================================================
with tab_diag:

    st.subheader(
        "Model diagnostics"
    )


    st.caption(
        f"Model root: {MODEL_ROOT}"
    )


    st.write(
        "Use this page to check whether the "
        "saved scalers/models expect the same "
        "number and order of input features "
        "assumed by this app."
    )


    diag_df = model_diagnostics_table(
        models
    )


    st.dataframe(
        diag_df,
        hide_index=True,
        use_container_width=True,
    )


    st.markdown(
        """
        **Expected input dimensionality**

        - Preparation → property models: **model-specific input count**  
          The app reads each saved scaler's `feature_names_in_` / `n_features_in_`.  
          A 3-input model receives only its original 3 variables; a 4-input model receives all 4.

        - Property → adsorption/separation models: **14 features**  
          13 material descriptors + one operating variable (`P`, `RH`, or CO₂ partial pressure)
        """
    )


    st.markdown(
        "**Default material-descriptor order "
        "used in this app**"
    )


    st.code(
        "\n".join(
            [
                f"{i+1:02d}. {name}"

                for i, name
                in enumerate(
                    PERFORMANCE_BASE_FEATURES
                )
            ]

            + [
                (
                    "14. operating variable "
                    "(P / RH / CO2 partial pressure)"
                )
            ]
        ),

        language=None,
    )


    st.warning(
        "Scientific-use note: first-stage "
        "preparation inputs are reconstructed "
        "from each saved scaler's own training "
        "columns/order. If a scaler lacks feature "
        "names and uses fewer than four inputs, "
        "the app intentionally stops and asks for "
        "an explicit override instead of guessing. "
        "Verify several known samples before "
        "publication."
    )