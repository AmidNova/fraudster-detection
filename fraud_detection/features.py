"""User-level feature engineering for fraudster detection.

Every function returns one row per user (indexed by user id) so the
outputs can be joined into a single modelling table.
"""
import numpy as np
import pandas as pd

# Category levels kept, chosen from the exploratory analysis (notebook Step 2)
TOP_COUNTRIES = ["GB", "FR", "PL", "LT", "IE", "ES", "RO"]
SHARE_LEVELS = {
    "TYPE": ["ATM", "BANK_TRANSFER", "CARD_PAYMENT", "P2P", "TOPUP"],
    "STATE": ["DECLINED", "FAILED", "REVERTED"],
    "ENTRY_METHOD": ["manu", "chip", "cont", "mags", "misc"],
    "SOURCE": ["MINOS", "INTERNAL", "GAIA", "HERA"],
}
EVENING_HOURS = range(18, 24)
NIGHT_HOURS = range(0, 6)
CASH_OUT_TYPES = ["ATM", "BANK_TRANSFER"]
FAST_CASH_OUT_HOURS = 24
BAD_STATES = ["DECLINED", "FAILED"]


def profile_features(users: pd.DataFrame) -> pd.DataFrame:
    """One row per user built from users.csv (STATE excluded: target leak)."""
    phone_match = [country in str(phones).split("||")
                   for country, phones in zip(users["COUNTRY"], users["PHONE_COUNTRY"])]
    df = pd.DataFrame({
        "USER_ID": users["ID"],
        "HAS_EMAIL": users["HAS_EMAIL"],
        "FAILED_SIGN_IN_ATTEMPTS": users["FAILED_SIGN_IN_ATTEMPTS"],
        "AGE": users["CREATED_DATE"].dt.year - users["BIRTH_YEAR"],
        "PHONE_MATCHES_COUNTRY": np.array(phone_match, dtype=int),
        "KYC": users["KYC"],
        "COUNTRY_GROUP": users["COUNTRY"].where(users["COUNTRY"].isin(TOP_COUNTRIES), "OTHER"),
    })
    return pd.get_dummies(df, columns=["KYC", "COUNTRY_GROUP"], dtype=int).set_index("USER_ID")


def share_features(tx: pd.DataFrame) -> pd.DataFrame:
    """Share of each selected category per user (TYPE, STATE, ENTRY_METHOD, SOURCE, ATM merchant)."""
    parts = [
        pd.crosstab(tx["USER_ID"], tx[col], normalize="index")
          .reindex(columns=levels, fill_value=0)
          .add_prefix(f"{col}_")
        for col, levels in SHARE_LEVELS.items()
    ]
    atm_share = (tx["MERCHANT_CATEGORY"] == "atm").groupby(tx["USER_ID"]).mean()
    return pd.concat(parts, axis=1).assign(MERCHANT_ATM_SHARE=atm_share)


def activity_features(tx: pd.DataFrame, users: pd.DataFrame, country_iso3: pd.Series) -> pd.DataFrame:
    """Volume, amount, timing, geography and cash-in/cash-out aggregates per user.

    `country_iso3` maps the 2-letter user country to the 3-letter merchant country code."""
    user_info = users.set_index("ID")
    home_iso3 = tx["USER_ID"].map(user_info["COUNTRY"]).map(country_iso3)
    hour = tx["CREATED_DATE"].dt.hour
    enriched = tx.assign(
        IS_EVENING=hour.isin(EVENING_HOURS),
        IS_NIGHT=hour.isin(NIGHT_HOURS),
        IS_WEEKEND=tx["CREATED_DATE"].dt.dayofweek >= 5,
        IS_FOREIGN=tx["MERCHANT_COUNTRY"].notna() & (tx["MERCHANT_COUNTRY"] != home_iso3),
        CASH_IN_USD=tx["AMOUNT_USD"].where(tx["TYPE"] == "TOPUP", 0),
        CASH_OUT_USD=tx["AMOUNT_USD"].where(tx["TYPE"].isin(CASH_OUT_TYPES), 0),
    )
    agg = enriched.groupby("USER_ID").agg(
        N_TX=("ID", "size"),
        AMOUNT_MEAN=("AMOUNT_USD", "mean"),
        AMOUNT_MAX=("AMOUNT_USD", "max"),
        AMOUNT_SUM=("AMOUNT_USD", "sum"),
        N_CURRENCIES=("CURRENCY", "nunique"),
        N_MERCHANT_COUNTRIES=("MERCHANT_COUNTRY", "nunique"),
        EVENING_SHARE=("IS_EVENING", "mean"),
        NIGHT_SHARE=("IS_NIGHT", "mean"),
        WEEKEND_SHARE=("IS_WEEKEND", "mean"),
        FOREIGN_MERCHANT_SHARE=("IS_FOREIGN", "mean"),
        CASH_IN=("CASH_IN_USD", "sum"),
        CASH_OUT=("CASH_OUT_USD", "sum"),
        FIRST_TX=("CREATED_DATE", "min"),
    )
    hours_to_first = (agg["FIRST_TX"] - agg.index.map(user_info["CREATED_DATE"])).dt.total_seconds() / 3600
    return (agg
            .assign(LOG_AMOUNT_MEAN=np.log1p(agg["AMOUNT_MEAN"].fillna(0)),
                    LOG_AMOUNT_MAX=np.log1p(agg["AMOUNT_MAX"].fillna(0)),
                    LOG_AMOUNT_SUM=np.log1p(agg["AMOUNT_SUM"]),
                    LOG_HOURS_TO_FIRST_TX=np.log1p(hours_to_first.clip(lower=0)),
                    CASH_OUT_SHARE=(agg["CASH_OUT"] / (agg["CASH_IN"] + agg["CASH_OUT"])).fillna(0))
            .drop(columns=["AMOUNT_MEAN", "AMOUNT_MAX", "AMOUNT_SUM", "CASH_IN", "CASH_OUT", "FIRST_TX"]))


def build_feature_table(users: pd.DataFrame, tx: pd.DataFrame, country_iso3: pd.Series) -> pd.DataFrame:
    """Final modelling table: one row per user, features + target.

    Users without any transaction get 0 for transaction features;
    HAS_TX lets the model tell "no activity" apart from "zero values"."""
    tx_feats = activity_features(tx, users, country_iso3).join(share_features(tx))
    features = (profile_features(users)
                .join(tx_feats)
                .assign(HAS_TX=lambda d: d["N_TX"].notna().astype(int))
                .fillna(0))
    target = users.set_index("ID")["IS_FRAUDSTER"].astype(int).rename("IS_FRAUDSTER")
    return features.join(target)


def sequence_features(tx: pd.DataFrame) -> pd.DataFrame:
    """Order-aware features per user: velocity, declined streaks, top-up -> cash-out speed.

    FAST_CASH_OUT_SHARE is undefined without any cash-out and LOG_MEDIAN_GAP_H without
    a second transaction: both are set to 0 and flagged by HAS_CASH_OUT / HAS_REPEAT_TX,
    so a 0 can be told apart from "not applicable"."""
    tx = tx.sort_values(["USER_ID", "CREATED_DATE"])
    by_user = tx.groupby("USER_ID")

    max_per_hour = tx.groupby(["USER_ID", tx["CREATED_DATE"].dt.floor("h")]).size().groupby(level=0).max()
    max_per_day = tx.groupby(["USER_ID", tx["CREATED_DATE"].dt.floor("D")]).size().groupby(level=0).max()
    gap_hours = by_user["CREATED_DATE"].diff().dt.total_seconds() / 3600
    median_gap = gap_hours.groupby(tx["USER_ID"]).median()

    # Longest run of consecutive declined / failed transactions
    is_bad = tx["STATE"].isin(BAD_STATES)
    run_id = (is_bad != is_bad.groupby(tx["USER_ID"]).shift()).cumsum()
    max_bad_streak = is_bad.groupby(run_id).transform("sum").groupby(tx["USER_ID"]).max()

    # For every cash-out, time since the user's previous top-up (no previous top-up -> not fast)
    topups = (tx.loc[tx["TYPE"] == "TOPUP", ["USER_ID", "CREATED_DATE"]]
                .rename(columns={"CREATED_DATE": "TOPUP_DATE"}).sort_values("TOPUP_DATE"))
    cash_outs = tx.loc[tx["TYPE"].isin(CASH_OUT_TYPES), ["USER_ID", "CREATED_DATE"]].sort_values("CREATED_DATE")
    linked = pd.merge_asof(cash_outs, topups, left_on="CREATED_DATE", right_on="TOPUP_DATE",
                           by="USER_ID", direction="backward")
    delay_hours = (linked["CREATED_DATE"] - linked["TOPUP_DATE"]).dt.total_seconds() / 3600
    fast_share = (delay_hours <= FAST_CASH_OUT_HOURS).groupby(linked["USER_ID"]).mean()

    features = pd.DataFrame({
        "FAST_CASH_OUT_SHARE": fast_share,
        "MAX_TX_PER_HOUR": max_per_hour,
        "MAX_TX_PER_DAY": max_per_day,
        "MAX_DECLINED_STREAK": max_bad_streak,
        "LOG_MEDIAN_GAP_H": np.log1p(median_gap),
    })
    return features.assign(HAS_CASH_OUT=features.index.isin(fast_share.index).astype(int),
                           HAS_REPEAT_TX=(by_user.size() >= 2).astype(int)).fillna(0)
