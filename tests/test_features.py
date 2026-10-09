import numpy as np
import pandas as pd
import pytest

from fraud_detection.features import (
    activity_features,
    build_feature_table,
    profile_features,
    sequence_features,
    share_features,
)

COUNTRY_ISO3 = pd.Series({"GB": "GBR", "FR": "FRA", "BR": "BRA"})


@pytest.fixture
def users():
    return pd.DataFrame({
        "ID": ["u1", "u2", "u3"],
        "HAS_EMAIL": [1, 1, 0],
        "FAILED_SIGN_IN_ATTEMPTS": [0, 2, 0],
        "CREATED_DATE": pd.to_datetime(["2018-01-01 08:00", "2018-01-01 08:00", "2018-01-02 08:00"]),
        "BIRTH_YEAR": [1990, 1980, 2000],
        "PHONE_COUNTRY": ["GB||JE", "GB", "FR"],
        "COUNTRY": ["GB", "FR", "BR"],
        "KYC": ["PASSED", "PENDING", "NONE"],
        "STATE": ["ACTIVE", "LOCKED", "ACTIVE"],
        "IS_FRAUDSTER": [False, True, False],
    })


def make_tx(rows):
    """rows: (user, time, type, state, amount, merchant_country)"""
    df = pd.DataFrame(rows, columns=["USER_ID", "CREATED_DATE", "TYPE", "STATE", "AMOUNT_USD", "MERCHANT_COUNTRY"])
    return df.assign(
        ID=[f"t{i}" for i in range(len(df))],
        CREATED_DATE=pd.to_datetime(df["CREATED_DATE"]),
        ENTRY_METHOD="chip",
        SOURCE="GAIA",
        MERCHANT_CATEGORY=np.where(df["TYPE"] == "ATM", "atm", "shop"),
        CURRENCY="GBP",
    )


@pytest.fixture
def tx():
    return make_tx([
        ("u1", "2018-01-01 09:00", "TOPUP", "COMPLETED", 100.0, None),
        ("u1", "2018-01-01 20:00", "CARD_PAYMENT", "COMPLETED", 30.0, "GBR"),
        ("u1", "2018-01-02 10:00", "CARD_PAYMENT", "COMPLETED", 20.0, "FRA"),
        ("u2", "2018-01-01 08:30", "TOPUP", "COMPLETED", 500.0, None),
        ("u2", "2018-01-01 09:00", "ATM", "DECLINED", 500.0, "FRA"),
        ("u2", "2018-01-01 09:10", "ATM", "DECLINED", 500.0, "FRA"),
        ("u2", "2018-01-01 09:20", "ATM", "COMPLETED", 400.0, "FRA"),
        ("u2", "2018-01-05 09:00", "BANK_TRANSFER", "COMPLETED", 100.0, None),
    ])


class TestProfileFeatures:
    def test_one_row_per_user_without_target_leak(self, users):
        out = profile_features(users)
        assert list(out.index) == ["u1", "u2", "u3"]
        assert "STATE" not in out.columns and "IS_FRAUDSTER" not in out.columns

    def test_phone_match_handles_multiple_phone_countries(self, users):
        out = profile_features(users)
        assert out["PHONE_MATCHES_COUNTRY"].tolist() == [1, 0, 0]

    def test_rare_countries_grouped_as_other(self, users):
        out = profile_features(users)
        assert out.loc["u3", "COUNTRY_GROUP_OTHER"] == 1
        assert out.loc["u1", "COUNTRY_GROUP_GB"] == 1

    def test_age_at_sign_up(self, users):
        assert profile_features(users)["AGE"].tolist() == [28, 38, 18]


class TestShareFeatures:
    def test_shares_with_unseen_levels_filled_with_zero(self, tx):
        out = share_features(tx)
        assert out.loc["u2", "TYPE_ATM"] == pytest.approx(3 / 5)
        assert out.loc["u2", "STATE_DECLINED"] == pytest.approx(2 / 5)
        assert out.loc["u1", "TYPE_P2P"] == 0
        assert out.loc["u2", "MERCHANT_ATM_SHARE"] == pytest.approx(3 / 5)

    def test_type_shares_sum_to_one(self, tx):
        out = share_features(tx)
        type_cols = [c for c in out.columns if c.startswith("TYPE_")]
        assert np.allclose(out[type_cols].sum(axis=1), 1)


class TestActivityFeatures:
    def test_counts_foreign_share_and_cash_out(self, tx, users):
        out = activity_features(tx, users, COUNTRY_ISO3)
        assert out.loc["u1", "N_TX"] == 3
        # u1 lives in GB: one payment in GBR, one in FRA, the top-up has no merchant country
        assert out.loc["u1", "FOREIGN_MERCHANT_SHARE"] == pytest.approx(1 / 3)
        assert out.loc["u1", "CASH_OUT_SHARE"] == 0
        # u2: 500 in, 1400 + 100 out
        assert out.loc["u2", "CASH_OUT_SHARE"] == pytest.approx(1500 / 2000)

    def test_hours_to_first_tx(self, tx, users):
        out = activity_features(tx, users, COUNTRY_ISO3)
        assert out.loc["u1", "LOG_HOURS_TO_FIRST_TX"] == pytest.approx(np.log1p(1))
        assert out.loc["u2", "LOG_HOURS_TO_FIRST_TX"] == pytest.approx(np.log1p(0.5))


class TestBuildFeatureTable:
    def test_user_without_transactions_gets_zeros_and_flag(self, tx, users):
        out = build_feature_table(users, tx, COUNTRY_ISO3)
        assert out.isna().sum().sum() == 0
        assert out.loc["u3", "HAS_TX"] == 0 and out.loc["u3", "N_TX"] == 0
        assert out.loc["u1", "HAS_TX"] == 1

    def test_target_attached_as_int(self, tx, users):
        out = build_feature_table(users, tx, COUNTRY_ISO3)
        assert out["IS_FRAUDSTER"].tolist() == [0, 1, 0]


class TestSequenceFeatures:
    def test_fast_cash_out_share(self, tx):
        out = sequence_features(tx)
        # u2: 3 ATM within 1h of the top-up (fast), transfer 4 days later (slow)
        assert out.loc["u2", "FAST_CASH_OUT_SHARE"] == pytest.approx(3 / 4)

    def test_cash_out_without_previous_topup_is_not_fast(self):
        out = sequence_features(make_tx([
            ("u1", "2018-01-01 09:00", "ATM", "COMPLETED", 50.0, "GBR"),
            ("u1", "2018-01-01 10:00", "TOPUP", "COMPLETED", 50.0, None),
        ]))
        assert out.loc["u1", "FAST_CASH_OUT_SHARE"] == 0
        assert out.loc["u1", "HAS_CASH_OUT"] == 1

    def test_no_cash_out_is_flagged_not_applicable(self, tx):
        out = sequence_features(tx)
        assert out.loc["u1", "HAS_CASH_OUT"] == 0
        assert out.loc["u1", "FAST_CASH_OUT_SHARE"] == 0
        assert out.loc["u2", "HAS_CASH_OUT"] == 1

    def test_declined_streak_does_not_cross_users(self):
        out = sequence_features(make_tx([
            ("u1", "2018-01-01 09:00", "CARD_PAYMENT", "COMPLETED", 1.0, "GBR"),
            ("u1", "2018-01-01 10:00", "CARD_PAYMENT", "DECLINED", 1.0, "GBR"),
            ("u2", "2018-01-01 08:00", "CARD_PAYMENT", "FAILED", 1.0, "GBR"),
            ("u2", "2018-01-01 08:05", "CARD_PAYMENT", "DECLINED", 1.0, "GBR"),
            ("u2", "2018-01-01 08:10", "CARD_PAYMENT", "COMPLETED", 1.0, "GBR"),
            ("u2", "2018-01-01 08:15", "CARD_PAYMENT", "DECLINED", 1.0, "GBR"),
        ]))
        assert out.loc["u1", "MAX_DECLINED_STREAK"] == 1
        assert out.loc["u2", "MAX_DECLINED_STREAK"] == 2

    def test_velocity(self, tx):
        out = sequence_features(tx)
        assert out.loc["u2", "MAX_TX_PER_HOUR"] == 3   # 09:00, 09:10, 09:20
        assert out.loc["u2", "MAX_TX_PER_DAY"] == 4

    def test_single_transaction_gap_flagged(self):
        out = sequence_features(make_tx([("u1", "2018-01-01 09:00", "TOPUP", "COMPLETED", 10.0, None)]))
        assert out.loc["u1", "HAS_REPEAT_TX"] == 0
        assert out.loc["u1", "LOG_MEDIAN_GAP_H"] == 0

    def test_median_gap(self, tx):
        out = sequence_features(tx)
        # u1 gaps: 11h, 14h -> median 12.5h
        assert out.loc["u1", "LOG_MEDIAN_GAP_H"] == pytest.approx(np.log1p(12.5))
        assert out.loc["u1", "HAS_REPEAT_TX"] == 1

    def test_input_order_does_not_matter(self, tx):
        shuffled = tx.sample(frac=1, random_state=0)
        pd.testing.assert_frame_equal(sequence_features(shuffled), sequence_features(tx))

    def test_no_missing_values(self, tx):
        assert sequence_features(tx).isna().sum().sum() == 0
