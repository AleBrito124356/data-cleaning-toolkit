"""Contact standardizers as audited pipeline operations."""

import pandas as pd
import pytest

from cleankit import Cleaner, load_pipeline, region_for_country, run_pipeline
from cleankit.contacts import fix_email_domain, standardize_email, to_e164_bare_international


def _crm() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["  JUAN  pérez ", "o'brien smith", None],
            "email": ["JUAN.PEREZ@GMAIL.COM", "maria@gmial.con", "valentina.rojasgmail.com"],
            "phone": ["3105551234", "6123-4567", "12"],
            "country": ["Colombia", "Rep. de Panamá", "Panamá"],
            "address": ["Ave Balboa, Edif Torre", "cra 7 #45, bogota", None],
        }
    )


def test_region_for_country_variants():
    assert region_for_country("Rep. de Panamá") == "PA"
    assert region_for_country("méxico") == "MX"
    assert region_for_country("Columbia") == "CO"  # one typo
    assert region_for_country("EEUU") == "US"
    assert region_for_country("República Dominicana") == "DO"
    assert region_for_country("Gambia") is None
    assert region_for_country(None) is None


def test_standardize_phones_uses_each_rows_country():
    c = Cleaner(_crm()).standardize_phones(["phone"], default_region="PA", region_column="country")
    assert c.df["phone"].tolist() == ["+573105551234", "+50761234567", "12"]
    entry = c.log[-1]
    assert entry["regions"] == {"CO": 1, "PA": 1}
    assert entry["invalid"] == 1
    assert entry["invalid_examples"][0]["value"] == "12"


def test_standardize_phones_falls_back_and_logs_it():
    df = pd.DataFrame(
        {"phone": ["6777 8888", "507-6987-6543"], "country": ["Colombia", "Colombia"]}
    )
    c = Cleaner(df).standardize_phones("phone", default_region="PA", region_column="country")
    assert c.df["phone"].tolist() == ["+50767778888", "+50769876543"]
    fallbacks = c.log[-1]["fallbacks"]
    assert [f["row_region"] for f in fallbacks] == ["CO", "CO"]
    assert fallbacks[0]["via"] == "default region PA"


def test_standardize_phones_invalid_modes():
    df = pd.DataFrame({"phone": ["6123-4567", "12"]})
    nulled = Cleaner(df).standardize_phones(["phone"], invalid="null").df
    assert nulled["phone"].iloc[0] == "+50761234567" and pd.isna(nulled["phone"].iloc[1])
    flagged = Cleaner(df).standardize_phones(["phone"], invalid="flag").df
    assert flagged["phone_is_invalid"].tolist() == [False, True]
    assert flagged["phone"].iloc[1] == "12"


def test_standardize_phones_missing_region_column_is_reported():
    c = Cleaner(pd.DataFrame({"phone": ["6123-4567"]})).standardize_phones(
        ["phone"], region_column="pais"
    )
    assert "Region column 'pais' not found" in c.log[-1]["message"]


def test_standardize_emails_fixes_typos_and_reports_invalid():
    c = Cleaner(_crm()).standardize_emails(["email"])
    assert c.df["email"].tolist() == [
        "juan.perez@gmail.com",
        "maria@gmail.com",
        "valentina.rojasgmail.com",
    ]
    entry = c.log[-1]
    assert entry["typos_fixed"] == 1
    assert entry["invalid"] == 1
    assert entry["invalid_examples"][0]["suggestion"] == "valentina.rojas@gmail.com"


def test_standardize_emails_invalid_null_and_flag():
    nulled = Cleaner(_crm()).standardize_emails("email", invalid="null").df
    assert pd.isna(nulled["email"].iloc[2])
    flagged = Cleaner(_crm()).standardize_emails("email", invalid="flag").df
    assert flagged["email_is_invalid"].tolist() == [False, False, True]


def test_email_domain_fixes_are_conservative():
    assert fix_email_domain("gmial.con") == "gmail.com"
    assert fix_email_domain("hotmail.co") == "hotmail.com"
    assert fix_email_domain("ymail.com") == "ymail.com"  # real Yahoo domain
    assert fix_email_domain("company.co") == "company.co"
    assert standardize_email("a@b..com").valid is False


def test_standardize_names_and_addresses_ops():
    c = Cleaner(_crm()).normalize_whitespace().standardize_names("name").clean_addresses(["address"])
    assert c.df["name"].tolist()[:2] == ["Juan Pérez", "O'Brien Smith"]
    assert c.df["address"].tolist()[:2] == ["Avenida Balboa, Edificio Torre", "Carrera 7 #45, Bogota"]
    ops = [e["op"] for e in c.log]
    assert ops == ["normalize_whitespace", "standardize_names", "clean_addresses"]
    assert c.log[1]["cells_changed"] == 2


def test_contact_ops_require_columns_and_skip_missing_ones():
    with pytest.raises(ValueError, match="columns"):
        Cleaner(_crm()).standardize_emails([])
    c = Cleaner(_crm()).standardize_names(["nombre"])
    assert c.log[-1]["skipped"] is True


def test_bare_international_reading():
    r = to_e164_bare_international("507-6987-6543")
    assert r is not None and r.value == "+50769876543" and r.region == "PA"
    assert to_e164_bare_international("6123-4567") is None


def test_contact_ops_in_a_pipeline(tmp_path):
    steps = tmp_path / "steps.yaml"
    steps.write_text(
        "name: contacts\n"
        "steps:\n"
        "  - op: standardize_emails\n    columns: [email]\n"
        "  - op: standardize_phones\n    columns: [phone]\n    default_region: PA\n"
        "    region_column: country\n"
        "  - op: standardize_names\n    columns: [name]\n"
        "  - op: clean_addresses\n    columns: address\n",
        encoding="utf-8",
    )
    result = run_pipeline(_crm(), load_pipeline(steps))
    assert result.df["phone"].iloc[0] == "+573105551234"
    assert [e["op"] for e in result.audit_log] == [
        "standardize_emails",
        "standardize_phones",
        "standardize_names",
        "clean_addresses",
    ]


def test_pipeline_rejects_bad_contact_options(tmp_path):
    steps = tmp_path / "steps.yaml"
    steps.write_text(
        "steps:\n  - op: standardize_phones\n    columns: [phone]\n    invalid: drop\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid='drop'"):
        load_pipeline(steps)
