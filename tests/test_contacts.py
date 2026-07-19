"""Contact standardizers: emails, phones (E.164), names, addresses."""

from cleankit import clean_address, standardize_email, standardize_name, to_e164


def test_phone_panama_local():
    r = to_e164("6123-4567", default_region="PA")
    assert r.value == "+50761234567"
    assert r.valid is True


def test_phone_panama_with_spaces_and_country_code():
    assert to_e164("+507 6123 4567").value == "+50761234567"
    assert to_e164("(507) 6222 3333").value == "+50762223333"


def test_phone_double_zero_prefix():
    r = to_e164("0050762001111", default_region="PA")
    assert r.value == "+50762001111"


def test_phone_us_number():
    r = to_e164("+1 305 555 0199")
    assert r.value == "+13055550199"
    assert r.valid is True


def test_phone_mexico_local_region():
    r = to_e164("55 1234 5678", default_region="MX")
    assert r.value == "+525512345678"
    assert r.valid is True


def test_phone_extension_preserved():
    r = to_e164("6123 4567 ext 22", default_region="PA")
    assert r.value == "+50761234567;ext=22"


def test_phone_empty():
    assert to_e164("").value is None
    assert to_e164(None).value is None


def test_email_lowercase_and_trim():
    r = standardize_email("  JUAN.PEREZ@GMAIL.COM ")
    assert r.value == "juan.perez@gmail.com"
    assert r.valid is True


def test_email_typo_fix():
    r = standardize_email("maria@gmial.com")
    assert r.value == "maria@gmail.com"
    assert r.note == "fixed domain typo"


def test_email_invalid():
    r = standardize_email("not-an-email")
    assert r.valid is False


def test_email_mailto_stripped():
    assert standardize_email("mailto:a@b.com").value == "a@b.com"


def test_name_casing():
    assert standardize_name("  JUAN   Pérez ") == "Juan Pérez"
    assert standardize_name("maría josé gonzález") == "María José González"


def test_name_particles_lowercased():
    assert standardize_name("JUAN DE LA CRUZ") == "Juan de la Cruz"


def test_name_mc_and_apostrophe():
    assert standardize_name("o'brien") == "O'Brien"
    assert standardize_name("mcdonald") == "McDonald"


def test_name_hyphenated():
    assert standardize_name("jean-paul sartre") == "Jean-Paul Sartre"


def test_clean_address_whitespace_and_abbrev():
    out = clean_address("  ave  balboa ,  edif torre  ")
    assert "Avenida" in out
    assert "Edificio" in out
    assert "  " not in out


def test_clean_address_empty():
    assert clean_address(None) is None
    assert clean_address("   ") is None
