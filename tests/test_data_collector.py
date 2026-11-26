"""Tests for CSV parsing, validation and deduplication."""
from datetime import date

import database as db
from data_collector import deduplicate, discover_csv_urls, parse_csv_text, validate_draw, year_from_filename
from database import Draw

YEARLY = (
    "Datum;Reihenfolge;Zahl1;Zahl2;Zahl3;Zahl4;Zahl5;Zahl6;ZZ;Zusatzzahl;Rang_1_5\n"
    "04.01.;aufsteigend;1;4;15;16;22;38;ZZ;11;6er;1;à;1.200.000,00\n"
    "04.01.;;;;;;;;;;4er;3.027;à;62,70;3er + ZZ;4.852;à;15,60;ZZ;173.512\n"
    "07.01.;aufsteigend;4;5;16;21;35;45;ZZ;33;6er;JP\n"
)
ARCHIVE_2010 = (
    ";;;;\n2010 Lotto - Beträge in EUR;;;;\n"
    "Mi;08.9.;aufsteigend;4;30;31;32;34;38;Zz;33;6er;JP\n"
    ";;gezogen;32;30;38;4;34;31;Zz;33;4er\n"
)
ARCHIVE_1986 = (
    "1986 Lotto - Beträge in ATS;;;\n"
    "So.;07.09.;1;20;22;24;27;40;Zz:;12;1;à;6.542.159,00;27;20;22;1;24;40;Zz:;12\n"
)


def test_yearly_layout():
    res = parse_csv_text(YEARLY, "NN_W2D_STAT_Lotto_2026.csv", 2026)
    assert [d.draw_date for d in res.draws] == [date(2026, 1, 4), date(2026, 1, 7)]
    assert res.draws[0].numbers == (1, 4, 15, 16, 22, 38) and res.draws[0].additional_number == 11
    assert res.issues == []


def test_archive_layouts_use_header_year():
    r1 = parse_csv_text(ARCHIVE_2010, "a.csv")
    assert r1.draws[0].draw_date == date(2010, 9, 8) and r1.draws[0].additional_number == 33
    assert len(r1.draws) == 1
    r2 = parse_csv_text(ARCHIVE_1986, "b.csv")
    assert r2.draws[0] == Draw(date(1986, 9, 7), (1, 20, 22, 24, 27, 40), 12, "b.csv")


def test_invalid_rows_logged_not_guessed():
    bad = "01.01.;aufsteigend;1;1;15;16;22;38;ZZ;11\n02.01.;aufsteigend;1;2;3;4;5;46;ZZ;7\n"
    res = parse_csv_text(bad, "x.csv", 2020)
    assert res.draws == [] and len(res.issues) == 2


def test_missing_year_logged():
    res = parse_csv_text("01.01.;aufsteigend;1;2;3;4;5;6;ZZ;7\n", "x.csv")
    assert res.draws == [] and "year" in res.issues[0]


def test_weekday_mismatch_flagged():
    res = parse_csv_text("2010 Lotto;\nSo;08.9.;aufsteigend;4;30;31;32;34;38;Zz;33\n", "x.csv")
    assert any("weekday" in i for i in res.issues)


def test_validate_draw():
    assert validate_draw([1, 2, 3, 4, 5, 6], 7) == []
    assert validate_draw([1, 2, 3, 4, 5], 7)
    assert validate_draw([1, 2, 3, 4, 5, 6], 6)
    assert validate_draw([0, 2, 3, 4, 5, 6], 7)
    assert validate_draw([1, 2, 3, 4, 5, 6], None)


def test_deduplicate():
    a = Draw(date(2020, 1, 1), (1, 2, 3, 4, 5, 6), 7, "a")
    b = Draw(date(2020, 1, 1), (1, 2, 3, 4, 5, 6), 7, "b")
    c1 = Draw(date(2020, 1, 5), (1, 2, 3, 4, 5, 6), 7, "a")
    c2 = Draw(date(2020, 1, 5), (1, 2, 3, 4, 5, 9), 7, "b")
    unique, issues = deduplicate([a, b, c1, c2])
    assert unique == [a] and len(issues) == 2


def test_db_insert_idempotent_and_conflicts():
    conn = db.connect(":memory:")
    d = Draw(date(2020, 1, 1), (1, 2, 3, 4, 5, 6), 7, "a")
    assert db.insert_draws(conn, [d])["inserted"] == 1
    assert db.insert_draws(conn, [d])["existing"] == 1
    assert db.insert_draws(conn, [Draw(d.draw_date, (1, 2, 3, 4, 5, 8), 7)])["conflicts"] == 1
    assert db.count_draws(conn) == 1


def test_discover_and_year():
    html = ('<a href="https://statics.win2day.at/media/NN_W2D_STAT_Lotto_2024.csv?v=1">x</a>'
            '<a href="/media/1986-2010-Lotto.csv">y</a><a href="/other.pdf">z</a>')
    urls = discover_csv_urls(html, "https://www.win2day.at/page")
    assert len(urls) == 2
    assert year_from_filename(urls[0]) == 2024 and year_from_filename(urls[1]) is None
