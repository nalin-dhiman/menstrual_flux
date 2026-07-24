from streamlit.testing.v1 import AppTest


def test_gui_home_and_interactive_research_pages_load_without_exceptions():
    app = AppTest.from_file("app/streamlit_app.py", default_timeout=30).run()
    assert not app.exception
    assert app.radio[0].value == "✦  Observatory"

    app.radio[0].set_value("◌  Synthetic Cycle Lab").run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    assert len(app.dataframe) >= 1

    app.radio[0].set_value("↝  Flux & First Passage").run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception


def test_gui_data_quality_example_passes():
    app = AppTest.from_file("app/streamlit_app.py", default_timeout=30).run()
    app.radio[0].set_value("▦  Data Quality Studio").run()
    assert not app.exception
    assert len(app.success) == 1
