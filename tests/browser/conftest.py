"""Start an isolated local app; require the real upstream parser for browser tests."""
import os
from pathlib import Path
import sys
from threading import Thread

import pytest
from werkzeug.serving import make_server

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope='session')
def app_url():
    from app import app
    from parser_validator import validate_config

    result = validate_config('--network_type=fully_connected\n--input_shape\n2\n')
    if result['status'] != 'valid':
        pytest.fail('Browser tests require NEURO_REPOSITORY_PATH pointing to a Zero2Neuro checkout: '
                    + result['message'])
    app.config.update(TESTING=True, SECRET_KEY=os.urandom(32))
    server = make_server('127.0.0.1', 0, app, threaded=True)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}'
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


@pytest.fixture(autouse=True)
def open_app(page, app_url):
    page.goto(app_url)
    yield
