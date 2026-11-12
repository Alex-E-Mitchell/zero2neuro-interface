"""Real browser checks: forms, JavaScript, validation, sessions and downloaded bytes."""
from pathlib import Path
from zipfile import ZipFile

import pytest
from playwright.sync_api import expect


def navigate(page, label):
    page.get_by_role('navigation', name='Configuration pages').get_by_role('link', name=label, exact=True).click()
    expect(page.get_by_role('navigation').get_by_role('link', name=label, exact=True)).to_have_attribute('aria-current', 'page')


def generate(page):
    page.get_by_role('button', name='Generate Configuration').click()


def fill_network(page, cnn=False):
    page.locator('#network_type').select_option('cnn' if cnn else 'fully_connected')
    page.locator('#input_shape').fill('28, 28, 1' if cnn else '2')
    page.locator('#hidden_units').fill('4')
    page.locator('#hidden_activation').select_option('tanh')
    page.locator('#output_shape').fill('1')
    page.locator('#output_activation').select_option('sigmoid')
    if cnn:
        expect(page.locator('#cnn-fields')).to_be_visible()
        page.locator('#conv_number_filters').fill('8')
        page.locator('#conv_kernel_size').fill('3')
        page.locator('#conv_pool_size').fill('2')
        page.locator('#conv_padding').select_option('same')


def fill_data(page):
    page.locator('#data_files').fill('xor_data.csv')
    page.locator('#data_inputs').fill('In 0\nIn 1')
    page.locator('#data_outputs').fill('Out 0')


def fill_experiment(page):
    page.locator('#experiment_name').fill('xor-demo')
    page.locator('#learning_rate').fill('0.001')
    page.locator('#epochs').fill('5000')


@pytest.mark.parametrize('cnn', [False, True], ids=['fully-connected', 'cnn'])
def test_forms_to_review_and_downloaded_zip(page, tmp_path, cnn):
    # CNN here checks serialization/export, not compatibility with the XOR CSV.
    fill_network(page, cnn)
    generate(page)
    expect(page.locator('#config-text')).to_be_visible()
    network = page.locator('#config-text').inner_text()
    assert '--network_type=' + ('cnn' if cnn else 'fully_connected') in network
    assert '--input_shape\n' + ('28\n28\n1' if cnn else '2') in network
    if cnn:
        assert '--conv_number_filters\n8' in network
    navigate(page, 'Data')
    fill_data(page)
    generate(page)
    expect(page.locator('#config-text')).to_contain_text('In 0\nIn 1')
    data = page.locator('#config-text').inner_text()
    navigate(page, 'Experiment')
    fill_experiment(page)
    generate(page)
    expect(page.locator('#config-text')).to_contain_text('--epochs=5000')
    experiment = page.locator('#config-text').inner_text()
    navigate(page, 'Review')
    expect(page.get_by_text('All three configurations are available for export.')).to_be_visible()
    expected = {'network.txt': network, 'data.txt': data, 'experiment.txt': experiment}
    if not cnn:
        reference = Path(__file__).resolve().parents[2] / 'examples/xor'
        for name, content in expected.items():
            assert content.strip() == (reference / name).read_text().strip()
    # Check individual downloads as well as the ZIP, not just the button labels.
    for name, content in expected.items():
        with page.expect_download() as event:
            page.get_by_role('link', name=f'Download {name}', exact=True).click()
        download = event.value
        assert download.suggested_filename == name
        assert Path(download.path()).read_text().strip() == content.strip()
    with page.expect_download() as event:
        page.get_by_role('link', name='Download All Configurations (.zip)').click()
    download = event.value
    assert download.suggested_filename == 'zero2neuro_config.zip'
    target = tmp_path / download.suggested_filename
    download.save_as(target)
    with ZipFile(target) as archive:
        assert set(archive.namelist()) == set(expected)
        for name, content in expected.items():
            assert archive.read(name).decode().strip() == content.strip()


def test_network_error_correction_removes_stale_download(page):
    fill_network(page)
    generate(page)
    expect(page.locator('#config-text')).to_be_visible()
    page.locator('#hidden_units').fill('4, nope')
    generate(page)
    expect(page.locator('#form-errors')).to_be_visible()
    expect(page.locator('#hidden_units')).to_have_value('4, nope')
    expect(page.locator('#input_shape')).to_have_value('2')
    expect(page.locator('#config-text')).to_have_count(0)
    # Read Review in a second tab without discarding the form being corrected.
    review = page.context.new_page()
    review.goto(page.url + 'review')
    expect(review.get_by_role('link', name='Download network.txt', exact=True)).to_have_count(0)
    expect(review.get_by_role('link', name='Download All Configurations (.zip)')).to_have_count(0)
    review.close()
    page.locator('#form-errors').get_by_role('link').first.click()
    expect(page.locator('#hidden_units')).to_be_focused()
    page.locator('#hidden_units').fill('4')
    generate(page)
    expect(page.locator('#form-errors')).to_have_count(0)
    expect(page.locator('#config-text')).to_be_visible()


def test_data_dynamic_fields_and_error_correction(page):
    navigate(page, 'Data')
    fill_data(page)
    page.locator('#data_fold_split').select_option('random')
    expect(page.locator('#data_n_folds')).to_be_visible()
    page.locator('#data_n_folds').fill('3')
    page.locator('#data_outputs').fill('-bad')
    generate(page)
    expect(page.locator('.error-message')).to_be_visible()
    expect(page.locator('#data_inputs')).to_have_value('In 0\nIn 1')
    expect(page.locator('#config-text')).to_have_count(0)
    page.locator('#data_outputs').fill('Out 0')
    generate(page)
    expect(page.locator('#config-text')).to_contain_text('--data_n_folds=3')
    expect(page.locator('.error-message')).to_have_count(0)


def test_experiment_early_stopping_and_error_correction(page):
    navigate(page, 'Experiment')
    fill_experiment(page)
    expect(page.locator('#early-stopping-fields')).to_be_hidden()
    page.locator('#early_stopping').check()
    expect(page.locator('#early-stopping-fields')).to_be_visible()
    page.locator('#metric_mae').check()
    page.locator('#metric_mse').check()
    page.locator('#epochs').fill('1.5')
    generate(page)
    expect(page.locator('#form-errors')).to_be_visible()
    expect(page.locator('#experiment_name')).to_have_value('xor-demo')
    expect(page.locator('#early_stopping')).to_be_checked()
    page.locator('#epochs').fill('50')
    generate(page)
    expect(page.locator('#config-text')).to_contain_text('--early_stopping')
    expect(page.locator('#config-text')).to_contain_text('--metrics\nmse\nmae')
    page.locator('#early_stopping').uncheck()
    generate(page)
    expect(page.locator('#config-text')).not_to_contain_text('--early_stopping')


def test_fresh_browser_session_cannot_download_another_users_config(page, new_context, app_url):
    fill_network(page)
    generate(page)
    navigate(page, 'Review')
    expect(page.get_by_role('link', name='Download network.txt', exact=True)).to_be_visible()
    other = new_context().new_page()
    other.goto(app_url + '/review')
    expect(other.get_by_text('Configuration is incomplete.', exact=True)).to_be_visible()
    expect(other.get_by_role('link', name='Download network.txt', exact=True)).to_have_count(0)
    assert other.request.get(app_url + '/review/download/network').status == 404


def test_unsent_drafts_return_and_clear_is_scoped_to_one_page(page):
    fill_network(page, cnn=True)
    page.locator('#batch_normalization').check()
    navigate(page, 'Data')
    fill_data(page)
    page.locator('#data_fold_split').select_option('random')
    page.locator('#data_n_folds').fill('3')
    navigate(page, 'Experiment')
    fill_experiment(page)
    page.locator('#early_stopping').check()
    page.locator('#metric_mae').check()
    navigate(page, 'Network')
    expect(page.locator('#network_type')).to_have_value('cnn')
    expect(page.locator('#cnn-fields')).to_be_visible()
    expect(page.locator('#conv_number_filters')).to_have_value('8')
    expect(page.locator('#batch_normalization')).to_be_checked()
    # Cancel preserves entries; accepting resets only Network's browser draft.
    page.once('dialog', lambda dialog: dialog.dismiss())
    page.get_by_role('button', name='Clear Inputs', exact=True).click()
    expect(page.locator('#input_shape')).to_have_value('28, 28, 1')
    page.once('dialog', lambda dialog: dialog.accept())
    page.get_by_role('button', name='Clear Inputs', exact=True).click()
    expect(page.locator('#input_shape')).to_have_value('')
    expect(page.locator('#network_type')).to_have_value('fully_connected')
    navigate(page, 'Data')
    expect(page.locator('#data_inputs')).to_have_value('In 0\nIn 1')
    expect(page.locator('#data_n_folds')).to_be_visible()
    expect(page.locator('#data_n_folds')).to_have_value('3')
    navigate(page, 'Experiment')
    expect(page.locator('#experiment_name')).to_have_value('xor-demo')
    expect(page.locator('#early_stopping')).to_be_checked()
    expect(page.locator('#early-stopping-fields')).to_be_visible()
    expect(page.locator('#metric_mae')).to_be_checked()
