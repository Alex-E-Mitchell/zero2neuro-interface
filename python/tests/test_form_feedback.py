import sys
import unittest
from html.parser import HTMLParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app import app


class FormPage(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.elements = {}
        self.selected = {}
        self.links = []
        self.select = None
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.elements[attrs['id']] = attrs
        if tag == 'select':
            self.select = attrs['id']
        if tag == 'option' and 'selected' in attrs:
            self.selected[self.select] = attrs.get('value', '')
        if tag == 'a':
            self.links.append(attrs.get('href'))

    def handle_endtag(self, tag):
        if tag == 'select':
            self.select = None


class TestFormFeedback(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.fields = {
            'network_type': 'fully_connected', 'input_shape': ' 128, 128, 3 ',
            'hidden_units': '64, 32', 'hidden_activation': 'tanh',
            'output_shape': '10, 20', 'output_activation': 'softmax',
            'batch_normalization': 'on',
        }

    def post(self, fields):
        response = self.client.post('/', data=fields)
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        return html, FormPage(html)

    def test_preserves_entries_and_checkbox_states_after_success_or_error(self):
        for hidden in ('64, 32', '64, nope'):
            for checked in (False, True):
                with self.subTest(hidden=hidden, checked=checked):
                    fields = dict(self.fields, hidden_units=hidden)
                    if checked:
                        fields['batch_normalization_input'] = 'on'
                    html, page = self.post(fields)
                    for name in ('input_shape', 'hidden_units', 'output_shape'):
                        self.assertEqual(page.elements[name]['value'], fields[name])
                    for name in ('network_type', 'hidden_activation', 'output_activation'):
                        self.assertEqual(page.selected[name], fields[name])
                    self.assertIn('checked', page.elements['batch_normalization'])
                    self.assertEqual('checked' in page.elements['batch_normalization_input'], checked)
                    self.assertEqual('config-text' in page.elements, hidden == '64, 32')

    def test_reports_all_invalid_fields_with_accessible_links(self):
        html, page = self.post(dict(self.fields, input_shape='0', hidden_units='1.5', output_shape=''))
        self.assertEqual(page.elements['form-errors']['role'], 'alert')
        self.assertEqual(page.elements['form-errors']['tabindex'], '-1')
        for name in ('input_shape', 'hidden_units', 'output_shape'):
            self.assertEqual(page.elements[name]['aria-invalid'], 'true')
            error_id = page.elements[name]['aria-describedby']
            self.assertIn(error_id, page.elements)
            self.assertIn('#' + name, page.links)
        self.assertNotIn('config-text', page.elements)

    def test_missing_required_fields_have_helpful_feedback(self):
        html, page = self.post({})
        for name in ('input_shape', 'output_shape', 'hidden_activation', 'output_activation'):
            self.assertIn(name + '-error', page.elements)
        self.assertNotIn('Bad Request', html)
        self.assertNotIn('hidden_units-error', page.elements)

    def test_unsupported_dropdown_values_cannot_generate_configuration(self):
        for name in ('network_type', 'hidden_activation', 'output_activation'):
            with self.subTest(name=name):
                html, page = self.post(dict(self.fields, **{name: 'invalid\n--extra_option'}))
                self.assertIn(name + '-error', page.elements)
                self.assertEqual(page.selected[name], '')
                self.assertNotIn('config-text', page.elements)
                self.assertEqual(page.elements['input_shape']['value'], self.fields['input_shape'])

    def test_correction_removes_errors_and_generates_the_selected_settings(self):
        self.post(dict(self.fields, input_shape='-1'))
        html, page = self.post(self.fields)
        self.assertNotIn('form-errors', page.elements)
        self.assertNotIn('aria-invalid', page.elements['input_shape'])
        for setting in ('--hidden_activation=tanh', '--output_activation=softmax', '--batch_normalization'):
            self.assertIn(setting, html)
        self.assertNotIn('--batch_normalization_input', html)

    def test_repopulated_input_is_escaped(self):
        payload = '\"><script>alert(1)</script>'
        html, page = self.post(dict(self.fields, input_shape=payload))
        self.assertEqual(page.elements['input_shape']['value'], payload)
        self.assertNotIn(payload, html)
        self.assertIn('&lt;script&gt;', html)
        self.assertNotIn('config-text', page.elements)

    def test_fresh_visit_has_defaults_without_another_requests_entries(self):
        self.post(self.fields)
        page = FormPage(self.client.get('/').get_data(as_text=True))
        self.assertEqual(page.elements['input_shape']['value'], '')
        self.assertEqual(page.selected['hidden_activation'], 'elu')
        self.assertEqual(page.selected['output_activation'], 'elup1')
        self.assertNotIn('checked', page.elements['batch_normalization'])
        self.assertNotIn('form-errors', page.elements)
