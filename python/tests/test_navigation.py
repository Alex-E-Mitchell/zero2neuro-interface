import sys
import unittest
from html.parser import HTMLParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app import app


class NavigationParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_navigation = False
        self.links = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'nav' and attrs.get('aria-label') == 'Configuration pages':
            self.in_navigation = True
        if tag == 'a' and self.in_navigation:
            self.links.append(attrs)

    def handle_endtag(self, tag):
        if tag == 'nav':
            self.in_navigation = False


class TestSharedNavigation(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_every_page_links_to_all_pages_and_marks_only_itself_current(self):
        for path in ('/', '/data', '/experiment'):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                parser = NavigationParser()
                parser.feed(response.get_data(as_text=True))
                self.assertEqual([link['href'] for link in parser.links], ['/', '/data', '/experiment'])
                self.assertEqual([link['href'] for link in parser.links
                                  if link.get('aria-current') == 'page'], [path])
                for link in parser.links:
                    self.assertEqual(self.client.get(link['href']).status_code, 200)

    def test_placeholders_explain_unavailable_functionality(self):
        for path, title in (('/experiment', 'Experiment'),):
            with self.subTest(path=path):
                page = self.client.get(path).get_data(as_text=True)
                self.assertIn(title + ' Configuration', page)
                self.assertIn('This configuration page is in progress.', page)
                self.assertIn('does not collect settings or generate configuration text yet', page)
                self.assertNotIn('<form', page)
                self.assertEqual(self.client.post(path, data={}).status_code, 405)

    def test_network_submission_keeps_shared_navigation(self):
        response = self.client.post('/', data={
            'input_shape': '34', 'hidden_units': '64,32',
            'hidden_activation': 'elu', 'output_shape': '1',
            'output_activation': 'linear',
        })
        page = response.get_data(as_text=True)
        self.assertIn('--network_type=fully_connected', page)
        parser = NavigationParser()
        parser.feed(page)
        self.assertEqual(len(parser.links), 3)
        self.assertEqual(parser.links[0].get('aria-current'), 'page')
