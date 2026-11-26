import json, sys, unittest
sys.path.insert(0, ".")
from eva.skills.fresh import parse_moex_index, index_quote
from eva.skills.intent import detect

PAYLOAD = {
    "marketdata": {
        "columns": ["SECID", "CURRENTVALUE", "LASTCHANGEPRC", "LASTCHANGE", "UPDATETIME"],
        "data": [["IMOEX", 2333.84, -0.85, -20.1, "16:40:39"]],
    }
}

class IndexTests(unittest.TestCase):
    def test_parse_index(self):
        quote = parse_moex_index(json.dumps(PAYLOAD))
        self.assertEqual(quote.ticker, "IMOEX")
        self.assertAlmostEqual(quote.price, 2333.84, places=2)
        self.assertAlmostEqual(quote.change_pct, -0.85, places=2)

    def test_parse_index_empty(self):
        self.assertIsNone(parse_moex_index(b'{"marketdata":{"columns":[],"data":[]}}'))

    def test_index_intent_before_stock(self):
        for q in ("индекс мосбиржи", "какой индекс мосбиржи", "IMOEX", "индекс ртс"):
            self.assertEqual(detect(q).kind, "index", q)
        self.assertEqual(detect("индекс мосбиржи").param, "IMOEX")
        self.assertEqual(detect("индекс ртс").param, "RTSI")
        self.assertEqual(detect("цена акции Сбер").kind, "stock")

if __name__ == "__main__":
    unittest.main()
