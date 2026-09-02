from __future__ import annotations

import unittest

from rynnvalue_vlm.analysis import anomaly_decision, parse_analysis


class RynnValueAnalysisTests(unittest.TestCase):
    def test_parses_released_analysis_layout(self) -> None:
        parsed = parse_analysis(
            "Analysis:\n"
            "- Video Description: The robot places the part in the fixture.\n"
            "- Match: Yes\n"
            "- Success: No"
        )
        self.assertEqual(
            parsed,
            {
                "description": "The robot places the part in the fixture.",
                "match": "Yes",
                "success": "No",
            },
        )

    def test_failure_if_either_native_check_is_no(self) -> None:
        self.assertEqual(anomaly_decision("No", "No"), "failure")
        self.assertEqual(anomaly_decision("Yes", "No"), "failure")

    def test_success_requires_both_native_checks(self) -> None:
        self.assertEqual(anomaly_decision("Yes", "Yes"), "success")
        self.assertEqual(anomaly_decision("Yes", None), "uncertain")
        self.assertEqual(anomaly_decision(None, None), "uncertain")


if __name__ == "__main__":
    unittest.main()
