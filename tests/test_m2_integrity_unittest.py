import json
import math
from pathlib import Path
import queue
import re
import unittest
from unittest import mock

import atlas_based_patient_generation as m2


def _schema2_time_model(underlying_duration, current_episode_duration, symptom_timeline):
    return {
        "schema_version": 2,
        "underlying_duration": underlying_duration,
        "current_episode_duration": current_episode_duration,
        "symptom_timeline": symptom_timeline,
    }


def _schema2_time_model_response(underlying_duration, current_episode_duration, symptom_timeline):
    return json.dumps(
        _schema2_time_model(underlying_duration, current_episode_duration, symptom_timeline),
        ensure_ascii=False,
    )


def _complete_unvisited_row(prior_count="0"):
    row = m2._empty_row()
    row[m2.COL_GENDER] = "男"
    selected = [[("咳嗽", "轻度", "2天", "", "")], [], [], [], []]
    absent = [[], [], [], [], []]
    for column, value in zip((
        m2.COL_SYMPTOMS, m2.COL_SIGNS, m2.COL_LAB_TESTS,
        m2.COL_IMAGING, m2.COL_FUNCTIONAL_TESTS,
    ), selected):
        row[column] = repr(value)
    for column, value in zip((
        m2.COL_ABSENT_SYMPTOMS, m2.COL_ABSENT_SIGNS,
        m2.COL_ABSENT_LAB_TESTS, m2.COL_ABSENT_IMAGING,
        m2.COL_ABSENT_FUNCTIONAL,
    ), absent):
        row[column] = repr(value)
    row[m2.COL_M24_OUTPUT] = json.dumps({
        "schema_version": 2,
        "status": "converged",
        "rounds": [],
        "final_hash": m2._phenotype_state_hash_from_lists(selected, absent),
    })
    timeline = [("咳嗽", "症状", "起病", "D-2")]
    row[m2.COL_M25_OUTPUT] = _schema2_time_model_response(
        None, "3天", [
            {"name": "咳嗽", "category": "症状", "phase": "起病", "time_label": "D-2"},
        ]
    )
    row[m2.COL_TIME_ORDER] = repr(timeline)
    row[m2.COL_DURATION_TOTAL] = "3天"
    row[m2.COL_ACUITY] = "急性"
    row[m2.COL_QUANTIFIED] = repr(timeline)
    row[m2.COL_SPECIFIC] = repr(timeline)
    row[m2.COL_CHIEF_COMPLAINT] = "咳嗽"
    row[m2.COL_PRIOR_VISITED] = "未就诊"
    row[m2.COL_PRIOR_VISIT_COUNT] = prior_count
    return row


class M2IntegrityTests(unittest.TestCase):
    def test_resume_rejects_legacy_time_model_sidecar_and_timeline(self):
        row = _complete_unvisited_row()
        selected_symptoms = [("咳嗽", "轻度", "2天", "", "")]
        row[m2.COL_M25_OUTPUT] = "[('咳嗽','症状','起病','D-2')]\n患病总时长=3天"
        row[m2.COL_TIME_ORDER] = repr([("咳嗽", "症状", "起病", "D-2")])
        row[m2.COL_DURATION_TOTAL] = "3天"

        self.assertIsNone(m2._validated_existing_time_model(row, selected_symptoms))
        self.assertFalse(m2._is_m2_row_complete(row))

    def test_completion_rejects_duration_total_mismatch_with_time_model_sidecar(self):
        row = _complete_unvisited_row()
        row[m2.COL_DURATION_TOTAL] = "99天"

        self.assertFalse(m2._is_m2_row_complete(row))

    def test_completion_rejects_symptom_projection_mismatch_with_time_model_sidecar(self):
        row = _complete_unvisited_row()
        projected = [("咳嗽", "症状", "就诊", "D-2")]
        row[m2.COL_TIME_ORDER] = repr(projected)
        row[m2.COL_QUANTIFIED] = repr(projected)
        row[m2.COL_SPECIFIC] = repr(projected)

        self.assertFalse(m2._is_m2_row_complete(row))

    def test_module_25_accepts_decimal_day_offsets(self):
        response = _schema2_time_model_response(None, "3天", [
            {"name": "胸痛", "category": "症状", "phase": "起病", "time_label": "D-2"},
            {"name": "大汗", "category": "症状", "phase": "起病", "time_label": "D-0.25"},
            {"name": "濒死感", "category": "症状", "phase": "就诊", "time_label": "D0"},
        ])
        with mock.patch.object(m2, "call_gpt5", return_value=response):
            result = m2.module_2_5_build_timeline(
                "急性心肌梗死 # 50-80 0.65 急性",
                "I级",
                [
                    ("胸痛", "I级", "2天", "活动", "压榨"),
                    ("大汗", "I级", "数小时", "", ""),
                    ("濒死感", "I级", "当前", "", ""),
                ],
                [], [], [], [],
            )

        self.assertIsNotNone(result)
        self.assertEqual(result[2]["symptom_timeline"][1], ("大汗", "症状", "起病", "D-0.25"))
        realized = m2._realize_symptom_durations(
            [("大汗", "I级", "", "", "")],
            [("大汗", "症状", "起病", "D-0.25")],
        )
        self.assertEqual(realized[0][2], "0.25天")

    def test_module_25_accepts_decimal_hour_offsets(self):
        items = [("胸痛", "症状", "起病", "H-4.5"), ("大汗", "症状", "就诊", "H0")]

        self.assertEqual(
            m2._validate_symptom_timeline(items, ["胸痛", "大汗"], "6小时", "急性"),
            items,
        )

    def test_quantification_only_targets_atomic_continuous_measurements(self):
        self.assertFalse(m2._is_obviously_quantifiable(
            ("血压升高(>140/90mmHg)", "体征", "就诊", "D0")
        ))
        self.assertFalse(m2._is_obviously_quantifiable(
            ("呼吸浅快（潮气量降低且呼吸频率>20次/分）", "体征", "就诊", "D0")
        ))
        self.assertFalse(m2._is_obviously_quantifiable(
            ("C反应蛋白及降钙素原升高", "实验室检查", "就诊", "D0")
        ))
        self.assertFalse(m2._is_obviously_quantifiable(
            ("白细胞计数/血小板计数降低", "实验室检查", "就诊", "D0")
        ))
        self.assertFalse(m2._is_obviously_quantifiable(
            ("白细胞计数 ／ 血小板计数降低", "实验室检查", "就诊", "D0")
        ))
        self.assertFalse(m2._is_obviously_quantifiable(
            ("股动脉搏动减弱", "体征", "就诊", "D0")
        ))
        self.assertFalse(m2._is_obviously_quantifiable(
            ("心率不齐（听诊R-R间期不等）", "体征", "就诊", "D0")
        ))
        self.assertFalse(m2._is_obviously_quantifiable(
            ("脉搏短绌（心尖率高于桡动脉率>10次/分）", "体征", "就诊", "D0")
        ))
        self.assertFalse(m2._is_obviously_quantifiable(
            ("奔马律（心率>100次/分时额外心音形成三音律）", "体征", "就诊", "D0")
        ))
        self.assertFalse(m2._is_obviously_quantifiable(
            ("头颅CT示急性颅内出血高密度灶", "影像检查", "就诊", "D0")
        ))
        self.assertFalse(m2._is_obviously_quantifiable(
            ("压力-流率测定示膀胱出口梗阻", "功能检查", "就诊", "D0")
        ))
        self.assertFalse(m2._is_obviously_quantifiable(
            ("Barthel指数降低", "功能检查", "就诊", "D0")
        ))
        self.assertTrue(m2._is_obviously_quantifiable(
            ("心率增快", "体征", "就诊", "D0")
        ))
        self.assertTrue(m2._is_obviously_quantifiable(
            ("心动过速（心率>100次/分）", "体征", "就诊", "D0")
        ))
        self.assertTrue(m2._is_obviously_quantifiable(
            ("高热（>39.0℃）", "体征", "就诊", "D0")
        ))
        self.assertTrue(m2._is_obviously_quantifiable(
            ("低体温（<36.0℃）", "体征", "就诊", "D0")
        ))
        self.assertTrue(m2._is_obviously_quantifiable(
            ("血氧饱和度降低", "体征", "就诊", "D0")
        ))
        self.assertTrue(m2._is_obviously_quantifiable(
            ("白细胞计数升高", "实验室检查", "就诊", "D0")
        ))
        self.assertTrue(m2._is_obviously_quantifiable(
            ("中性粒细胞/淋巴细胞比值升高", "实验室检查", "就诊", "D0")
        ))
        self.assertTrue(m2._is_obviously_quantifiable(
            ("冠状动脉CTA示冠状动脉管腔狭窄≥50%", "影像检查", "就诊", "D0")
        ))

    def test_quantified_range_must_respect_explicit_directional_threshold(self):
        with self.assertRaisesRegex(ValueError, "阈值"):
            m2._validate_quantified_name(
                "体温升高（≥38.0℃）",
                "体温升高（≥38.0℃）{37.5,0.5,36.0,39.0}℃",
                "体征",
            )
        with self.assertRaisesRegex(ValueError, "阈值"):
            m2._validate_quantified_name(
                "左室射血分数降低（<40%）",
                "左室射血分数降低（<40%）{35,8,20,55}%",
                "功能检查",
            )

    def test_quantified_range_accepts_distribution_inside_explicit_threshold(self):
        self.assertEqual(
            m2._validate_quantified_name(
                "体温升高（≥38.0℃）",
                "体温升高（≥38.0℃）{39,0.5,38,41}℃",
                "体征",
            ),
            "体温升高（≥38.0℃）{39,0.5,38,41}℃",
        )
        self.assertEqual(
            m2._validate_quantified_name(
                "左室射血分数降低（<40%）",
                "左室射血分数降低（<40%）{30,5,15,39}%",
                "功能检查",
            ),
            "左室射血分数降低（<40%）{30,5,15,39}%",
        )
        self.assertEqual(
            m2._validate_quantified_name(
                "白细胞计数>=10×10^9/L",
                "白细胞计数>=10×10^9/L{14,2,10,20}×10^9/L",
                "实验室检查",
            ),
            "白细胞计数>=10×10^9/L{14,2,10,20}×10^9/L",
        )

    def test_quantified_range_supports_full_width_threshold_and_preserves_unit(self):
        with self.assertRaisesRegex(ValueError, "阈值"):
            m2._validate_quantified_name(
                "体温升高（＞38℃）",
                "体温升高（＞38℃）{37,0.2,36.5,37.5}℃",
                "体征",
            )
        with self.assertRaisesRegex(ValueError, "单位"):
            m2._validate_quantified_name(
                "C反应蛋白>10mg/L",
                "C反应蛋白>10mg/L{15,2,10.1,30}kg",
                "实验室检查",
            )
        self.assertEqual(
            m2._validate_quantified_name(
                "白细胞计数升高（>10×10^9/L）",
                "白细胞计数升高（>10×10^9/L）{14,2,10.1,20}10^9/L",
                "实验室检查",
            ),
            "白细胞计数升高（>10×10^9/L）{14,2,10.1,20}×10^9/L",
        )
        self.assertEqual(
            m2._validate_quantified_name(
                "体温升高（>38°C）",
                "体温升高（>38°C）{39,0,38.1,41}℃",
                "体征",
            ),
            "体温升高（>38°C）{39,0,38.1,41}℃",
        )

    def test_agatston_unit_aliases_match_au_threshold_unit(self):
        source_names = [
            "冠状动脉钙化积分>400AU",
            "冠状动脉钙化积分>400Agatston单位",
            "冠状动脉钙化积分>400Agatston unit",
            "冠状动脉钙化积分>400Agatston units",
            "冠状动脉钙化积分>400agatston Unit",
            "冠状动脉钙化积分>400Agatston  units",
        ]
        output_suffixes = [
            "AU",
            "Agatston单位",
            "Agatston unit",
            "Agatston units",
            "agatston unit",
            "Agatston  units",
        ]

        for source_name in source_names:
            with self.subTest(source_name=source_name):
                self.assertEqual(
                    m2._canonical_explicit_threshold_unit(
                        m2._EXPLICIT_THRESHOLD_RE.search(source_name).group("unit"),
                        source_name,
                    ),
                    "agatston_unit",
                )
            for suffix in output_suffixes:
                quantified_name = f"{source_name}{{500,20,401,800}}{suffix}"
                with self.subTest(source_name=source_name, suffix=suffix):
                    self.assertEqual(
                        m2._validate_quantified_name(source_name, quantified_name, "影像检查"),
                        quantified_name,
                    )

        with self.assertRaisesRegex(ValueError, "单位"):
            m2._validate_quantified_name(
                "吸光度>10AU",
                "吸光度>10AU{12,1,11,20}Agatston unit",
                "实验室检查",
            )
        with self.assertRaisesRegex(ValueError, "单位"):
            m2._validate_quantified_name(
                "C反应蛋白>10mg/L", "C反应蛋白>10mg/L{15,2,10.1,30}kg", "实验室检查"
            )
        with self.assertRaisesRegex(ValueError, "单位"):
            m2._validate_quantified_name(
                "血钠>145mmol/L", "血钠>145mmol/L{150,2,146,160}mg/dL", "实验室检查"
            )

    def test_agatston_threshold_fallback_uses_canonical_au_suffix(self):
        for source_name in (
                "冠状动脉钙化积分>400AU",
                "冠状动脉钙化积分>400Agatston单位",
                "冠状动脉钙化积分>400Agatston unit",
                "冠状动脉钙化积分>400Agatston units",
                "冠状动脉钙化积分>400Agatston  units"):
            with self.subTest(source_name=source_name):
                result = m2._fallback_quantified_name_for_threshold(
                    (source_name, "影像检查", "就诊", "D0")
                )

                self.assertIsNotNone(result)
                _, values, suffix = m2._parse_quantified_name_parts(source_name, result)
                self.assertEqual(suffix, "AU")
                self.assertGreater(values[2], 400)
                self.assertEqual(
                    m2._validate_quantified_name(source_name, result, "影像检查"),
                    result,
                )

    def test_decimal_ratio_threshold_normalizes_spurious_percent_suffix(self):
        self.assertEqual(
            m2._validate_quantified_name(
                "肺功能示FEV1/FVC<0.70",
                "肺功能示FEV1/FVC<0.70{0.68,0.02,0.3,0.69}%",
                "功能检查",
            ),
            "肺功能示FEV1/FVC<0.70{0.68,0.02,0.3,0.69}",
        )

    def test_decimal_ratio_fallback_does_not_invent_percent_unit(self):
        result = m2._fallback_quantified_name_for_threshold(
            ("指标比例<0.7", "实验室检查", "就诊", "D0")
        )

        self.assertIsNotNone(result)
        self.assertFalse(result.endswith("%"))
        _, values, suffix = m2._parse_quantified_name_parts("指标比例<0.7", result)
        self.assertEqual(suffix, "")
        self.assertGreaterEqual(values[2], 0.0)
        self.assertLessEqual(values[3], 1.0)
        self.assertEqual(
            m2._validate_quantified_name("指标比例<0.7", result, "实验室检查"),
            result,
        )

        greater = m2._fallback_quantified_name_for_threshold(
            ("指标比例>0.7", "实验室检查", "就诊", "D0")
        )
        _, greater_values, greater_suffix = m2._parse_quantified_name_parts(
            "指标比例>0.7", greater
        )
        self.assertEqual(greater_suffix, "")
        self.assertGreater(greater_values[2], 0.7)
        self.assertLessEqual(greater_values[3], 1.0)

    def test_two_sided_temperature_fallback_preserves_full_interval(self):
        source = "低热（37.5℃≤体温<38.3℃）"

        result = m2._fallback_quantified_name_for_threshold(
            (source, "体征", "就诊", "D0")
        )

        self.assertIsNotNone(result)
        _, values, suffix = m2._parse_quantified_name_parts(source, result)
        self.assertEqual(suffix, "℃")
        self.assertGreaterEqual(values[2], 37.5)
        self.assertLess(values[3], 38.3)

    def test_blood_ph_direction_fallback_is_quantifiable(self):
        source = "血pH降低"

        result = m2._fallback_quantified_name_for_threshold(
            (source, "实验室检查", "就诊", "D0")
        )

        self.assertIsNotNone(result)
        _, values, suffix = m2._parse_quantified_name_parts(source, result)
        self.assertEqual(suffix, "")
        self.assertLessEqual(values[3], 7.35)

    def test_blood_acidity_direction_aliases_use_ph_boundary(self):
        for source in ("血液酸碱度降低", "酸碱度降低"):
            with self.subTest(source=source):
                result = m2._fallback_quantified_name_for_threshold(
                    (source, "实验室检查", "就诊", "D0")
                )

                self.assertIsNotNone(result)
                _, values, suffix = m2._parse_quantified_name_parts(
                    source, result
                )
                self.assertEqual(suffix, "")
                self.assertLessEqual(values[3], 7.35)

    def test_unparseable_compound_quantification_never_drops_phenotype(self):
        source = [("血压{120/80,10,90,160}mmHg", "体征", "就诊", "D0")]

        with self.assertRaisesRegex(ValueError, "无法解析"):
            m2._postprocess_compound_vitals(source)


    def test_module_27_repairs_explicit_threshold_boundaries_without_dropping_phenotypes(self):
        source = [
            ("体温升高（体温>38.0℃）", "体征", "就诊", "D0"),
            ("肺功能示FEV1/FVC<0.70", "功能检查", "就诊", "D0"),
            ("胸痛", "症状", "起病", "D-1"),
        ]
        response = json.dumps({
            "schema_version": 1,
            "updates": [
                {
                    "source_index": 0,
                    "quantified_name": "体温升高（体温>38.0℃）{38.4,0.5,37.5,40.0}℃",
                },
                {
                    "source_index": 1,
                    "quantified_name": "肺功能示FEV1/FVC<0.70{0.62,0.08,0.30,0.80}",
                },
            ],
        }, ensure_ascii=False)

        with mock.patch.object(m2, "call_gpt5", return_value=response):
            result = m2.module_2_7_quantify_values(
                "慢阻肺 # 40-80 0.5 急性", "轻度", repr(source)
            )

        self.assertIsNotNone(result)
        repaired = result[2]
        self.assertEqual(len(repaired), 3)
        self.assertEqual(repaired[2], source[2])
        self.assertRegex(repaired[0][0], r"体温升高（体温>38\.0℃）\{38\.4,0\.5,38\.1,40\}℃")
        self.assertRegex(repaired[1][0], r"肺功能示FEV1/FVC<0\.70\{0\.62,0\.08,0\.3,0\.69\}")

    def test_module_27_converts_percent_scale_when_ratio_threshold_is_decimal(self):
        source = [("肺功能示FEV1/FVC<0.70", "功能检查", "就诊", "D0")]
        response = json.dumps({
            "schema_version": 1,
            "updates": [{
                "source_index": 0,
                "quantified_name": "肺功能示FEV1/FVC<0.70{62,8,30,80}%",
            }],
        }, ensure_ascii=False)

        with mock.patch.object(m2, "call_gpt5", return_value=response):
            result = m2.module_2_7_quantify_values(
                "慢阻肺 # 40-80 0.5 急性", "轻度", repr(source)
            )

        self.assertIsNotNone(result)
        self.assertEqual(result[2][0][0], "肺功能示FEV1/FVC<0.70{0.62,0.08,0.3,0.69}")

    def test_module_27_harmonizes_alias_ranges_before_returning(self):
        source = [
            ("血氧饱和度显著下降（SpO2<90%）", "体征", "就诊", "D0"),
            ("血氧饱和度下降（SpO2<95%）", "体征", "就诊", "D0"),
        ]
        response = json.dumps({
            "schema_version": 1,
            "updates": [
                {"source_index": 0, "quantified_name": f"{source[0][0]}{{84,3,70,89}}%"},
                {"source_index": 1, "quantified_name": f"{source[1][0]}{{91,2,80,94}}%"},
            ],
        }, ensure_ascii=False)

        with mock.patch.object(m2, "call_gpt5", return_value=response):
            result = m2.module_2_7_quantify_values(
                "慢阻肺 # 40-80 0.5 慢性急性加重", "重度", repr(source),
                absent_by_category={},
            )

        self.assertIsNotNone(result)
        ranges = [m2._parse_quantified_name_parts(src[0], out[0])[1][2:]
                  for src, out in zip(source, result[2])]
        self.assertEqual(ranges[0], ranges[1])

    def test_module_28_separates_threshold_text_from_actual_value(self):
        quantified = [
            ("体温升高（体温>38.0℃）{38.8,0,38.1,39.5}℃", "体征", "就诊", "D0"),
            ("肺功能示FEV1/FVC<0.70{0.68,0,0.3,0.69}", "功能检查", "就诊", "D0"),
            ("胸痛", "症状", "起病", "D-1"),
        ]

        _, _, _, specific_text = m2.module_2_8_specific_values(
            "慢阻肺 # 40-80 0.5 急性",
            repr(quantified),
            preset_age=65,
            preset_gender="男",
        )
        specific = m2.parse_list_from_response(specific_text)

        self.assertEqual(len(specific), len(quantified))
        self.assertEqual(
            specific[0][0],
            "体温升高（体温>38.0℃）；实际值=38.8℃",
        )
        self.assertEqual(
            specific[1][0],
            "肺功能示FEV1/FVC<0.70；实际值=0.68",
        )
        self.assertEqual(specific[2], quantified[2])
        self.assertTrue(m2._validate_specific_timeline(quantified, specific))

    def test_objective_aliases_share_one_value_without_dropping_phenotypes(self):
        quantified = [
            ("血氧饱和度显著下降（SpO2<90%）{84,3,70,89}%", "体征", "就诊", "D0"),
            ("血氧饱和度下降（SpO2<95%）{91,2,80,94}%", "体征", "就诊", "D0"),
        ]

        harmonized = m2._harmonize_objective_metric_ranges(quantified, {})
        self.assertEqual(len(harmonized), 2)
        _, _, _, specific_text = m2.module_2_8_specific_values(
            "慢阻肺 # 40-80 0.5 慢性急性加重",
            repr(harmonized),
            preset_age=65,
            preset_gender="男",
        )
        specific = m2.parse_list_from_response(specific_text)
        values = [float(re.search(r"实际值=([0-9.]+)", item[0]).group(1)) for item in specific]

        self.assertEqual(len(specific), 2)
        self.assertEqual(values[0], values[1])
        self.assertLess(values[0], 90)

    def test_systolic_blood_pressure_aliases_share_one_value(self):
        quantified = [
            ("重度收缩压升高（SBP>180mmHg）{195,10,181,230}mmHg", "体征", "就诊", "D0"),
            ("收缩压升高（SBP>140mmHg）{172,18,141,230}mmHg", "体征", "就诊", "D0"),
        ]

        harmonized = m2._harmonize_objective_metric_ranges(quantified, {})
        specific = m2._specific_values_with_shared_objective_values(harmonized)
        values = [float(re.search(r"实际值=([0-9.]+)", item[0]).group(1)) for item in specific]

        self.assertEqual(m2._objective_metric_key(quantified[0][0], "体征"), "sign:systolic_bp")
        self.assertEqual(m2._objective_metric_key(quantified[1][0], "体征"), "sign:systolic_bp")
        self.assertEqual(values[0], values[1])
        self.assertGreater(values[0], 180)

    def test_blood_pressure_measurements_in_distinct_contexts_are_not_shared(self):
        quantified = [
            ("卧位收缩压升高{182,0,182,182}mmHg", "体征", "就诊", "D0"),
            ("站立位收缩压升高{150,0,150,150}mmHg", "体征", "就诊", "D0"),
        ]

        harmonized = m2._harmonize_objective_metric_ranges(quantified, {})
        specific = m2._specific_values_with_shared_objective_values(harmonized)

        self.assertIn("实际值=182mmHg", specific[0][0])
        self.assertIn("实际值=150mmHg", specific[1][0])

    def test_blood_pressure_measurements_in_distinct_arms_are_not_shared(self):
        quantified = [
            ("左臂收缩压升高{182,0,182,182}mmHg", "体征", "就诊", "D0"),
            ("右臂收缩压升高{150,0,150,150}mmHg", "体征", "就诊", "D0"),
        ]

        harmonized = m2._harmonize_objective_metric_ranges(quantified, {})
        specific = m2._specific_values_with_shared_objective_values(harmonized)

        self.assertIn("实际值=182mmHg", specific[0][0])
        self.assertIn("实际值=150mmHg", specific[1][0])

    def test_disjoint_blood_pressure_alias_ranges_use_one_stricter_value(self):
        quantified = [
            ("重度收缩压升高（SBP>180mmHg）{185,2,181,190}mmHg", "体征", "就诊", "D0"),
            ("收缩压升高（SBP>140mmHg）{170,2,141,175}mmHg", "体征", "就诊", "D0"),
        ]

        harmonized = m2._harmonize_objective_metric_ranges(quantified, {})
        specific = m2._specific_values_with_shared_objective_values(harmonized)
        values = [float(re.search(r"实际值=([0-9.]+)", item[0]).group(1)) for item in specific]
        bounds = [
            m2._quantified_objective_record(item, index)["values"][2:]
            for index, item in enumerate(harmonized)
        ]

        self.assertEqual(values[0], values[1])
        self.assertGreater(values[0], 180)
        self.assertEqual(bounds[0], bounds[1])
        self.assertLessEqual(bounds[0][0], values[0])
        self.assertGreaterEqual(bounds[0][1], values[0])

    def test_blood_pressure_metric_aliases_are_classified_conservatively(self):
        self.assertEqual(
            m2._objective_metric_key("舒张压升高（DBP>90mmHg）", "体征"),
            "sign:diastolic_bp",
        )
        self.assertEqual(
            m2._objective_metric_key("平均动脉压降低（MAP<65mmHg）", "体征"),
            "sign:mean_arterial_pressure",
        )
        self.assertIsNone(m2._objective_metric_key(
            "奇脉（吸气时收缩压下降>10mmHg）", "体征"
        ))

    def test_blood_pressure_chinese_threshold_unit_is_normalized(self):
        item = (
            "收缩压升高（SBP>140毫米汞柱）{160,5,141,180}", "体征", "就诊", "D0"
        )

        record = m2._quantified_objective_record(item, 0)
        self.assertEqual(record["unit_key"], "mmhg")


    def test_relative_blood_pressure_differences_are_not_absolute_sbp_aliases(self):
        relative_measurements = [
            "双上肢收缩压差增大（差值>20mmHg）",
            "下肢收缩压降低（踝部收缩压较上臂低>20mmHg）",
            "一侧上肢收缩压较对侧低>15mmHg",
            "踝臂收缩压差增大（踝臂差>20mmHg）",
            "上下肢收缩压差异常（下肢较上肢低>20mmHg）",
            "血压波动增大（收缩压波动>30mmHg）",
            "体位性低血压（站立后收缩压下降>20mmHg）",
            "直立性低血压（站立后收缩压下降≥20mmHg）",
            "收缩期血压体位性下降（站立3分钟收缩压下降≥20mmHg）",
            "血压进行性下降（收缩压较基础值下降≥40mmHg）",
            "脉搏奇异（吸气时收缩压下降>10mmHg）",
            "24小时动态血压示收缩压负荷升高",
            "吞咽压力测定示咽部收缩压力降低",
            "肺动脉收缩压升高（PASP>50mmHg）",
            "右心室收缩压升高（>40mmHg）",
            "心室收缩压升高（>140mmHg）",
            "食管收缩压降低（<30mmHg）",
            "括约肌收缩压降低（<40mmHg）",
            "膀胱收缩压升高（>40mmHg）",
            "尿道收缩压降低（<40mmHg）",
            "宫缩压力升高（>60mmHg）",
        ]

        for name in relative_measurements:
            with self.subTest(name=name):
                self.assertIsNone(m2._objective_metric_key(name, "体征"))


    def test_blood_pressure_initial_repeat_and_treatment_contexts_are_not_shared(self):
        quantified = [
            ("初测收缩压升高{182,0,182,182}mmHg", "体征", "就诊", "D0"),
            ("复测收缩压升高{150,0,150,150}mmHg", "体征", "就诊", "D0"),
            ("治疗后收缩压升高{160,0,160,160}mmHg", "体征", "就诊", "D0"),
            ("降压后收缩压升高{145,0,145,145}mmHg", "体征", "就诊", "D0"),
        ]

        harmonized = m2._harmonize_objective_metric_ranges(quantified, {})
        specific = m2._specific_values_with_shared_objective_values(harmonized)

        self.assertIn("实际值=182mmHg", specific[0][0])
        self.assertIn("实际值=150mmHg", specific[1][0])
        self.assertIn("实际值=160mmHg", specific[2][0])
        self.assertIn("实际值=145mmHg", specific[3][0])

    def test_absent_stricter_threshold_constrains_shared_value(self):
        quantified = [
            ("血氧饱和度下降（SpO2<95%）{91,2,80,94}%", "体征", "就诊", "D0"),
        ]
        absent = {"体征": ["血氧饱和度显著下降（SpO2<90%）"]}

        harmonized = m2._harmonize_objective_metric_ranges(quantified, absent)
        _, _, _, specific_text = m2.module_2_8_specific_values(
            "慢阻肺 # 40-80 0.5 慢性急性加重",
            repr(harmonized),
            preset_age=65,
            preset_gender="男",
        )
        specific = m2.parse_list_from_response(specific_text)
        value = float(re.search(r"实际值=([0-9.]+)", specific[0][0]).group(1))

        self.assertGreaterEqual(value, 90)
        self.assertLess(value, 95)
        self.assertTrue(m2._validate_objective_value_consistency(
            harmonized, specific, absent
        ))

    def test_absent_two_sided_range_is_evaluated_as_one_phenotype(self):
        quantified = [
            ("发热（体温≥38.3℃）{38.4,0,38.3,38.4}℃", "体征", "就诊", "D0"),
        ]
        specific = [
            ("发热（体温≥38.3℃）；实际值=38.38℃", "体征", "就诊", "D0"),
        ]
        absent = {"体征": ["低热（37.5℃≤体温<38.3℃）"]}

        self.assertTrue(m2._validate_objective_value_consistency(
            quantified, specific, absent
        ))

    def test_harmonization_chooses_compatible_side_of_absent_two_sided_range(self):
        quantified = [
            ("发热（体温≥38.3℃）{38.4,0.03,38.3,38.4}℃", "体征", "就诊", "D0"),
        ]
        absent = {"体征": ["低热（37.5℃≤体温<38.3℃）"]}

        harmonized = m2._harmonize_objective_metric_ranges(quantified, absent)
        specific = m2._specific_values_with_shared_objective_values(harmonized)

        self.assertTrue(m2._validate_objective_value_consistency(
            harmonized, specific, absent
        ))

    def test_quantified_two_sided_range_must_respect_both_boundaries(self):
        source = "低热（37.5℃≤体温<38.3℃）"

        with self.assertRaisesRegex(ValueError, "显式阈值"):
            m2._validate_quantified_name(
                source,
                f"{source}{{37.0,0.1,36.8,38.2}}℃",
                "体征",
            )

    def test_blood_ph_direction_has_a_deterministic_clinical_boundary(self):
        with self.assertRaisesRegex(ValueError, "阈值"):
            m2._validate_quantified_name(
                "血pH降低",
                "血pH降低{7.50,0.02,7.46,7.54}",
                "实验室检查",
            )

        repaired = m2._repair_quantified_name_threshold_bounds(
            "血pH降低",
            "血pH降低{7.50,0.02,7.20,7.54}",
            "实验室检查",
        )
        self.assertIsNotNone(repaired)
        _, values, suffix = m2._parse_quantified_name_parts("血pH降低", repaired)
        self.assertEqual(suffix, "")
        self.assertLessEqual(values[3], 7.35)
        self.assertEqual(
            m2._validate_quantified_name(
                "血pH降低",
                "血pH降低{7.30,0.02,7.20,7.35}",
                "实验室检查",
            ),
            "血pH降低{7.30,0.02,7.20,7.35}",
        )
        self.assertEqual(
            m2._validate_quantified_name(
                "血pH升高",
                "血pH升高{7.50,0.02,7.45,7.55}",
                "实验室检查",
            ),
            "血pH升高{7.50,0.02,7.45,7.55}",
        )

    def test_objective_consistency_rejects_incoherent_blood_gas_triad(self):
        quantified = [
            ("血pH降低{7.29,0.06,7.05,7.35}", "实验室检查", "就诊", "D0"),
            ("动脉血二氧化碳分压降低{28,5,15,34.9}mmHg", "实验室检查", "就诊", "D0"),
            ("碳酸氢根降低{18,3,8,21.9}mmol/L", "实验室检查", "就诊", "D0"),
        ]
        specific = [
            ("血pH降低；实际值=7.2946", "实验室检查", "就诊", "D0"),
            ("动脉血二氧化碳分压降低；实际值=25.83mmHg", "实验室检查", "就诊", "D0"),
            ("碳酸氢根降低；实际值=18.76mmol/L", "实验室检查", "就诊", "D0"),
        ]

        with self.assertRaisesRegex(ValueError, "Henderson-Hasselbalch"):
            m2._validate_objective_value_consistency(quantified, specific, {})

    def test_blood_gas_triad_is_realized_as_one_physiologic_state(self):
        quantified = [
            ("血pH降低{7.29,0.06,7.05,7.35}", "实验室检查", "就诊", "D0"),
            ("动脉血二氧化碳分压降低{28,5,15,34.9}mmHg", "实验室检查", "就诊", "D0"),
            ("碳酸氢根降低{18,3,8,21.9}mmol/L", "实验室检查", "就诊", "D0"),
        ]
        incompatible = [
            ("血pH降低；实际值=7.2946", "实验室检查", "就诊", "D0"),
            ("动脉血二氧化碳分压降低；实际值=25.83mmHg", "实验室检查", "就诊", "D0"),
            ("碳酸氢根降低；实际值=18.76mmol/L", "实验室检查", "就诊", "D0"),
        ]

        corrected = m2._harmonize_acid_base_specific_values(quantified, incompatible)
        values = [float(re.search(r"实际值=([-0-9.]+)", item[0]).group(1)) for item in corrected]
        expected_ph = 6.1 + math.log10(values[2] / (0.03 * values[1]))

        self.assertLessEqual(abs(values[0] - expected_ph), 0.02)
        self.assertTrue(m2._validate_objective_value_consistency(
            quantified, corrected, {}
        ))

    def test_blood_gas_direction_labels_enforce_clinical_ranges(self):
        invalid = (
            ("动脉血二氧化碳分压降低", "动脉血二氧化碳分压降低{80,2,70,90}mmHg"),
            ("动脉血二氧化碳分压升高", "动脉血二氧化碳分压升高{30,2,20,34}mmHg"),
            ("碳酸氢根降低", "碳酸氢根降低{38,2,30,45}mmol/L"),
            ("碳酸氢根升高", "碳酸氢根升高{18,2,10,21}mmol/L"),
        )
        for source_name, quantified_name in invalid:
            with self.subTest(source_name=source_name):
                with self.assertRaises(ValueError):
                    m2._validate_quantified_name(
                        source_name, quantified_name, "实验室检查"
                    )

    def test_blood_gas_aliases_must_agree_after_unit_conversion(self):
        quantified = [
            ("血pH{7.4,0,7.4,7.4}", "实验室检查", "就诊", "D0"),
            ("动脉血二氧化碳分压{40,0,30,50}mmHg", "实验室检查", "就诊", "D0"),
            ("动脉血PaCO2{6.5,0,4,7}kPa", "实验室检查", "就诊", "D0"),
            ("碳酸氢根{29.25,0,25,35}mmol/L", "实验室检查", "就诊", "D0"),
        ]
        specific = [
            ("血pH；实际值=7.4", "实验室检查", "就诊", "D0"),
            ("动脉血二氧化碳分压；实际值=40mmHg", "实验室检查", "就诊", "D0"),
            ("动脉血PaCO2；实际值=6.5kPa", "实验室检查", "就诊", "D0"),
            ("碳酸氢根；实际值=29.25mmol/L", "实验室检查", "就诊", "D0"),
        ]

        with self.assertRaisesRegex(ValueError, "同一血气指标"):
            m2._validate_objective_value_consistency(quantified, specific, {})

    def test_exertional_test_classifier_does_not_match_static_load_findings(self):
        for name in (
            "心肺运动试验峰值摄氧量降低",
            "平板运动试验阳性",
            "6分钟步行距离缩短",
            "CPET无氧阈降低",
            "无法完成运动试验",
        ):
            self.assertTrue(m2._is_exertional_functional_test(name), name)
        for name in (
            "静息心电图异常",
            "动态心电图异常",
            "心脏后负荷增加",
            "容量负荷增加",
            "静息肺功能下降",
            "糖负荷试验异常",
            "葡萄糖负荷试验异常",
        ):
            self.assertFalse(m2._is_exertional_functional_test(name), name)

    def test_instability_does_not_treat_pulsus_paradoxus_as_systolic_hypotension(self):
        quantified = [
            ("奇脉（吸气时收缩压下降>10mmHg）{15,3,10.1,30}mmHg",
             "体征", "就诊", "D0"),
        ]
        specific = [
            ("奇脉（吸气时收缩压下降>10mmHg）；实际值=14mmHg",
             "体征", "就诊", "D0"),
        ]

        self.assertFalse(m2._has_acute_vital_instability(
            "急性", [], [], [], quantified, specific
        ))
        self.assertFalse(m2._critical_threshold_in_name(
            "心律绝对不齐（相邻RR间期差>120ms）"
        ))

    def test_chronic_objective_instability_prohibits_exercise(self):
        functional = [("6分钟步行试验示步行距离缩短", "IV级")]

        self.assertTrue(m2._has_exercise_functional_contraindication(
            "慢性",
            [("意识障碍", "IV级", "当前", "", "")],
            [("平均动脉压降低（MAP<65mmHg）", "IV级")],
            [], functional,
            diagnosis="心肌病",
        ))

    def test_rest_dyspnea_prohibits_exercise_even_with_chronic_label(self):
        functional = [("平板运动试验示运动耐量下降", "极重度")]

        self.assertTrue(m2._has_exercise_functional_contraindication(
            "慢性", [("静息性呼吸困难", "极重度", "2天", "", "")],
            [("端坐呼吸体位（不能平卧）", "极重度")], [], functional,
            diagnosis="主动脉瓣狭窄",
        ))
        self.assertFalse(m2._has_exercise_functional_contraindication(
            "慢性", [("静息性呼吸困难", "中度", "2天", "", "")],
            [("端坐呼吸体位（不能平卧）", "中度")], [],
            [("平板运动试验示运动耐量下降", "中度")],
            diagnosis="主动脉瓣狭窄",
        ))

    def test_kussmaul_requires_acidosis(self):
        signs = [("库斯莫尔呼吸（深大呼吸频率>20次/分）", "轻度")]
        absent = ["碳酸氢根降低", "动脉血pH降低"]

        self.assertTrue(m2._has_kussmaul_without_acidosis(signs, [], absent))
        self.assertTrue(m2._has_kussmaul_without_acidosis(
            signs, [("尿酮体阳性", "轻度")], absent
        ))
        self.assertFalse(m2._has_kussmaul_without_acidosis(
            signs, [("碳酸氢根降低", "轻度")], absent
        ))
        for variant in (
            "库斯莫尔样呼吸（深大呼吸频率>20次/分）",
            "Kussmaul呼吸（深大呼吸频率>20次/分）",
        ):
            self.assertTrue(m2._has_kussmaul_without_acidosis(
                [(variant, "轻度")], [], absent
            ))
            self.assertIsNone(m2._objective_metric_key(variant, "体征"))
        self.assertFalse(m2._has_kussmaul_without_acidosis(
            [("Kussmaul征（吸气时JVP升高>1cm）", "轻度")], [], absent
        ))
        acid_base_absent = ["碳酸氢根降低", "血液酸碱度降低"]
        self.assertTrue(m2._has_kussmaul_without_acidosis(
            [("库斯莫尔样呼吸", "轻度")], [], acid_base_absent
        ))
        self.assertFalse(m2._has_kussmaul_without_acidosis(
            signs, [("血液酸碱度降低", "轻度")], absent
        ))

    def test_correlation_does_not_promote_kussmaul_from_tachypnea(self):
        clean = json.dumps({
            "status": "clean", "issues": [], "changes": [],
            "resolved_previous_issues": False,
        }, ensure_ascii=False)
        with mock.patch.object(m2, "call_gpt5", return_value=clean) as call:
            result = m2.module_2_4_correlation_correction(
                [], [("呼吸频率增快（呼吸频率>20次/分）", "轻度")],
                [], [], [], "[]",
                "[('呼吸频率增快（呼吸频率>20次/分）', 0.8),"
                "('库斯莫尔呼吸（深大呼吸频率>20次/分）', 0.2)]",
                "[]", "[]", "[]", "肾盂肾炎 # 33-86 0.40 急性", "轻度",
                absent_signs=["库斯莫尔呼吸（深大呼吸频率>20次/分）"],
                absent_lab_tests=["碳酸氢根降低", "动脉血pH降低"],
                max_rounds=2,
            )

        self.assertIsNotNone(result)
        self.assertEqual(call.call_count, 1)
        self.assertNotIn(
            "库斯莫尔呼吸（深大呼吸频率>20次/分）",
            [item[0] for item in result[3]],
        )
        self.assertIn(
            "库斯莫尔呼吸（深大呼吸频率>20次/分）", result[8]
        )

    def test_correlation_accepts_clean_after_pending_issue(self):
        responses = [
            json.dumps({
                "status": "issues",
                "issues": ["需要再检查"],
                "changes": [],
            }, ensure_ascii=False),
            json.dumps({
                "status": "clean",
                "issues": [],
                "changes": [],
                "resolved_previous_issues": False,
            }, ensure_ascii=False),
        ]
        with mock.patch.object(m2, "call_gpt5", side_effect=responses) as call:
            result = m2.module_2_4_correlation_correction(
                [("咳嗽", "轻度", "2天", "", "")], [], [], [], [],
                "[('咳嗽', ('2天','','',0.8))]", "[]", "[]", "[]", "[]",
                "肺炎 # 40-80 0.5 急性", "轻度", max_rounds=2,
            )

        self.assertIsNotNone(result)
        audit = json.loads(result[1])
        self.assertEqual(audit["status"], "converged")
        self.assertEqual([item["status"] for item in audit["rounds"]], ["issues", "clean"])
        self.assertEqual(call.call_count, 2)

    def test_worker_reruns_correlation_for_stale_kussmaul_state(self):
        calls = []
        kussmaul = "库斯莫尔呼吸（深大呼吸频率>20次/分）"
        selected = [
            [("咳嗽", "轻度", "2天", "", "")],
            [(kussmaul, "轻度")], [], [], [],
        ]
        absent = [[], [], ["碳酸氢根降低", "动脉血pH降低"], [], []]
        row = _complete_unvisited_row()
        for column, value in zip((
            m2.COL_SYMPTOMS, m2.COL_SIGNS, m2.COL_LAB_TESTS,
            m2.COL_IMAGING, m2.COL_FUNCTIONAL_TESTS,
        ), selected):
            row[column] = repr(value)
        for column, value in zip((
            m2.COL_ABSENT_SYMPTOMS, m2.COL_ABSENT_SIGNS,
            m2.COL_ABSENT_LAB_TESTS, m2.COL_ABSENT_IMAGING,
            m2.COL_ABSENT_FUNCTIONAL,
        ), absent):
            row[column] = repr(value)
        row[m2.COL_DIAGNOSIS] = "肾盂肾炎"
        row[m2.COL_STAGE] = "轻度"
        row[m2.COL_ACUITY] = "急性"
        row[m2.COL_COMORBIDITIES] = "[]"
        row[m2.COL_M24_OUTPUT] = json.dumps({
            "schema_version": 2,
            "status": "converged",
            "rounds": [],
            "final_hash": m2._phenotype_state_hash_from_lists(selected, absent),
        })

        def fake_correlation(selected_symptoms, selected_signs, selected_labs,
                             selected_imaging, selected_functional, *args, **kwargs):
            calls.append("2.4")
            corrected_signs = [
                item for item in selected_signs if item[0] != kussmaul
            ]
            corrected_absent_signs = list(kwargs["absent_signs"]) + [kussmaul]
            corrected_selected = [
                selected_symptoms, corrected_signs, selected_labs,
                selected_imaging, selected_functional,
            ]
            corrected_absent = [
                kwargs["absent_symptoms"], corrected_absent_signs,
                kwargs["absent_lab_tests"], kwargs["absent_imaging"],
                kwargs["absent_functional"],
            ]
            audit = json.dumps({
                "schema_version": 2,
                "status": "converged",
                "rounds": [],
                "final_hash": m2._phenotype_state_hash_from_lists(
                    corrected_selected, corrected_absent
                ),
            })
            return ("p", audit, *corrected_selected, *corrected_absent)

        def fake_quantify(_seed, _stage, corrected, **kwargs):
            items = m2.parse_list_from_response(corrected)
            quantified = []
            for item in items:
                if "库斯莫尔" in item[0]:
                    quantified.append((f"{item[0]}{{25,5,21,35}}次/分", *item[1:]))
                else:
                    quantified.append(tuple(item))
            return "p", "r", quantified

        with mock.patch.object(
            m2, "module_2_4_correlation_correction", fake_correlation
        ), mock.patch.object(
            m2, "module_2_5_build_timeline",
            return_value=(
                "p",
                _schema2_time_model_response(None, "3天", [
                    {"name": "咳嗽", "category": "症状", "phase": "起病", "time_label": "D-2"},
                ]),
                _schema2_time_model(None, "3天", [("咳嗽", "症状", "起病", "D-2")]),
            ),
        ), mock.patch.object(
            m2, "module_2_7_quantify_values",
            side_effect=fake_quantify,
        ), mock.patch.object(
            m2, "module_2_8_specific_values",
            side_effect=lambda *args, **kwargs: (
                65, "男", "肾盂肾炎", m2._quantize_values(args[1])
            ),
        ), mock.patch.object(
            m2, "module_2_9_derive_chief_complaint", return_value="咳嗽"
        ):
            m2._generate_single_patient_m2((
                1,
                "肾盂肾炎 # 33-86 0.40 急性",
                "[('咳嗽', ('2天','','',0.8))]",
                f"[('{kussmaul}', 0.2)]",
                "[]", "[]", "[]", "[]",
                {"name": "分级", "levels": [{"name": "轻度", "proportion": 1.0}]},
                "轻度", queue.Queue(), row, "either", "[]",
            ))

        self.assertIn("2.4", calls)

    def test_correlation_keeps_exercise_tests_for_acute_unstable_patient(self):
        clean = json.dumps({
            "status": "clean", "issues": [], "changes": [],
            "resolved_previous_issues": False,
        }, ensure_ascii=False)
        with mock.patch.object(m2, "call_gpt5", side_effect=[clean]):
            result = m2.module_2_4_correlation_correction(
                [("突发呼吸困难", "高危", "1小时", "", "")],
                [("休克", "高危")], [], [],
                [
                    ("心肺运动试验峰值摄氧量降低", "高危"),
                    ("静息心电图异常", "高危"),
                ],
                "[]", "[]", "[]", "[]", "[]",
                "肺栓塞 # 50-80 0.5 急性", "高危",
                absent_symptoms=[], absent_signs=[], absent_lab_tests=[],
                absent_imaging=[], absent_functional=[], max_rounds=3,
            )

        selected_functional = [item[0] for item in result[6]]
        absent_functional = result[11]
        audit = json.loads(result[1])
        self.assertIn("心肺运动试验峰值摄氧量降低", selected_functional)
        self.assertNotIn("心肺运动试验峰值摄氧量降低", absent_functional)
        self.assertIn("静息心电图异常", selected_functional)
        self.assertEqual(audit["rounds"][0]["deterministic_changes"], [])

    def test_acute_high_risk_diagnosis_prohibits_exercise_without_vital_instability(self):
        functional = [("心肺运动试验峰值摄氧量降低", "中危")]

        self.assertTrue(m2._has_exercise_functional_contraindication(
            "急性", [], [], [], functional, diagnosis="肺栓塞"
        ))
        self.assertFalse(m2._has_exercise_functional_contraindication(
            "慢性", [], [], [], functional, diagnosis="肺栓塞"
        ))
        self.assertTrue(m2._has_exercise_functional_contraindication(
            "慢性急性加重", [], [], [], functional, diagnosis="心力衰竭"
        ))
        self.assertFalse(m2._has_exercise_functional_contraindication(
            "急性", [], [], [], functional, diagnosis="急性胃肠炎"
        ))

    def test_completion_allows_acute_shock_with_exercise_test(self):
        row = _complete_unvisited_row()
        selected = [
            [("咳嗽", "轻度", "2天", "", "")],
            [("休克", "高危")], [], [],
            [("心肺运动试验异常", "高危")],
        ]
        absent = [[], [], [], [], []]
        for column, value in zip((
            m2.COL_SYMPTOMS, m2.COL_SIGNS, m2.COL_LAB_TESTS,
            m2.COL_IMAGING, m2.COL_FUNCTIONAL_TESTS,
        ), selected):
            row[column] = repr(value)
        row[m2.COL_M24_OUTPUT] = json.dumps({
            "schema_version": 2,
            "status": "converged",
            "rounds": [],
            "final_hash": m2._phenotype_state_hash_from_lists(selected, absent),
        })
        timeline = [
            ("咳嗽", "症状", "起病", "D-2"),
            ("休克", "体征", "就诊", "D0"),
            ("心肺运动试验异常", "功能检查", "就诊", "D0"),
        ]
        row[m2.COL_TIME_ORDER] = row[m2.COL_QUANTIFIED] = row[m2.COL_SPECIFIC] = repr(timeline)

        self.assertTrue(m2._is_m2_row_complete(row))

    def test_nonoverlapping_egfr_alias_ranges_are_harmonized_and_shared(self):
        quantified = [
            ("估算肾小球滤过率降低（eGFR<60mL/min/1.73m²）{45,8,15,59}mL/min/1.73m²", "实验室检查", "就诊", "D0"),
            ("eGFR降低{75,8,61,90}mL/min/1.73m²", "实验室检查", "就诊", "D0"),
        ]

        harmonized = m2._harmonize_objective_metric_ranges(quantified, {})
        _, _, _, specific_text = m2.module_2_8_specific_values(
            "慢性肾病 # 40-80 0.5 慢性",
            repr(harmonized),
            preset_age=65,
            preset_gender="男",
        )
        specific = m2.parse_list_from_response(specific_text)
        values = [float(re.search(r"实际值=([0-9.]+)", item[0]).group(1)) for item in specific]

        self.assertEqual(values[0], values[1])
        self.assertLess(values[0], 60)

    def test_same_metric_at_different_times_is_not_forced_to_share(self):
        quantified = [
            ("体温升高{38.2,0,38.2,38.2}℃", "体征", "起病", "D-2"),
            ("体温升高{39.1,0,39.1,39.1}℃", "体征", "就诊", "D0"),
        ]

        _, _, _, specific_text = m2.module_2_8_specific_values(
            "肺炎 # 20-80 0.5 急性", repr(quantified),
            preset_age=65, preset_gender="男",
        )
        specific = m2.parse_list_from_response(specific_text)

        self.assertIn("实际值=38.2℃", specific[0][0])
        self.assertIn("实际值=39.1℃", specific[1][0])

    def test_same_metric_under_different_measurement_contexts_is_not_shared(self):
        quantified = [
            ("静息状态SpO2降低（SpO2<95%）{92,0,92,92}%", "体征", "就诊", "D0"),
            ("运动后SpO2降低（SpO2<90%）{84,0,84,84}%", "体征", "就诊", "D0"),
            ("支气管舒张前FEV1/FVC<0.70{0.62,0,0.62,0.62}", "功能检查", "就诊", "D0"),
            ("支气管舒张后FEV1/FVC<0.70{0.68,0,0.68,0.68}", "功能检查", "就诊", "D0"),
        ]

        harmonized = m2._harmonize_objective_metric_ranges(quantified, {})
        _, _, _, specific_text = m2.module_2_8_specific_values(
            "慢阻肺 # 40-80 0.5 慢性", repr(harmonized),
            preset_age=65, preset_gender="男",
        )
        specific = m2.parse_list_from_response(specific_text)

        self.assertIn("实际值=92%", specific[0][0])
        self.assertIn("实际值=84%", specific[1][0])
        self.assertIn("实际值=0.62", specific[2][0])
        self.assertIn("实际值=0.68", specific[3][0])

    def test_correlation_does_not_promote_threshold_from_different_context(self):
        state = m2._build_phenotype_state(
            {'symptoms': [], 'signs': [(
                "运动后SpO2降低（SpO2<90%）", "重度"
            )], 'lab_tests': [], 'imaging': [], 'functional': []},
            {'symptoms': [], 'signs': [
                "静息状态SpO2降低（SpO2<95%）"
            ], 'lab_tests': [], 'imaging': [], 'functional': []},
            {'symptoms': {}, 'signs': {}, 'lab_tests': {}, 'imaging': {}, 'functional': {}},
        )

        repairs = m2._repair_objective_threshold_state(state, "重度")

        self.assertEqual(repairs, [])
        self.assertIn("静息状态SpO2降低（SpO2<95%）", state['signs']['negative'])

    def test_absent_threshold_with_incompatible_unit_is_not_applied(self):
        quantified = [
            ("血清肌酐升高（>1.5mg/dL）{2.2,0,2.2,2.2}mg/dL", "实验室检查", "就诊", "D0"),
        ]
        absent = {"实验室检查": ["血清肌酐降低（<180μmol/L）"]}

        harmonized = m2._harmonize_objective_metric_ranges(quantified, absent)

        self.assertIn("{2.2,0,2.2,2.2}mg/dL", harmonized[0][0])

    def test_egfr_square_meter_unit_spellings_share_value(self):
        quantified = [
            ("eGFR降低{45,0,45,45}mL/min/1.73m²", "实验室检查", "就诊", "D0"),
            ("估算肾小球滤过率降低{55,0,55,55}mL/min/1.73㎡", "实验室检查", "就诊", "D0"),
        ]

        harmonized = m2._harmonize_objective_metric_ranges(quantified, {})
        specific = m2._specific_values_with_shared_objective_values(harmonized)
        values = [float(re.search(r"实际值=([0-9.]+)", item[0]).group(1)) for item in specific]

        self.assertEqual(values[0], values[1])

    def test_serum_creatinine_and_hscrp_aliases_share_without_merging_ratio(self):
        quantified = [
            ("血清肌酐升高{150,0,150,150}μmol/L", "实验室检查", "就诊", "D0"),
            ("肌酐升高{200,0,200,200}μmol/L", "实验室检查", "就诊", "D0"),
            ("尿白蛋白肌酐比值升高{4,0,4,4}mg/mmol", "实验室检查", "就诊", "D0"),
            ("高敏C反应蛋白轻度升高{8,0,8,8}mg/L", "实验室检查", "就诊", "D0"),
            ("超敏C反应蛋白升高{18,0,18,18}mg/L", "实验室检查", "就诊", "D0"),
        ]

        harmonized = m2._harmonize_objective_metric_ranges(quantified, {})
        specific = m2._specific_values_with_shared_objective_values(harmonized)
        values = [float(re.search(r"实际值=([0-9.]+)", item[0]).group(1))
                  for item in specific]

        self.assertEqual(values[0], values[1])
        self.assertEqual(values[2], 4)
        self.assertEqual(values[3], values[4])

    def test_pulse_deficit_and_gallop_are_not_heart_rate_measurements(self):
        self.assertIsNone(m2._objective_metric_key(
            "脉搏短绌（心率高于脉率>10次/分）", "体征"
        ))
        self.assertIsNone(m2._objective_metric_key(
            "奔马律（心率>100次/分时额外心音形成三音律）", "体征"
        ))

    def test_liver_enzyme_ratio_is_not_classified_as_atomic_alt(self):
        self.assertIsNone(m2._objective_metric_key(
            "AST/ALT比值升高", "实验室检查"
        ))
        self.assertIsNone(m2._objective_metric_key(
            "天冬氨酸氨基转移酶/丙氨酸氨基转移酶比值升高", "实验室检查"
        ))

    def test_correlation_threshold_repair_does_not_compare_incompatible_units(self):
        positive = "血清肌酐升高（>150μmol/L）"
        negative = "血清肌酐升高（>2mg/dL）"
        state = m2._build_phenotype_state(
            {'symptoms': [], 'signs': [], 'lab_tests': [(positive, '重度')],
             'imaging': [], 'functional': []},
            {'symptoms': [], 'signs': [], 'lab_tests': [negative],
             'imaging': [], 'functional': []},
            {'symptoms': {}, 'signs': {}, 'lab_tests': {},
             'imaging': {}, 'functional': {}},
        )

        repairs = m2._repair_objective_threshold_state(state, "重度")

        self.assertEqual(repairs, [])
        self.assertIn(negative, state['lab_tests']['negative'])

    def test_objective_consistency_rejects_value_that_hits_absent_threshold(self):
        quantified = [
            ("呼吸频率增快（>20次/分）{25,3,21,30}次/分", "体征", "就诊", "D0"),
        ]
        specific = [
            ("呼吸频率增快（>20次/分）；实际值=30次/分", "体征", "就诊", "D0"),
        ]
        absent = {"体征": ["严重呼吸急促（呼吸频率≥30次/分）"]}

        with self.assertRaisesRegex(ValueError, "阴性阈值"):
            m2._validate_objective_value_consistency(quantified, specific, absent)

    def test_resume_rejects_legacy_specific_value_concatenation(self):
        quantified = [
            ("体温升高（体温>38.0℃）{38.8,0,38.1,39.5}℃", "体征", "就诊", "D0"),
        ]
        legacy = [
            ("体温升高（体温>38.0℃）38.8℃", "体征", "就诊", "D0"),
        ]
        current = [
            ("体温升高（体温>38.0℃）；实际值=38.8℃", "体征", "就诊", "D0"),
        ]

        self.assertIsNone(
            m2._validated_existing_specific_timeline(quantified, repr(legacy))
        )
        self.assertEqual(
            m2._validated_existing_specific_timeline(quantified, repr(current)),
            current,
        )

    def test_quantified_timeline_rejects_historical_non_atomic_values(self):
        source = [("血压升高(>140/90mmHg)", "体征", "就诊", "D0")]
        quantified = [(
            "血压升高(>140/90mmHg){150,10,120,180}mmHg",
            "体征", "就诊", "D0",
        )]
        with self.assertRaisesRegex(ValueError, "不适合单值量化"):
            m2._validate_quantified_timeline(source, quantified)

    def test_module_27_preserves_rich_non_atomic_phenotypes(self):
        source = [
            ("心率增快", "体征", "就诊", "D0"),
            ("白细胞计数升高", "实验室检查", "就诊", "D0"),
            ("血压升高(>140/90mmHg)", "体征", "就诊", "D0"),
            ("呼吸浅快（潮气量降低且呼吸频率>20次/分）", "体征", "就诊", "D0"),
            ("股动脉搏动减弱", "体征", "就诊", "D0"),
        ]

        def fake_call(prompt, **_kwargs):
            match = re.search(
                r"候选条目（source_index 为 0 起始的稳定索引）：\n(\[.*?\])\n\n只输出",
                prompt,
                re.S,
            )
            candidates = json.loads(match.group(1))
            self.assertEqual([item["source_index"] for item in candidates], [0, 1])
            names = {
                0: "心率增快{110,10,91,140}次/分",
                1: "白细胞计数升高{14,2,10,20}×10^9/L",
            }
            return json.dumps({
                "schema_version": 1,
                "updates": [
                    {"source_index": item["source_index"],
                     "quantified_name": names[item["source_index"]]}
                    for item in candidates
                ],
            }, ensure_ascii=False)

        with mock.patch.object(m2, "call_gpt5", side_effect=fake_call):
            result = m2.module_2_7_quantify_values(
                "疾病 # 20-80 0.5 急性", "轻度", repr(source)
            )

        self.assertEqual(len(result[2]), len(source))
        self.assertEqual(result[2][2:], source[2:])

    def test_sampling_filters_opposite_sex_and_personalizes_dual_threshold(self):
        signs = repr([
            ("男性腹型肥胖（腰围≥90cm）", 1.0),
            ("女性腹型肥胖（腰围≥85cm）", 1.0),
            ("腰围增大(男性≥90cm或女性≥85cm)", 1.0),
        ])

        result = m2.module_2_1_sample_phenotypes(
            "[('胸痛', ('1天','','',1.0))]", signs, "[]", "[]", "[]",
            patient_level_index=0,
            num_levels=1,
            patient_stage="轻度",
            patient_gender="男",
        )

        selected_names = [item[0] for item in result[1]]
        absent_names = result[6]
        self.assertIn("男性腹型肥胖（腰围≥90cm）", selected_names)
        self.assertIn("腰围增大(男性≥90cm)", selected_names)
        self.assertNotIn("女性腹型肥胖（腰围≥85cm）", selected_names)
        self.assertFalse(any("女性" in name for name in absent_names))

    def test_complication_sampling_personalizes_dual_sex_threshold(self):
        result = m2._sample_complication_phenotypes(
            "肥胖症",
            "[('肥胖症','体征','腰围增大(男性≥90cm或女性≥85cm)',1.0)]",
            patient_gender="女",
        )

        self.assertEqual(result[0], [("腰围增大(女性≥85cm)", "伴随疾病")])

    def test_gender_personalization_handles_common_separators_and_case(self):
        self.assertEqual(
            m2._personalize_phenotype_name_for_gender(
                "血红蛋白降低（男性<130g/L，女性<120g/L）", "女"
            ),
            "血红蛋白降低（女性<120g/L）",
        )
        self.assertIsNone(
            m2._personalize_phenotype_name_for_gender("psa升高", "女")
        )
        self.assertEqual(
            m2._personalize_phenotype_name_for_gender("女性化乳房", "男"),
            "女性化乳房",
        )

    def test_gender_applies_accepts_common_neutral_labels(self):
        for label in ("不限", "男女", "男、女", "女/男"):
            self.assertTrue(m2._gender_applies(label, "男"), label)
            self.assertTrue(m2._gender_applies(label, "女"), label)

    def test_correlation_candidate_universe_cannot_restore_opposite_sex(self):
        captured = {}

        def fake_call(prompt, **_kwargs):
            captured["prompt"] = prompt
            return json.dumps({"status": "clean", "issues": [], "changes": []},
                              ensure_ascii=False)

        with mock.patch.object(m2, "call_gpt5", side_effect=fake_call):
            result = m2.module_2_4_correlation_correction(
                [], [("女性腹型肥胖（腰围≥85cm）", "轻度")], [], [], [],
                "[]",
                repr([
                    ("女性腹型肥胖（腰围≥85cm）", 1.0),
                    ("腰围增大(男性≥90cm或女性≥85cm)", 1.0),
                ]),
                "[]", "[]", "[]", "肥胖症", "轻度",
                absent_signs=[
                    "女性腹型肥胖（腰围≥85cm）",
                    "腰围增大(男性≥90cm或女性≥85cm)",
                ],
                patient_gender="男",
            )

        self.assertIsNotNone(result)
        self.assertNotIn("女性腹型肥胖", captured["prompt"])
        self.assertIn("腰围增大(男性≥90cm)", captured["prompt"])
        self.assertEqual(result[2], [])
        self.assertEqual(result[3], [])
        self.assertEqual(result[8], ["腰围增大(男性≥90cm)"])

    def test_correlation_promotes_implied_negative_threshold_without_losing_label(self):
        calls = []

        def clean_response(*_args, **_kwargs):
            calls.append(1)
            return json.dumps({"status": "clean", "issues": [], "changes": []},
                              ensure_ascii=False)

        severe = "血氧饱和度显著下降（SpO2<90%）"
        mild = "血氧饱和度下降（SpO2<95%）"
        with mock.patch.object(m2, "call_gpt5", side_effect=clean_response):
            result = m2.module_2_4_correlation_correction(
                [], [(severe, "重度")], [], [], [],
                "[]", repr([(severe, 1.0), (mild, 1.0)]), "[]", "[]", "[]",
                "慢阻肺", "重度",
                absent_signs=[mild],
                max_rounds=3,
            )

        self.assertIsNotNone(result)
        self.assertEqual([item[0] for item in result[3]], [severe, mild])
        self.assertEqual(result[8], [])
        self.assertGreaterEqual(len(calls), 2)

    def test_correlation_does_not_promote_disjoint_two_sided_temperature_range(self):
        calls = []

        def clean_response(*_args, **_kwargs):
            calls.append(1)
            return json.dumps({"status": "clean", "issues": [], "changes": []},
                              ensure_ascii=False)

        hypothermia = "体温降低（体温<35.0℃）"
        low_fever = "低热（37.5℃≤体温<38.3℃）"
        with mock.patch.object(m2, "call_gpt5", side_effect=clean_response):
            result = m2.module_2_4_correlation_correction(
                [], [(hypothermia, "高危")], [], [], [],
                "[]", repr([(hypothermia, 0.8), (low_fever, 0.2)]),
                "[]", "[]", "[]", "肺栓塞", "高危",
                absent_signs=[low_fever], max_rounds=2,
            )

        self.assertIsNotNone(result)
        self.assertEqual([item[0] for item in result[3]], [hypothermia])
        self.assertEqual(result[8], [low_fever])
        self.assertEqual(len(calls), 1)

    def test_correlation_prompt_allows_genuinely_asymptomatic_patients(self):
        captured = {}

        def clean_response(prompt, **_kwargs):
            captured['prompt'] = prompt
            return json.dumps({"status": "clean", "issues": [], "changes": []},
                              ensure_ascii=False)

        with mock.patch.object(m2, "call_gpt5", side_effect=clean_response):
            result = m2.module_2_4_correlation_correction(
                [], [("心律绝对不齐", "I级")], [], [], [],
                "[('心悸', ('数天', '', '', 0.2))]",
                "[('心律绝对不齐', 1.0)]", "[]", "[]", "[]",
                "心房颤动", "I级：无症状", absent_symptoms=["心悸"],
            )

        self.assertIn("允许真实无主观症状", captured['prompt'])
        self.assertEqual(result[2], [])
        self.assertEqual(result[7], ["心悸"])

    def test_sampling_allows_true_zero_symptom_patient_if_all_miss(self):
        symptom_library = repr([
            ("乏力", ("数天", "", "持续", 0.2)),
            ("胸痛", ("数小时", "活动后", "压榨样", 0.9)),
        ])

        with mock.patch.object(m2.random, "random", return_value=1.0):
            result = m2.module_2_1_sample_phenotypes(
                symptom_library, "[]", "[]", "[]", "[]",
                patient_level_index=0,
                num_levels=1,
                patient_stage="轻度",
            )

        selected_symptoms, absent_symptoms = result[0], result[5]
        self.assertEqual(selected_symptoms, [])
        self.assertEqual(absent_symptoms, ["乏力", "胸痛"])

    def test_no_symptom_suggested_none_falls_back_to_objective_chief_complaint(self):
        with mock.patch.object(m2, "call_gpt5", side_effect=AssertionError("不应调用GPT")):
            chief = m2.module_2_9_derive_chief_complaint(
                repr([(
                    "心电图提示心房颤动",
                    "功能检查",
                    "就诊",
                    "D0",
                )]),
                symptoms_list=[],
                diagnosis="心房颤动",
                suggested="无",
            )

        self.assertEqual(chief, "体检发现心房颤动相关异常")

    def test_no_symptom_prior_treatment_does_not_claim_symptom_treatment(self):
        result = m2.module_2_6_prior_visit_history(
            "心房颤动", "慢性急性加重", 1, "1年",
            [], [("心律绝对不齐", "I级")], [], symptom_timeline=[],
        )

        self.assertIsNotNone(result)
        self.assertIn("客观异常", result[2]["history_text"])
        self.assertNotIn("本次症状对症处理", result[2]["history_text"])

    def test_m2_entry_validation_rejects_malformed_symptom_atlas(self):
        row = m2._empty_row()
        row[m2.COL_STAGING_SYSTEM] = json.dumps({
            "name": "临床严重度分级",
            "levels": [
                {"level": 1, "name": "轻度", "description": "症状轻", "proportion": 0.6},
                {"level": 2, "name": "重度", "description": "症状重", "proportion": 0.4},
            ],
        }, ensure_ascii=False)
        row[m2.COL_SYMPTOMS] = "[('数天','活动后','胸闷',0.5)]"
        for column in (
            m2.COL_SIGNS, m2.COL_LAB_TESTS, m2.COL_IMAGING,
            m2.COL_FUNCTIONAL_TESTS,
        ):
            row[column] = "[]"

        with self.assertRaisesRegex(ValueError, "模块1.11"):
            m2._validate_m1_row_for_m2(row)

    def test_integral_text_parser_accepts_csv_float_spelling(self):
        self.assertEqual(m2._parse_integral_text("91.0", minimum=1, maximum=120), 91)
        self.assertEqual(m2._parse_integral_text("0.0", minimum=0), 0)
        with self.assertRaises(ValueError):
            m2._parse_integral_text("91.5", minimum=1, maximum=120)

    def test_completion_accepts_integral_zero_loaded_as_float_text(self):
        self.assertTrue(m2._is_m2_row_complete(_complete_unvisited_row("0.0")))

    def test_completion_rejects_fixed_right_laterality_mismatch(self):
        row = _complete_unvisited_row()
        row[m2.COL_DIAGNOSIS] = "急性阑尾炎"
        row[m2.COL_PATIENT_LATERALITY] = "左侧"

        self.assertFalse(m2._is_m2_row_complete(row))

    def test_completion_keeps_mirror_anatomy_left_laterality(self):
        row = _complete_unvisited_row()
        row[m2.COL_DIAGNOSIS] = "急性阑尾炎伴内脏反位"
        row[m2.COL_PATIENT_LATERALITY] = "左侧"

        self.assertTrue(m2._is_m2_row_complete(row))

    def test_completion_rejects_opposite_sex_phenotype_in_final_outputs(self):
        row = _complete_unvisited_row()
        row[m2.COL_TIME_ORDER] = repr([
            ("咳嗽", "症状", "起病", "D-2"),
            ("女性腹型肥胖（腰围≥85cm）", "体征", "就诊", "D0"),
        ])
        row[m2.COL_QUANTIFIED] = row[m2.COL_TIME_ORDER]
        row[m2.COL_SPECIFIC] = row[m2.COL_TIME_ORDER]

        self.assertFalse(m2._is_m2_row_complete(row))

    def test_completion_rejects_opposite_sex_comorbidity(self):
        row = _complete_unvisited_row()
        row[m2.COL_GENDER] = "女"
        row[m2.COL_COMORBIDITIES] = repr(["前列腺增生"])

        self.assertFalse(m2._is_m2_row_complete(row))

    def test_resolve_patient_demographics_preserves_injected_values(self):
        row = m2._empty_row()
        row[m2.COL_CASE_ID] = "vp_fixed"
        row[m2.COL_AGE] = "91.0"
        row[m2.COL_GENDER] = "女"
        with mock.patch.object(m2.random, "randint", side_effect=AssertionError("不得重采样")):
            age, gender = m2._resolve_patient_demographics(
                row, "肺炎 # 40-80 0.5 急性"
            )
        self.assertEqual((age, gender), (91, "女"))
        self.assertEqual(row[m2.COL_AGE], "91")
        self.assertEqual(row[m2.COL_GENDER], "女")

    def test_realize_symptom_durations_uses_final_timeline(self):
        symptoms = [
            ("咳嗽", "轻度", "持续数周", "受凉后", "间断"),
            ("气短", "轻度", "数天", "活动后", "反复"),
        ]
        timeline = [
            ("咳嗽", "症状", "起病", "D-2"),
            ("气短", "症状", "就诊", "D0"),
        ]
        result = m2._realize_symptom_durations(symptoms, timeline)
        self.assertEqual(result, [
            ("咳嗽", "轻度", "2天", "受凉后", "间断"),
            ("气短", "轻度", "不足1天", "活动后", "反复"),
        ])

    def test_module_25_prompt_includes_symptom_attributes(self):
        captured = {}

        def fake_call(prompt, **_kwargs):
            captured["prompt"] = prompt
            return _schema2_time_model_response(None, "3天", [
                {"name": "咳嗽", "category": "症状", "phase": "起病", "time_label": "D-2"},
            ])

        with mock.patch.object(m2, "call_gpt5", side_effect=fake_call):
            result = m2.module_2_5_build_timeline(
                "肺炎 # 40-80 0.5 急性", "轻度",
                [("咳嗽", "轻度", "持续数天", "受凉后", "间断")],
                [], [], [], [],
            )
        self.assertIsNotNone(result)
        self.assertIn("持续数天", captured["prompt"])
        self.assertIn("受凉后", captured["prompt"])

    def test_quantified_unit_inserts_multiplication_sign(self):
        result = m2._validate_quantified_name(
            "白细胞计数升高",
            "白细胞计数升高{14,2,10,20}10^9/L",
            "实验室检查",
        )
        self.assertEqual(result, "白细胞计数升高{14,2,10,20}×10^9/L")

    def test_large_quantification_is_chunked_without_dropping_phenotypes(self):
        source = [
            (f"指标{i}升高", "实验室检查", "就诊", "D0")
            for i in range(45)
        ] + [("咳嗽", "症状", "起病", "D-2")]
        calls = []

        def fake_call(prompt, **_kwargs):
            match = re.search(
                r"候选条目（source_index 为 0 起始的稳定索引）：\n(\[.*?\])\n\n只输出",
                prompt,
                re.S,
            )
            self.assertIsNotNone(match)
            candidates = json.loads(match.group(1))
            calls.append(candidates)
            return json.dumps({
                "schema_version": 1,
                "updates": [
                    {
                        "source_index": item["source_index"],
                        "quantified_name": f'{item["name"]}{{1,0,0,2}}U',
                    }
                    for item in candidates
                ],
            }, ensure_ascii=False)

        with mock.patch.object(m2, "call_gpt5", side_effect=fake_call):
            result = m2.module_2_7_quantify_values(
                "疾病 # 20-80 0.5 急性", "轻度", repr(source)
            )

        self.assertIsNotNone(result)
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(len(batch) <= 20 for batch in calls))
        self.assertEqual(len(result[2]), len(source))
        self.assertEqual(result[2][-1], source[-1])
        self.assertTrue(all("{" in item[0] for item in result[2][:-1]))


    def test_module_27_fills_missing_threshold_vital_without_dropping_phenotypes(self):
        source = [
            ("高热（>39.0℃）", "体征", "就诊", "D0"),
            ("心动过速（心率>100次/分）", "体征", "就诊", "D0"),
            ("胸痛", "症状", "起病", "D-1"),
        ]
        response = json.dumps({
            "schema_version": 1,
            "updates": [
                {
                    "source_index": 1,
                    "quantified_name": "心动过速（心率>100次/分）{110,5,101,130}次/分",
                },
            ],
        }, ensure_ascii=False)

        with mock.patch.object(m2, "call_gpt5", return_value=response):
            result = m2.module_2_7_quantify_values(
                "感染性心内膜炎 # 20-80 0.5 急性", "中度", repr(source)
            )

        self.assertIsNotNone(result)
        quantified = result[2]
        self.assertEqual(len(quantified), len(source))
        self.assertRegex(quantified[0][0], r"^高热（>39\.0℃）\{[0-9.]+,[0-9.]+,39\.1,[0-9.]+\}℃$")
        self.assertEqual(quantified[2], source[2])


if __name__ == "__main__":
    unittest.main()


def test_worker_runs_m281_between_specific_values_and_chief_and_persists_sidecar(tmp_path, monkeypatch):
    csv_path = tmp_path / "肺炎.csv"
    row_1 = m2._empty_row()
    row_1[m2.COL_SEED] = "肺炎 # 40-80 1.0 急性"
    row_1[m2.COL_STAGING_SYSTEM] = json.dumps({
        "name": "临床严重度分级",
        "levels": [
            {"level": 1, "name": "轻度", "description": "轻", "proportion": 0.99},
            {"level": 2, "name": "重度", "description": "重", "proportion": 0.01},
        ],
    }, ensure_ascii=False)
    for column in (
        m2.COL_SYMPTOMS, m2.COL_SIGNS, m2.COL_LAB_TESTS,
        m2.COL_IMAGING, m2.COL_FUNCTIONAL_TESTS, m2.COL_COMORBIDITIES,
        m2.COL_COMPLICATION_PHENOTYPES,
    ):
        row_1[column] = "[]"
    row_1[m2.COL_SYMPTOMS] = "[('咳嗽', ('2天','','',1.0), ('2天','','',1.0))]"

    patient = _complete_unvisited_row()
    patient[m2.COL_CASE_ID] = "case_00001"
    patient[m2.COL_AGE] = "60"
    patient[m2.COL_SEED] = row_1[m2.COL_SEED]
    patient[m2.COL_DIAGNOSIS] = "肺炎"
    patient[m2.COL_STAGE] = "轻度"
    patient[m2.COL_SPECIFIC] = ""
    patient[m2.COL_CHIEF_COMPLAINT] = ""
    m2._save_csv(str(csv_path), [row_1, patient])

    calls = []
    real_finalize = m2.module_2_81_finalize_fact_ledger

    def fake_specific(*args, **kwargs):
        calls.append("2.8")
        return 60, "男", "肺炎", repr([("咳嗽", "症状", "起病", "D-2")])

    def spy_finalize(row, staging_system, max_rounds=5):
        calls.append("2.81")
        return real_finalize(row, staging_system, max_rounds=max_rounds)

    def fake_chief(*args, **kwargs):
        calls.append("2.9")
        return "咳嗽2天"

    monkeypatch.setattr(m2, "module_2_8_specific_values", fake_specific)
    monkeypatch.setattr(m2, "module_2_81_finalize_fact_ledger", spy_finalize)
    monkeypatch.setattr(m2, "module_2_9_derive_chief_complaint", fake_chief)

    result = m2.process_csv_module2(str(csv_path), num_patients=1)

    assert result["status"] == "success"
    assert calls == ["2.8", "2.81", "2.9"]
    row_payload = json.loads((Path(str(csv_path) + ".module_io") / "row_1.json").read_text(encoding="utf-8"))
    assert row_payload["模块2.81_输出"]["status"] == "converged"
    assert (Path(str(csv_path) + ".module_io") / "row_1.fact_ledger.json").exists()
    reloaded_row_1, reloaded = m2._load_existing_csv(str(csv_path))
    assert "模块2.81_输出" not in reloaded[1]
    assert reloaded[1][m2.COL_CHIEF_COMPLAINT] == "咳嗽2天"
