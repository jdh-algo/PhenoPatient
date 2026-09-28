import tempfile
import sys
import types
import unittest
from unittest import mock

gateway_stub = types.ModuleType("JD_gateway_caller")
gateway_stub.analyze_text = lambda *args, **kwargs: ""
sys.modules.setdefault("JD_gateway_caller", gateway_stub)

import phenotypic_atlas as m1
import atlas_based_patient_generation as m2


STAGING_SYSTEM = {
    "name": "测试分级",
    "levels": [
        {"level": 1, "name": "轻度", "description": "", "proportion": 0.5},
        {"level": 2, "name": "重度", "description": "", "proportion": 0.5},
    ],
}

STAGING_TEXT = repr(STAGING_SYSTEM)
VALID_SYMPTOMS = "[('胸痛', ('数小时', '活动后', '压榨样', 0.8), ('持续', '静息时', '剧烈', 0.9))]"
VALID_FLAT = "[('白细胞升高', 0.4, 0.8)]"


class M1SchemaValidationTest(unittest.TestCase):
    def test_module_111_requests_large_output_budget_for_rich_four_level_library(self):
        seen = {}

        def fake_llm(_prompt, **kwargs):
            seen.update(kwargs)
            return VALID_SYMPTOMS

        with mock.patch.object(m1, "call_gpt5", side_effect=fake_llm):
            _prompt, response = m1._module_1_11_v2_fallback(
                "测试病 # 20-80 0.5 急性",
                STAGING_SYSTEM,
            )

        self.assertEqual(response, VALID_SYMPTOMS)
        self.assertEqual(seen.get("max_tokens"), 16384)

    def test_compound_symptom_split_does_not_salvage_truncated_nested_output(self):
        truncated = "[('胸痛', ('数小时', '活动后', '压榨样', 0.8), ('持续', '静息时', '剧烈', 0.9)"

        result = m1._split_compound_symptoms_in_response(truncated)

        self.assertEqual(result, truncated)

    def test_gold_standard_audit_uses_explicit_no_match_result(self):
        def fake_llm(prompt, **_kwargs):
            if "没有符合条件的条目" in prompt and "[]" in prompt:
                return "[]"
            return None

        with mock.patch.object(m1, "call_gpt5", side_effect=fake_llm):
            _prompt, response, corrected, names = m1.module_1_211_audit_lab_tests(
                VALID_FLAT,
                "测试病 # 20-80 0.5 急性",
                STAGING_SYSTEM,
            )

        self.assertEqual(response, "[]")
        self.assertEqual(corrected, VALID_FLAT)
        self.assertEqual(names, [])

    def test_gold_standard_audit_fails_closed_after_empty_response(self):
        with mock.patch.object(m1, "call_gpt5", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "模块1\\.211.*空响应"):
                m1.module_1_211_audit_lab_tests(
                    VALID_FLAT,
                    "测试病 # 20-80 0.5 急性",
                    STAGING_SYSTEM,
                )

    def test_gold_standard_audit_normalizes_textual_no_match(self):
        with mock.patch.object(m1, "call_gpt5", return_value="无"):
            _prompt, response, corrected, names = m1.module_1_211_audit_lab_tests(
                VALID_FLAT,
                "测试病 # 20-80 0.5 急性",
                STAGING_SYSTEM,
            )

        self.assertEqual(response, "[]")
        self.assertEqual(corrected, VALID_FLAT)
        self.assertEqual(names, [])

    def test_gold_standard_audit_parses_list_and_matches_exact_names(self):
        source = repr([
            ("血培养", 0.2, 0.4),
            ("血培养阳性", 0.5, 0.8),
        ])

        with mock.patch.object(m1, "call_gpt5", return_value='["血培养阳性"]'):
            _prompt, _response, corrected, names = m1.module_1_211_audit_lab_tests(
                source,
                "测试病 # 20-80 0.5 急性",
                STAGING_SYSTEM,
            )

        parsed = m1.parse_list_from_response(corrected)
        self.assertEqual(parsed[0], ("血培养", 0.2, 0.4))
        self.assertEqual(parsed[1], ("血培养阳性", 0.5, 1.0))
        self.assertEqual(names, ["血培养阳性"])

    def test_gold_standard_audit_preserves_large_library_order_and_size(self):
        tuples = [(f"检查{i}升高", 0.1, 0.2) for i in range(120)]
        source = repr(tuples)

        with mock.patch.object(m1, "call_gpt5", return_value="检查119升高"):
            _prompt, _response, corrected, names = m1.module_1_211_audit_lab_tests(
                source,
                "测试病 # 20-80 0.5 急性",
                STAGING_SYSTEM,
            )

        parsed = m1.parse_list_from_response(corrected)
        self.assertEqual(len(parsed), 120)
        self.assertEqual([item[0] for item in parsed], [item[0] for item in tuples])
        self.assertEqual(parsed[-1], ("检查119升高", 0.1, 1.0))
        self.assertEqual(names, ["检查119升高"])

    def test_staging_generation_rejects_diagnostic_certainty_as_severity(self):
        diagnostic_certainty = repr({
            "name": "改良Duke诊断标准",
            "levels": [
                {"level": 1, "name": "明确诊断", "description": "", "proportion": 0.6},
                {"level": 2, "name": "可能诊断", "description": "", "proportion": 0.3},
                {"level": 3, "name": "排除诊断", "description": "", "proportion": 0.1},
            ],
        })

        with mock.patch.object(m1, "call_gpt5", return_value=diagnostic_certainty):
            _prompt, response = m1.module_1_0_generate_staging_system(
                "感染性心内膜炎 # 20-80 0.5 急性"
            )

        self.assertIsNone(response)

    def test_staging_generation_rejects_risk_stratification_axis(self):
        risk_stratification = repr({
            "name": "心血管风险分层",
            "levels": [
                {"level": 1, "name": "低危", "description": "未来事件风险较低", "proportion": 0.5},
                {"level": 2, "name": "中危", "description": "未来事件风险中等", "proportion": 0.3},
                {"level": 3, "name": "高危", "description": "未来事件风险较高", "proportion": 0.2},
            ],
        })

        with mock.patch.object(m1, "call_gpt5", return_value=risk_stratification):
            _prompt, response = m1.module_1_0_generate_staging_system(
                "冠心病 # 20-80 0.5 慢性"
            )

        self.assertIsNone(response)

    def test_staging_validation_rejects_named_outcome_risk_scores(self):
        for name in ("ABCD2评分", "PESI分级", "Wells评分", "改良Geneva评分"):
            with self.subTest(name=name):
                risk_score = repr({
                    "name": name,
                    "levels": [
                        {"level": 1, "name": "1级", "description": "30天死亡率较低", "proportion": 0.6},
                        {"level": 2, "name": "2级", "description": "30天死亡率中等", "proportion": 0.3},
                        {"level": 3, "name": "3级", "description": "30天死亡率较高", "proportion": 0.1},
                    ],
                })

                with self.assertRaisesRegex(ValueError, "风险"):
                    m1.validate_m1_staging_system(risk_score)

    def test_staging_validation_rejects_death_risk_descriptions(self):
        mortality_risk = repr({
            "name": "GRACE评分",
            "levels": [
                {"level": 1, "name": "1级", "description": "院内死亡风险较低", "proportion": 0.6},
                {"level": 2, "name": "2级", "description": "院内死亡风险中等", "proportion": 0.3},
                {"level": 3, "name": "3级", "description": "院内死亡风险较高", "proportion": 0.1},
            ],
        })

        with self.assertRaisesRegex(ValueError, "风险"):
            m1.validate_m1_staging_system(mortality_risk)

    def test_staging_validation_allows_curb65_as_current_pneumonia_severity(self):
        curb65 = repr({
            "name": "CURB-65评分",
            "levels": [
                {"level": 1, "name": "轻度", "description": "低死亡风险，适合门诊治疗", "proportion": 0.6},
                {"level": 2, "name": "中度", "description": "中等死亡风险，建议住院", "proportion": 0.3},
                {"level": 3, "name": "重度", "description": "高死亡风险，需评估重症监护", "proportion": 0.1},
            ],
        })

        parsed = m1.validate_m1_staging_system(curb65)

        self.assertEqual(parsed["name"], "CURB-65评分")

    def test_staging_validation_rejects_anatomical_classification_aliases(self):
        stanford = repr({
            "name": "Stanford分类",
            "levels": [
                {"level": 1, "name": "Stanford B型", "description": "轻度表现，不累及升主动脉", "proportion": 0.4},
                {"level": 2, "name": "Stanford A型", "description": "重度表现，累及升主动脉", "proportion": 0.6},
            ],
        })

        with self.assertRaisesRegex(ValueError, "解剖"):
            m1.validate_m1_staging_system(stanford)

    def test_staging_validation_rejects_subtypes_hidden_under_generic_name(self):
        level_name_pairs = [
            ("缺铁性贫血", "溶血性贫血"),
            ("轻度缺铁性贫血", "重度溶血性贫血"),
        ]
        for first_name, second_name in level_name_pairs:
            with self.subTest(level_names=(first_name, second_name)):
                hidden_subtypes = repr({
                    "name": "临床严重度分级",
                    "levels": [
                        {"level": 1, "name": first_name, "description": "症状轻", "proportion": 0.5},
                        {"level": 2, "name": second_name, "description": "症状重", "proportion": 0.5},
                    ],
                })

                with self.assertRaisesRegex(ValueError, "level name"):
                    m1.validate_m1_staging_system(hidden_subtypes)

    def test_staging_validation_allows_disease_names_with_anatomical_words(self):
        valid_systems = [
            {
                "name": "糖尿病周围神经病变临床严重度分级",
                "levels": [
                    {"level": 1, "name": "轻度", "description": "感觉症状轻", "proportion": 0.6},
                    {"level": 2, "name": "重度", "description": "明显功能受损", "proportion": 0.4},
                ],
            },
            {
                "name": "周围动脉疾病Fontaine分期",
                "levels": [
                    {"level": 1, "name": "1期", "description": "无症状", "proportion": 0.4},
                    {"level": 2, "name": "2期", "description": "间歇性跛行", "proportion": 0.3},
                    {"level": 3, "name": "3期", "description": "静息痛", "proportion": 0.2},
                    {"level": 4, "name": "4期", "description": "溃疡或坏疽", "proportion": 0.1},
                ],
            },
        ]

        for staging_system in valid_systems:
            with self.subTest(name=staging_system["name"]):
                parsed = m1.validate_m1_staging_system(staging_system)
                self.assertEqual(parsed["name"], staging_system["name"])

    def test_staging_retry_uses_last_invalid_output_and_validation_error(self):
        risk_stratification = repr({
            "name": "心血管风险分层",
            "levels": [
                {"level": 1, "name": "低危", "description": "未来事件风险较低", "proportion": 0.5},
                {"level": 2, "name": "高危", "description": "未来事件风险较高", "proportion": 0.5},
            ],
        })
        clinical_severity = repr({
            "name": "临床严重度分级",
            "levels": [
                {"level": 1, "name": "轻度", "description": "症状轻，无器官功能障碍", "proportion": 0.6},
                {"level": 2, "name": "重度", "description": "症状重或有器官功能障碍", "proportion": 0.4},
            ],
        })
        prompts = []

        def fake_call(prompt, **_kwargs):
            prompts.append(prompt)
            if (
                "上一轮输出" in prompt
                and "心血管风险分层" in prompt
                and "严重程度分级不得使用风险分层" in prompt
            ):
                return clinical_severity
            return risk_stratification

        with mock.patch.object(m1, "call_gpt5", side_effect=fake_call):
            prompt, response = m1._retry_m10_generate_staging_system(
                "冠心病 # 20-80 0.5 慢性",
                max_retries=2,
                retry_sleep=0,
            )

        self.assertEqual(response, clinical_severity)
        self.assertEqual(prompt, prompts[-1])
        self.assertEqual(len(prompts), 2)
        self.assertIn("上一轮输出", prompts[1])
        self.assertIn("严重程度分级不得使用风险分层", prompts[1])

    def test_staging_generation_rejects_etiology_or_subtype_axis(self):
        etiology_subtype = repr({
            "name": "贫血病因分型",
            "levels": [
                {"level": 1, "name": "缺铁性贫血", "description": "铁缺乏导致", "proportion": 0.4},
                {"level": 2, "name": "巨幼细胞性贫血", "description": "叶酸或维生素B12缺乏导致", "proportion": 0.3},
                {"level": 3, "name": "溶血性贫血", "description": "红细胞破坏增加导致", "proportion": 0.3},
            ],
        })

        with mock.patch.object(m1, "call_gpt5", return_value=etiology_subtype):
            _prompt, response = m1.module_1_0_generate_staging_system(
                "贫血 # 20-80 0.5 慢性"
            )

        self.assertIsNone(response)

    def test_staging_generation_rejects_mixed_etiology_and_severity_axis(self):
        mixed_axis = repr({
            "name": "哮喘病因严重度分型",
            "levels": [
                {"level": 1, "name": "过敏性轻度", "description": "过敏诱发且症状轻", "proportion": 0.45},
                {"level": 2, "name": "感染性中度", "description": "感染诱发且症状中等", "proportion": 0.35},
                {"level": 3, "name": "运动性重度", "description": "运动诱发且症状重", "proportion": 0.20},
            ],
        })

        with mock.patch.object(m1, "call_gpt5", return_value=mixed_axis):
            _prompt, response = m1.module_1_0_generate_staging_system(
                "支气管哮喘 # 20-80 0.5 慢性"
            )

        self.assertIsNone(response)

    def test_staging_generation_rejects_unlisted_histology_prefix_mixed_axis(self):
        mixed_histology_axis = repr({
            "name": "临床严重度分级",
            "levels": [
                {"level": 1, "name": "腺癌轻度", "description": "腺癌且症状轻", "proportion": 0.5},
                {"level": 2, "name": "鳞癌重度", "description": "鳞癌且症状重", "proportion": 0.5},
            ],
        })

        with mock.patch.object(m1, "call_gpt5", return_value=mixed_histology_axis):
            _prompt, response = m1.module_1_0_generate_staging_system(
                "肺癌 # 20-80 0.5 慢性"
            )

        self.assertIsNone(response)

    def test_staging_generation_accepts_same_disease_prefix_severity_axis(self):
        same_prefix_axis = repr({
            "name": "临床严重度分级",
            "levels": [
                {"level": 1, "name": "心衰轻度", "description": "心衰症状轻", "proportion": 0.5},
                {"level": 2, "name": "心衰中度", "description": "心衰症状中等", "proportion": 0.3},
                {"level": 3, "name": "心衰重度", "description": "心衰症状重", "proportion": 0.2},
            ],
        })

        parsed = m1.validate_m1_staging_system(same_prefix_axis)

        self.assertEqual(parsed["name"], "临床严重度分级")

    def test_staging_generation_rejects_duplicate_level_names(self):
        duplicate_levels = repr({
            "name": "临床严重度分级",
            "levels": [
                {"level": 1, "name": "轻度", "description": "症状轻", "proportion": 0.5},
                {"level": 2, "name": "轻度", "description": "症状中等", "proportion": 0.3},
                {"level": 3, "name": "重度", "description": "症状重", "proportion": 0.2},
            ],
        })

        with mock.patch.object(m1, "call_gpt5", return_value=duplicate_levels):
            _prompt, response = m1.module_1_0_generate_staging_system(
                "肺炎 # 20-80 0.5 急性"
            )

        self.assertIsNone(response)

    def test_staging_generation_accepts_ordered_severity_levels(self):
        severity = repr({
            "name": "临床严重度分级",
            "levels": [
                {"level": 1, "name": "轻度", "description": "无器官功能障碍", "proportion": 0.6},
                {"level": 2, "name": "中度", "description": "局部并发症", "proportion": 0.3},
                {"level": 3, "name": "重度", "description": "器官功能障碍", "proportion": 0.1},
            ],
        })

        with mock.patch.object(m1, "call_gpt5", return_value=severity):
            _prompt, response = m1.module_1_0_generate_staging_system(
                "感染性心内膜炎 # 20-80 0.5 急性"
            )

        self.assertIsNotNone(response)
        self.assertEqual(m1.parse_staging_system(response)["name"], "临床严重度分级")

    def test_staging_generation_accepts_kdigo_and_nyha_severity_axes(self):
        accepted = [
            {
                "name": "KDIGO急性肾损伤诊断标准",
                "levels": [
                    {"level": 1, "name": "1期", "description": "轻度肾功能变化", "proportion": 0.6},
                    {"level": 2, "name": "2期", "description": "中度肾功能变化", "proportion": 0.3},
                    {"level": 3, "name": "3期", "description": "重度肾功能变化", "proportion": 0.1},
                ],
            },
            {
                "name": "NYHA心功能分级",
                "levels": [
                    {"level": 1, "name": "I级", "description": "体力活动不受限", "proportion": 0.25},
                    {"level": 2, "name": "II级", "description": "轻度体力活动受限", "proportion": 0.35},
                    {"level": 3, "name": "III级", "description": "明显体力活动受限", "proportion": 0.25},
                    {"level": 4, "name": "IV级", "description": "静息时也有症状", "proportion": 0.15},
                ],
            },
        ]

        for staging_system in accepted:
            with self.subTest(name=staging_system["name"]):
                parsed = m1.validate_m1_staging_system(repr(staging_system))
                self.assertEqual(parsed["name"], staging_system["name"])

    def test_m2_rejects_m1_row_with_non_severity_staging_axis(self):
        row = m1._empty_row()
        row[m1.COL_STAGING_SYSTEM] = repr({
            "name": "血栓栓塞风险分层",
            "levels": [
                {"level": 1, "name": "低危", "description": "血栓风险低", "proportion": 0.5},
                {"level": 2, "name": "高危", "description": "血栓风险高", "proportion": 0.5},
            ],
        })

        with self.assertRaisesRegex(ValueError, "严重程度"):
            m2._validate_m1_row_for_m2(row)

    def test_staging_generation_allows_diagnosis_word_inside_severity_description(self):
        severity = repr({
            "name": "临床严重度分级",
            "levels": [
                {"level": 1, "name": "轻度", "description": "确诊后无器官功能障碍", "proportion": 0.7},
                {"level": 2, "name": "重度", "description": "确诊后出现器官功能障碍", "proportion": 0.3},
            ],
        })

        with mock.patch.object(m1, "call_gpt5", return_value=severity):
            _prompt, response = m1.module_1_0_generate_staging_system(
                "感染性心内膜炎 # 20-80 0.5 急性"
            )

        self.assertIsNotNone(response)

    def test_staging_generation_allows_diagnostic_standard_name_with_severity_levels(self):
        severity = repr({
            "name": "KDIGO急性肾损伤诊断标准",
            "levels": [
                {"level": 1, "name": "1期", "description": "轻度肾功能变化", "proportion": 0.6},
                {"level": 2, "name": "2期", "description": "中度肾功能变化", "proportion": 0.3},
                {"level": 3, "name": "3期", "description": "重度肾功能变化", "proportion": 0.1},
            ],
        })

        with mock.patch.object(m1, "call_gpt5", return_value=severity):
            _prompt, response = m1.module_1_0_generate_staging_system(
                "急性肾损伤 # 20-80 0.5 急性"
            )

        self.assertIsNotNone(response)

    def test_m1_check_prompt_does_not_force_all_stages_to_probability_one(self):
        with mock.patch.object(m1, "call_gpt5", return_value=VALID_FLAT):
            prompt, _response = m1._module_1_21_v2_fallback(
                "测试病 # 20-80 0.5 急性",
                STAGING_SYSTEM,
            )

        self.assertNotIn("在所有分期均应为 1.0", prompt)

    def test_module_111_returns_none_for_attribute_quadruple_without_symptom_name(self):
        with mock.patch.object(
            m1,
            "call_gpt5",
            return_value="[('数天', '活动后', '胸闷', 0.5)]",
        ):
            _prompt, response = m1._module_1_11_v2_fallback(
                "测试病 # 20-80 0.5 急性",
                STAGING_SYSTEM,
            )

        self.assertIsNone(response)

    def test_flat_m1_modules_return_none_for_combined_numeric_measure_name(self):
        cases = [
            ("1.12", m1._module_1_12_v2_fallback),
            ("1.21", m1._module_1_21_v2_fallback),
            ("1.22", m1._module_1_22_v2_fallback),
            ("1.23", m1._module_1_23_v2_fallback),
        ]
        for label, func in cases:
            with self.subTest(module=label):
                with mock.patch.object(
                    m1,
                    "call_gpt5",
                    return_value="[('平均跨瓣压差升高并瓣口面积缩小', 0.4, 0.8)]",
                ):
                    _prompt, response = func(
                        "主动脉瓣狭窄 # 20-80 0.5 慢性",
                        STAGING_SYSTEM,
                    )
                self.assertIsNone(response)

    def test_flat_schema_rejects_shared_suffix_composite_measurement(self):
        for name in (
            "C反应蛋白及降钙素原升高",
            "白细胞计数/血小板计数降低",
            "白细胞计数 ／ 血小板计数降低",
        ):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "单一表型"):
                m1.validate_m1_flat_probability_schema(
                    repr([(name, 0.4, 0.8)]),
                    STAGING_SYSTEM,
                    module_name="模块1.21",
                )

    def test_flat_schema_does_not_treat_saturation_or_palpation_as_connectors(self):
        result = m1.validate_m1_flat_probability_schema(
            "[('血氧饱和度降低', 0.4, 0.8), "
            "('皮下结节（可触及直径≥5mm）', 0.2, 0.6), "
            "('中性粒细胞/淋巴细胞比值升高', 0.3, 0.7), "
            "('心动过速（心率>100次/分）', 0.3, 0.7), "
            "('呼吸音减弱（患侧或双侧肺野呼吸音较弱）', 0.2, 0.5)]",
            STAGING_SYSTEM,
            module_name="模块1.12",
        )

        self.assertIn("血氧饱和度降低", result)
        self.assertIn("可触及直径", result)
        self.assertIn("中性粒细胞/淋巴细胞比值升高", result)
        self.assertIn("心动过速", result)
        self.assertIn("患侧或双侧", result)

    def test_functional_schema_rejects_imaging_studies(self):
        for name in (
            "超声心动图示瓣膜赘生物",
            "心脏MRI示延迟强化",
            "冠状动脉CT血管成像示管腔狭窄",
        ):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "功能检查混入影像学项目"):
                m1.validate_m1_flat_probability_schema(
                    repr([(name, 0.4, 0.8)]),
                    STAGING_SYSTEM,
                    module_name="模块1.23",
                )

    def test_evidence_probability_stage_rejects_silently_truncated_flat_list(self):
        with mock.patch.object(m1, "batch_evidence_queries", return_value={}), \
             mock.patch.object(
                 m1,
                 "call_gpt5",
                 return_value="[('白细胞计数升高', 0.4, 0.8)]",
             ):
            _prompt, response = m1._phase3_evidence_probabilities(
                "肺炎 # 20-80 0.5 急性",
                "实验室检查",
                ["白细胞计数升高", "C反应蛋白升高"],
                STAGING_SYSTEM,
                tag="模块1.21",
            )

        self.assertIsNone(response)

    def test_run_with_subdiagnoses_returns_pure_list_and_keeps_raw_responses_in_prompt(self):
        outputs = {
            "糖尿病": "[('糖化血红蛋白升高', 0.7, 0.9)]",
            "糖尿病肾病": "[('糖化血红蛋白升高', 0.5, 0.8), ('尿白蛋白升高', 0.6, 0.9)]",
        }

        def single_fn(sub_seed):
            diagnosis = sub_seed.split("#", 1)[0].strip()
            return f"prompt for {diagnosis}", outputs[diagnosis]

        prompt, response = m1._run_with_subdiagnoses(
            "糖尿病肾病 # 20-80 0.5 慢性",
            ["糖尿病", "糖尿病肾病"],
            single_fn,
            tag="模块1.21",
        )

        self.assertNotIn("[合并后表型项目数", response)
        self.assertNotIn("[原始子诊断响应]", response)
        self.assertEqual(
            m1.parse_list_from_response(response),
            [("糖化血红蛋白升高", 0.7, 0.9), ("尿白蛋白升高", 0.6, 0.9)],
        )
        self.assertIn("[原始子诊断响应]", prompt)
        self.assertIn("糖尿病肾病", prompt)
        self.assertIn("尿白蛋白升高", prompt)

    def test_process_seed_reruns_invalid_existing_m1_and_reaudits_lab(self):
        row = m1._empty_row()
        row[m1.COL_SEED] = "测试病 # 20-80 0.5 急性"
        row[m1.COL_STAGING_SYSTEM] = STAGING_TEXT
        row[m1.COL_LATERALITY_TYPE] = "either"
        row[m1.COL_DIAGNOSIS_COMPONENTS] = "['测试病']"
        row[m1.COL_SYMPTOMS] = "[('数天', '活动后', '胸闷', 0.5)]"
        row[m1.COL_SIGNS] = "[('心动过速', 0.3, 0.8)]"
        row[m1.COL_LAB_TESTS] = "[合并后表型项目数: 1]\n[('白细胞升高', 0.4, 0.8)]\n\n[原始子诊断响应]"
        row[m1.COL_IMAGING] = "[('胸片异常', 0.2, 0.6)]"
        row[m1.COL_FUNCTIONAL_TESTS] = "[('心电图异常', 0.2, 0.6)]"
        row[m1.COL_COMORBIDITIES] = "[]"
        row[m1.COL_COMPLICATION_PHENOTYPES] = "[]"
        row[m1.COL_M211_OUTPUT] = "old audit"
        row[m1.COL_M221_OUTPUT] = "old imaging audit"
        row[m1.COL_M231_OUTPUT] = "old functional audit"

        calls = []
        saved = []
        audit_inputs = []

        def direct_retry(fn, args=(), kwargs=None, module_name=""):
            return fn(*args, **(kwargs or {}))

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(m1, "_load_existing_csv", return_value=(row, {})), \
                 mock.patch.object(m1, "_save_csv", side_effect=lambda _path, rows: saved.append(rows)), \
                 mock.patch.object(m1, "_retry_module_call", side_effect=direct_retry), \
                 mock.patch.object(m1, "module_1_11_generate_symptoms", side_effect=lambda *args, **kwargs: calls.append("1.11") or ("p111", VALID_SYMPTOMS)), \
                 mock.patch.object(m1, "module_1_21_generate_lab_tests", side_effect=lambda *args, **kwargs: calls.append("1.21") or ("p121", VALID_FLAT)), \
                 mock.patch.object(m1, "module_1_211_audit_lab_tests", side_effect=lambda tuples, *args: audit_inputs.append(tuples) or ("p211", "new audit", tuples, [])):
                result = m1.process_seed_module1(
                    "测试病 # 20-80 0.5 急性",
                    tmpdir,
                    enable_evidence=False,
                )

        self.assertEqual(result["status"], "success")
        self.assertIn("1.11", calls)
        self.assertIn("1.21", calls)
        self.assertEqual(audit_inputs, [VALID_FLAT])
        self.assertTrue(saved)
        final_row = saved[-1][0]
        self.assertEqual(final_row[m1.COL_SYMPTOMS], VALID_SYMPTOMS)
        self.assertEqual(final_row[m1.COL_LAB_TESTS], VALID_FLAT)
        self.assertEqual(final_row[m1.COL_M211_OUTPUT], "new audit")
        self.assertEqual(final_row[m1.COL_M221_OUTPUT], "old imaging audit")
        self.assertEqual(final_row[m1.COL_M231_OUTPUT], "old functional audit")

    def test_process_seed_fails_closed_when_m10_returns_no_staging_system(self):
        downstream_calls = []
        saved = []

        def direct_retry(fn, args=(), kwargs=None, module_name=""):
            return fn(*args, **(kwargs or {}))

        def forbidden_downstream(*args, **kwargs):
            downstream_calls.append("downstream")
            raise AssertionError("M1.0 failure must stop before downstream modules")

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(m1, "_load_existing_csv", return_value=(None, {})), \
                 mock.patch.object(m1, "_save_csv", side_effect=lambda _path, rows: saved.append(rows)), \
                 mock.patch.object(m1, "_retry_module_call", side_effect=direct_retry), \
                 mock.patch.object(m1, "_retry_m10_generate_staging_system", return_value=("p10", None)), \
                 mock.patch.object(m1, "module_1_02_generate_laterality", side_effect=forbidden_downstream):
                result = m1.process_seed_module1(
                    "测试病 # 20-80 0.5 急性",
                    tmpdir,
                    enable_evidence=False,
                )

        self.assertEqual(result["status"], "error")
        self.assertIn("模块1.0", result["error"])
        self.assertEqual(downstream_calls, [])
        self.assertTrue(saved)
        self.assertEqual(saved[-1][0][m1.COL_STAGING_SYSTEM], "")


if __name__ == "__main__":
    unittest.main()
