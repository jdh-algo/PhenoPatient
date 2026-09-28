import virtual_clinical_interaction as m3


def test_hmy_negative_times_render_as_onset_not_visit():
    rendered = m3.format_specific_for_prompt(str([
        ("胸痛", "症状", "未标注", "H-3"),
        ("慢性咳嗽", "症状", "未标注", "M-3"),
        ("活动后气短", "症状", "未标注", "Y-2"),
        ("白细胞升高", "实验室检查", "就诊", "D0"),
    ]))

    onset, visit = rendered.split("【就诊期", 1)
    assert "胸痛" in onset
    assert "慢性咳嗽" in onset
    assert "活动后气短" in onset
    assert "白细胞升高" in visit


def test_legacy_three_tuple_specific_items_are_ignored():
    rendered = m3.format_specific_for_prompt(str([
        ("胸痛", "症状", "H-3"),
    ]))

    assert rendered == ""


def test_patient_prompt_contains_total_duration():
    prompt = m3._build_patient_prompt(
        60,
        "男",
        "[('胸痛','症状','起病','D-2')]",
        "心肌梗死",
        "",
        [],
        duration_total="3天",
    )

    assert "患病总时长：3天" in prompt
