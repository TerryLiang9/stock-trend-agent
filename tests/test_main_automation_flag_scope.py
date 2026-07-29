import main


def test_automation_flag_is_resolved_from_module_scope():
    """默认启动路径必须能调用自动化开关，不能被 main 内局部导入遮蔽。"""
    assert callable(main.automation_enabled)
    assert "automation_enabled" not in main.main.__code__.co_varnames
