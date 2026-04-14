# ai_service_voucher_platform/api_demo.py
import json
import time
import sys
from src.compliance_classifier import ComplianceClassifier

def run_single_demo():
    """
    一个简单的、同步的API请求模拟，用于快速演示核心审核能力。
    """
    print("\n" + "="*70)
    print("🚀 [模式B: 快速演示] 启动单次审核模拟...")

    # --- 模拟三种典型场景 ---
    test_cases = {
        "通过场景 (智能制造)": "我们的服务为客户提供MES生产执行系统的定制开发与产线集成服务。",
        "不通过场景 (通用营销)": "此项服务旨在帮助企业进行市场拓展，包括策划和参与各类线上线下宣传推广活动。",
        "部分通过场景 (描述模糊)": "为企业提供基础的技术升级支持，优化内部管理流程，提高员工办公效率。"
    }
    
    try:
        classifier = ComplianceClassifier()
    except Exception as e:
        print(f"\n❌ [错误] 无法初始化审核器: {e}", file=sys.stderr)
        print("💡 请确认您已将政策文件放入 policy_source 并已运行 watcher.py 来构建知识库。", file=sys.stderr)
        return

    for case_name, description in test_cases.items():
        print(f"\n--- 正在测试场景: {case_name} ---")
        print(f"  输入描述: '{description[:40]}...'")

        start_time = time.time()
        result = classifier.classify_service(description)
        end_time = time.time()
        
        response_payload = {
            "conclusion": result.conclusion,
            "reason": result.primary_reason,
            "suggestion": result.suggestions[0] if result.suggestions else None,
            "process_time_ms": round((end_time - start_time) * 1000)
        }

        print("  [AI审核结果]:")
        # 使用更美观的格式打印
        print(f"    结论: {response_payload['conclusion']}")
        print(f"    原因: {response_payload['reason']}")
        if response_payload['suggestion']:
            print(f"    建议: {response_payload['suggestion']}")
        print(f"    耗时: {response_payload['process_time_ms']} ms")

    print("\n" + "="*70)

if __name__ == "__main__":
    run_single_demo()