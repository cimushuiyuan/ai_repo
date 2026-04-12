# ai_service_voucher_platform/main_api.py
import time
import httpx
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List
import logging
import os

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AI_Audit_API")

# --- 导入我们的核心审核逻辑 ---
from src.compliance_classifier import ComplianceClassifier

app = FastAPI(
    title="制造业服务券AI审核API",
    description="根据广西制造业培优育强政策，提供服务产品上架内容的合规性审核。",
    version="2.0-beta"
)

# --- 定义甲方要求的通知回调地址 (从环境变量获取，如果未设置则使用文档中的默认值) ---
CLIENT_NOTIFY_URL = os.getenv("NOTIFY_URL", "https://www.gxgeq.com/sme-api/smeback/product/ai/notify/")
logger.info(f"回调通知地址配置为: {CLIENT_NOTIFY_URL}")

# --- 定义数据模型以匹配甲方接口规范 ---
class ProductCheckRequest(BaseModel):
    batch: str = Field(..., description="批次号")
    size: int = Field(..., description="本批次产品总数")
    productContentMap: Dict[str, str] = Field(..., description="产品ID到产品描述的映射")

# --- 核心后台处理任务 ---
def process_batch_and_notify(batch_id: str, product_map: Dict[str, str]):
    logger.info(f"[{batch_id}] 开始在后台处理 {len(product_map)} 个产品...")
    try:
        classifier = ComplianceClassifier()
    except Exception as e:
        logger.error(f"[{batch_id}] 致命错误: 无法加载审核器内核。审核中止。错误: {e}")
        return

    product_ids = list(product_map.keys())
    total = len(product_ids)
    
    # 结果映射: (我们的结论 -> 甲方的数字代码)
    result_mapping = {"通过": 1, "不通过": 2, "部分通过": 3, "无法判断": 4}

    with httpx.Client(timeout=15.0) as client:
        for i, product_id in enumerate(product_ids):
            description = product_map.get(product_id, "")
            logger.info(f"[{batch_id}] 正在审核产品 {i+1}/{total} (ID: {product_id})...")
            
            # 调用我们的专家模型
            ai_result = classifier.classify_service(description)
            
            # 准备发送给甲方的数据
            reason_text = ai_result.primary_reason
            if ai_result.suggestions:
                # 将建议也并入原因字段
                reason_text += f" [修改建议: {ai_result.suggestions[0]}]"

            payload = {
                "result": result_mapping.get(ai_result.conclusion, 4),
                "reason": reason_text[:255], # 按甲方要求限制原因长度
                "productId": product_id,
                "hasNext": 1 if (i + 1) < total else 0, # 判断是否为最后一条
                "batch": batch_id
            }

            # 异步回调通知甲方
            try:
                logger.info(f"[{batch_id}] 正在回调通知产品 {product_id} 的结果...")
                response = client.post(CLIENT_NOTIFY_URL, json=payload)
                response.raise_for_status() # 如果HTTP状态码是4xx或5xx，则抛出异常
                logger.info(f"[{batch_id}] 通知成功，产品ID: {product_id}，甲方返回: {response.text}")
            except httpx.RequestError as exc:
                logger.error(f"[{batch_id}] 回调通知甲方失败 (ID: {product_id})。错误: {exc}")
            
            # 根据实际情况可以调整休眠时间
            time.sleep(0.2)
    
    logger.info(f"[{batch_id}] 所有产品处理和通知完成。")

# --- API 端点 ---
@app.post(
    "/servceName1/check_service_content/",
    summary="接收服务产品批量审核请求",
    description="接收甲方系统发送的待审核产品列表，立即返回接收成功响应，并在后台异步处理及回调通知。"
)
async def check_service_content(request: ProductCheckRequest, background_tasks: BackgroundTasks):
    """
    符合甲方规范的主接口。
    """
    if not request.productContentMap:
        raise HTTPException(status_code=400, detail="productContentMap不能为空。")
    
    # 立即将耗时任务添加到后台
    background_tasks.add_task(process_batch_and_notify, request.batch, request.productContentMap)

    # 按照规范，立即返回成功接收的响应
    logger.info(f"已接收批次 {request.batch}，包含 {request.size} 个产品。已转入后台异步处理。")
    return {
        "code": 200,
        "msg": f"batch {request.batch}接收成功"
    }

@app.get("/", summary="服务状态检查")
def read_root():
    return {"status": "AI审核API服务正在运行", "version": app.version}