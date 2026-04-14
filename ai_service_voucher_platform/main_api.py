# ai_service_voucher_platform/main_api.py
import time
import httpx
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from typing import Dict
import logging
import os

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AI_Audit_API")

# --- 导入核心审核逻辑 ---
from src.compliance_classifier import ComplianceClassifier

app = FastAPI(
    title="制造业服务券AI审核API",
    description="根据广西制造业培优育强政策，提供服务产品上架内容的合规性审核。",
    version="2.0-beta"
)

# --- 定义甲方要求的通知回调地址 ---
# CLIENT_NOTIFY_URL = os.getenv("NOTIFY_URL", "https://www.gxgeq.com/sme-api/smeback/product/ai/notify/")
CLIENT_NOTIFY_URL = os.getenv("NOTIFY_URL", "https://webhook.site/3d6a1f2b-b15d-4517-9e3f-85498db104c1")

logger.info(f"回调通知地址配置为: {CLIENT_NOTIFY_URL}")

# --- 初始化审核器对象 ---
classifier = ComplianceClassifier()

# --- 定义数据模型以匹配甲方接口规范 ---
class ProductCheckRequest(BaseModel):
    batch: str = Field(..., description="批次号")
    size: int = Field(..., description="本批次产品总数")
    productContentMap: Dict[str, str] = Field(..., description="产品ID到产品描述的映射")

# --- 核心后台处理任务 ---
def process_batch_and_notify(batch_id: str, product_map: Dict[str, str]):
    """
    后台任务：逐条调用AI审核，并把结果推给甲方。
    """
    product_ids = list(product_map.keys())
    total = len(product_ids)
    
    logger.info(f"[{batch_id}] 开始在后台处理 {total} 个产品...")
    
    # 结果结论映射
    result_mapping = {"通过": 1, "不通过": 2, "部分通过": 3, "无法判断": 4}

    with httpx.Client(timeout=15.0) as client:
        
        # ==================== 【重点：1. 登录获取 Token】 ====================
        login_url = "https://www.gxgeq.com/sme-api/login"
        login_payload = {
            "username": "guet",
            "password": "Guet20260406!"
        }
        
        try:
            logger.info(f"[{batch_id}] 正在请求甲方系统获取 Token...")
            login_resp = client.post(login_url, json=login_payload)
            login_resp.raise_for_status() 
            
            # 解析返回的 JSON数据，拿到 token
            token = login_resp.json().get("token")
            if not token:
                logger.error(f"[{batch_id}] 获取Token失败: 甲方未返回token字段。响应内容: {login_resp.text}")
                return # 没拿到token就不往下走了
                
            logger.info(f"[{batch_id}] 成功获取Token！")
        except Exception as e:
            logger.error(f"[{batch_id}] 登录请求失败，中止回调。错误: {e}")
            return
            
        # ==================== 【重点：2. 组装请求头】 ====================
        # 通常采用 Authorization: Bearer <token> 的格式
        notify_headers = {
            "Authorization": f"Bearer {token}"
            # 如果依然报401，请尝试将其替换为: "token": token
        }

        # ==================== 【原有逻辑：3. 循环审核并带Token发通知】 ====================
        for i, product_id in enumerate(product_ids):
            description = product_map.get(product_id, "")
            logger.info(f"[{batch_id}] 正在审核产品 {i+1}/{total} (ID: {product_id})...")
            
            # 调用你的AI专家模型
            try:
                ai_result = classifier.classify_service(description)
                reason_text = ai_result.primary_reason
                if ai_result.suggestions:
                    reason_text += f" [修改建议: {ai_result.suggestions[0]}]"
            except Exception as e:
                logger.error(f"[{batch_id}] AI 审核因异常失败: {e}")
                ai_result = type('obj', (object,), {'conclusion': '无法判断'})()
                reason_text = "AI审核模型内部异常"

            # 组装返回给甲方的 JSON 数据
            payload = {
                "result": result_mapping.get(getattr(ai_result, 'conclusion', '无法判断'), 4),
                "reason": reason_text[:255], # 截断一下防止甲方数据库塞不下报错
                "productId": product_id,
                "hasNext": 1 if (i + 1) < total else 0,
                "batch": batch_id
            }

            # 带着刚才获取的 headers 强推给甲方
            try:
                logger.info(f"[{batch_id}] 正在回调通知产品 {product_id} 的结果...")
                
                # 这里加入了 headers=notify_headers
                response = client.post(CLIENT_NOTIFY_URL, json=payload, headers=notify_headers)
                response.raise_for_status() 
                
                logger.info(f"[{batch_id}] 通知成功，产品ID: {product_id}，甲方返回: {response.text}")
            except httpx.RequestError as exc:
                logger.error(f"[{batch_id}] 回调通知甲方失败 (ID: {product_id})。HTTP请求错误: {exc}")
            except Exception as e:
                logger.error(f"[{batch_id}] 回调时发生未捕获的异常: {e}")
            
            # 停顿一下，防止一口气几十个请求把对方测试服务器炸掉
            time.sleep(0.5)
    
    logger.info(f"[{batch_id}] 所有产品处理和通知完成。")

# --- 主 API 接口端点 ---
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
    
    # 将批量跑模型和连环请求的耗时操作放进后台任务中
    background_tasks.add_task(process_batch_and_notify, request.batch, request.productContentMap)
    
    # 瞬间给前端(甲方)回一个 200，证明收到了
    return {
        "code": 200, 
        "msg": f"batch {request.batch}接收成功"
    }