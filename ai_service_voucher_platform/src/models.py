from typing import List, Dict, Literal, Optional
from pydantic import BaseModel, Field

ConclusionType = Literal["通过", "不通过", "部分通过", "无法判断"]

class ComplianceResult(BaseModel):
    conclusion: ConclusionType = Field(..., description="审核结论")
    confidence_score: float = Field(..., description="置信度分数")
    primary_reason: str = Field(..., description="主要原因/判定依据")
    suggestions: List[str] = Field(default_factory=list, description="修改建议列表")
    matched_rules: List[Dict] = Field(default_factory=list, description="命中的规则详情")
    matched_keywords: List[str] = Field(default_factory=list, description="命中的关键词")