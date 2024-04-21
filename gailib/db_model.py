from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Type, Union

from pydantic import BaseModel, Field
from werkzeug.security import check_password_hash, generate_password_hash


class PurchaseType(str, Enum):
    ANNUAL = "年度"
    QUARTERLY = "季度"
    MONTHLY = "月度"
    WEEKLY = "每周"
    DAILY = "每日"


class UserRole(str, Enum):
    WORD = "单词"
    WORD_VIP = "单词VIP"
    SPOKEN = "口语"
    WRITING = "写作"
    USER = "用户"
    VIP = "VIP"
    SVIP = "超级成员"
    ADMIN = "管理员"


class PaymentStatus(str, Enum):
    PENDING = "待定"
    RETURNED = "返还"
    IN_SERVICE = "服务中"
    COMPLETED = "完成"


class LoginEvent(BaseModel):
    phone_number: str = Field("", max_length=15)
    login_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    logout_time: Optional[datetime] = Field(default=None)
    session_id: str = Field("")

    @classmethod
    def from_doc(cls, doc: dict):
        return cls(**doc)


class TokenUsageRecord(BaseModel):
    phone_number: str = Field("", max_length=15)
    token_type: str = Field(default="")
    used_token_count: int = Field(default=0)
    used_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_doc(cls, doc: dict):
        return cls(**doc)


def str_to_enum(s: str, enum_type: Type[Enum]) -> Union[Enum, str]:
    for t in enum_type:
        if t.value == s:
            return t
    return None  # type: ignore


class Payment(BaseModel):
    phone_number: str = Field("", max_length=15)
    payment_id: str
    order_id: str
    payment_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    registration_time: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    receivable: float = Field(gt=0.0, le=100000.0)
    discount_rate: float = Field(0.0, ge=0.0)
    payment_amount: float = Field(ge=0.0, le=100000.0)
    purchase_type: PurchaseType = Field(default=PurchaseType.DAILY)
    payment_method: str = Field("")
    status: PaymentStatus = Field(default=PaymentStatus.PENDING)
    is_approved: bool = Field(False)
    sales_representative: str = Field("")
    expiry_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    remark: str = Field("")

    @classmethod
    def from_doc(cls, doc: dict):
        return cls(**doc)


class User(BaseModel):
    email: str = Field("")
    real_name: str = Field("")
    country: str = Field("")
    province: str = Field("")
    timezone: str = Field("")
    phone_number: str = Field("", max_length=15)
    display_name: str = Field("", max_length=100)
    current_level: str = Field("A1")
    target_level: str = Field("C2")
    password: str = Field("")
    personal_vocabulary: List[str] = Field(default_factory=list)
    user_role: UserRole = Field(default=UserRole.USER)
    voice_style: str = Field("en-US-JennyMultilingualNeural")
    registration_time: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    total_tokens: int = Field(default=0)
    memo: Optional[str] = Field("")  # 新增备注字段

    def hash_password(self):
        self.password = generate_password_hash(self.password)

    def check_password(self, password):
        return check_password_hash(self.password, password)

    @classmethod
    def from_doc(cls, doc: dict):
        return cls(**doc)
