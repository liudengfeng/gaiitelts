import datetime
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import List

import pandas as pd

# import mimetypes
import plotly.express as px
import plotly.graph_objects as go
import pytz
import streamlit as st
from google.cloud import firestore

from menu import menu
from gailib.db_interface import PRICES
from gailib.db_model import Payment, PaymentStatus, PurchaseType, str_to_enum
from gailib.google_cloud_configuration import PROJECT_ID
from gailib.st_helper import (
    add_exercises_to_db,
    check_access,
    configure_google_apis,
    get_blob_container_client,
    get_blob_service_client,
    on_project_changed,
    setup_logger,
)

from gailib.utils import get_current_monday

# region 配置

# 创建或获取logger对象
logger = logging.getLogger("streamlit")
setup_logger(logger)

CURRENT_CWD: Path = Path(__file__).parent.parent

st.set_page_config(
    page_title="系统管理",
    page_icon=":gear:",
    layout="wide",
)
menu()
check_access(True)
configure_google_apis()
on_project_changed("系统管理")
add_exercises_to_db()

tz = pytz.timezone(
    st.session_state.dbi.cache.get("user_info", {}).get("timezone", "Asia/Shanghai")
)
# endregion

# region 常量配置

PM_OPTS = list(PaymentStatus)

PAYMENT_COLUMN_CONFIG = {
    "phone_number": "手机号码",
    "payment_id": "付款编号",
    "order_id": "订单编号",
    "payment_time": st.column_config.DatetimeColumn(
        "支付时间",
        min_value=datetime.datetime(2024, 1, 1),
        max_value=datetime.datetime(2134, 1, 1),
        step=60,
    ),
    "registration_time": st.column_config.DatetimeColumn(
        "登记时间",
        min_value=datetime.datetime(2024, 1, 1),
        max_value=datetime.datetime(2134, 1, 1),
        step=60,
    ),
    "sales_representative": "销售代表",
    "purchase_type": st.column_config.SelectboxColumn(
        "套餐类型",
        help="✨ 购买的套餐类型",
        width="small",
        options=list(PurchaseType),
        default=list(PurchaseType)[-1],
        required=True,
    ),
    "receivable": st.column_config.NumberColumn(
        "应收 (元)",
        help="✨ 购买套餐应支付的金额",
        min_value=0.00,
        max_value=10000.00,
        step=0.01,
        format="￥%.2f",
    ),
    "discount_rate": st.column_config.NumberColumn(
        "折扣率",
        help="✨ 享受的折扣率",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        format="%.2f",
    ),
    "payment_method": "付款方式",
    # "real_name": "姓名",
    # "display_name": "显示名称",
    "payment_amount": st.column_config.NumberColumn(
        "实收 (元)",
        help="✨ 购买套餐实际支付的金额",
        min_value=0.01,
        max_value=10000.00,
        step=0.01,
        format="￥%.2f",
    ),
    "is_approved": st.column_config.CheckboxColumn(
        "是否批准",
        help="✨ 选中表示允许用户使用系统",
        default=False,
    ),
    "expiry_time": st.column_config.DatetimeColumn(
        "服务截至时间",
        min_value=datetime.datetime(2024, 1, 1),
        max_value=datetime.datetime(2134, 1, 1),
        step=60,
    ),
    "status": st.column_config.SelectboxColumn(
        "服务状态",
        help="✨ 服务状态",
        width="small",
        options=PM_OPTS,
        default=PM_OPTS[-1],
        required=True,
    ),
    "remark": "服务备注",
}

PAYMENT_COLUMN_ORDER = [
    "phone_number",
    "payment_id",
    "order_id",
    "payment_time",
    "registration_time",
    "sales_representative",
    "purchase_type",
    "receivable",
    "discount_rate",
    "payment_method",
    "payment_amount",
    "is_approved",
    "expiry_time",
    "status",
    "remark",
]

PAYMENT_TIME_COLS = ["payment_time", "expiry_time", "registration_time"]

PAYMENT_EDITABLE_COLS: list[str] = [
    "is_approved",
    "payment_time",
    "expiry_time",
    "phone_number",
    "payment_id",
    "registration_time",
    "sales_representative",
    "purchase_type",
    "receivable",
    "discount_rate",
    "payment_method",
    "payment_amount",
    "status",
    "remark",
]


# endregion

# region 函数

# region 支付管理辅助函数


def get_new_order_id():
    db = st.session_state.dbi.db

    # 获取用于生成订单编号的文档
    doc_ref = db.collection("system").document("order_id_generator")

    # 定义一个事务函数
    @firestore.transactional
    def update_order_id(transaction):
        # 在事务中获取文档的内容
        doc = doc_ref.get(transaction=transaction)

        # 如果文档不存在，创建一个新的文档，设置 "last_order_id" 为 0
        if not doc.exists:
            transaction.set(doc_ref, {"last_order_id": "0"})
            last_order_id = "0"
        else:
            # 获取 "last_order_id" 的值
            last_order_id = doc.get("last_order_id")

        # 将 "last_order_id" 转换为整数，然后加 1
        new_order_id = str(int(last_order_id) + 1).zfill(10)

        # 在事务中更新文档，设置 "last_order_id" 为新的订单编号
        transaction.update(doc_ref, {"last_order_id": new_order_id})

        return new_order_id

    # 开始事务
    transaction = db.transaction()
    new_order_id = update_order_id(transaction)

    return new_order_id


def generate_timestamp(key: str, type: str, idx: int):
    # 获取日期和时间
    if type:
        date = st.session_state.get(f"{key}_{type}_date-{idx}")
        time = st.session_state.get(f"{key}_{type}_time-{idx}")
    else:
        date = st.session_state.get(f"{key}_date-{idx}")
        time = st.session_state.get(f"{key}_time-{idx}")

    # 将日期和时间组合成一个 datetime 对象
    datetime_obj = datetime.datetime.combine(date, time)

    # 设置时区
    datetime_obj = tz.localize(datetime_obj)

    # 转换为 UTC 时区
    datetime_utc = datetime_obj.astimezone(pytz.UTC)

    # 返回字典
    if type:
        return {f"{type}_" + key: datetime_utc}
    else:
        return {key: datetime_utc}


# endregion

# region 处理反馈辅助函数


@st.cache_data(ttl=datetime.timedelta(hours=1))  # 缓存有效期为1小时
def get_feedbacks():
    container_name = "feedback"
    container_client = get_blob_container_client(container_name)

    # 获取blob列表
    blobs_list = container_client.list_blobs()

    # 获取一周前的日期
    one_week_ago = datetime.datetime.now(tz=pytz.UTC) - datetime.timedelta(weeks=1)

    feedbacks = {}
    for blob in blobs_list:
        # 检查 blob 是否在最近一周内创建
        if blob.last_modified >= one_week_ago:
            name, ext = os.path.splitext(blob.name)
            if name not in feedbacks:
                feedbacks[name] = {
                    "txt": None,
                    "webm": None,
                    "delete": False,
                    "view": False,
                }
            if ext == ".txt":
                feedbacks[name]["txt"] = blob.name
            elif ext == ".webm":
                feedbacks[name]["webm"] = blob.name

    return feedbacks


# endregion

# region 统计分析辅助函数


@st.cache_data(ttl=datetime.timedelta(hours=1))  # 缓存有效期为1小时
def get_phone_numbers():
    return st.session_state.dbi.list_usages_phone_number()


@st.cache_data(ttl=datetime.timedelta(hours=1))  # 缓存有效期为1小时
def get_usage_records(phone_number, start_date, end_date):
    data = st.session_state.dbi.get_usage_records(phone_number, start_date, end_date)
    df = pd.DataFrame(data)
    if not df.empty:
        df["cost"] = df["cost"].round(2)
    return df


def display_metric_by_item(df: pd.DataFrame, item: str):
    """
    在给定的数据框中，按照指定的服务项目对成本进行统计，并以大号粗体字展示各个服务项目的总成本和总成本。
    可以选择显示指标的变化情况。

    参数：
    df (pd.DataFrame): 包含成本数据的数据框。
    item (str): 用于分组的服务项目列名。

    返回：
    无返回值，直接显示指标。
    """

    # 计算服务项目的数量
    num_services = df.groupby(item).size().count()

    # 创建等于服务项目数量+1的列
    columns = st.columns(int(num_services) + 1)

    # 计算每个服务项目的总成本
    item_total_costs = df.groupby(item)["cost"].sum()
    # 计算所有服务项目的总成本
    total_cost = df["cost"].sum()

    # 在第一列中显示总成本
    columns[0].metric("总成本", f"{total_cost:.2f} 元")

    # 在剩余的列中显示各个服务项目的总成本
    for i, (item_name, cost) in enumerate(item_total_costs.items(), start=1):
        columns[i].metric(item_name, f"{cost:.2f} 元")


def display_service_costs(df: pd.DataFrame, period: str = "天"):
    # 使用数据框副本
    df = df.copy()
    if period == "天":
        df["timestamp"] = df["timestamp"].dt.date
    else:
        df["timestamp"] = df["timestamp"].dt.hour

    # 对数据进行分组
    grouped = df.groupby("service_name")

    display_metric_by_item(df, "service_name")

    # 创建一个空的 Figure 对象
    fig = go.Figure()

    # 为每个 service_item 创建一个 Bar 对象
    for name, group in grouped:
        fig.add_trace(go.Bar(x=group["timestamp"], y=group["cost"], name=name))

    # 设置图表的布局
    fig.update_layout(
        barmode="stack",  # 设置为堆积柱状图
        xaxis_title="日期",
        xaxis=dict(
            tickformat=(
                "%Y-%m-%d" if period == "天" else "%m-%d %H"
            ),  # 根据周期调整日期的显示格式
        ),
        yaxis_title="成本",
        title="服务项目成本随时间变化的堆积柱状图",
    )
    st.plotly_chart(fig, use_container_width=True)


def display_cost_pie_chart(df: pd.DataFrame):
    df_grouped = df.groupby(["timestamp", "item_name"])["cost"].sum().reset_index()
    # 创建饼图
    fig = px.pie(df_grouped, values="cost", names="item_name", title="项目成本分布")
    # 使用 Streamlit 显示图形
    st.plotly_chart(fig, use_container_width=True)


def display_cost_bar_chart(df: pd.DataFrame, period: str = "天"):
    # 使用数据框副本
    df = df.copy()
    if period == "天":
        df["timestamp"] = df["timestamp"].dt.date
    else:
        df["timestamp"] = df["timestamp"].dt.strftime("%m-%d %H")

    df_grouped = df.groupby(["timestamp", "item_name"])["cost"].sum().reset_index()

    # 对数据进行分组
    grouped = df_grouped.groupby("item_name")

    # 创建一个空的 Figure 对象
    fig = go.Figure()

    # 为每个 item_name 创建一个 Bar 对象
    for name, group in grouped:
        fig.add_trace(go.Bar(x=group["timestamp"], y=group["cost"], name=name))

    # 设置图表的布局
    fig.update_layout(
        barmode="stack",  # 设置为堆积柱状图
        xaxis_title="日期",
        xaxis=dict(
            tickformat=(
                "%Y-%m-%d" if period == "天" else "%m-%d %H"
            ),  # 根据周期调整日期的显示格式
        ),
        yaxis_title="成本",
        title="服务项目成本随时间变化的堆积柱状图",
    )

    # 创建两列
    cols = st.columns(2)

    # 在左侧列中显示堆积柱状图
    cols[0].plotly_chart(fig, use_container_width=True)

    # 在右侧列中单独绘制 item_name 的成本柱状图
    item_costs = df.groupby("item_name")["cost"].sum()
    fig_item = go.Figure([go.Bar(x=item_costs.index, y=item_costs.values)])
    fig_item.update_layout(
        xaxis_title="服务项目",
        yaxis_title="成本",
        title="服务项目成本柱状图",
    )
    cols[1].plotly_chart(fig_item, use_container_width=True)


# endregion

# endregion

# region 侧边栏

item_menu = st.sidebar.selectbox("菜单", options=["支付管理", "处理反馈", "统计分析"])
sidebar_status = st.sidebar.empty()

# endregion

# region 主页

# region 支付管理


if item_menu == "支付管理":
    items = ["订阅登记", "支付管理", "权限设置"]
    tabs = st.tabs(items)
    with tabs[items.index("订阅登记")]:
        st.subheader("订阅登记")
        with st.form(key="payment_form", clear_on_submit=True):
            cols = st.columns(2)
            phone_number = cols[0].text_input(
                "手机号码",
                key="phone_number",
                help="✨ 请输入订阅者可接收短信的手机号码",
                placeholder="请输入订阅者可接收短信的手机号码[必须]",
            )
            sales_representative = cols[1].text_input(
                "销售代表",
                key="sales_representative",
                help="✨ 请提供销售代表的名称（选填）",
                placeholder="请提供销售代表的名称（选填）",
            )
            purchase_type = cols[0].selectbox(
                "套餐类型",
                key="purchase_type",
                help="✨ 请选择套餐类型",
                options=list(PurchaseType),
                index=1,
                format_func=lambda x: x.value,
                # on_change=compute_discount,
            )
            payment_amount = cols[1].number_input(
                "实收金额",
                key="payment_amount",
                help="✨ 请输入实际收款金额",
                value=0.0,
                # on_change=compute_discount,
            )
            payment_method = cols[0].text_input(
                "付款方式",
                key="payment_method",
                help="✨ 请输入付款方式",
                placeholder="必填。付款方式",
            )
            payment_id = cols[1].text_input(
                "付款编号",
                key="payment_id",
                help="✨ 请输入付款编号",
                placeholder="必填。请在付款凭证上查找付款编号",
            )
            cols[0].date_input(
                "支付日期",
                key="payment_time_date-0",
                value=datetime.datetime.now(tz).date(),
                help="✨ 请选择支付日期。登记日期默认为今天。",
            )
            cols[1].time_input(
                "时间",
                key="payment_time_time-0",
                value=datetime.time(0, 0, 0),
                help="✨ 请选择支付时间。登记时间默认为系统处理时间。",
            )
            remark = st.text_input(
                "备注",
                key="remark",
                help="✨ 请输入备注信息",
            )
            is_approved = st.toggle("是否批准")
            if st.form_submit_button(label="登记"):
                if not phone_number:
                    st.error("手机号码不能为空")
                    st.stop()
                if not payment_id:
                    st.error("付款编号不能为空")
                    st.stop()

                order_id = get_new_order_id()

                receivable = PRICES[purchase_type]  # type: ignore
                discount_rate = payment_amount / receivable
                key = "payment_time"
                payment_time = generate_timestamp(key, "", 0)[key]
                payment = Payment(
                    phone_number=phone_number,
                    payment_id=payment_id,
                    registration_time=datetime.datetime.now(datetime.timezone.utc),
                    payment_time=payment_time,
                    expiry_time=datetime.datetime.now(datetime.timezone.utc),
                    receivable=receivable,
                    payment_amount=payment_amount,  # type: ignore
                    purchase_type=str_to_enum(purchase_type, PurchaseType),  # type: ignore
                    order_id=order_id,
                    payment_method=payment_method,
                    discount_rate=discount_rate,
                    sales_representative=sales_representative,
                    is_approved=is_approved,
                    remark=remark,
                )
                st.session_state.dbi.add_payment(payment)
                st.toast(f"成功登记，订单号:{order_id}", icon="🎉")

    with tabs[items.index("支付管理")]:
        st.subheader("查询参数")
        # with st.form(key="query_form", clear_on_submit=True):
        with st.form(key="query_form"):
            # 精确匹配
            t_0_cols = st.columns(4)
            t_0_cols[0].markdown(":rainbow[精确匹配查询]")
            t0 = t_0_cols[1].toggle(
                label="包含",
                key="is_include-0",
                help="✨ 选中表示包含该查询条件，否则表示不包含",
            )
            payment_0_cols = st.columns(4)
            payment_0_cols[0].text_input(label="手机号码", key="phone_number-1")
            payment_0_cols[1].text_input(label="付款编号", key="payment_id-1")
            payment_0_cols[2].text_input(label="订单编号", key="order_id-1")
            payment_0_cols[3].text_input(label="销售代表", key="sales_representative-1")
            # 选项查询
            t_1_cols = st.columns(4)
            t_1_cols[0].markdown(":rainbow[状态查询]")
            t1 = t_1_cols[1].toggle(
                label="包含",
                key="is_include-1",
                help="✨ 选中表示包含该查询条件，否则表示不包含",
            )
            payment_1_cols = st.columns(4)
            payment_1_cols[0].selectbox(
                label="套餐类型",
                key="purchase_type-1",
                options=["All"] + [x.value for x in PurchaseType],
            )
            payment_1_cols[1].selectbox(
                label="支付状态",
                key="status-1",
                options=["All"] + [x.value for x in PaymentStatus],
            )
            payment_1_cols[2].selectbox(
                label="是否批准",
                key="is_approved-1",
                options=["All", False, True],
            )

            # 支付时间
            t_2_cols = st.columns(4)
            t_2_cols[0].markdown(":rainbow[支付期间查询]")
            t2 = t_2_cols[1].toggle(
                label="包含",
                key="is_include-2",
                help="✨ 选中表示包含该查询条件，否则表示不包含",
            )
            payment_2_cols = st.columns(4)
            payment_2_cols[0].date_input(
                "支付【开始日期】",
                key="payment_time_start_date-1",
                value=datetime.datetime.now(tz).date(),
            )
            payment_2_cols[1].time_input(
                "支付【开始时间】",
                key="payment_time_start_time-1",
                value=datetime.time(0, 0, 0),
            )
            payment_2_cols[2].date_input(
                "支付【结束日期】",
                key="payment_time_end_date-1",
                value=datetime.datetime.now(tz).date(),
            )
            payment_2_cols[3].time_input(
                "支付【结束时间】",
                key="payment_time_end_time-1",
                value=datetime.time(23, 59, 59),
            )

            # 服务时间查询
            t_3_cols = st.columns(4)
            t_3_cols[0].markdown(":rainbow[服务期间查询]")
            t3 = t_3_cols[1].toggle(
                label="包含",
                key="is_include-3",
                help="✨ 选中表示包含该查询条件，否则表示不包含",
            )
            payment_3_cols = st.columns(4)
            payment_3_cols[0].date_input(
                "服务【开始日期】",
                key="expiry_time_start_date-1",
                value=datetime.datetime.now(tz).date(),
            )
            payment_3_cols[1].time_input(
                "服务【开始时间】",
                key="expiry_time_start_time-1",
                value=datetime.time(0, 0, 0),
            )
            payment_3_cols[2].date_input(
                "服务【结束日期】",
                key="expiry_time_end_date-1",
                value=datetime.datetime.now(tz).date(),
            )
            payment_3_cols[3].time_input(
                "服务【结束时间】",
                key="expiry_time_end_time-1",
                value=datetime.time(23, 59, 59),
            )

            # 模糊查询
            t_4_cols = st.columns(4)
            t_4_cols[0].markdown(":rainbow[模糊查询]")
            t4 = t_4_cols[1].toggle(
                label="包含",
                key="is_include-4",
                help="✨ 选中表示包含该查询条件，否则表示不包含",
            )
            payment_4_cols = st.columns(2)
            payment_4_cols[0].text_input(
                "支付方式",
                key="payment_method-1",
                help="✨ 要查询的支付方式信息",
            )
            payment_4_cols[1].text_input(
                "备注",
                key="remark-1",
                help="✨ 要查询的备注信息",
            )
            query_button = st.form_submit_button(label="查询")

            if query_button:
                kwargs = {}
                if t0:
                    kwargs.update(
                        {
                            "phone_number": st.session_state.get(
                                "phone_number-1", None
                            ),
                            "payment_id": st.session_state.get("payment_id-1", None),
                            "order_id": st.session_state.get("order_id-1", None),
                            "sales_representative": st.session_state.get(
                                "sales_representative-1", None
                            ),
                        }
                    )
                if t1:
                    kwargs.update(
                        {
                            "purchase_type": (
                                None
                                if st.session_state.get("purchase_type-1", None)
                                == "ALL"
                                else str_to_enum(
                                    st.session_state.get("purchase_type-1", None),
                                    PurchaseType,
                                )
                            ),
                            "status": (
                                None
                                if st.session_state.get("status-1", None) == "ALL"
                                else str_to_enum(
                                    st.session_state.get("status-1", None),
                                    PaymentStatus,
                                )
                            ),
                            "is_approved": (
                                None
                                if st.session_state.get("is_approved-1", None) == "ALL"
                                else st.session_state.get("is_approved-1", None)
                            ),
                        }
                    )

                if t2:
                    kwargs.update(generate_timestamp("payment_time", "start", 1))
                    kwargs.update(generate_timestamp("payment_time", "end", 1))

                if t3:
                    kwargs.update(generate_timestamp("expiry_time", "start", 1))
                    kwargs.update(generate_timestamp("expiry_time", "end", 1))

                if t4:
                    kwargs.update(
                        {
                            "payment_method": st.session_state.get(
                                "payment_method-1", None
                            ),
                            "remark": st.session_state.get("remark-1", None),
                        }
                    )

                # 删除字典中的空值部分【None ""】
                kwargs = {k: v for k, v in kwargs.items() if v}
                st.write(f"{kwargs=}")

                # 检查数据生成的参数及其类型
                # st.write(kwargs)
                # for k, v in kwargs.items():
                #     st.write(f"{k=}, {type(v)=}")
                results = st.session_state.dbi.query_payments(kwargs)
                # 将每个文档转换为字典
                dicts = [{"order_id": doc.id, **doc.to_dict()} for doc in results]
                st.write(f"{dicts=}")
                st.session_state["queried_payments"] = dicts

        st.subheader("支付清单")
        df = pd.DataFrame(st.session_state.get("queried_payments", {}))

        placeholder = st.empty()
        status = st.empty()
        pay_cols = st.columns([1, 1, 8])
        upd_btn = pay_cols[0].button(
            "更新", key="upd_btn", help="✨ 更新数据库中选中的支付记录"
        )
        del_btn = pay_cols[1].button(
            "删除", key="del_btn", help="✨ 在数据库中删除选中的支付记录"
        )
        # # st.divider()
        if df.empty:
            placeholder.info("没有记录")
        else:
            # 将时间列转换为本地时区
            for col in PAYMENT_TIME_COLS:
                if col in df.columns:
                    df[col] = df[col].dt.tz_convert(tz)
            edited_df = placeholder.data_editor(
                df,
                column_config=PAYMENT_COLUMN_CONFIG,
                column_order=PAYMENT_COLUMN_ORDER,
                hide_index=True,
                num_rows="dynamic",
                key="users_payments",
                disabled=[
                    col for col in df.columns if col not in PAYMENT_EDITABLE_COLS
                ],
            )

        # # Access edited data
        if upd_btn and st.session_state.get("users_payments", None):
            users_payments = st.session_state["users_payments"]
            # st.write(f"{users_payments=}")
            for idx, d in users_payments["edited_rows"].items():
                order_id = df.iloc[idx]["order_id"]  # type: ignore
                for key in d.keys():
                    if key in PAYMENT_TIME_COLS:
                        # 检查返回的对象的类型及其值
                        # st.write(f"{type(d[key])=}, {d[key]=}")
                        value = d[key]
                        # 将 'Z' 替换为 '+00:00'
                        value = value.replace("Z", "+00:00")
                        # 将字符串转换为 datetime 对象
                        timestamp = datetime.datetime.fromisoformat(value).astimezone(
                            datetime.timezone.utc
                        )
                        d[key] = timestamp
                st.session_state.dbi.update_payment(order_id, d)
                st.toast(f"更新支付记录，订单号：{order_id}", icon="🎉")
            users_payments["edited_rows"] = {}

        if del_btn and st.session_state.get("users_payments", None):
            users_payments = st.session_state["users_payments"]
            # st.write(f'{users_payments["deleted_rows"]=}')
            for idx in users_payments["deleted_rows"]:
                order_id = df.iloc[idx]["order_id"]  # type: ignore
                st.session_state.dbi.delete_payment(order_id)
                st.toast(f"删除支付记录，订单号：{order_id}", icon="⚠️")
            # 清除删除的行
            users_payments["deleted_rows"] = []

    # with tabs[items.index("权限设置")]:
    #     st.subheader("查询参数")
    #     # with st.form(key="query_form", clear_on_submit=True):
    #     with st.form(key="query_form-2"):
    #         payment_0_cols = st.columns(2)
    #         payment_0_cols[0].text_input(label="手机号码", key="phone-number-2")
    #         payment_0_cols[1].text_input(label="用户姓名", key="user-name-2")
    #         query_button = st.form_submit_button(label="查询")

    #         if query_button:
    #             kwargs = {}
    #             kwargs.update(
    #                 {
    #                     "phone_number": st.session_state.get(
    #                         "phone-number-2", None
    #                     ),
    #                     "user-name": st.session_state.get("user-name-2", None),
    #                 }
    #             )
    #             # 删除字典中的空值部分【None ""】
    #             kwargs = {k: v for k, v in kwargs.items() if v}
    #             st.write(f"{kwargs=}")

    #             results = st.session_state.dbi.query_payments(kwargs)
    #             # 将每个文档转换为字典
    #             dicts = [{"order_id": doc.id, **doc.to_dict()} for doc in results]
    #             st.write(f"{dicts=}")
    #             st.session_state["queried_payments"] = dicts

    #     st.subheader("支付清单")
    #     df = pd.DataFrame(st.session_state.get("queried_payments", {}))

    #     placeholder = st.empty()
    #     status = st.empty()
    #     pay_cols = st.columns([1, 1, 8])
    #     upd_btn = pay_cols[0].button(
    #         "更新", key="upd_btn", help="✨ 更新数据库中选中的支付记录"
    #     )
    #     del_btn = pay_cols[1].button(
    #         "删除", key="del_btn", help="✨ 在数据库中删除选中的支付记录"
    #     )
    #     # # st.divider()
    #     if df.empty:
    #         placeholder.info("没有记录")
    #     else:
    #         # 将时间列转换为本地时区
    #         for col in PAYMENT_TIME_COLS:
    #             if col in df.columns:
    #                 df[col] = df[col].dt.tz_convert(tz)
    #         edited_df = placeholder.data_editor(
    #             df,
    #             column_config=PAYMENT_COLUMN_CONFIG,
    #             column_order=PAYMENT_COLUMN_ORDER,
    #             hide_index=True,
    #             num_rows="dynamic",
    #             key="users_payments",
    #             disabled=[
    #                 col for col in df.columns if col not in PAYMENT_EDITABLE_COLS
    #             ],
    #         )

    #     # # Access edited data
    #     if upd_btn and st.session_state.get("users_payments", None):
    #         users_payments = st.session_state["users_payments"]
    #         # st.write(f"{users_payments=}")
    #         for idx, d in users_payments["edited_rows"].items():
    #             order_id = df.iloc[idx]["order_id"]  # type: ignore
    #             for key in d.keys():
    #                 if key in PAYMENT_TIME_COLS:
    #                     # 检查返回的对象的类型及其值
    #                     # st.write(f"{type(d[key])=}, {d[key]=}")
    #                     value = d[key]
    #                     # 将 'Z' 替换为 '+00:00'
    #                     value = value.replace("Z", "+00:00")
    #                     # 将字符串转换为 datetime 对象
    #                     timestamp = datetime.datetime.fromisoformat(value).astimezone(
    #                         datetime.timezone.utc
    #                     )
    #                     d[key] = timestamp
    #             st.session_state.dbi.update_payment(order_id, d)
    #             st.toast(f"更新支付记录，订单号：{order_id}", icon="🎉")
    #         users_payments["edited_rows"] = {}

    #     if del_btn and st.session_state.get("users_payments", None):
    #         users_payments = st.session_state["users_payments"]
    #         # st.write(f'{users_payments["deleted_rows"]=}')
    #         for idx in users_payments["deleted_rows"]:
    #             order_id = df.iloc[idx]["order_id"]  # type: ignore
    #             st.session_state.dbi.delete_payment(order_id)
    #             st.toast(f"删除支付记录，订单号：{order_id}", icon="⚠️")
    #         # 清除删除的行
    #         users_payments["deleted_rows"] = []


# endregion

# region 处理反馈

elif item_menu == "处理反馈":
    st.subheader("处理反馈", divider="rainbow", anchor=False)
    container_name = "feedback"
    # connect_str = st.secrets["Microsoft"]["AZURE_STORAGE_CONNECTION_STRING"]
    blob_service_client = get_blob_service_client()
    # container_client = blob_service_client.get_container_client(container_name)
    container_client = get_blob_container_client(container_name)
    # 设置缓存为 1 小时，不能实时查看反馈
    feedbacks = get_feedbacks()
    # st.write(f"{feedbacks=}")
    if len(feedbacks):
        # 将反馈字典转换为一个DataFrame
        feedbacks_df = pd.DataFrame(feedbacks.values())
        feedbacks_df.columns = ["文件文件", "视频文件", "删除", "显示"]

        feedbacks_edited_df = st.data_editor(
            feedbacks_df, hide_index=True, key="feedbacks"
        )

        cols = st.columns(2)
        # 添加一个按钮来删除反馈
        if cols[0].button("删除", help="✨ 删除选中的反馈"):
            # 获取要删除的反馈
            edited_rows = st.session_state["feedbacks"]["edited_rows"]
            for idx, vs in edited_rows.items():
                if vs.get("删除", False):
                    try:
                        txt = feedbacks_df.iloc[idx]["文件文件"]
                        webm = feedbacks_df.iloc[idx]["视频文件"]
                        if txt is not None:
                            container_client.delete_blob(txt)
                            feedbacks_df.iloc[idx]["删除"] = True
                            st.toast(f"从blob中删除：{txt}", icon="🎉")
                        if webm is not None:
                            container_client.delete_blob(webm)
                            st.toast(f"从blob中删除：{webm}", icon="🎉")
                    except Exception as e:
                        pass

        if cols[1].button("显示", help="✨ 显示选中的反馈"):
            # 显示反馈
            edited_rows = st.session_state["feedbacks"]["edited_rows"]
            for idx, vs in edited_rows.items():
                if vs.get("显示", False):
                    deleted = feedbacks_df.iloc[idx]["删除"]
                    if not deleted:
                        try:
                            st.divider()
                            txt = feedbacks_df.iloc[idx]["文件文件"]
                            if txt is not None:
                                text_blob_client = blob_service_client.get_blob_client(
                                    container_name, txt
                                )
                                text_data = (
                                    text_blob_client.download_blob()
                                    .readall()
                                    .decode("utf-8")
                                )
                                st.text(f"{text_data}")
                            webm = feedbacks_df.iloc[idx]["视频文件"]
                            if webm is not None:
                                video_blob_client = blob_service_client.get_blob_client(
                                    container_name, webm
                                )
                                video_data = video_blob_client.download_blob().readall()
                                st.video(video_data)
                        except Exception as e:
                            pass

# endregion


# # endregion


# region 创建统计分析页面
elif item_menu == "统计分析":
    st.subheader("统计分析", divider="rainbow", anchor=False)
    tabs = st.tabs(["运行费用", "用户"])
    with tabs[0]:
        phone_number = st.selectbox("选择用户", options=["ALL"] + get_phone_numbers())
        cols = st.columns(2)
        start_date = cols[0].date_input(
            "开始日期",
            value=get_current_monday(
                st.session_state.dbi.cache["user_info"]["timezone"]
            ),
            key="start_date",
        )
        end_date = cols[1].date_input("结束日期", value=None, key="end_date")

        if start_date is None and end_date is None:
            st.warning("请先选择开始日期和结束日期。")
            st.stop()

        if end_date is not None and start_date >= end_date:
            st.error("错误: 结束日期必须大于开始日期.")
            st.stop()

        column_config = {
            "item_name": "项目名称",
            "stars": st.column_config.NumberColumn(
                "Github Stars",
                help="Number of stars on GitHub",
                format="%d ⭐",
            ),
            "url": st.column_config.LinkColumn("App URL"),
            "views_history": st.column_config.LineChartColumn(
                "Views (past 30 days)", y_min=0, y_max=5000
            ),
        }

        st.markdown("##### 运行费用")
        if st.button("统计"):
            df = get_usage_records(phone_number, start_date, end_date)
            if df.empty:
                st.warning("没有记录")
                st.stop()

            if phone_number != "ALL":
                # 显示用户每日各服务项目的费用
                # 使用函数将 'timestamp' 列的值转换为日期
                df["timestamp"] = df["timestamp"].dt.date
                # 按 'service_name' 和 'timestamp' 分组，对 'cost' 进行汇总
                df_grouped = (
                    df.groupby(["service_name", "timestamp"])["cost"]
                    .sum()
                    .reset_index()
                )

                # 使用 plotly 绘制项目柱状图，x 轴为 'timestamp'，y 轴为 'cost'，颜色为 'service_name'
                fig = px.bar(
                    df_grouped,
                    x="timestamp",
                    y="cost",
                    color="service_name",
                    barmode="group",
                )

                # 更新图形的布局
                fig.update_layout(
                    title="一周内各项服务成本",  # 设置图形的标题
                    xaxis_title="日期",  # 设置 x 轴的标题
                    yaxis_title="成本",  # 设置 y 轴的标题
                    legend_title="服务名称",  # 设置图例的标题
                    font=dict(  # 设置字体
                        family="Courier New, monospace", size=18, color="RebeccaPurple"
                    ),
                    xaxis=dict(
                        title="日期", tickformat="%Y年%m月%d日"  # 定义日期的显示格式
                    ),
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                display_service_costs(df)
                display_cost_bar_chart(df)
                display_cost_pie_chart(df)


# endregion

# endregion
