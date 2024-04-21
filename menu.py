import streamlit as st

from gailib.db_interface import DbInterface
from gailib.st_helper import get_firestore_client


def return_home():
    st.sidebar.page_link("Home.py", label="主页", icon="🏠", help="✨ 返回主页。")


def help_page():
    st.sidebar.page_link(
        "pages/30_🛠️_帮助.py",
        label="帮助文档",
        help="✨ 请参考帮助文档以获取详细的使用指南和信息。",
        icon="🛠️",
    )


def authenticated_menu():
    # Show a navigation menu for authenticated users
    return_home()
    st.sidebar.page_link(
        "pages/02_👥_用户.py", label="用户中心", icon="👥", help="✨ 进入用户中心页面。"
    )
    st.sidebar.page_link(
        "pages/12_📚_单词.py",
        label="记忆单词",
        icon="📚",
        help="✨ 进入记忆单词页面。",
        disabled=st.session_state.role
        not in [
            "单词",
            "单词VIP",
            "用户",
            "超级用户",
            "管理员",
        ],
    )
    st.sidebar.page_link(
        "pages/13_💪_练习.py", label="听说练习", icon="💪", help="✨ 进入听说练习页面。"
    )
    st.sidebar.page_link(
        "pages/14_🏄‍♀️_写作.py",
        label="写作练习",
        icon="🏄‍♀️",
        help="✨ 进入写作练习页面。",
    )
    st.sidebar.page_link(
        "pages/15_🔖_评估.py", label="能力评估", icon="🔖", help="✨ 进入能力评估页面。"
    )
    # st.sidebar.page_link(
    #     "pages/29_♊_GAI.py", label="智能AI", icon="♊", help="✨ 进入智能AI页面。"
    # )
    # st.sidebar.page_link(
    #     "pages/31_🧮_数学作业帮手.py",
    #     label="数学作业帮手",
    #     icon="🧮",
    #     help="✨ 数学作业帮手。",
    #     disabled=st.session_state.role
    #     not in [
    #         "超级用户",
    #         "管理员",
    #     ],
    # )
    help_page()
    # if st.session_state.role in ["管理员"]:
    #     st.sidebar.page_link("pages/40_⚙️_系统.py", label="系统管理", icon="⚙️")
    #     st.sidebar.page_link(
    #         "pages/50_test.py",
    #         label="测试",
    #         icon="🧮",
    #         help="✨ 测试",
    #     )


def unauthenticated_menu():
    # Show a navigation menu for unauthenticated users
    return_home()
    st.sidebar.page_link(
        "pages/00_📇_注册.py",
        label="用户注册",
        help="✨ 请注意，您需要先完成注册才能继续。",
        icon="📇",
    )
    # st.sidebar.page_link(
    #     "pages/01_💰_订阅.py",
    #     label="订阅续费",
    #     help="✨ 请选择适合您的套餐选项。",
    #     icon="💰",
    # )
    help_page()
    # st.sidebar.page_link(
    #     "pages/60_🎧_us_voices.py",
    #     label="美式发音",
    #     help="✨ 美式发音示例。",
    #     icon="🎧",
    # )


def menu():
    if "dbi" not in st.session_state:
        st.session_state["dbi"] = DbInterface(get_firestore_client())
    # Determine if a user is logged in or not, then show the correct
    # navigation menu
    if "role" not in st.session_state or st.session_state.role is None:
        unauthenticated_menu()
        return
    authenticated_menu()


def menu_with_redirect():
    # Redirect users to the main page if not logged in, otherwise continue to
    # render the navigation menu
    if "role" not in st.session_state or st.session_state.role is None:
        st.switch_page("Home.py")
    menu()
