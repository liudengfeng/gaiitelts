from pathlib import Path

import streamlit as st
from menu import menu

from gailib.st_helper import add_exercises_to_db, on_project_changed

st.set_page_config(
    page_title="帮助中心",
    page_icon="🛠️",
    layout="centered",
)
menu()
on_project_changed("帮助中心")
add_exercises_to_db()

CURRENT_CWD: Path = Path(__file__).parent.parent
VIDEO_DIR = CURRENT_CWD / "resource/video_tip"


# region 常见问题

st.subheader(":information_source: 产品介绍", divider="rainbow", anchor="产品介绍")

st.subheader(":woman-tipping-hand: 常见问题", divider="rainbow", anchor="常见问题")

with st.expander(":bulb: 如何注册？", expanded=False):
    # vfp = VIDEO_DIR / "单词" / "基础词库整体加入个人词库.mp4"
    # st.video(str(vfp))
    pass

with st.expander(":bulb: 如何订阅？", expanded=False):
    # vfp = VIDEO_DIR / "单词" / "基础词库整体加入个人词库.mp4"
    # st.video(str(vfp))
    pass

with st.expander(":bulb: 如何登录？", expanded=False):
    fp = VIDEO_DIR / "如何登录.mp4"
    st.video(str(fp))

with st.expander(":bulb: 忘记密码怎么办？", expanded=False):
    # vfp = VIDEO_DIR / "单词" / "基础词库整体加入个人词库.mp4"
    # st.video(str(vfp))
    pass

with st.expander(":bulb: 如何调整布局让屏幕显示更美观？", expanded=False):
    fp = VIDEO_DIR / "如何调整布局.mp4"
    st.video(str(fp))

# endregion

# region 使用指南
st.subheader("使用指南-记忆单词", divider="rainbow", anchor="使用指南")

with st.expander(":bulb: 如何使用'闪卡'方式记忆单词？", expanded=False):
    fp = str(VIDEO_DIR / "单词" / "闪卡记忆单词.mp4")
    video_file = open(fp, "rb")
    video_bytes = video_file.read()
    st.video(video_bytes)

with st.expander(":bulb: 如何使用'单词拼图'方式记忆单词？", expanded=False):
    fp = str(VIDEO_DIR / "单词" / "单词拼图.mp4")
    video_file = open(fp, "rb")
    video_bytes = video_file.read()
    st.video(video_bytes)

with st.expander(":bulb: 如何将单词添加到个人词库？", expanded=False):
    fp = str(VIDEO_DIR / "单词" / "个人词库逐词添加.mp4")
    video_file = open(fp, "rb")
    video_bytes = video_file.read()
    st.video(video_bytes)

with st.expander(":bulb: 如何将单词从个人词库中删除？", expanded=False):
    fp = str(VIDEO_DIR / "单词" / "个人词库逐词删除.mp4")
    video_file = open(fp, "rb")
    video_bytes = video_file.read()
    st.video(video_bytes)

with st.expander(":bulb: 如何把一个基础词库整体添加到个人词库？", expanded=False):
    fp = str(VIDEO_DIR / "单词" / "基础词库整体加入个人词库.mp4")
    video_file = open(fp, "rb")
    video_bytes = video_file.read()
    st.video(video_bytes)

with st.expander(":bulb: 如何删除个人词库？", expanded=False):
    fp = str(VIDEO_DIR / "单词" / "删除个人词库.mp4")
    video_file = open(fp, "rb")
    video_bytes = video_file.read()
    st.video(video_bytes)

with st.expander(":bulb: 如何进行单词拼图游戏？", expanded=False):
    fp = str(VIDEO_DIR / "单词" / "如何进行单词拼图游戏.mp4")
    video_file = open(fp, "rb")
    video_bytes = video_file.read()
    st.video(video_bytes)

st.subheader("使用指南-阅读练习", divider="rainbow", anchor="使用指南")

# with st.expander(":bulb: 如何进行阅读练习？", expanded=False):
#     fp = str(VIDEO_DIR / "reading_excise.mp4")
#     video_file = open(fp, "rb")
#     video_bytes = video_file.read()
#     st.video(video_bytes)

# endregion


# region 联系我们
st.subheader("联系我们", divider="rainbow", anchor="联系我们")
# endregion
