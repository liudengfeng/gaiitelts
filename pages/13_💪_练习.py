import io
import json
import logging
import random
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import pytz
import streamlit as st
import streamlit.components.v1 as components
from streamlit_mic_recorder import mic_recorder
from menu import menu

from gailib.constants import (
    CEFR_LEVEL_MAPS,
    CEFR_LEVEL_TOPIC,
    CONTENTS,
    CONTENTS_EN,
    GENRES,
    GENRES_EN,
    NAMES,
    SCENARIO_MAPS,
    TOPICS,
)

# from mypylib.db_model import LearningTime
from gailib.google_ai import (
    generate_dialogue,
    generate_listening_test,
    generate_reading_comprehension_article,
    generate_reading_comprehension_test,
    generate_scenarios,
    load_vertex_model,
    summarize_in_one_sentence,
)
from gailib.st_helper import (
    PRONUNCIATION_SCORE_BADGE_MAPS,
    WORD_COUNT_BADGE_MAPS,
    add_exercises_to_db,
    autoplay_audio_and_display_text,
    check_access,
    configure_google_apis,
    count_non_none,
    display_assessment_score,
    load_mini_dict,
    pronunciation_assessment_for,
    update_and_display_progress,
    get_synthesis_speech,
    is_answer_correct,
    is_aside,
    on_project_changed,
    setup_logger,
    translate_text,
    update_sidebar_status,
    view_md_badges,
)
from gailib.utils import combine_audio_data
from gailib.word_utils import audio_autoplay_elem, count_words_and_get_levels

# region 配置

CURRENT_CWD: Path = Path(__file__).parent.parent
VOICES_FP = CURRENT_CWD / "resource" / "voices.json"

# 创建或获取logger对象
logger = logging.getLogger("streamlit")
setup_logger(logger)

st.set_page_config(
    page_title="练习",
    page_icon=":muscle:",
    layout="wide",
)

menu()
check_access(False)
if st.session_state.role not in [
        "用户",
        "超级用户",
        "管理员",
    ]:
    st.error("您没有权限访问此页面。")
    st.stop()

configure_google_apis()

add_exercises_to_db()

menu_emoji = [
    "🗣️",
    "📖",
    "✍️",
]
menu_names = ["听说练习", "阅读练习"]
menu_opts = [e + " " + n for e, n in zip(menu_emoji, menu_names)]


item_menu = st.sidebar.radio(
    "菜单",
    menu_opts,
    key="menu-radio",
    help="✨ 请选择您要进行的练习项目",
    # on_change=on_menu_changed,
)

st.sidebar.divider()
sidebar_status = st.sidebar.empty()

if "text_model" not in st.session_state:
    st.session_state["text_model"] = load_vertex_model("gemini-pro")


EXERCISE_TYPE_MAPPING = {
    "单项选择": "single_choice",
    "多选选择": "multiple_choice",
    "填空题": "reading_comprehension_fill_in_the_blank",
    "逻辑题": "reading_comprehension_logic",
}

AI_TIPS = {
    "A1": """
* 建议体裁：`A1`是CEFR英语语言能力分级的最低级别，学习者具有基本的英语听说读写能力，能够理解和使用简单的日常用语，能够进行简单的对话，能够阅读和理解简单的文本。
* **记叙文、说明文、新闻报道和人物传记**是适合`A1`级别学习者的文章体裁，这些体裁的文章语言简单，内容易懂，适合学习者阅读和理解。
* 建议内容：**社会、文化、科技、自然和教育**是适合`A1`级别学习者的文章内容，这些内容与学习者的日常生活息息相关，学习者对这些内容有兴趣，也容易理解。
""",
    "A2": """
- 建议体裁：**记叙文、说明文、新闻报道、人物传记、艺术评论**通常语言简单易懂，适合`A2`水平的学习者阅读。
- **议论文、应用文、科研报告**通常语言复杂，包含大量专业术语，不适合`A2`水平的学习者阅读。
- 建议内容：**社会、文化、科技、经济、历史、艺术、自然、体育、教育**等话题通常与`A2`水平学习者的生活息息相关，容易理解。
- **政治**话题通常涉及复杂的概念和术语，不适合`A2`水平的学习者阅读。
""",
    "B1": """
- 建议体裁：**记叙文、说明文、议论文、新闻报道、人物传记和艺术评论**。
- 对`B1`而言，**议论文**内容抽象，逻辑性强；**应用文**内容专业，术语较多；**科研报告**内容专业，术语较多；这类体裁理解难度大。
- 建议内容：除**政治**因内容敏感，理解难度大不合适外，其余**全部合适**。
""",
    "B2": """
- 建议体裁：**记叙文、说明文、议论文、新闻报道、人物传记和艺术评论**。
- **应用文、科研报告**通常对`B2`级英语学习者来说不合适，要么过于简单或要么过于复杂。
- 建议内容：**全部合适**。
""",
    "C1": """
- 建议体裁：**记叙文、说明文、议论文、新闻报道、人物传记和艺术评论**。
- **应用文、科研报告**通常对`C1`级英语学习者来说不合适，要么过于简单或要么过于复杂。
- 建议内容：**全部合适**。
""",
    "C2": """
- 建议体裁：**记叙文、说明文、议论文、新闻报道、人物传记、艺术评论、科研报告**都是适合`C2`级英语学习者的体裁，因为这些体裁通常涉及复杂的概念和思想，需要较高的语言能力才能理解。
- **应用文**通常涉及日常生活中常见的任务，如写信、写电子邮件、填写表格等，对`C2`级英语学习者来说过于简单。
- 建议内容：**社会、文化、科技、经济、历史、政治、艺术、自然、体育、教育**都是适合`C2`级英语学习者的内容，因为这些内容通常涉及广泛的知识和观点，需要较高的语言能力才能理解。
""",
}
# endregion

# region 函数


# region 通用


@st.cache_data(ttl=timedelta(days=1))
def count_words_and_get_levels_for(text, excluded_words=[]):
    mini_dict = load_mini_dict()
    return count_words_and_get_levels(text, mini_dict, True, True, excluded_words)


def display_text_word_count_summary(container, text, excluded_words=[]):
    total_words, level_dict = count_words_and_get_levels_for(text, excluded_words)
    container.markdown(f"**字数统计：{len(text.split())}字**")
    level_dict.update({"单词总量": total_words})
    view_md_badges(container, level_dict, WORD_COUNT_BADGE_MAPS)


def get_voice_style(
    sentence, boy_name, girl_name, m_voice_style, fm_voice_style, default_style
):
    # 如果是旁白，使用默认的声音
    if is_aside(sentence):
        return default_style

    # 发言人姓名
    speaker_name = sentence.split(":", 1)[0]

    # 判断句子是否以男孩或女孩的姓名开头，然后返回相应的声音风格
    if boy_name in speaker_name:
        return m_voice_style[0]
    elif girl_name in speaker_name:
        return fm_voice_style[0]
    else:
        return default_style


# endregion


# region 听力练习


def display_dialogue_summary(container, dialogue, summarize):
    container.markdown("**对话概要**")
    container.markdown(f"{summarize}")
    text = dialogue["text"]
    dialogue_text = " ".join(text)
    boy_name = dialogue["boy_name"]
    girl_name = dialogue["girl_name"]
    display_text_word_count_summary(container, dialogue_text, [boy_name, girl_name])
    container.markdown("**对话内容**")
    for d in text:
        container.markdown(d)


# endregion


@st.cache_data(ttl=timedelta(days=1), show_spinner="正在生成听力测试题，请稍候...")
def generate_listening_test_for(difficulty: str, conversation: str):
    return generate_listening_test(
        st.session_state["text_model"], difficulty, conversation, 5
    )


@st.cache_data(ttl=timedelta(days=1), show_spinner="正在生成阅读理解测试题，请稍候...")
def generate_reading_test_for(difficulty: str, exercise_type, article: List[str]):
    test = generate_reading_comprehension_test(
        st.session_state["text_model"],
        exercise_type,
        5,
        difficulty,
        "\n".join(article),
    )
    # logger.info(test)
    return test


@st.cache_data(ttl=timedelta(days=1), show_spinner="正在加载场景类别，请稍候...")
def generate_scenarios_for(category: str):
    return generate_scenarios(st.session_state["text_model"], category)


@st.cache_data(ttl=timedelta(days=1), show_spinner="正在生成模拟场景，请稍候...")
def generate_dialogue_for(selected_scenario, interesting_plot, difficulty):
    boy_name = random.choice(NAMES["en-US"]["male"])
    girl_name = random.choice(NAMES["en-US"]["female"])
    scenario = selected_scenario.split(".")[1]
    dialogue = generate_dialogue(
        st.session_state["text_model"],
        boy_name,
        girl_name,
        scenario,
        interesting_plot if interesting_plot else "",
        difficulty,
    )
    return {"text": dialogue, "boy_name": boy_name, "girl_name": girl_name}


@st.cache_data(
    ttl=timedelta(days=1), show_spinner="正在生成阅读理解练习文章，请稍候..."
)
def generate_reading_comprehension_article_for(genre, contents, plot, difficulty):
    content = ",".join(contents)
    return generate_reading_comprehension_article(
        st.session_state["text_model"],
        genre,
        content,
        plot,
        difficulty,
    )


@st.cache_data(ttl=timedelta(days=1), show_spinner="正在生成对话概要，请稍候...")
def summarize_in_one_sentence_for(dialogue: str):
    return summarize_in_one_sentence(st.session_state["text_model"], dialogue)


def get_and_combine_audio_data():
    dialogue = st.session_state.conversation_scene["text"]
    audio_data_list = []
    for i, sentence in enumerate(dialogue):
        voice_style = m_voice_style if i % 2 == 0 else fm_voice_style
        style = "en-US-AnaNeural" if is_aside(sentence) else voice_style[0]
        sentence_without_speaker_name = re.sub(
            r"^\w+:\s", "", sentence.replace("**", "")
        )
        with st.spinner(f"使用 Azure 将文本合成语音..."):
            result = get_synthesis_speech(sentence_without_speaker_name, style)
        audio_data_list.append(result["audio_data"])
    return combine_audio_data(audio_data_list)


def autoplay_audio_and_display_article(container):
    container.empty()
    content_cols = container.columns(2)
    article = st.session_state["reading-article"]
    audio_data_list = []
    durations = []
    total = 0
    for i, paragraph in enumerate(article):
        voice_style = m_voice_style if i % 2 == 0 else fm_voice_style
        with st.spinner(f"微软语音合成 第{i+1}段文本..."):
            result = get_synthesis_speech(paragraph, voice_style[0])
        audio_data_list.append(result["audio_data"])
        duration = result["audio_duration"]
        total += duration.total_seconds()
        durations.append(duration)

    # 创建一个空的插槽
    slot_1 = content_cols[0].empty()
    slot_2 = content_cols[1].empty()
    # 如果需要显示中文，那么翻译文本
    if st.session_state.get("ra-display-state", "英文") != "英文":
        cns = translate_text("阅读理解练习", article, "zh-CN", True)

    # 播放音频并同步显示文本
    for i, duration in enumerate(durations):
        # 计算这一段音频的播放长度与总长度的占比
        # 播放音频
        audio_html = audio_autoplay_elem(audio_data_list[i], fmt="wav")
        components.html(audio_html)

        # 检查 session state 的值
        if st.session_state.get("ra-display-state", "英文") == "英文":
            # 显示英文
            slot_1.markdown(f"**{article[i]}**")
        elif st.session_state.get("ra-display-state", "中文") == "中文":
            # 显示中文
            slot_2.markdown(cns[i])
        else:
            # 同时显示英文和中文
            slot_1.markdown(f"**{article[i]}**")
            slot_2.markdown(cns[i])
        # 等待音频播放完毕
        t = duration.total_seconds() + 0.5
        time.sleep(t)
    return total


def process_play_and_record_article(
    container, m_voice_style, fm_voice_style, difficulty, genre
):
    container.empty()
    content_cols = container.columns(2)
    paragraphs = st.session_state["reading-article"]
    cns = translate_text("阅读理解练习", paragraphs, "zh-CN", True)

    idx = st.session_state["reading-exercise-idx"]
    paragraph = paragraphs[idx]
    voice_style = m_voice_style if idx % 2 == 0 else fm_voice_style
    with st.spinner(f"使用 Azure 将文本合成语音..."):
        result = get_synthesis_speech(paragraph, voice_style[0])

    audio_html = audio_autoplay_elem(result["audio_data"], fmt="wav")
    components.html(audio_html)

    if st.session_state["ra-display-state"] == "英文":
        content_cols[0].markdown("英文")
        content_cols[0].markdown(paragraph)
    elif st.session_state["ra-display-state"] == "中文":
        content_cols[1].markdown("中文")
        content_cols[1].markdown(cns[idx])
    else:
        content_cols[0].markdown("英文")
        content_cols[0].markdown(paragraph)
        content_cols[1].markdown("中文")
        content_cols[1].markdown(cns[idx])
    t = result["audio_duration"].total_seconds() + 0.5
    time.sleep(t)


def display_dialogue(dialogue_placeholder):
    dialogue = st.session_state.conversation_scene.get("text", [])
    if len(dialogue) == 0:
        return
    idx = st.session_state["listening-idx"]
    if idx == -1:
        return
    cns = translate_text("听说练习", dialogue, "zh-CN", True)
    sentence = dialogue[idx]

    content_cols = dialogue_placeholder.columns(2)
    if st.session_state["listening-display-state"] == "英文":
        content_cols[0].markdown("英文")
        content_cols[0].markdown(sentence)
    elif st.session_state["listening-display-state"] == "中文":
        content_cols[1].markdown("中文")
        content_cols[1].markdown(cns[idx])
    else:
        content_cols[0].markdown("英文")
        content_cols[0].markdown(sentence)
        content_cols[1].markdown("中文")
        content_cols[1].markdown(cns[idx])


def play_and_record_dialogue(m_voice_style, fm_voice_style):
    dialogue = st.session_state.conversation_scene.get("text")
    if dialogue is None or len(dialogue) == 0:
        return
    idx = st.session_state["listening-idx"]
    if idx == -1:
        return

    boy_name = st.session_state.conversation_scene["boy_name"]
    girl_name = st.session_state.conversation_scene["girl_name"]
    sentence = dialogue[idx]
    # voice_style = m_voice_style if idx % 2 == 0 else fm_voice_style
    # style = "en-US-AnaNeural" if is_aside(sentence) else voice_style[0]
    style = get_voice_style(
        sentence, boy_name, girl_name, m_voice_style, fm_voice_style, "en-US-AnaNeural"
    )
    sentence_without_speaker_name = re.sub(r"^\w+:\s", "", sentence.replace("**", ""))
    with st.spinner(f"使用 Azure 将文本合成语音..."):
        result = get_synthesis_speech(sentence_without_speaker_name, style)

    audio_html = audio_autoplay_elem(result["audio_data"], fmt="wav")
    components.html(audio_html)


def on_prev_btn_click(key):
    st.session_state[key] -= 1


def on_next_btn_click(key):
    st.session_state[key] += 1


def on_word_test_radio_change(idx, options):
    current = st.session_state["listening-test-options"]
    # 转换为索引
    st.session_state["listening-test-answer"][idx] = options.index(current)


def play_listening_test():
    idx = st.session_state["listening-test-idx"]
    test = st.session_state["listening-test"][idx]
    question = test["question"]

    t = 0
    if st.session_state["listening-test-display-state"] == "语音":
        with st.spinner(f"使用 Azure 将文本合成语音..."):
            question_audio = get_synthesis_speech(question, m_voice_style[0])
        audio_html = audio_autoplay_elem(question_audio["audio_data"], fmt="wav")
        components.html(audio_html)
        t = question_audio["audio_duration"].total_seconds() + 0.5
        time.sleep(t)


def view_listening_test(container):
    container.empty()
    idx = st.session_state["listening-test-idx"]
    test = st.session_state["listening-test"][idx]
    question = test["question"]
    options = test["options"]
    user_answer_idx = st.session_state["listening-test-answer"][idx]
    if st.session_state["listening-test-display-state"] != "语音":
        container.markdown(question)
    container.radio(
        "选项",
        options,
        index=user_answer_idx,
        label_visibility="collapsed",
        on_change=on_word_test_radio_change,
        args=(idx, options),
        key="listening-test-options",
    )


def on_reading_test_radio_change(idx, options):
    current = st.session_state["reading-test-options"]
    # 转换为索引
    st.session_state["reading-test-answer"][idx] = options.index(current)


def view_reading_test(container, difficulty, exercise_type, genre):
    idx = st.session_state["reading-test-idx"]
    test = st.session_state["reading-test"][idx]
    question = test["question"]
    options = test["options"]
    user_answer_idx = st.session_state["reading-test-answer"][idx]
    t = 0
    if st.session_state["reading-test-display-state"] == "语音":
        with st.spinner(f"使用 Azure 将文本合成语音..."):
            question_audio = get_synthesis_speech(question, m_voice_style[0])
        audio_html = audio_autoplay_elem(question_audio["audio_data"], fmt="wav")
        components.html(audio_html)
        t = question_audio["audio_duration"].total_seconds() + 0.5
        time.sleep(t)
    else:
        container.markdown(question)

    container.radio(
        "选项",
        options,
        index=user_answer_idx,
        label_visibility="collapsed",
        on_change=on_reading_test_radio_change,
        args=(idx, options),
        key="reading-test-options",
    )

    # 添加一个学习时间记录
    # record = LearningTime(
    #     phone_number=st.session_state.dbi.cache["user_info"]["phone_number"],
    #     project="阅读理解测验",
    #     content=f"{difficulty}-{genre}-{exercise_type}",
    #     word_count=len(question.split()),
    #     duration=t,
    # )
    # st.session_state.dbi.add_record_to_cache(record)


def check_listening_test_answer(container, level, selected_scenario):
    score = 0
    n = count_non_none(st.session_state["listening-test"])
    for idx, test in enumerate(st.session_state["listening-test"]):
        question = test["question"]
        options = test["options"]
        answer = test["answer"]
        explanation = test["explanation"]
        related_sentence = test["related_sentence"]

        # 存储的是 None 或者 0、1、2、3
        user_answer_idx = st.session_state["listening-test-answer"][idx]
        container.divider()
        container.markdown(question)
        container.radio(
            "选项",
            options,
            # horizontal=True,
            index=user_answer_idx,
            disabled=True,
            label_visibility="collapsed",
            key=f"test-options-{idx}",
        )
        msg = ""
        # 用户答案是选项序号，而提供的标准答案是A、B、C、D
        if is_answer_correct(user_answer_idx, answer):
            score += 1
            msg = f"正确答案：{answer} :white_check_mark:"
        else:
            msg = f"正确答案：{answer} :x:"
        container.markdown(msg)
        container.markdown(f"解释：{explanation}")
        container.markdown(f"相关对话：{related_sentence}")
    percentage = score / n * 100
    if percentage >= 75:
        container.balloons()
    container.divider()
    container.markdown(f":red[得分：{percentage:.0f}%]")
    test_dict = {
        # "phone_number": st.session_state.dbi.cache["user_info"]["phone_number"],
        "item": "听力测验",
        "topic": selected_scenario,
        "level": level,
        "score": percentage,
        "duration": (time.time() - st.session_state["listening-start-time"]),
        "record_time": datetime.now(pytz.UTC),
    }
    # st.session_state.dbi.save_daily_quiz_results(test_dict)
    st.session_state.dbi.add_documents_to_user_history("performances", [test_dict])


def check_reading_test_answer(container, difficulty, exercise_type, genre):
    score = 0
    n = count_non_none(st.session_state["reading-test"])
    for idx, test in enumerate(st.session_state["reading-test"]):
        question = test["question"]
        options = test["options"]
        answer = test["answer"]
        explanation = test["explanation"]

        # 存储的是 None 或者 0、1、2、3
        user_answer_idx = st.session_state["reading-test-answer"][idx]
        container.divider()
        container.markdown(question)
        container.radio(
            "选项",
            options,
            # horizontal=True,
            index=user_answer_idx,
            disabled=True,
            label_visibility="collapsed",
            key=f"test-options-{idx}",
        )
        msg = ""
        # 用户答案是选项序号，而提供的标准答案是A、B、C、D
        if is_answer_correct(user_answer_idx, answer):
            score += 1
            msg = f"正确答案：{answer} :white_check_mark:"
        else:
            msg = f"正确答案：{answer} :x:"
        container.markdown(msg)
        container.markdown(f"解释：{explanation}")

    percentage = score / n * 100
    if percentage >= 75:
        container.balloons()
    container.divider()
    container.markdown(f":red[得分：{percentage:.0f}%]")
    test_dict = {
        # "phone_number": st.session_state.dbi.cache["user_info"]["phone_number"],
        "item": "阅读理解测验",
        "topic": genre,
        "level": f"{difficulty}-{exercise_type}",
        "score": percentage,
        "duration": (
            datetime.now(pytz.UTC) - st.session_state["reading-start-time"]
        ).total_seconds(),
        "record_time": datetime.now(pytz.UTC),
    }
    # st.session_state.dbi.save_daily_quiz_results(test_dict)
    st.session_state.dbi.add_documents_to_user_history("performances", [test_dict])


# endregion

# region 会话状态

if "m_voices" not in st.session_state and "fm_voices" not in st.session_state:
    with open(VOICES_FP, "r", encoding="utf-8") as f:
        voices = json.load(f)["en-US"]
    st.session_state["m_voices"] = [v for v in voices if v[1] == "Male"]
    st.session_state["fm_voices"] = [v for v in voices if v[1] == "Female"]

if "conversation_scene" not in st.session_state:
    st.session_state["conversation_scene"] = {}

if "summarize_in_one" not in st.session_state:
    st.session_state["summarize_in_one"] = ""

if "listening-learning-times" not in st.session_state:
    st.session_state["listening-learning-times"] = 0

if "reading-learning-times" not in st.session_state:
    st.session_state["reading-learning-times"] = 0

if "listening-test" not in st.session_state:
    st.session_state["listening-test"] = []

if "listening-test-idx" not in st.session_state:
    st.session_state["listening-test-idx"] = -1

if "listening-test-answer" not in st.session_state:
    st.session_state["listening-test-answer"] = []

if "reading-test" not in st.session_state:
    st.session_state["reading-test"] = []

if "reading-test-idx" not in st.session_state:
    st.session_state["reading-test-idx"] = -1

if "reading-test-answer" not in st.session_state:
    st.session_state["reading-test-answer"] = []

if "listening-test-display-state" not in st.session_state:
    st.session_state["listening-test-display-state"] = "文本"

if "listening-display-state" not in st.session_state:
    st.session_state["listening-display-state"] = "英文"

if "ra-display-state" not in st.session_state:
    st.session_state["ra-display-state"] = "英文"

if "reading-test-display-state" not in st.session_state:
    st.session_state["reading-test-display-state"] = "英文"

if "scenario-list" not in st.session_state:
    st.session_state["scenario-list"] = []

if "reading-article" not in st.session_state:
    st.session_state["reading-article"] = []

if "listening-pronunciation-assessment" not in st.session_state:
    st.session_state["listening-pronunciation-assessment"] = None

# endregion

# region 通用

update_sidebar_status(sidebar_status)

if "stage" not in st.session_state:
    st.session_state["stage"] = 0


def set_state(i):
    st.session_state.stage = i


# endregion

# region 听说练习

if item_menu is not None and item_menu.endswith("听说练习"):
    on_project_changed("听说练习")
    m_voice_style = st.sidebar.selectbox(
        "合成男声风格",
        st.session_state["m_voices"],
        # on_change=on_voice_changed,
        help="✨ 选择您喜欢的合成男声语音风格",
        format_func=lambda x: f"{x[2]}",  # type: ignore
    )
    fm_voice_style = st.sidebar.selectbox(
        "合成女声风格",
        st.session_state["fm_voices"],
        # on_change=on_voice_changed,
        help="✨ 选择您喜欢的合成女声语音风格",
        format_func=lambda x: f"{x[2]}",  # type: ignore
    )

    listening_tabs = st.tabs(["配置场景", "开始练习", "小测验"])

    # region "配置场景"

    with listening_tabs[0]:
        st.subheader("配置场景", divider="rainbow", anchor="配置场景")
        st.markdown("依次执行以下步骤，生成听说练习模拟场景。")
        steps = [
            "1. CEFR等级",
            "2. 场景类别",
            "3. 选择场景",
            "4. 添加情节",
            "5. 预览场景",
        ]
        sub_tabs = st.tabs(steps)
        scenario_category = None
        selected_scenario = None
        interesting_plot = None
        difficulty = None

        with sub_tabs[0]:
            st.info("第一步：点击下拉框选择CEFR等级", icon="🚨")
            difficulty = st.selectbox(
                "CEFR等级",
                list(CEFR_LEVEL_MAPS.keys()),
                key="listening-difficulty",
                index=0,
                format_func=lambda x: f"{x}({CEFR_LEVEL_MAPS[x]})",
                on_change=set_state,
                args=(1,),
                placeholder="请选择CEFR等级",
            )
            on_project_changed("听说练习-CEFR等级")

        with sub_tabs[1]:
            st.info("第二步：点击下拉框选定场景类别", icon="🚨")
            if st.session_state.stage == 1 or difficulty is not None:
                scenario_category = st.selectbox(
                    "场景类别",
                    CEFR_LEVEL_TOPIC[difficulty],
                    # index=None,
                    index=0,
                    on_change=set_state,
                    args=(2,),
                    key="scenario_category",
                    placeholder="请选择场景类别",
                )
                on_project_changed("听说练习-场景类别")
            # logger.info(f"{st.session_state.stage=}")

        with sub_tabs[2]:
            st.info(
                "第三步：点击下拉框，选择您感兴趣的场景。如果下拉框中没有可选项目，或者您希望 AI 生成新的场景，只需点击 '刷新[🔄]' 按钮。请注意，AI 生成新场景的过程可能需要 6-12 秒。",
                icon="🚨",
            )
            if st.session_state.stage == 2 or scenario_category is not None:
                if st.button(
                    "刷新[:arrows_counterclockwise:]", key="generate-scenarios"
                ):
                    st.session_state["scenario-list"] = generate_scenarios_for(
                        scenario_category
                    )
                    # st.write(st.session_state["scenario-list"])

                # st.write(scenario_list)
                selected_scenario = st.selectbox(
                    "选择场景",
                    st.session_state["scenario-list"],
                    key="selected_scenario",
                    index=0,
                    on_change=set_state,
                    args=(3,),
                    placeholder="请选择您感兴趣的场景",
                )
                on_project_changed(f"听说练习-选择场景")

        with sub_tabs[3]:
            st.info(
                "第四步：可选。可在文本框内添加一些有趣的情节以丰富听力练习材料。如果您想跳过这一步，可以选择'跳过'。",
                icon="🚨",
            )
            ignore = st.toggle("跳过", key="add_interesting_plot", value=True)
            if ignore:
                st.session_state.stage = 4
            st.divider()
            if st.session_state.stage == 3 or selected_scenario is not None:
                interesting_plot = st.text_area(
                    "添加一些有趣的情节【可选】",
                    height=200,
                    key="interesting_plot",
                    on_change=set_state,
                    args=(4,),
                    placeholder="""您可以在这里添加一些有趣的情节。比如：
- 同事问了一个非常奇怪的问题，让您忍俊不禁。
- 同事在工作中犯了一个错误，但他能够及时发现并改正。
                """,
                )
                on_project_changed("听说练习-添加情节")

        with sub_tabs[4]:
            on_project_changed("听说练习-生成对话")
            st.info(
                """在完成所有步骤后，您可以在此处生成并查看详细的对话场景。生成对话场景后，您可以切换到最上方👆的 "开始练习" 标签页，开始进行听力和口语练习。""",
                icon="🚨",
            )
            if selected_scenario is None or difficulty is None:
                st.warning("您需要先完成之前的所有步骤")

            session_cols = st.columns(8)

            container = st.container()

            gen_btn = session_cols[0].button(
                "刷新[:arrows_counterclockwise:]",
                key="generate-dialogue",
                help="✨ 点击按钮，生成对话场景。",
            )

            if gen_btn:
                if selected_scenario is None:
                    st.warning("需要完成第三步，选择您感兴趣的场景")
                    st.stop()

                container.empty()
                # 学习次数重置为0
                st.session_state["listening-learning-times"] = 0

                dialogue = generate_dialogue_for(
                    selected_scenario, interesting_plot, difficulty
                )
                summarize = summarize_in_one_sentence_for(dialogue["text"])

                display_dialogue_summary(container, dialogue, summarize)

                st.session_state.conversation_scene = dialogue
                st.session_state.summarize_in_one = summarize

            elif len(st.session_state.conversation_scene.get("text", [])) > 0:
                display_dialogue_summary(
                    container,
                    st.session_state.conversation_scene,
                    st.session_state.summarize_in_one,
                )

    # endregion

    # region "听说练习"

    with listening_tabs[1]:
        st.subheader("听说练习", divider="rainbow", anchor="听说练习")
        st.markdown(
            """
您可以通过反复播放和跟读每条对话样例来提升您的听力和口语技能。点击 '全文[🎞️]' 可以一次性收听整个对话。另外，您可以通过点击左侧的按钮调整合成语音的风格，以更好地适应您的听力习惯。      
"""
        )
        st.warning(
            "请注意，练习过程中会使用喇叭播放音频。为了避免音量过大或过小影响您的体验，请提前调整到适合的音量。",
            icon="🚨",
        )
        with st.expander("✨ 跟读录音提示", expanded=False):
            st.markdown(
                """\
- 跟读当前显示的对话内容【不包括发言人的名称】，以进行发音练习。
- 首次点击 '录音[⏸️]' 按钮，开始录音。
- 再次点击 '停止[🔴]' 按钮，结束录音。
- 录音结束后，点击 '评估[🔖]' 按钮，系统将对用户的发音进行全面评估。评估标准包括发音的准确度、语流的流畅性、发音的完整性和韵律感，以及一个综合性的评分。
- 点击 '回放[▶️]' 按钮，可以回放用户的跟读录音。
- 通过这种方式，用户可以得到关于其发音水平的反馈，从而有针对性地进行改进和提高。  """
            )
        if len(st.session_state.conversation_scene.get("text", [])) == 0:
            st.warning("请先配置场景")
            # st.stop()

        if "listening-idx" not in st.session_state:
            st.session_state["listening-idx"] = -1

        pronunciation_evaluation_container = st.container()
        replay_text_placeholder = st.empty()
        st.divider()
        ls_btn_cols = st.columns(9)
        st.divider()

        refresh_btn = ls_btn_cols[0].button(
            "刷新[:arrows_counterclockwise:]",
            key="listening-refresh",
            help="✨ 点击按钮，从头开始练习。",
        )
        display_status_button = ls_btn_cols[1].button(
            "切换[:recycle:]",
            key="listening-mask",
            help="✨ 点击按钮可以在中英对照、只显示英文和只显示中文三种显示状态之间切换。初始状态为中英对照。",
        )
        prev_btn = ls_btn_cols[2].button(
            "上一[:leftwards_arrow_with_hook:]",
            key="listening-prev",
            help="✨ 点击按钮，切换到上一轮对话。",
            on_click=on_prev_btn_click,
            args=("listening-idx",),
            disabled=st.session_state["listening-idx"] < 0,
        )
        next_btn = ls_btn_cols[3].button(
            "下一[:arrow_right_hook:]",
            key="listening-next",
            help="✨ 点击按钮，切换到下一轮对话。",
            on_click=on_next_btn_click,
            args=("listening-idx",),
            disabled=len(st.session_state.conversation_scene.get("text", [])) == 0
            or (st.session_state["listening-idx"] != -1 and st.session_state["listening-idx"] == len(st.session_state.conversation_scene.get("text", [])) - 1),  # type: ignore
        )
        replay_btn = ls_btn_cols[4].button(
            "重放[:headphones:]",
            key="listening-replay",
            help="✨ 点击按钮，重新播放当前对话。",
            disabled=st.session_state["listening-idx"] == -1
            or len(st.session_state.conversation_scene.get("text", [])) == 0,
        )

        full_btn = ls_btn_cols[5].button(
            "全文[:film_frames:]",
            key="listening-full",
            help="✨ 点击按钮，收听对话全文。",
            disabled=len(st.session_state.conversation_scene.get("text", [])) == 0,
        )

        audio_key = "listening-mic-recorder"
        audio_session_output_key = f"{audio_key}_output"

        with ls_btn_cols[6]:
            audio_info = mic_recorder(
                start_prompt="录音[⏸️]",
                stop_prompt="停止[🔴]",
                key=audio_key,
            )

        pro_btn = ls_btn_cols[7].button(
            "评估[🔖]",
            disabled=not audio_info,
            key="pronunciation-evaluation-btn",
            help="✨ 点击按钮，开始发音评估。",
        )

        play_btn = ls_btn_cols[8].button(
            "回放[▶️]",
            disabled=not audio_info,
            key="listening-play-btn",
            help="✨ 点击按钮，播放您的跟读录音。",
        )

        dialogue_placeholder = st.empty()

        if pro_btn and audio_info is not None:
            on_project_changed("听说练习-发音评估")
            # 去掉发言者的名字
            reference_text = st.session_state.conversation_scene["text"][
                st.session_state["listening-idx"]
            ]
            reference_text = reference_text.replace("**", "")
            reference_text = re.sub(r"^\w+:\s", "", reference_text)
            start = datetime.now(pytz.UTC)
            st.session_state["listening-pronunciation-assessment"] = (
                pronunciation_assessment_for(
                    audio_info,
                    reference_text,
                )
            )

            display_assessment_score(
                pronunciation_evaluation_container,
                PRONUNCIATION_SCORE_BADGE_MAPS,
                "listening-pronunciation-assessment",
            )

            # 添加成绩记录
            test_dict = {
                # "phone_number": st.session_state.dbi.cache["user_info"]["phone_number"],
                "item": "发音评估",
                "topic": scenario_category,
                "level": f"{difficulty}-{len(reference_text.split())}",
                "score": st.session_state["listening-pronunciation-assessment"][
                    "pronunciation_result"
                ]["pronunciation_score"],
                "duration": (datetime.now(pytz.UTC) - start).total_seconds(),
                "record_time": datetime.now(pytz.UTC),
            }
            # st.session_state.dbi.save_daily_quiz_results(test_dict)
            st.session_state.dbi.add_documents_to_user_history(
                "performances", [test_dict]
            )

        if (
            play_btn
            and audio_info
            and st.session_state["listening-pronunciation-assessment"]
        ):
            on_project_changed("听说练习-回放录音")
            autoplay_audio_and_display_text(
                replay_text_placeholder,
                audio_info["bytes"],
                st.session_state["listening-pronunciation-assessment"][
                    "recognized_words"
                ],
            )

        if refresh_btn:
            on_project_changed("听说练习-刷新")
            st.session_state["listening-idx"] = -1
            st.session_state["listening-learning-times"] = 0
            st.session_state["listening-pronunciation-assessment"] = None
            st.rerun()

        if display_status_button:
            if st.session_state["listening-display-state"] == "英文":
                st.session_state["listening-display-state"] = "全部"
            elif st.session_state["listening-display-state"] == "全部":
                st.session_state["listening-display-state"] = "中文"
            else:
                st.session_state["listening-display-state"] = "英文"

        if prev_btn or next_btn or replay_btn:
            on_project_changed("听说练习-练习")
            play_and_record_dialogue(
                m_voice_style,
                fm_voice_style,
            )

        if full_btn:
            dialogue = st.session_state.conversation_scene
            text = dialogue["text"]
            boy_name = dialogue["boy_name"]
            girl_name = dialogue["girl_name"]
            audio_data_list = []
            duration_list = []
            total = 0
            for i, sentence in enumerate(text):
                # 如果是旁白，使用小女孩的声音
                # voice_style = m_voice_style if i % 2 == 0 else fm_voice_style
                # style = "en-US-AnaNeural" if is_aside(sentence) else voice_style[0]
                style = get_voice_style(
                    sentence,
                    boy_name,
                    girl_name,
                    m_voice_style,
                    fm_voice_style,
                    "en-US-AnaNeural",
                )
                sentence_without_speaker_name = re.sub(
                    r"^\w+:\s", "", sentence.replace("**", "")
                )
                with st.spinner(f"微软语音合成 第 {i+1:2d} 轮对话..."):
                    result = get_synthesis_speech(sentence_without_speaker_name, style)
                audio_data_list.append(result["audio_data"])
                duration_list.append(result["audio_duration"])
                total += result["audio_duration"].total_seconds()

            current_idx = st.session_state["listening-idx"]

            for i, duration in enumerate(duration_list):
                on_project_changed(f"听说练习-第{i:2d}轮")
                st.session_state["listening-idx"] = i
                display_dialogue(dialogue_placeholder)
                play_and_record_dialogue(m_voice_style, fm_voice_style)
                time.sleep(duration.total_seconds() + 0.5)
            # 恢复指针
            st.session_state["listening-idx"] = current_idx
            st.session_state["listening-learning-times"] = len(
                st.session_state.conversation_scene["text"]
            )
            dialogue_text = " ".join(st.session_state.conversation_scene["text"])
            word_count = len(dialogue_text.split())
            # 防止重复播放
            st.rerun()

        # 始终显示当前对话文本
        display_dialogue(dialogue_placeholder)

    # endregion

    # region "听力测验"

    with listening_tabs[2]:
        st.subheader("听力测验(五道题)", divider="rainbow", anchor="听力测验")

        if "listening-start-time" not in st.session_state:
            st.session_state["listening-start-time"] = None

        if len(st.session_state.conversation_scene.get("text", [])) == 0:
            st.warning("请先配置场景")
            # st.stop()

        if st.session_state["listening-learning-times"] == 0:
            st.warning("请先完成听说练习")
            # st.stop()

        cols = st.columns(2)
        update_and_display_progress(
            (
                st.session_state["listening-test-idx"] + 1
                if st.session_state["listening-test-idx"] != -1
                else 0
            ),
            (
                len(st.session_state["listening-test"])
                if len(st.session_state["listening-test"]) != 0
                else 1
            ),
            cols[0].empty(),
        )

        st.divider()

        ls_text_btn_cols = st.columns(8)

        st.divider()

        refresh_test_btn = ls_text_btn_cols[0].button(
            "刷新[:arrows_counterclockwise:]",
            key="listening-test-refresh",
            help="✨ 点击按钮，生成听力测试题。",
        )
        display_test_btn = ls_text_btn_cols[1].button(
            "切换[:recycle:]",
            key="listening-test-mask",
            help="✨ 此按钮可切换题目展示方式：文本或语音。默认为文本形式。",
        )
        listening_prev_test_btn = ls_text_btn_cols[2].button(
            "上一[:leftwards_arrow_with_hook:]",
            key="listening-test-prev",
            help="✨ 点击按钮，切换到上一道听力测试题。",
            on_click=on_prev_btn_click,
            args=("listening-test-idx",),
            disabled=st.session_state["listening-test-idx"] <= 0,
        )
        listening_next_test_btn = ls_text_btn_cols[3].button(
            "下一[:arrow_right_hook:]",
            key="listening-test-next",
            help="✨ 点击按钮，切换到下一道听力测试题。",
            on_click=on_next_btn_click,
            args=("listening-test-idx",),
            disabled=len(st.session_state["listening-test"]) == 0
            or st.session_state["listening-test-idx"] == len(st.session_state["listening-test"]) - 1,  # type: ignore
        )
        rpl_test_btn = ls_text_btn_cols[4].button(
            "听题[:headphones:]",
            key="listening-test-replay",
            help="✨ 点击此按钮，可以重新播放当前测试题目的语音。",
            disabled=len(st.session_state["listening-test"]) == 0
            or st.session_state["listening-test-idx"] == -1
            or st.session_state["listening-test-display-state"] == "文本",  # type: ignore
        )
        sumbit_test_btn = ls_text_btn_cols[5].button(
            "检查[:mag:]",
            key="submit-listening-test",
            disabled=st.session_state["listening-test-idx"] == -1
            or len(st.session_state["listening-test-answer"]) == 0,
            help="✨ 至少完成一道测试题后，才可点击按钮，检查听力测验得分。",
        )

        container = st.container()

        if refresh_test_btn:
            on_project_changed(f"听说练习-测试题目")
            st.session_state["listening-test"] = generate_listening_test_for(
                difficulty, st.session_state.conversation_scene["text"]
            )
            st.session_state["listening-test-idx"] = -1
            st.session_state["listening-test-answer"] = [None] * len(
                st.session_state["listening-test"]
            )
            st.session_state["listening-start-time"] = time.time()
            # 更新
            st.rerun()

        if display_test_btn:
            if st.session_state["listening-test-display-state"] == "文本":
                st.session_state["listening-test-display-state"] = "语音"
            else:
                st.session_state["listening-test-display-state"] = "文本"

            if st.session_state["listening-test-idx"] != -1:
                play_listening_test()

        if rpl_test_btn:
            on_project_changed("听说练习-测试-听题")
            if st.session_state["listening-test-idx"] != -1:
                play_listening_test()

        if listening_prev_test_btn:
            idx = st.session_state["listening-test-idx"]
            on_project_changed(f"听说练习-测试-{idx}")
            play_listening_test()

        if listening_next_test_btn:
            idx = st.session_state["listening-test-idx"]
            on_project_changed(f"听说练习-测试-{idx}")
            play_listening_test()

        if sumbit_test_btn:
            on_project_changed("听说练习-测试-答题")
            container.empty()

            if count_non_none(st.session_state["listening-test-answer"]) == 0:
                container.warning("您尚未答题。")
                container.stop()

            if count_non_none(
                st.session_state["listening-test-answer"]
            ) != count_non_none(st.session_state["listening-test"]):
                container.warning("您尚未完成测试。")

            check_listening_test_answer(container, difficulty, selected_scenario)
        else:
            if st.session_state["listening-test-idx"] != -1:
                idx = st.session_state["listening-test-idx"]
                on_project_changed(f"听说练习-测试-答题-{idx}")
                view_listening_test(container)

    # endregion

# endregion

# region 阅读练习

if item_menu is not None and item_menu.endswith("阅读练习"):
    on_project_changed("阅读练习")
    m_voice_style = st.sidebar.selectbox(
        "合成男声风格",
        st.session_state["m_voices"],
        # on_change=on_voice_changed,
        help="✨ 选择您喜欢的合成男声语音风格",
        format_func=lambda x: f"{x[2]}",  # type: ignore
    )
    fm_voice_style = st.sidebar.selectbox(
        "合成女声风格",
        st.session_state["fm_voices"],
        # on_change=on_voice_changed,
        help="✨ 选择您喜欢的合成女声语音风格",
        format_func=lambda x: f"{x[2]}",  # type: ignore
    )

    exercise_type = st.sidebar.selectbox(
        "考题类型", list(EXERCISE_TYPE_MAPPING.keys()), help="✨ 选择您喜欢的考题类型"
    )

    # 获取英文的考题类型
    english_exercise_type = EXERCISE_TYPE_MAPPING[exercise_type]

    reading_tabs = st.tabs(["配置场景", "开始练习", "小测验"])

    # region "配置场景"



    with reading_tabs[0]:
        st.subheader("配置场景", divider="rainbow", anchor="配置场景")
        st.markdown("依次执行以下步骤，生成阅读理解练习模拟场景。")
        steps = ["1. CEFR等级", "2. 体裁内容", "3. 添加情节", "4. 预览场景"]
        sub_tabs = st.tabs(steps)

        difficulty = None
        genre = None
        contents = None
        plot = None

        with sub_tabs[0]:
            on_project_changed("阅读练习-难度")
            st.info("第一步：点击下拉框选择CEFR等级", icon="🚨")
            difficulty = st.selectbox(
                "CEFR等级",
                list(CEFR_LEVEL_MAPS.keys()),
                key="reading-difficulty",
                index=0,
                format_func=lambda x: f"{x}({CEFR_LEVEL_MAPS[x]})",
                on_change=set_state,
                args=(1,),
                placeholder="请选择CEFR等级",
            )

        with sub_tabs[1]:
            on_project_changed("阅读练习-文章体裁")
            st.info("第二步：设置文章体裁和内容", icon="🚨")
            st.markdown(AI_TIPS[difficulty], unsafe_allow_html=True)
            if st.session_state.stage == 1 or difficulty is not None:
                genre = st.selectbox(
                    "请选择文章体裁",
                    GENRES,
                    index=0,
                    on_change=set_state,
                    args=(2,),
                    key="reading-genre",
                    placeholder="请选择文章体裁",
                )
                contents = st.multiselect(
                    "请选择文章内容",
                    CONTENTS,
                    key="reading-contents",
                    max_selections=3,
                    on_change=set_state,
                    args=(2,),
                    placeholder="请选择文章内容（可多选）",
                    help="✨ 选择文章内容（可多选）。",
                )

        with sub_tabs[2]:
            on_project_changed("阅读练习-情节")
            st.info(
                "第三步：可选。可在文本框内添加一些有趣的情节以丰富练习材料。如果您想跳过这一步，可以选择'跳过'。",
                icon="🚨",
            )
            ignore = st.toggle("跳过", key="add_interesting_plot", value=True)
            if ignore:
                st.session_state.stage = 3
            st.divider()
            if st.session_state.stage == 2 or genre is not None:
                plot = st.text_area(
                    "添加一些有趣的情节【可选】",
                    height=200,
                    key="interesting_plot",
                    on_change=set_state,
                    args=(3,),
                    placeholder="""您可以在这里添加一些有趣的情节。比如：
- 同事问了一个非常奇怪的问题，让您忍俊不禁。
- 同事在工作中犯了一个错误，但他能够及时发现并改正。
                """,
                )

        with sub_tabs[3]:
            on_project_changed("阅读练习-生成场景")
            st.info(
                """在完成所有步骤后，您可以在此处生成并查看场景。生成场景后，您可以切换到最上方👆的 "开始练习" 标签页，开始进行阅读理解练习。""",
                icon="🚨",
            )
            st.warning(
                "我们使用的生成式AI的主要目标是丰富阅读理解的文本材料。然而，由于其生成的内容具有虚幻特性，可能并非真实或准确，因此请不要完全依赖其生成的内容或将其视为事实。",
                icon="🚨",
            )
            if genre is None or difficulty is None or contents is None:
                st.warning("您需要先完成之前的所有步骤")
                st.stop()

            session_cols = st.columns(8)

            container = st.container()

            gen_btn = session_cols[0].button(
                "刷新[:arrows_counterclockwise:]",
                key="generate-readings",
                help="✨ 点击按钮，生成阅读理解练习材料。",
            )

            if gen_btn:
                container.empty()
                # 学习次数重置为0
                st.session_state["reading-learning-times"] = 0

                genre_index = GENRES.index(genre)
                genre_en = GENRES_EN[genre_index]

                contents_index = [CONTENTS.index(c) for c in contents]
                contents_en = [CONTENTS_EN[i] for i in contents_index]

                article = generate_reading_comprehension_article_for(
                    genre_en, contents_en, plot if plot else "", difficulty
                )
                paragraphs = [
                    paragraph for paragraph in article.split("\n") if paragraph.strip()
                ]
                st.session_state["reading-article"] = paragraphs
                display_text_word_count_summary(container, " ".join(paragraphs))
                st.markdown("\n\n".join(paragraphs))

            elif len(st.session_state["reading-article"]):
                paragraphs = st.session_state["reading-article"]
                display_text_word_count_summary(container, " ".join(paragraphs))
                st.markdown("\n\n".join(paragraphs))

    # endregion

    # region 阅读练习

    with reading_tabs[1]:
        st.subheader("阅读练习", divider="rainbow", anchor="阅读练习")
        st.markdown(
            """
您可以通过反复阅读和理解文章来提升您的阅读理解技能。点击`全文`可以一次性阅读整篇文章。另外，您可以通过点击左侧的按钮调整合成语音风格，以更好地适应您的听力习惯。
"""
        )
        st.warning(
            "请注意，练习过程中会使用喇叭播放音频。为了避免音量过大或过小影响您的体验，请提前调整到适合的音量。",
            icon="🚨",
        )
        if len(st.session_state["reading-article"]) == 0:
            st.warning("请先配置阅读材料")
            st.stop()

        if "reading-exercise-idx" not in st.session_state:
            st.session_state["reading-exercise-idx"] = -1

        ra_btn_cols = st.columns(8)

        st.divider()

        refresh_btn = ra_btn_cols[0].button(
            "刷新[:arrows_counterclockwise:]",
            key="refresh-reading-exercise",
            help="✨ 点击按钮，从头开始练习。",
        )
        display_status_button = ra_btn_cols[1].button(
            "切换[:recycle:]",
            key="toggle-display-status",
            help="✨ 点击按钮可以在中英对照、只显示英文和只显示中文三种显示状态之间切换。初始状态为中英对照。",
        )
        prev_btn = ra_btn_cols[2].button(
            "上一[:leftwards_arrow_with_hook:]",
            key="ra-prev",
            help="✨ 点击按钮，切换到上一段落。",
            on_click=on_prev_btn_click,
            args=("reading-exercise-idx",),
            disabled=st.session_state["reading-exercise-idx"] <= 0,
        )
        next_btn = ra_btn_cols[3].button(
            "下一[:arrow_right_hook:]",
            key="ra-next",
            help="✨ 点击按钮，切换到下一段落。",
            on_click=on_next_btn_click,
            args=("reading-exercise-idx",),
            disabled=st.session_state["reading-exercise-idx"]
            == len(st.session_state["reading-article"]) - 1,
        )
        replay_btn = ra_btn_cols[4].button(
            "重放[:headphones:]",
            key="ra-replay",
            help="✨ 点击按钮，重新播放当前段落。",
            disabled=len(st.session_state["reading-article"]) == 0
            or st.session_state["reading-exercise-idx"] == -1,
        )
        full_reading_btn = ra_btn_cols[5].button(
            "全文[:film_frames:]",
            key="reading-exercise-full",
            help="✨ 点击按钮，收听整个文章。",
            disabled=len(st.session_state["reading-article"]) == 0,
        )

        container = st.container()

        if refresh_btn:
            on_project_changed("阅读练习-刷新")
            st.session_state["reading-exercise-idx"] = -1
            st.session_state["reading-learning-times"] = 0
            st.rerun()

        if display_status_button:
            if st.session_state["ra-display-state"] == "英文":
                st.session_state["ra-display-state"] = "全部"
            elif st.session_state["ra-display-state"] == "全部":
                st.session_state["ra-display-state"] = "中文"
            else:
                st.session_state["ra-display-state"] = "英文"

            if st.session_state["reading-exercise-idx"] != -1:
                process_play_and_record_article(
                    container,
                    m_voice_style,
                    fm_voice_style,
                    difficulty,
                    genre,
                )

        if prev_btn or next_btn or replay_btn:
            st.session_state["reading-learning-times"] += 1
            idx = st.session_state["reading-exercise-idx"]
            on_project_changed(f"阅读练习-练习-{idx}")
            process_play_and_record_article(
                container,
                m_voice_style,
                fm_voice_style,
                difficulty,
                genre,
            )

        if full_reading_btn:
            on_project_changed("阅读练习-练习-全文")
            total = autoplay_audio_and_display_article(container)
            st.session_state["reading-learning-times"] = len(
                st.session_state["reading-article"]
            )
            text = " ".join(st.session_state["reading-article"])
            word_count = len(text.split())
            st.rerun()

    # endregion

    # region 阅读测验

    with reading_tabs[2]:
        on_project_changed("阅读练习-理解测验")
        st.subheader("阅读理解测验", divider="rainbow", anchor="阅读理解测验")
        if "reading-start-time" not in st.session_state:
            st.session_state["reading-start-time"] = None

        if len(st.session_state["reading-article"]) == 0:
            st.warning("请先配置阅读理解练习材料")
            st.stop()

        if st.session_state["reading-learning-times"] == 0:
            st.warning("请先完成练习")
            st.stop()

        cols = st.columns(2)
        update_and_display_progress(
            (
                st.session_state["reading-test-idx"] + 1
                if st.session_state["reading-test-idx"] != -1
                else 0
            ),
            (
                len(st.session_state["reading-test"])
                if len(st.session_state["reading-test"]) != 0
                else 1
            ),
            cols[0].empty(),
        )

        st.divider()

        ra_test_btn_cols = st.columns(8)

        st.divider()

        refresh_test_btn = ra_test_btn_cols[0].button(
            "刷新[:arrows_counterclockwise:]",
            key="ra-test-refresh",
            help="✨ 点击按钮，生成阅读理解测试题。",
        )
        prev_test_btn = ra_test_btn_cols[1].button(
            "上一[:leftwards_arrow_with_hook:]",
            key="ra-test-prev",
            help="✨ 点击按钮，切换到上一道测试题。",
            on_click=on_prev_btn_click,
            args=("reading-test-idx",),
            disabled=st.session_state["reading-test-idx"] <= 0,
        )
        next_test_btn = ra_test_btn_cols[2].button(
            "下一[:arrow_right_hook:]",
            key="ra-test-next",
            help="✨ 点击按钮，切换到下一道测试题。",
            on_click=on_next_btn_click,
            args=("reading-test-idx",),
            disabled=len(st.session_state["reading-test"]) == 0
            or st.session_state["reading-test-idx"] == len(st.session_state["reading-test"]) - 1,  # type: ignore
        )
        rpl_test_btn = ra_test_btn_cols[3].button(
            "听题[:headphones:]",
            key="ra-test-replay",
            help="✨ 点击此按钮，使用语音播放问题。",
            disabled=len(st.session_state["reading-test"]) == 0
            or st.session_state["reading-test-idx"] == -1,
        )
        sumbit_test_btn = ra_test_btn_cols[4].button(
            "检查[:mag:]",
            key="submit-reading-test",
            disabled=st.session_state["reading-test-idx"] == -1
            or len(st.session_state["reading-test-answer"]) == 0,
            help="✨ 至少完成一道测试题后，才可点击按钮，检查测验得分。",
        )

        container = st.container()

        if refresh_test_btn:
            on_project_changed("阅读练习-理解测验-刷新")
            st.session_state["reading-test"] = generate_reading_test_for(
                difficulty, english_exercise_type, st.session_state["reading-article"]
            )
            # logger.info(st.session_state["reading-test"])
            st.session_state["reading-test-idx"] = -1
            st.session_state["reading-test-answer"] = [None] * len(
                st.session_state["reading-test"]
            )
            st.session_state["reading-start-time"] = datetime.now(pytz.UTC)
            # 更新
            st.rerun()

        if prev_test_btn:
            idx = st.session_state["reading-test-idx"]
            on_project_changed(f"阅读练习-理解测验-{idx}")
            view_reading_test(container, difficulty, exercise_type, genre)

        if next_test_btn:
            idx = st.session_state["reading-test-idx"]
            on_project_changed(f"阅读练习-理解测验-{idx}")
            view_reading_test(container, difficulty, exercise_type, genre)

        if rpl_test_btn:
            idx = st.session_state["reading-test-idx"]
            on_project_changed(f"阅读练习-理解测验-{idx}")
            test = st.session_state["reading-test"][idx]
            question = test["question"]
            with st.spinner(f"使用 Azure 将文本合成语音..."):
                question_audio = get_synthesis_speech(question, m_voice_style[0])
            audio_html = audio_autoplay_elem(question_audio["audio_data"], fmt="wav")
            components.html(audio_html)
            view_reading_test(container, difficulty, exercise_type, genre)

        if sumbit_test_btn:
            on_project_changed("阅读练习-理解测验-评分")
            container.empty()

            if count_non_none(st.session_state["reading-test-answer"]) == 0:
                container.warning("您尚未答题。")
                container.stop()

            if count_non_none(
                st.session_state["reading-test-answer"]
            ) != count_non_none(st.session_state["reading-test"]):
                container.warning("您尚未完成测试。")

            check_reading_test_answer(container, difficulty, exercise_type, genre)

    # endregion

# endregion
