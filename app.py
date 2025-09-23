# 1. 필요한 라이브러리들을 모두 가져옵니다.
import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import io
import google.generativeai as genai
from dotenv import load_dotenv
import os

# --------------------------------------------------------------------------
# AI 두뇌 연결 및 설정 최적화
# --------------------------------------------------------------------------

# .env 파일에서 API 키를 로드합니다.
load_dotenv()

# Google Gemini API 설정
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("GOOGLE_API_KEY가 .env 파일에 설정되지 않았습니다. API 키를 확인해주세요.")
        st.stop()
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"API 키 설정 중 오류가 발생했습니다: {e}")
    st.stop()

# --------------------------------------------------------------------------
# AI 전문가 함수 정의 (스트리밍 기능 추가)
# --------------------------------------------------------------------------

def get_gemini_summary_stream(full_text):
    """Gemini 1.5 Flash 모델을 사용하여 텍스트 요약을 스트리밍으로 생성하는 함수"""
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    prompt = """
    당신은 세계적인 수준의 재료공학 분야 논문 리뷰어입니다.
    제공된 논문 전체 텍스트를 바탕으로, 다음 항목에 대해 한국어로 명확하게 요약해주세요.
    각 항목은 반드시 소제목(마크다운 볼드체)과 함께 줄바꿈하여 구분해주세요.

    1.  **핵심 연구 목표**: 이 연구가 궁극적으로 해결하고자 하는 문제는 무엇입니까?
    2.  **연구의 핵심 방법론**: 어떤 재료와 공정을 사용했으며, 가장 중요한 측정 및 분석 방법은 무엇이었습니까?
    3.  **가장 중요한 발견 및 결론**: 이 연구를 통해 새롭게 밝혀낸 가장 중요한 사실과 최종 결론은 무엇입니까?
    """
    try:
        response_stream = model.generate_content([prompt, full_text], stream=True)
        return response_stream
    except Exception as e:
        # 오류 발생 시 사용자에게 보여줄 수 있는 텍스트를 생성(yield)
        yield f"텍스트 요약 스트림 생성 중 오류 발생: {e}"


def get_gemini_vision_analysis(image_bytes, context_text):
    """Gemini 1.5 Flash 모델을 사용하여 이미지와 텍스트(컨텍스트)를 함께 분석하는 함수"""
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    img = Image.open(io.BytesIO(image_bytes))
    
    prompt_parts = [
        "당신은 데이터 시각화 및 재료 분석 전문가입니다.",
        "이 이미지는 아래 논문의 일부입니다. 전체적인 맥락 파악에 참고하세요:\n\n" + context_text,
        "\n\n---\n\n",
        "위 텍스트를 참고하여, 아래 이미지에 대해 다음 항목을 한국어로 분석해주세요:",
        img,
        "\n\n1. **이미지 종류 및 내용**: 이 이미지는 무엇인가요? (예: SEM 현미경 사진, XRD 회절 그래프, 공정 모식도 등) 그리고 무엇을 보여주고 있나요?",
        "2. **핵심 정보 및 데이터**: 이 이미지가 전달하는 가장 중요한 정보나 데이터는 무엇인가요? (예: 그래프의 경향성, 특정 피크의 의미, 미세구조의 특징 등)"
    ]
    try:
        response = model.generate_content(prompt_parts)
        return response.text
    except Exception as e:
        return f"이미지 분석 중 오류 발생: {e}"

# --------------------------------------------------------------------------
# 캐싱(Caching)을 통한 PDF 처리 속도 향상
# --------------------------------------------------------------------------

@st.cache_data
def extract_text_from_pdf(file_id, file_bytes):
    """PDF 파일에서 텍스트를 추출하는 함수"""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    full_text = "".join(page.get_text() for page in doc)
    doc.close()
    return full_text

@st.cache_data
def extract_images_from_pdf(file_id, file_bytes):
    """PDF 파일에서 이미지를 추출하는 함수"""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    image_list = []
    for page in doc:
        for img in page.get_images(full=True):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_list.append(image_bytes)
            except Exception:
                continue
    doc.close()
    return image_list

# --------------------------------------------------------------------------
# Streamlit 웹사이트 UI 구성 (오류 수정 및 안정화)
# --------------------------------------------------------------------------

st.set_page_config(layout="wide")
st.title("✨ Summize: AI 논문 분석 솔루션 (v0.8 ErrorFixed)")
st.write("AI가 논문의 핵심을 실시간으로 요약해 드립니다. 이미지는 개별적으로 심층 분석할 수 있습니다.")

# 안정적인 UI를 위한 세션 상태 변수 추가
if 'current_file_id' not in st.session_state:
    st.session_state.current_file_id = None
if 'image_analysis_results' not in st.session_state:
    st.session_state.image_analysis_results = {}
if 'summary_requested' not in st.session_state:
    st.session_state.summary_requested = False
if 'summary_result' not in st.session_state:
    st.session_state.summary_result = ""

uploaded_file = st.file_uploader("여기에 분석할 논문 PDF 파일을 업로드하세요.", type="pdf")

if uploaded_file is not None:
    file_id = uploaded_file.file_id
    file_bytes = uploaded_file.getvalue()

    # 새 파일이 업로드되면 모든 상태 초기화
    if st.session_state.current_file_id != file_id:
        st.session_state.current_file_id = file_id
        st.session_state.image_analysis_results = {}
        st.session_state.summary_requested = False
        st.session_state.summary_result = ""
        st.cache_data.clear()

    with st.spinner('PDF 파일에서 텍스트와 이미지를 추출하고 있습니다...'):
        extracted_text = extract_text_from_pdf(file_id, file_bytes)
        extracted_images = extract_images_from_pdf(file_id, file_bytes)
    st.success("텍스트 및 이미지 추출 완료!")
    st.divider()

    tab1, tab2 = st.tabs(["📄 실시간 텍스트 요약", f"🖼️ 이미지 개별 분석 ({len(extracted_images)}개)"])

    with tab1:
        st.header("논문 텍스트 핵심 요약")
        st.write("아래 버튼을 누르면 AI가 생성하는 요약을 실시간으로 확인할 수 있습니다.")

        if st.button("실시간 요약 시작하기", type="primary"):
            st.session_state.summary_requested = True
            st.session_state.summary_result = ""

        if st.session_state.get('summary_requested', False):
            st.markdown("---")
            with st.spinner("AI가 논문을 읽고 실시간으로 요약 중입니다..."):
                response_generator = get_gemini_summary_stream(extracted_text)
                summary_container = st.empty()
                full_summary = ""
                
                try:
                    for chunk in response_generator:
                        # ⭐️ 핵심 수정: chunk에 text 속성이 있는지 확인하여 오류를 방지합니다.
                        if hasattr(chunk, 'text'):
                            full_summary += chunk.text
                            summary_container.markdown(full_summary + "▌")
                except Exception as e:
                    st.error(f"요약 내용을 처리하는 중 오류가 발생했습니다: {e}")
                
                st.session_state.summary_result = full_summary
                st.session_state.summary_requested = False
                st.rerun()

        if st.session_state.get('summary_result', ""):
            st.markdown("---")
            st.markdown(st.session_state.summary_result)

    with tab2:
        st.header("이미지 상세 분석 (원하는 이미지를 선택하여 분석)")
        if not extracted_images:
            st.warning("이 PDF에서는 이미지를 찾을 수 없습니다.")
        else:
            cols = st.columns(2)
            for i, img_bytes in enumerate(extracted_images):
                col = cols[i % 2]
                with col:
                    st.image(img_bytes, caption=f"이미지 #{i+1}")
                    
                    if st.button(f"이미지 #{i+1} AI 분석", key=f"img_btn_{file_id}_{i}"):
                        with st.spinner(f"Gemini AI가 이미지 #{i+1}을(를) 분석하고 있습니다..."):
                            analysis = get_gemini_vision_analysis(img_bytes, extracted_text)
                            st.session_state.image_analysis_results[i] = analysis
                            st.rerun()
                    
                    if i in st.session_state.image_analysis_results:
                        with st.expander(f"**이미지 #{i+1} 분석 결과 보기**", expanded=True):
                            st.markdown(st.session_state.image_analysis_results[i])
                    
                    st.divider()