# 1. 필요한 라이브러리들을 모두 가져옵니다.
import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import io
import google.generativeai as genai
from dotenv import load_dotenv
import os

# --------------------------------------------------------------------------
# ⭐️ 핵심 개선 사항: AI 두뇌 연결 및 설정 최적화
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
# AI 전문가 함수 정의 (모델 변경 및 컨텍스트 강화)
# --------------------------------------------------------------------------

def get_gemini_summary(full_text):
    """Gemini 1.5 Flash 모델을 사용하여 텍스트를 요약하는 함수"""
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
        response = model.generate_content([prompt, full_text])
        return response.text
    except Exception as e:
        return f"텍스트 요약 중 오류 발생: {e}"

def get_gemini_vision_analysis(image_bytes, context_text):
    """Gemini 1.5 Flash 모델을 사용하여 이미지와 텍스트(컨텍스트)를 함께 분석하는 함수"""
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    img = Image.open(io.BytesIO(image_bytes))
    
    # ⭐️ 핵심 개선 사항: 이미지 분석 시 논문 전체 텍스트를 컨텍스트로 제공하여 정확도 향상
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
# ⭐️ 핵심 개선 사항: 캐싱(Caching)을 통한 PDF 처리 속도 향상
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
# Streamlit 웹사이트 UI 구성 (상호작용 중심 버전)
# --------------------------------------------------------------------------

st.set_page_config(layout="wide")
st.title("✨ Summize: AI 논문 분석 솔루션 (v0.5 Interactive)")
st.write("PDF 파일을 업로드하면, AI가 논문의 핵심을 요약하고 이미지를 개별적으로 심층 분석해드립니다.")

# 세션 상태 초기화 (더 안정적인 상태 관리)
if 'current_file_id' not in st.session_state:
    st.session_state.current_file_id = None
if 'summary_result' not in st.session_state:
    st.session_state.summary_result = ""
if 'image_analysis_results' not in st.session_state:
    st.session_state.image_analysis_results = {}

uploaded_file = st.file_uploader("여기에 분석할 논문 PDF 파일을 업로드하세요.", type="pdf")

if uploaded_file is not None:
    file_id = uploaded_file.file_id
    file_bytes = uploaded_file.getvalue()

    # 새로운 파일이 업로드되면 이전 분석 결과 초기화
    if st.session_state.current_file_id != file_id:
        st.session_state.current_file_id = file_id
        st.session_state.summary_result = ""
        st.session_state.image_analysis_results = {}
        # 캐시된 결과도 초기화하기 위해 st.cache_data.clear() 호출
        st.cache_data.clear()

    with st.spinner('PDF 파일에서 텍스트와 이미지를 추출하고 있습니다...'):
        # 캐싱 함수에 파일 바이트를 직접 전달
        extracted_text = extract_text_from_pdf(file_id, file_bytes)
        extracted_images = extract_images_from_pdf(file_id, file_bytes)
    st.success("텍스트 및 이미지 추출 완료!")
    st.divider()

    # ⭐️ 핵심 개선 사항: 텍스트 요약과 이미지 분석 탭 분리
    tab1, tab2 = st.tabs(["📄 텍스트 요약 분석", f"🖼️ 이미지 개별 분석 ({len(extracted_images)}개)"])

    with tab1:
        st.header("논문 텍스트 핵심 요약")
        if st.button("텍스트 요약 실행하기", type="primary"):
            with st.spinner("Gemini AI가 논문 전체를 읽고 요약 중입니다..."):
                summary = get_gemini_summary(extracted_text)
                st.session_state.summary_result = summary
        
        if st.session_state.summary_result:
            st.markdown(st.session_state.summary_result)

    with tab2:
        st.header("이미지 상세 분석 (원하는 이미지를 선택하여 분석)")
        if not extracted_images:
            st.warning("이 PDF에서는 이미지를 찾을 수 없습니다.")
        else:
            for i, img_bytes in enumerate(extracted_images):
                st.image(img_bytes, caption=f"이미지 #{i+1}", width=300)
                
                # 각 이미지에 대한 분석 결과를 세션 상태에 저장 및 표시
                if st.button(f"이미지 #{i+1} AI 분석", key=f"img_btn_{file_id}_{i}"):
                    with st.spinner(f"Gemini AI가 이미지 #{i+1}을(를) 분석하고 있습니다..."):
                        analysis = get_gemini_vision_analysis(img_bytes, extracted_text)
                        st.session_state.image_analysis_results[i] = analysis
                
                if i in st.session_state.image_analysis_results:
                    with st.expander(f"**이미지 #{i+1} 분석 결과 보기**", expanded=True):
                        st.markdown(st.session_state.image_analysis_results[i])
                
                st.divider()

