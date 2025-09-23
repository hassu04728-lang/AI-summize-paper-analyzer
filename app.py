# 1. 필요한 라이브러리들을 모두 가져옵니다.
import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import io
import google.generativeai as genai
from dotenv import load_dotenv
import os
import concurrent.futures

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
# AI 전문가 함수 정의 (모델 변경 및 오류 처리 강화)
# --------------------------------------------------------------------------

def get_gemini_summary(full_text):
    """Gemini 1.5 Flash 모델을 사용하여 텍스트를 요약하는 함수"""
    # 더 빠르고 효율적인 Flash 모델 사용
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    prompt = """
    You are a world-class materials science paper reviewer.
    Based on the provided full text of the paper, please summarize the following items clearly in Korean.
    Each item must be separated by a newline with a subtitle (in Markdown bold).

    1.  **Core Research Objective**: What is the ultimate problem this research aims to solve?
    2.  **Key Research Methodology**: What materials and processes were used, and what were the most important measurement and analysis methods?
    3.  **Most Important Findings and Conclusion**: What are the most significant new facts discovered and the final conclusion of this study?
    """
    try:
        response = model.generate_content([prompt, full_text])
        return response.text
    except Exception as e:
        return f"텍스트 요약 중 오류 발생: {e}"

def get_gemini_vision_analysis(image_bytes):
    """Gemini 1.5 Flash 모델을 사용하여 이미지를 분석하는 함수"""
    # 더 빠르고 효율적인 Flash 모델 사용
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    img = Image.open(io.BytesIO(image_bytes))
    prompt_parts = [
        "You are an expert in data visualization and materials analysis.",
        "Please analyze the following image based on these criteria in Korean:",
        img,
        "\n\n1. **Image Type and Content**: What is this image? (e.g., SEM micrograph, XRD diffraction graph, process schematic, etc.) and what does it show?",
        "2. **Key Information and Data**: What is the most important information or data this image conveys? (e.g., trend in a graph, meaning of a specific peak, features of a microstructure, etc.)"
    ]
    try:
        response = model.generate_content(prompt_parts)
        return response.text
    except Exception as e:
        return f"이미지 분석 중 오류 발생: {e}"

# --------------------------------------------------------------------------
# ⭐️ 핵심 개선 사항: 캐싱(Caching)을 통한 PDF 처리 속도 향상
# Streamlit의 캐싱 기능을 사용하여 동일한 파일을 다시 업로드했을 때,
# 이전에 처리했던 결과를 즉시 반환하여 반복 작업을 피합니다.
# --------------------------------------------------------------------------

@st.cache_data
def extract_text_from_pdf(file_id):
    """PDF 파일에서 텍스트를 추출하는 함수"""
    # st.session_state에서 파일 바이트를 가져옵니다.
    pdf_file = st.session_state['uploaded_files'][file_id]
    pdf_file.seek(0)
    file_bytes = pdf_file.getvalue()
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    full_text = "".join(page.get_text() for page in doc)
    doc.close()
    return full_text

@st.cache_data
def extract_images_from_pdf(file_id):
    """PDF 파일에서 이미지를 추출하는 함수"""
    # st.session_state에서 파일 바이트를 가져옵니다.
    pdf_file = st.session_state['uploaded_files'][file_id]
    pdf_file.seek(0)
    file_bytes = pdf_file.getvalue()
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
                # 가끔 이미지를 추출할 수 없는 경우가 있어도 전체가 멈추지 않도록 예외 처리
                continue
    doc.close()
    return image_list

# --------------------------------------------------------------------------
# Streamlit 웹사이트 UI 구성 (개선된 버전)
# --------------------------------------------------------------------------

st.set_page_config(layout="wide")
st.title("✨ Summize: AI 논문 분석 솔루션 (v0.4 Optimized)")
st.write("PDF 파일을 업로드하면, AI가 논문의 핵심을 요약하고 이미지를 심층 분석해드립니다. **(속도 최적화 버전)**")

# 파일 업로드 세션 상태 초기화
if 'uploaded_files' not in st.session_state:
    st.session_state['uploaded_files'] = {}
if 'analysis_results' not in st.session_state:
    st.session_state['analysis_results'] = {}
if 'file_id' not in st.session_state:
    st.session_state['file_id'] = None

uploaded_file = st.file_uploader("여기에 분석할 논문 PDF 파일을 업로드하세요.", type="pdf")

if uploaded_file is not None:
    # 파일 ID를 생성하여 캐싱 및 세션 관리에 사용
    file_id = uploaded_file.file_id
    st.session_state['file_id'] = file_id
    
    # 이전에 업로드되지 않은 파일인 경우에만 저장
    if file_id not in st.session_state['uploaded_files']:
        st.session_state['uploaded_files'][file_id] = uploaded_file
        # 새로운 파일이므로 이전 분석 결과 초기화
        st.session_state['analysis_results'] = {}


    with st.spinner('PDF 파일에서 텍스트와 이미지를 추출하고 있습니다...'):
        extracted_text = extract_text_from_pdf(file_id)
        extracted_images = extract_images_from_pdf(file_id)
    st.success("텍스트 및 이미지 추출 완료!")
    st.divider()

    # ⭐️ 핵심 개선 사항: 단일 버튼으로 모든 분석을 병렬 실행
    if st.button("🔬 전체 AI 분석 시작하기", type="primary"):
        # 이전 결과가 있다면 초기화
        st.session_state.analysis_results = {}
        
        # ⭐️ 핵심 개선 사항: ThreadPoolExecutor를 사용한 병렬 처리
        # 여러 AI 분석 작업을 동시에 실행하여 전체 대기 시간을 획기적으로 단축
        with st.spinner("Gemini AI가 논문 요약 및 모든 이미지 분석을 동시에 진행하고 있습니다... 잠시만 기다려주세요."):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # 텍스트 요약 작업 제출
                future_summary = executor.submit(get_gemini_summary, extracted_text)
                
                # 이미지 분석 작업들을 제출
                future_images = {executor.submit(get_gemini_vision_analysis, img_bytes): i for i, img_bytes in enumerate(extracted_images)}
                
                # 결과 취합
                summary_result = future_summary.result()
                image_results = [None] * len(extracted_images)
                for future in concurrent.futures.as_completed(future_images):
                    index = future_images[future]
                    try:
                        image_results[index] = future.result()
                    except Exception as exc:
                        image_results[index] = f"이미지 분석 중 오류 발생: {exc}"

            st.session_state.analysis_results = {
                'summary': summary_result,
                'images': image_results
            }
        st.success("모든 AI 분석이 완료되었습니다!")


    # 분석 결과가 세션에 저장되어 있으면 화면에 표시
    if st.session_state.analysis_results:
        summary = st.session_state.analysis_results.get('summary')
        image_analyses = st.session_state.analysis_results.get('images', [])

        st.header("📄 논문 종합 요약")
        if summary:
            st.markdown(summary)
        else:
            st.info("아직 요약 분석이 실행되지 않았습니다. '전체 AI 분석 시작하기' 버튼을 눌러주세요.")

        st.divider()

        st.header(f"🖼️ 이미지 상세 분석 ({len(extracted_images)}개)")
        if not extracted_images:
            st.warning("이 PDF에서는 이미지를 찾을 수 없습니다.")
        else:
            # 결과를 2열로 보기 좋게 배치
            cols = st.columns(2)
            for i, (img_bytes, analysis_result) in enumerate(zip(extracted_images, image_analyses)):
                col = cols[i % 2]
                with col:
                    st.image(img_bytes, caption=f"이미지 #{i+1}")
                    with st.expander(f"**이미지 #{i+1} AI 분석 결과 보기**"):
                        st.markdown(analysis_result)
                    st.markdown("---")
