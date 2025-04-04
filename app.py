import streamlit as st
from google import genai
from google.genai import types
import os
import csv
import json
import time
import io

# --- Configuration ---
DEFAULT_MODEL = "gemini-2.5-pro-exp-03-25"
AVAILABLE_MODELS = ["gemini-2.5-pro-preview-03-25", "gemini-2.5-pro-exp-03-25", "gemini-2.0-flash"]

ANKI_DELIMITER = ';'
API_RETRY_DELAY = 5

# --- Core Logic Functions ---

def configure_gemini():
    """Configures the Gemini API key using Streamlit Secrets."""
    try:
        api_key = st.secrets["google_api_key"]
        if not api_key:
             st.error("Google API Key found in Streamlit Secrets, but it is empty.")
             st.info("Please ensure the 'google_api_key' in your secrets configuration has a valid value.")
             return False
        genai.Client(api_key=api_key) # Correct method.
        return True
    except KeyError:
        st.error("Google Gemini API Key not found in Streamlit Secrets.")
        st.info("""
            **How to Fix:**
            * **Local:** Create `.streamlit/secrets.toml` with `google_api_key = "YOUR_API_KEY_HERE"`
            * **Deployed:** Add the `google_api_key` secret in Streamlit Community Cloud settings.
            Restart the app after adding the secret.
        """)
        return False
    except Exception as e:
        st.error(f"Failed to configure Gemini using key from Streamlit Secrets: {e}")
        return False

def create_flashcard_prompt() -> str:
    """Creates the detailed prompt for Gemini requesting structured JSON output from a provided document."""

    prompt = f"""
    You are an expert academic assistant helping a PhD student prepare for comprehensive exams.
    Your task is to analyze the **provided academic paper document (PDF)** and generate high-quality Anki flashcards focused on understanding the paper's significance, arguments, methods, findings, and limitations.
    Create as many flashcards as necessary to effectively cover the key concepts and findings of the paper, following the below principles and aspects.

    Apply these learning science principles for card creation:
    1.  **Atomicity:** Each card should test ONLY ONE specific concept, finding, argument, or term.
    2. **Future-proofing:** Add more specificity to the questions and answers. Avoid vague or overly broad prompts. For example, instead of "What is the main finding?", use "What was the primary research question addressed in Author et al., 1999?". Further, add in more context like descriptors.
    3.  **Active Recall:** Frame cards as specific questions (What, Why, How, What is the significance of...) or prompts requiring recall (Define X, Explain Y's role). Avoid simple True/False or vague prompts.
    4.  **Significance Focus:** Prioritize information crucial for understanding the paper's contribution, limitations, and place in the field. Go beyond surface-level facts. Connect findings/discussion back to the core research question where possible.
    5.  **Conciseness:** Keep both the 'front' (question/prompt) and 'back' (answer/explanation) clear, brief, and short.
    6.  **Clarity:** Use clear, precise language. Be specific in your questions and answers. Always explicitly define which paper you are referring to in the question.
    7.  **Building Blocks:** Use the paper's abstract, introduction, and conclusion as a starting point for generating questions. These sections often contain the most important information and can help you identify key concepts and findings.
    8.  **Avoid Recitation:** Do not include verbatim text from the paper. Instead, paraphrase and summarize key points.
    9.  **Avoid Overlap:** Ensure that each card covers a unique aspect of the paper. Avoid redundancy and overlap between cards.
    10. **Avoid Recursion:** Do not include questions that ask for a list of items or concepts. Instead, focus on specific details and explanations.
    11. **Avoid Ambiguity:** Ensure that each question has a clear and unambiguous answer. Avoid questions that could be interpreted in multiple ways.
    12. **Keep it short:** Aim for a maximum of 2-3 sentences per card. Avoid lengthy explanations or complex language. More cards are better with less information is better than fewer cards with more information.
    13. **Build on the basics:** Start with cards covering basic concepts of the paper and gradually move to more complex ideas. This will help reinforce foundational knowledge before diving into more advanced topics.
    14. **Use examples:** Where appropriate, include examples or case studies to illustrate key concepts. This can help reinforce understanding and make the material more relatable.
    15. **Order by complexity:** Organize the cards in a logical order, starting with simpler concepts and gradually progressing to more complex ideas. This will help reinforce understanding and make the material more relatable.

    Generate cards covering these aspects based on the **entire paper provided**:
    * **Core Problem/Question:** What specific gap, problem, or research question does the paper address?
    * **Hypothesis/Objective:** What are the main hypotheses or objectives?
    * **Methodology:** What are the key methods/techniques, and their purpose/significance?
    * **Key Findings/Results:** What are the specific, significant findings reported (quantify if possible)? Frame as questions about the finding.
    * **Interpretations/Conclusions:** What interpretations or conclusions do the authors draw? What is the main takeaway/contribution?
    * **Limitations/Caveats:** What specific limitations or weaknesses are mentioned?
    * **Key Terms/Concepts:** Define or explain crucial technical terms or concepts.
    * **Connections (Optional but helpful):** If evident, briefly frame a question about how a finding relates to the initial problem or hypothesis.
    * **General Questions:** If the paper is a review or meta-analysis, generate questions about the overall findings or trends across studies.
    * **Theoretical Implications:** What are the theoretical implications of the findings? How do they contribute to existing theories or frameworks in the field?
    * **Practical Implications:** What are the practical implications of the findings? How can they be applied in real-world contexts?
    * **Future Directions:** What future research directions do the authors suggest? What are the potential areas for further investigation?
    * **Comparative Analysis:** If applicable, how do the findings compare to previous research in the field? What are the similarities and differences?

    **Output Format:**
    Return ONLY a valid JSON list containing flashcard objects. Each object must have exactly two keys: "front" and "back". Do NOT include any text, explanations, or markdown formatting outside the JSON list itself.

    Example JSON Output Structure (General Questions):
    [
      {{"front": "What was the primary research question addressed in Author et al., 1999?", "back": "The study investigated the impact of X on Y under Z conditions."}},
      {{"front": "Define 'epistemic uncertainty' as used in Author et al., 2005.", "back": "Uncertainty stemming from a lack of knowledge or data, as distinct from aleatoric uncertainty (inherent randomness)."}},
      {{"front": "In Author et al., 2006, What was the main finding regarding the effect of the intervention reported?", "back": "A statistically significant improvement (p < .01) was observed in the intervention group compared to the control."}},
      {{"front": "In Author et al., 2008, How was the negative correlation found between A and B interpreted?", "back": "It was suggested that factor A might inhibit process B under the studied conditions."}}
    ]

    **Now, analyze the provided PDF document and generate the JSON output based on its content, following all instructions:**
    """
    return prompt

def generate_flashcards_from_pdf(pdf_file_bytes: bytes, mime_type: str, model_name: str) -> list[tuple[str, str]] | None:
    """Generates Anki flashcard pairs from PDF bytes using Gemini with JSON output."""
    flashcards = []
    try:
        client = genai.Client()

        pdf_document_part = types.Part.from_bytes(data=pdf_file_bytes, mime_type=mime_type)
        st.info(f"Prepared {mime_type} part ({len(pdf_file_bytes) / 1024:.1f} KB).")

    except Exception as e:
        st.error(f"Error initializing Gemini model or preparing PDF part: {e}")
        return None

    st.info("Processing PDF with Gemini. This may take several minutes for large documents...")

    prompt_text = create_flashcard_prompt()

    attempts = 0
    max_attempts = 2
    processed = False

    while attempts < max_attempts and not processed:
        try:
            response = client.models.generate_content(model=model_name, contents=[prompt_text, pdf_document_part])

            try:
                response_text = response.text.strip()
                if response_text.startswith("```json"): response_text = response_text[7:]
                if response_text.endswith("```"): response_text = response_text[:-3]
                response_text = response_text.strip()

                if not response_text:
                    st.warning(f"Received empty response from Gemini (Attempt {attempts + 1}/{max_attempts}).")
                    processed = True; break

                cards_data = json.loads(response_text)

                if not isinstance(cards_data, list):
                    st.warning(f"Gemini response was not a JSON list (Attempt {attempts + 1}/{max_attempts}). Skipping. Response: {response_text[:100]}...")
                    processed = True; break

                total_generated = 0
                for card_obj in cards_data:
                    if isinstance(card_obj, dict) and 'front' in card_obj and 'back' in card_obj:
                        front = str(card_obj['front']).strip()
                        back = str(card_obj['back']).strip()
                        if front and back:
                            flashcards.append((front, back))
                            total_generated += 1
                    else:
                        st.warning(f"Invalid card structure found in response: {card_obj}")

                if total_generated > 0:
                    st.success(f"Flashcard generation complete. Total cards: {total_generated}")
                else:
                    st.warning("Processing complete, but no flashcards were generated from the content. The model may not have found relevant information or the format wasn't suitable.")
                processed = True

            except json.JSONDecodeError as json_e:
                st.warning(f"Failed to decode JSON response (Attempt {attempts + 1}/{max_attempts}).")
                st.text(f"Response (first 200 chars): {response.text[:200]}...")
                st.text(f"Decode Error: {json_e}")
                time.sleep(API_RETRY_DELAY)
            except Exception as parse_e:
                st.error(f"Error processing response content: {parse_e}")
                st.text(f"Response (first 200 chars): {response.text[:200]}...")
                processed = True; break
        except Exception as e:
            st.error(f"Gemini API call error (Attempt {attempts + 1}/{max_attempts}): {e}")
            try:
                response_details = None
                if 'response' in locals() and response:
                    response_details = response

                if response_details and response_details.prompt_feedback:
                    st.warning(f"Prompt Feedback: {response_details.prompt_feedback}")
                if response_details and response_details.candidates and response_details.candidates[0].finish_reason:
                     finish_reason = response_details.candidates[0].finish_reason
                     st.warning(f"Generation Finish Reason: {finish_reason}")
                     if finish_reason == genai.types.FinishReason.SAFETY:
                          st.error("Content generation stopped due to safety settings.")
                     elif finish_reason == genai.types.FinishReason.RECITATION:
                           st.warning("Content generation stopped due to potential recitation.")
                     elif finish_reason == genai.types.FinishReason.MAX_TOKENS:
                           st.warning("Content generation stopped because the maximum output token limit was reached.")
                     elif finish_reason != genai.types.FinishReason.STOP:
                           st.warning(f"Content generation stopped for reason: {finish_reason.name}")

            except AttributeError:
                st.warning("Could not access detailed feedback fields in the response object.")
            except Exception as feedback_e:
                st.warning(f"Could not access detailed feedback: {feedback_e}")
            time.sleep(API_RETRY_DELAY * (attempts + 1))

        attempts += 1
        if not processed and attempts == max_attempts:
             st.error(f"Failed to process PDF after {max_attempts} attempts.")
             return None

    return flashcards if processed and flashcards else None

def convert_to_anki_csv(flashcards: list[tuple[str, str]]) -> str:
    """Converts the list of flashcards to a CSV string."""
    output = io.StringIO()
    writer = csv.writer(output, delimiter=ANKI_DELIMITER, quoting=csv.QUOTE_MINIMAL)
    writer.writerows(flashcards)
    return output.getvalue()



st.set_page_config(page_title="Article to Anki Flashcards", layout="wide")

st.title("📄➡️🧠 Article to Anki Flashcards Generator")
st.markdown("""
Upload an academic paper (PDF). This tool uses Google Gemini to analyze the study and generate Anki-importable flashcards.
""")

with st.sidebar:
    st.header("⚙️ Configuration")

    selected_model = st.selectbox(
        "Select Gemini Model",
        options=AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(DEFAULT_MODEL),
        help="Select the Gemini model to use. Preview/Experimental models might have newer features but less stability."
    )

    st.markdown("---")
    st.markdown("Created with [Streamlit](https://streamlit.io) & [Google Gemini](https://ai.google.dev/).")

uploaded_file = st.file_uploader("1. Upload your PDF file", type="pdf", accept_multiple_files=False)

if 'flashcards' not in st.session_state:
    st.session_state.flashcards = None
if 'output_csv_data' not in st.session_state:
    st.session_state.output_csv_data = None
if 'processed_filename' not in st.session_state:
     st.session_state.processed_filename = None

if uploaded_file is not None:
    st.info(f"Uploaded file: `{uploaded_file.name}` (Type: {uploaded_file.type})")

    if st.button("🚀 Generate Flashcards from PDF", key="generate_button"):
        st.session_state.flashcards = None
        st.session_state.output_csv_data = None
        st.session_state.processed_filename = uploaded_file.name

        with st.spinner(f"Configuring API and sending PDF to Gemini ({selected_model}) for processing..."):
            if not configure_gemini():
                st.stop()

            pdf_bytes = uploaded_file.getvalue()
            mime_type = uploaded_file.type or "application/pdf"

            if not pdf_bytes:
                st.error("Could not read bytes from the uploaded PDF file.")
                st.stop()

            st.session_state.flashcards = generate_flashcards_from_pdf(
                pdf_bytes,
                mime_type,
                selected_model
            )

            if st.session_state.flashcards:
                st.session_state.output_csv_data = convert_to_anki_csv(st.session_state.flashcards)

if st.session_state.output_csv_data:
    st.markdown("---")
    st.header("✅ Flashcards Ready!")
    num_cards = len(st.session_state.flashcards)
    st.info(f"Generated {num_cards} flashcards from '{st.session_state.processed_filename}'.")
    safe_filename_base = os.path.splitext(st.session_state.processed_filename.replace(" ", "_"))[0]
    output_filename = f"{safe_filename_base}_anki_cards.txt"
    st.download_button(
        label="⬇️ Download Anki File (.txt)",
        data=st.session_state.output_csv_data,
        file_name=output_filename,
        mime='text/plain',
    )
    with st.expander("Anki Import Instructions"):
        st.markdown(f"""
        1. Open Anki > File > Import... > Select `{output_filename}`.
        2. Choose Deck & Note Type (e.g., `Basic`).
        3. **Separator:** Choose **Semicolon**.
        4. Check 'Allow HTML in fields'.
        5. Map `Field 1` -> `Front`, `Field 2` -> `Back`.
        6. Click 'Import' & **Review cards!**
        """)

elif st.session_state.processed_filename:
    if st.session_state.flashcards is None:
         st.error(f"Flashcard generation process failed for '{st.session_state.processed_filename}'. Please check error messages above. Ensure the selected model ({selected_model}) supports PDF input and JSON output.")
    elif not st.session_state.flashcards:
         st.warning(f"Processing complete for '{st.session_state.processed_filename}', but no flashcards were successfully generated. The model may not have found relevant information, the PDF content might be unsuitable (e.g., image-only), or the response format was invalid.")