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
        client = genai.Client(api_key=api_key)  # Correct method.
        return client
    except KeyError:
        st.error("Google Gemini API Key not found in Streamlit Secrets.")
        st.info("""
            **How to Fix:**
            * **Local:** Create `.streamlit/secrets.toml` with `google_api_key = "YOUR_API_KEY_HERE"`
            * **Deployed:** Add the `google_api_key` secret in Streamlit Community Cloud settings.
            Restart the app after adding the secret.
        """)
        return None
    except Exception as e:
        st.error(f"Failed to configure Gemini using key from Streamlit Secrets: {e}")
        return None


def create_flashcard_prompt() -> str:
    """Creates the detailed prompt for Gemini requesting structured JSON output from a provided document."""

    # Note the use of {{ and }} to escape literal braces within the f-string
    prompt = f"""
    You are an expert academic assistant tasked with aiding a PhD student preparing for comprehensive exams.

    **Input Source:**
    You will be provided with a **single academic paper document (PDF)**. All generated content MUST be derived **exclusively** from this document. Do not use any external knowledge or other sources.

    **Primary Goal:**
    Analyze the **entire provided academic paper** and generate high-quality Anki flashcards designed to facilitate deep understanding of the paper's core elements. The output MUST be a valid JSON list of flashcard objects.

    **Flashcard Creation Principles (Apply Rigorously):**

    1.  **Atomicity:** Each card MUST test ONLY ONE specific concept, finding, argument, method detail, term, or limitation.
    2.  **Specificity & Context (Future-proofing):** Questions and answers must be highly specific. Always include context, such as the paper reference (e.g., "In Author et al., 1999...") within the question (front). Avoid vague prompts like "What was the method?". Instead, ask "What specific statistical test was used in Author et al., 1999 to compare group A and group B?".
    3.  **Active Recall:** Frame cards as specific questions (What, Why, How, What is the significance of...) or direct prompts requiring recall (Define X, Explain Y's purpose in Author et al., 1999). Avoid True/False or overly simple recall.
    4.  **Significance Focus:** Prioritize information crucial for understanding the paper's contribution, core argument, limitations, and its place within its field. Go beyond surface-level facts. Where possible, link findings/interpretations back to the central research question or hypothesis.
    5.  **Conciseness:** Keep questions (front) and answers (back) clear, brief, and focused on the single atomic point.
    6.  **Clarity:** Use precise academic language appropriate to the paper's field. Be unambiguous in questions and answers.
    7.  **Avoid Overlap:** Ensure each card covers a unique piece of information. Do not create redundant cards testing the same point.
    8.  **Avoid List Generation Prompts:** Frame questions for single, specific answers, not lists. E.g., Instead of "List the limitations," create separate cards like "What was one key limitation mentioned regarding sample size in Author et al., 1999?".
    9.  **Keep Answers Short:** Answers (back) should ideally be 1-3 concise sentences. Use bullet points *only* if absolutely essential for clarity within a *single* atomic concept and cannot be broken into separate atomic cards.
    10. **Use Examples (If Applicable):** If the paper provides specific examples to illustrate a concept or finding, create a card asking about that example or its significance.
    11. **Verify Accuracy:** Ensure all information presented in the cards is factually correct and accurately reflects the content of the provided paper.

    **Content Coverage Checklist (Generate cards addressing these aspects, if present in the paper):**

    * **Core Problem/Question:** The specific gap, problem, or research question driving the study.
    * **Hypothesis/Objective:** The main hypotheses tested or the study's primary objectives.
    * **Methodology:** Key methods, techniques, apparatus, or data sources used. Include *why* a specific method was chosen if stated.
    * **Key Findings/Results:** Specific, significant findings. Quantify results (e.g., effect sizes, p-values) if reported and relevant to the core finding. Frame as questions about the specific result.
    * **Interpretations/Conclusions:** The authors' interpretations of the results and the main conclusions drawn. The paper's primary takeaway or contribution.
    * **Limitations/Caveats:** Specific weaknesses, limitations, or boundary conditions acknowledged by the authors.
    * **Key Terms/Concepts:** Definitions or explanations of crucial technical terms, theories, or concepts introduced or utilized in a specific way within the paper.
    * **Connections:** (If explicitly stated) How specific findings relate back to the initial problem, question, or hypothesis.
    * **Review/Meta-Analysis Specifics:** (If applicable) Overall trends, consensus findings, or major points of debate identified across the reviewed studies.
    * **Theoretical Implications:** How the findings contribute to, challenge, or refine existing theories or conceptual frameworks.
    * **Practical Implications:** Potential real-world applications or relevance of the findings.
    * **Future Directions:** Specific suggestions made by the authors for future research.
    * **Comparative Context:** (If discussed) How findings compare/contrast with specific prior research mentioned in the paper.

    **Strategic Hint:** Use the paper's Abstract, Introduction, and Conclusion sections as primary sources for identifying the most crucial elements (Problem, Hypothesis, Main Findings, Conclusion), then delve into the Methods, Results, and Discussion for specifics.

    **Mandatory Output Format:**
    Return **ONLY** a valid JSON list containing flashcard objects.
    Each object in the list MUST have exactly two keys: `"front"` (containing the question or prompt) and `"back"` (containing the answer or explanation).
    Do **NOT** include any introductory text, concluding remarks, explanations, code blocks, or markdown formatting outside the JSON list itself. The entire output must start with `[` and end with `]`.

    **Example JSON Output Structure:**
    ```json
    [
      {{"front": "In Author et al., 1999, what was the primary research question?", "back": "The study investigated the causal relationship between Factor X and Outcome Y in Population Z.", "tags": "research_question introduction causality"}},
      {{"front": "Define 'operationalization' as used in the methodology section of Author et al., 2005.", "back": "The process of defining variables into measurable factors. In this study, 'well-being' was operationalized using the SWLS score.", "tags": "definition methodology operationalization measurement"}},
      {{"front": "What specific statistical significance level was reported for the main finding in Author et al., 2006?", "back": "A statistically significant effect was found (p < 0.01) for the primary outcome measure.", "tags": "results finding statistics significance p-value"}},
      {{"front": "According to Author et al., 2008, what is one stated limitation regarding the study's generalizability?", "back": "The authors noted that the findings might be limited to the specific demographic group sampled (e.g., university students in North America).", "tags": "limitation generalizability discussion sampling"}}
    ]
    ```

    **Now, analyze the provided PDF document and generate the JSON output based strictly on its content, following all instructions.**
    """
    return prompt


def generate_flashcards_from_pdf(client, pdf_file_bytes: bytes, mime_type: str, model_name: str) -> list[tuple[
    str, str, str]] | None:
    """Generates Anki flashcard pairs from PDF bytes using Gemini with JSON output."""
    flashcards = []
    try:
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
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                response_text = response_text.strip()

                if not response_text:
                    st.warning(f"Received empty response from Gemini (Attempt {attempts + 1}/{max_attempts}).")
                    processed = True;
                    break

                cards_data = json.loads(response_text)

                if not isinstance(cards_data, list):
                    st.warning(
                        f"Gemini response was not a JSON list (Attempt {attempts + 1}/{max_attempts}). Skipping. Response: {response_text[:100]}...")
                    processed = True;
                    break

                total_generated = 0
                total_generated = 0
                for card_obj in cards_data:
                    # Check if it's a dict and has the required keys
                    if isinstance(card_obj, dict) and 'front' in card_obj and 'back' in card_obj:
                        front = str(card_obj['front']).strip()
                        back = str(card_obj['back']).strip()
                        # --- Tag Extraction ---
                        # Get tags if present, otherwise default to empty string.
                        # Ensure tags are treated as a single string.
                        tags_raw = card_obj.get('tags', '')  # Safely get 'tags', default to ''
                        if isinstance(tags_raw, list):
                            # If LLM mistakenly returns a list, join it. Adjust if needed.
                            tags = " ".join(str(t).strip() for t in tags_raw).strip()
                            st.warning(f"Received tags as list, joined to: '{tags}'")
                        else:
                            tags = str(tags_raw).strip()

                        # Only add if front and back are non-empty
                        if front and back:
                            flashcards.append((front, back, tags))  # Append tuple with tags
                            total_generated += 1
                    else:
                        st.warning(
                            f"Invalid card structure found or missing front/back: {str(card_obj)[:100]}...")  # Log truncated object

                if total_generated > 0:
                    st.success(f"Flashcard generation complete. Total cards: {total_generated}")
                else:
                    st.warning(
                        "Processing complete, but no flashcards were generated from the content. The model may not have found relevant information or the format wasn't suitable.")
                processed = True

            except json.JSONDecodeError as json_e:
                st.warning(f"Failed to decode JSON response (Attempt {attempts + 1}/{max_attempts}).")
                st.text(f"Response (first 200 chars): {response.text[:200]}...")
                st.text(f"Decode Error: {json_e}")
                time.sleep(API_RETRY_DELAY)
            except Exception as parse_e:
                st.error(f"Error processing response content: {parse_e}")
                st.text(f"Response (first 200 chars): {response.text[:200]}...")
                processed = True;
                break
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


def convert_to_anki_csv(flashcards: list[tuple[str, str, str]]) -> str:
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
            client = configure_gemini()
            if client is None:
                st.stop()

            pdf_bytes = uploaded_file.getvalue()
            mime_type = uploaded_file.type or "application/pdf"

            if not pdf_bytes:
                st.error("Could not read bytes from the uploaded PDF file.")
                st.stop()

            st.session_state.flashcards = generate_flashcards_from_pdf(client,
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
        st.error(
            f"Flashcard generation process failed for '{st.session_state.processed_filename}'. Please check error messages above. Ensure the selected model ({selected_model}) supports PDF input and JSON output.")
    elif not st.session_state.flashcards:
        st.warning(
            f"Processing complete for '{st.session_state.processed_filename}', but no flashcards were successfully generated. The model may not have found relevant information, the PDF content might be unsuitable (e.g., image-only), or the response format was invalid.")
