import streamlit as st
import PyPDF2
import json
import pandas as pd
import re
import os
import asyncio
import logging
from datetime import datetime
from io import BytesIO
from typing import List, TypedDict, Optional

# LangChain/LangGraph imports (replaces CrewAI)
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph, END

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('adk_app.log'), # Log to a new file
        logging.StreamHandler()
    ]
)

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets.get("OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "") or ""

# ----------------------------------------------------------------------------
# 1. PDF & KLP PROCESSING (UNCHANGED)
# Your PDFProcessor and KLPLoader classes are perfect as-is.
# ----------------------------------------------------------------------------

class PDFProcessor:
    """Handle PDF text extraction and parsing"""
    
    def __init__(self):
        self.cadet_responses = []
    
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from uploaded PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logging.error(f"Error extracting PDF text: {e}")
            return None
    
    def parse_pdf_content(self, pdf_text):
        """Parse PDF content to extract cadet responses"""
        if not pdf_text:
            return []
        
        # Split by page breaks and sections
        sections = re.split(r'Page \d+ of \d+|Essay Question|Test:', pdf_text)
        cadet_responses = []
        
        for section in sections:
            if not section.strip():
                continue
            
            # Accept both numbers AND letters as cadet identifiers
            cadet_match = re.search(r'For:\s*([A-Za-z0-9]+)', section)
            if not cadet_match:
                continue
            
            cadet_number = cadet_match.group(1)
            
            # Extract book number
            book_match = re.search(r'Book\s+(\d+)', section, re.IGNORECASE)
            book_number = f"Book_{book_match.group(1)}" if book_match else "Unknown"
            
            # Extract question
            question_match = re.search(r'Write a paragraph between[^.]*\.', section)
            question = question_match.group(0) if question_match else ""
            
            additional_question_match = re.search(r'-\s*Write about[^.]*\.', section)
            if additional_question_match:
                question += " " + additional_question_match.group(0)
            
            # Extract response
            response = ""
            question_end_pattern = r'-\s*Write about[^.]*\.'
            question_end_match = re.search(question_end_pattern, section)
            
            if question_end_match:
                after_question = section[section.find(question_end_match.group(0)) + len(question_end_match.group(0)):]
                response_match = re.search(r'^(.*?)(?:Instructor Comments:|$)', after_question, re.DOTALL)
                if response_match:
                    response = response_match.group(1).strip()
                    response = re.sub(r'\s+', ' ', response).strip()
            
            # Clean response
            response = self.clean_response(response)
            
            if cadet_number and question and response:
                cadet_responses.append({
                    'cadetNumber': cadet_number,
                    'bookNumber': book_number,
                    'question': question,
                    'response': response
                })
        
        logging.info(f"Extracted {len(cadet_responses)} cadet responses")
        return cadet_responses
    
    def clean_response(self, response):
        """Clean response text by removing unwanted sections"""
        response = re.sub(r'Section \d+', '', response, flags=re.IGNORECASE)
        response = re.sub(r'\s+', ' ', response).strip()
        return response

class KLPLoader:
    """Handle loading and management of Key Learning Points"""
    
    def __init__(self, klp_file_path):
        self.klp_file_path = klp_file_path
        self.klps = self.load_klps()
    
    def load_klps(self):
        """Load KLPs from JSON file"""
        try:
            with open(self.klp_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading KLPs: {e}")
            return {}
    
    def get_klp_for_book(self, book_number):
        """Get KLPs for a specific book number"""
        if book_number in self.klps:
            return self.klps[book_number]
        return {"Grammatical Structures": "No specific structures found", "Vocabulary Lists": "No vocabulary found"}
    
    def get_vocabulary_list(self, book_number):
        """Get ONLY vocabulary list for the book - direct injection"""
        klp = self.get_klp_for_book(book_number)
        return klp.get('Vocabulary Lists', 'No vocabulary found')
    
    def get_grammar_structures(self, book_number):
        """Get ONLY grammar structures for the book - direct injection"""
        klp = self.get_klp_for_book(book_number)
        return klp.get('Grammatical Structures', 'No grammar structures found')

# ----------------------------------------------------------------------------
# 2. FEEDBACK & MEMORY (UNCHANGED)
# Your FeedbackManager is the long-term memory system.
# We will now *use* this memory in the agent prompts.
# ----------------------------------------------------------------------------

class FeedbackManager:
    """Handle feedback and memory management"""
    
    def __init__(self, memory_file="long_term_memory.json"):
        self.memory_file = memory_file
        self.memory = self.load_memory()
    
    def load_memory(self):
        """Load existing memory from file"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logging.error(f"Error loading memory: {e}")
            return []
    
    def save_memory(self):
        """Save memory to file"""
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.memory, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Error saving memory: {e}")
    
    def add_feedback(self, input_text, output_result, rating, comment):
        """Add new feedback to memory"""
        feedback_entry = {
            "input": input_text[:500],
            "output": str(output_result)[:500],
            "feedback": {
                "rating": rating,
                "comment": comment,
                "timestamp": datetime.now().isoformat()
            }
        }
        self.memory.append(feedback_entry)
        self.save_memory()
        logging.info(f"Added feedback: rating={rating}, comment={comment[:50]}...")
    
    def get_latest_feedback(self):
        """Get the most recent feedback for adaptation"""
        if self.memory:
            latest = self.memory[-1]
            return f"A previous evaluation was rated {latest['feedback']['rating']}/5. The user's feedback was: '{latest['feedback']['comment']}'. Please adapt your evaluation to incorporate this feedback."
        return "No previous feedback available. This is the first run."

# ----------------------------------------------------------------------------
# 3. NEW AGENT FRAMEWORK (Replaces CrewAIManager)
# This section uses LangGraph for a reliable, stateful workflow.
# ----------------------------------------------------------------------------

# --- 3.1 Define reliable output schemas (Pydantic Models) ---
# This ensures we *always* get valid JSON with the correct structure.

class VocabScore(BaseModel):
    """Schema for Vocabulary Evaluation"""
    vocabulary_score: int = Field(description="Score from 0-10 based on criteria")
    vocab_words_found: List[str] = Field(description="List of KLP vocabulary words found")

class GrammarScore(BaseModel):
    """Schema for Grammar Evaluation"""
    grammar_score: int = Field(description="Score from 0-10 based on criteria")
    grammar_structures_found: List[str] = Field(description="List of KLP grammar structures found")

class TaskScore(BaseModel):
    """Schema for Task Achievement Evaluation"""
    task_achievement_score: int = Field(description="Score from 0-40 based on criteria")
    feedback: str = Field(description="One-sentence feedback")

class RangeScore(BaseModel):
    """Schema for Range & Accuracy Evaluation"""
    range_accuracy_score: int = Field(description="Score from 0-40 based on criteria")
    feedback: str = Field(description="One-sentence feedback")

class Synthesis(BaseModel):
    """Schema for Final Feedback Synthesis"""
    final_feedback: str = Field(description="Comprehensive 2-3 sentence assessment")

# --- 3.2 Define the State for the Graph ---
# This is the "memory" that passes between agents.

class GraphState(TypedDict):
    """The state of our agent graph."""
    # Inputs
    student_response: dict
    vocabulary_list: str
    grammar_structures: str
    latest_feedback: str  # For the synthesis agent
    
    # Results from each agent
    vocab_eval: Optional[VocabScore] = None
    grammar_eval: Optional[GrammarScore] = None
    task_eval: Optional[TaskScore] = None
    range_eval: Optional[RangeScore] = None
    final_feedback: Optional[Synthesis] = None
    
    # Error handling
    error: str = None

# --- 3.3 The new ADKManager ---

class ADKManager:
    """Manage LangGraph agents and evaluation process"""
    
    def __init__(self, klp_loader, feedback_manager):
        self.klp_loader = klp_loader
        self.feedback_manager = feedback_manager
        self.llm = ChatOpenAI(
            model="gpt-5.1-2025-11-13", # Using your specified model
        )
        
        # Build the graph once
        self.graph = self._create_graph()
        self.app = self.graph.compile()

    def _create_graph(self):
        """Defines the LangGraph workflow."""
        graph = StateGraph(GraphState)
        
        # Define the nodes (agents)
        graph.add_node("evaluate_vocabulary", self.evaluate_vocabulary)
        graph.add_node("evaluate_grammar", self.evaluate_grammar)
        graph.add_node("evaluate_task_achievement", self.evaluate_task_achievement)
        graph.add_node("evaluate_range_accuracy", self.evaluate_range_accuracy)
        graph.add_node("synthesize_feedback", self.synthesize_feedback)
        
        # Define the edges (workflow)
        graph.set_entry_point("evaluate_vocabulary")
        graph.add_edge("evaluate_vocabulary", "evaluate_grammar")
        graph.add_edge("evaluate_grammar", "evaluate_task_achievement")
        graph.add_edge("evaluate_task_achievement", "evaluate_range_accuracy")
        graph.add_edge("evaluate_range_accuracy", "synthesize_feedback")
        graph.add_edge("synthesize_feedback", END)
        
        return graph

    # --- Node 1: Vocabulary Agent ---
    def evaluate_vocabulary(self, state: GraphState):
        logging.info(f"Evaluating vocabulary for {state['student_response']['cadetNumber']}")
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a Vocabulary Evaluator. You ONLY count vocabulary words that appear in the provided KLP vocabulary list. 
You follow the exact scoring rubric without deviation. You always return valid JSON.

SCORING CRITERIA - Vocabulary Usage (0-10 points):
- 0 = No response
- 2 = Uses one vocabulary word but incorrectly
- 4 = Attempts one or two vocabulary words with limited success (errors or partial usage)
- 6 = Uses one or two vocabulary words but not fully accurate across response
- 8 = Uses three vocabulary words with minor issues
- 10 = Uses three vocabulary words correctly & appropriately

IMPORTANT: Be extremely strict - ONLY count vocabulary words that appear in the KLP vocabulary list below."""),
            ("user", """STUDENT RESPONSE:
{response}

KLP VOCABULARY LIST FOR THIS BOOK:
{vocabulary_list}

Evaluate the student's response based *only* on the list and criteria.
"""),
        ])
        
        # Create the LLM chain with reliable structured output
        eval_chain = prompt_template | self.llm.with_structured_output(VocabScore)
        
        result = eval_chain.invoke({
            "response": state['student_response']['response'],
            "vocabulary_list": state['vocabulary_list']
        })
        
        return {"vocab_eval": result}

    # --- Node 2: Grammar Agent ---
    def evaluate_grammar(self, state: GraphState):
        logging.info(f"Evaluating grammar for {state['student_response']['cadetNumber']}")
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a Grammar Evaluator. You ONLY count grammar structures that appear in the provided KLP grammar list. 
You follow the exact scoring rubric without deviation. You always return valid JSON.

SCORING CRITERIA - Grammar Usage (0-10 points):
- 0 = No response
- 2 = Uses one grammar point but incorrectly
- 4 = Attempts one grammar point with limited success (errors or partial usage)
- 6 = Uses one grammar point but not fully accurate across response
- 8 = Uses one grammar point with minor issues
- 10 = Uses one grammar point correctly & appropriately

IMPORTANT: Be extremely strict - ONLY count grammar structures that appear in the KLP grammar list below."""),
            ("user", """STUDENT RESPONSE:
{response}

KLP GRAMMAR STRUCTURES FOR THIS BOOK:
{grammar_structures}

Evaluate the student's response based *only* on the list and criteria.
"""),
        ])
        
        eval_chain = prompt_template | self.llm.with_structured_output(GrammarScore)
        
        result = eval_chain.invoke({
            "response": state['student_response']['response'],
            "grammar_structures": state['grammar_structures']
        })
        
        return {"grammar_eval": result}

    # --- Node 3: Task Achievement Agent ---
    def evaluate_task_achievement(self, state: GraphState):
        logging.info(f"Evaluating task achievement for {state['student_response']['cadetNumber']}")
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a Task Achievement Specialist. You evaluate how completely students answer all parts of the question.
You follow the exact scoring rubric without deviation. You always return valid JSON.

SCORING CRITERIA - Task Achievement (0-40 points):
- 0 = No response
- 8 = Minimal, very few words, barely addresses question
- 16 = Half response, covers about half with major gaps
- 24 = Partial, answers most parts, some gaps or short on words
- 32 = Good, mostly developed, minor improvements needed
- 40 = Full, complete, well-developed, within limit"""),
            ("user", """QUESTION:
{question}

STUDENT RESPONSE:
{response}

Evaluate the student's response based on the criteria.
"""),
        ])
        
        eval_chain = prompt_template | self.llm.with_structured_output(TaskScore)
        
        result = eval_chain.invoke({
            "question": state['student_response']['question'],
            "response": state['student_response']['response']
        })
        
        return {"task_eval": result}

    # --- Node 4: Range & Accuracy Agent ---
    def evaluate_range_accuracy(self, state: GraphState):
        logging.info(f"Evaluating range/accuracy for {state['student_response']['cadetNumber']}")
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a Range and Accuracy Specialist. You evaluate the overall quality of grammar, punctuation, and spelling.
You follow the exact scoring rubric without deviation. You always return valid JSON.

SCORING CRITERIA - Range & Accuracy (0-40 points):
- 0 = No response
- 8 = Minimal response. Numerous errors in structure, punctuation, spelling (hard to understand)
- 16 = Half response. Some correct structures but frequent errors in spelling, punctuation, sentence formation
- 24 = Partial response. Reasonable range of structures and vocabulary but noticeable errors
- 32 = Good response. Good range of vocabulary and grammatical structures with minor errors
- 40 = Full response. Excellent structure, punctuation, and accuracy with very few minor errors"""),
            ("user", """STUDENT RESPONSE:
{response}

Evaluate the student's response based on the criteria.
"""),
        ])
        
        eval_chain = prompt_template | self.llm.with_structured_output(RangeScore)
        
        result = eval_chain.invoke({
            "response": state['student_response']['response']
        })
        
        return {"range_eval": result}

    # --- Node 5: Synthesis Agent (with MEMORY) ---
    def synthesize_feedback(self, state: GraphState):
        logging.info(f"Synthesizing feedback for {state['student_response']['cadetNumber']}")
        
        # Here we use the long-term memory from FeedbackManager!
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are the Quality Assurance Lead. You review all specialist evaluations and provide a comprehensive, actionable feedback summary.

**CONTEXT FROM PAST RUNS (MEMORY):**
{latest_feedback}

Please synthesize the following evaluations into a 2-3 sentence summary.
If the memory context provides feedback, try to adapt your synthesis to address it.
"""),
            ("user", """CADET: {cadet_id}
BOOK: {book_number}
RESPONSE: {response}

EVALUATIONS:
- Vocabulary Score: {vocab_score} (Words: {vocab_words})
- Grammar Score: {grammar_score} (Structures: {grammar_structures})
- Task Achievement Score: {task_score} (Feedback: {task_feedback})
- Range & Accuracy Score: {range_score} (Feedback: {range_feedback})

Provide the final, synthesized feedback.
"""),
        ])
        
        eval_chain = prompt_template | self.llm.with_structured_output(Synthesis)
        
        result = eval_chain.invoke({
            "latest_feedback": state['latest_feedback'],
            "cadet_id": state['student_response']['cadetNumber'],
            "book_number": state['student_response']['bookNumber'],
            "response": state['student_response']['response'],
            "vocab_score": state['vocab_eval'].vocabulary_score,
            "vocab_words": ", ".join(state['vocab_eval'].vocab_words_found),
            "grammar_score": state['grammar_eval'].grammar_score,
            "grammar_structures": ", ".join(state['grammar_eval'].grammar_structures_found),
            "task_score": state['task_eval'].task_achievement_score,
            "task_feedback": state['task_eval'].feedback,
            "range_score": state['range_eval'].range_accuracy_score,
            "range_feedback": state['range_eval'].feedback,
        })
        
        return {"final_feedback": result}

    # --- 3.4 Main evaluation runners ---

    async def run_single_evaluation(self, response, vocabulary_list, grammar_structures, latest_feedback):
        """Run the compiled graph for a single student."""
        try:
            logging.info(f"Starting graph evaluation for cadet {response['cadetNumber']}")
            
            # The input for the graph
            graph_input = {
                "student_response": response,
                "vocabulary_list": vocabulary_list,
                "grammar_structures": grammar_structures,
                "latest_feedback": latest_feedback,
            }
            
            # Run the graph asynchronously
            final_state = await asyncio.wait_for(
                self.app.ainvoke(graph_input),
                timeout=120.0 # Increased timeout for full graph
            )
            
            # Format the final state into the flat dictionary you need
            result_data = {
                'cadet_id': response['cadetNumber'],
                'book_number': response['bookNumber'],
                'vocab_score': final_state['vocab_eval'].vocabulary_score,
                'grammar_score': final_state['grammar_eval'].grammar_score,
                'task_achievement_score': final_state['task_eval'].task_achievement_score,
                'range_accuracy_score': final_state['range_eval'].range_accuracy_score,
                'final_feedback': final_state['final_feedback'].final_feedback
            }
            
            logging.info(f"Completed evaluation for cadet {response['cadetNumber']}")
            return result_data

        except asyncio.TimeoutError:
            logging.warning(f"Timeout evaluating cadet {response['cadetNumber']}")
            return self.get_error_result(response, "Evaluation timeout")
        except Exception as e:
            logging.error(f"Error evaluating cadet {response['cadetNumber']}: {e}")
            return self.get_error_result(response, f"Error: {str(e)[:100]}")

    def get_error_result(self, response, feedback):
        """Helper to create a consistent error entry."""
        return {
            'cadet_id': response['cadetNumber'],
            'book_number': response['bookNumber'],
            'vocab_score': 0,
            'grammar_score': 0,
            'task_achievement_score': 0,
            'range_accuracy_score': 0,
            'final_feedback': feedback
        }

    async def evaluate_students(self, cadet_responses):
        """Run PARALLEL evaluation for all students using the LangGraph app."""
        if not cadet_responses:
            return []
        
        total = len(cadet_responses)
        logging.info(f"Starting PARALLEL LangGraph evaluation of {total} students")
        
        # Get context ONCE
        book_number = cadet_responses[0]['bookNumber']
        vocabulary_list = self.klp_loader.get_vocabulary_list(book_number)
        grammar_structures = self.klp_loader.get_grammar_structures(book_number)
        latest_feedback = self.feedback_manager.get_latest_feedback() # Get long-term memory
        
        logging.info(f"All students from {book_number}. Using latest feedback for context.")
        
        # Create parallel tasks for ALL students
        tasks = [
            self.run_single_evaluation(response, vocabulary_list, grammar_structures, latest_feedback)
            for response in cadet_responses
        ]
        
        # Run ALL graphs in parallel
        logging.info(f"Launching {len(tasks)} parallel graph evaluations...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logging.error(f"Student {i+1} graph failed: {result}")
                final_results.append(
                    self.get_error_result(cadet_responses[i], f"Exception: {str(result)[:100]}")
                )
            else:
                final_results.append(result)
        
        logging.info(f"Completed: {len(final_results)}/{total} students")
        return final_results

# ----------------------------------------------------------------------------
# 4. EXCEL DOWNLOAD (UNCHANGED)
# Your Excel function is perfect as-is.
# ----------------------------------------------------------------------------

def create_excel_download(results):
    """Create Excel file for download"""
    if not results:
        return None
    
    df = pd.DataFrame(results)
    
    # Reorder columns
    column_order = [
        'cadet_id', 'book_number', 'vocab_score', 'grammar_score',
        'task_achievement_score', 'range_accuracy_score', 'final_feedback'
    ]
    
    # Filter for columns that actually exist in the dataframe
    df_columns = [col for col in column_order if col in df.columns]
    df = df[df_columns]
    
    # Create Excel in memory
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Evaluation Results', index=False)
    
    output.seek(0)
    return output.getvalue()

# ----------------------------------------------------------------------------
# 5. STREAMLIT APP (UNCHANGED)
# Your Streamlit UI logic is perfect as-is.
# It will now call ADKManager instead of CrewAIManager.
# ----------------------------------------------------------------------------

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Student Writing Evaluator",
        page_icon="📝",
        layout="wide"
    )
    
    st.title("📝 Student Writing PDF Evaluator")
    st.markdown("Upload student writing PDFs for AI-powered evaluation using a reliable Agentic Framework")
    
    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'klp_loader' not in st.session_state:
        st.session_state.klp_loader = None
    if 'feedback_manager' not in st.session_state:
        st.session_state.feedback_manager = FeedbackManager()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key status
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            st.success("✅ OpenAI API Key: Configured")
        else:
            st.error("❌ OpenAI API Key: Missing")
            st.info("Add OPENAI_API_KEY to secrets or environment variables")
        
        # KLP file upload
        st.subheader("📚 Key Learning Points (KLPs)")
        klp_file = st.file_uploader(
            "Upload KLPs JSON file",
            type=['json'],
            help="Upload the book_vocabulary_grammar_lists.json file"
        )
        
        if klp_file:
            # Save uploaded file temporarily
            with open("temp_klps.json", "wb") as f:
                f.write(klp_file.getbuffer())
            st.session_state.klp_loader = KLPLoader("temp_klps.json")
            st.success("✅ KLPs loaded successfully!")
        
        # Memory management
        st.subheader("🧠 Memory Management")
        if st.button("Clear Memory"):
            st.session_state.feedback_manager.memory = []
            st.session_state.feedback_manager.save_memory()
            st.success("Memory cleared!")
        
        # Info section
        st.markdown("---")
        st.info("**Note:** Cadet IDs accept both numbers (1234) and letters (ABCD) as identifiers.")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📄 PDF Upload and Processing")
        
        # PDF upload
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a student writing PDF for evaluation"
        )
        
        if uploaded_file and st.session_state.klp_loader:
            if st.button("🚀 Process PDF", type="primary"):
                with st.spinner("Processing PDF..."):
                    try:
                        # Process PDF
                        processor = PDFProcessor()
                        pdf_text = processor.extract_text_from_pdf(uploaded_file)
                        
                        if pdf_text:
                            cadet_responses = processor.parse_pdf_content(pdf_text)
                            
                            if cadet_responses:
                                st.success(f"✅ Extracted {len(cadet_responses)} student responses")
                                
                                # Display extracted data
                                with st.expander("👀 Extracted Data Preview"):
                                    for i, response in enumerate(cadet_responses[:3]):
                                        st.write(f"**Cadet {response['cadetNumber']} ({response['bookNumber']})**")
                                        st.write(f"Question: {response['question'][:100]}...")
                                        st.write(f"Response: {response['response'][:200]}...")
                                        st.write("---")
                                
                                # Run evaluation
                                with st.spinner("🤖 Running AI evaluation graph..."):
                                    # THIS IS THE ONLY CHANGE
                                    adk_manager = ADKManager(
                                        st.session_state.klp_loader,
                                        st.session_state.feedback_manager
                                    )
                                    
                                    results = asyncio.run(adk_manager.evaluate_students(cadet_responses))
                                    st.session_state.results = results
                                
                                st.success(f"✅ Evaluation complete for {len(results)} students!")
                                
                            else:
                                st.error("❌ No student responses found in PDF")
                        else:
                            st.error("❌ Failed to text from PDF")
                            
                    except Exception as e:
                        st.error(f"❌ Error processing PDF: {str(e)}")
                        logging.error(f"PDF processing error: {e}")
        
        elif uploaded_file and not st.session_state.klp_loader:
            st.warning("⚠️ Please upload the KLPs JSON file first (in sidebar)")
        
        # Display results
        if st.session_state.results:
            st.header("📊 Evaluation Results")
            
            # Results table
            df = pd.DataFrame(st.session_state.results)
            st.dataframe(df, use_container_width=True)
            
            # Download Excel
            excel_data = create_excel_download(st.session_state.results)
            if excel_data:
                st.download_button(
                    label="📥 Download Excel Report",
                    data=excel_data,
                    file_name=f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheet_ml.sheet"
                )
    
    with col2:
        st.header("💬 Feedback System")
        
        if st.session_state.results:
            st.subheader("Rate the Evaluation")
            
            # Rating slider
            rating = st.slider(
                "How accurate was the evaluation?",
                min_value=1,
                max_value=5,
                value=3,
                help="Rate the quality of the AI evaluation (1=Poor, 5=Excellent)"
            )
            
            # Comment box
            comment = st.text_area(
                "Additional Feedback",
                placeholder="Provide specific feedback to improve future evaluations...",
                help="Your feedback will be used to adapt the AI evaluation for future runs"
            )
            
            if st.button("Submit Feedback"):
                if comment.strip():
                    # Get sample input/output for memory
                    sample_input = st.session_state.results[0]['final_feedback'] if st.session_state.results else ""
                    sample_output = str(st.session_state.results)
                    
                    st.session_state.feedback_manager.add_feedback(
                        sample_input, sample_output, rating, comment
                    )
                    
                    st.success("✅ Feedback saved! It will be used to improve future evaluations.")
                else:
                    st.warning("⚠️ Please provide a comment with your feedback.")
        
        # Memory status
        st.subheader("📈 Memory Status")
        memory_count = len(st.session_state.feedback_manager.memory)
        st.metric("Feedback Entries", memory_count)
        
        if memory_count > 0:
            latest_feedback = st.session_state.feedback_manager.memory[-1]
            st.write("**Latest Feedback:**")
            st.write(f"⭐ Rating: {latest_feedback['feedback']['rating']}/5")
            st.write(f"💭 Comment: {latest_feedback['feedback']['comment'][:100]}...")

if __name__ == "__main__":
    main()
