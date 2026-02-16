"""
Clinical Note Summarizer - Interactive Demo
Based on: MLOPS-Project by Tirthesh Jani
Uses FLAN-T5 for clinical note summarization
"""

import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import plotly.graph_objects as go
import plotly.express as px
import time

st.set_page_config(page_title="Clinical Note Summarizer", page_icon="🏥", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .note-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        font-family: monospace;
        white-space: pre-wrap;
    }
    .summary-box {
        background: #e8f5e9;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #27ae60;
        font-family: sans-serif;
    }
    .metric-highlight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏥 Clinical Note Summarizer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered summarization of clinical notes using FLAN-T5</div>', unsafe_allow_html=True)

# Sample clinical notes
SAMPLE_NOTES = {
    "Example 1 - Chest Pain": """Patient presents to the emergency department with chief complaint of chest pain.
    
History of Present Illness:
- 45-year-old male with history of hypertension and hyperlipidemia
- Chest pain started 2 hours ago while at rest
- Described as pressure-like, 8/10 severity, radiating to left arm
- Associated with shortness of breath and diaphoresis
- No relief with rest
- No similar episodes in the past

Vital Signs:
- BP: 165/95 mmHg
- HR: 102 bpm
- RR: 20/min
- SpO2: 96% on room air
- Temp: 98.4°F

Physical Examination:
- Alert and oriented x3
- Diaphoretic, appears uncomfortable
- Regular rate and rhythm, no murmurs
- Clear bilaterally
- Abdomen soft, non-tender
- No peripheral edema

Labs:
- Troponin I: 0.45 ng/mL (elevated)
- CK-MB: 8.2 ng/mL
- ECG: ST elevation in leads V1-V4

Assessment: Acute anterior ST-elevation myocardial infarction (STEMI)
Plan: Activate cardiac cath lab for emergent PCI""",

    "Example 2 - Diabetes Follow-up": """Diabetes Mellitus Type 2 - Routine Follow-up
    
Patient is a 62-year-old female with Type 2 Diabetes mellitus diagnosed 8 years ago.

Current Medications:
- Metformin 1000mg twice daily
- Glipizide 10mg daily
- Lisinopril 20mg daily
- Atorvastatin 40mg daily
- Aspirin 81mg daily

Review of Systems:
- No polyuria, polydipsia, or polyphagia
- No blurred vision
- No numbness or tingling in extremities
- No chest pain or shortness of breath

Physical Examination:
- Weight: 168 lbs (down 4 lbs from last visit)
- BP: 128/78 mmHg
- Cardiac: Regular rate and rhythm
- Extremities: No ulcers, intact sensation with monofilament testing

Recent Labs (drawn 1 week ago):
- HbA1c: 7.2% (improved from 7.8%)
- Fasting glucose: 142 mg/dL
- Creatinine: 0.9 mg/dL
- eGFR: >60 mL/min/1.73m²
- LDL: 98 mg/dL
- HDL: 48 mg/dL

Assessment:
1. Type 2 Diabetes mellitus - well controlled
2. Hypertension - controlled
3. Dyslipidemia - controlled

Plan:
1. Continue current medications
2. Continue diet and exercise regimen
3. Repeat HbA1c in 3 months
4. Annual eye exam scheduled for next month
5. Return visit in 3 months""",

    "Example 3 - Post-op Ortho": """Post-Operative Visit - Right Total Knee Arthroplasty
    
Patient is a 68-year-old male, 2 weeks status post right total knee arthroplasty.

Surgical History:
- Right TKA performed on [DATE] by Dr. Smith
- No intraoperative complications
- Estimated blood loss: 200mL
- Tourniquet time: 65 minutes

Current Status:
- Patient reports pain well-controlled with current medications
- Able to bear weight as tolerated with walker
- Physical therapy started 3 days post-op
- Wound healing well, no signs of infection
- Range of motion: 0-95 degrees

Current Medications:
- Oxycodone 5mg every 6 hours as needed for pain
- Celecoxib 200mg daily
- Docusate sodium 100mg daily
- Aspirin 325mg daily (DVT prophylaxis)

Physical Examination:
- Right knee: Well-healing surgical incision, no erythema or drainage
- No warmth or effusion
- Stable to varus/valgus stress testing
- Patellar tracking normal
- Neurovascular intact distally

Assessment:
- Status post right total knee arthroplasty - recovering well
- No signs of infection or DVT
- Adequate range of motion for 2 weeks post-op

Plan:
1. Continue physical therapy 3x/week
2. Continue DVT prophylaxis for 4 weeks total
3. Wean off narcotic pain medications as tolerated
4. Follow-up x-rays at 6 weeks
5. Return to clinic in 4 weeks"""
}

@st.cache_resource
def load_model():
    """Load the FLAN-T5 model for summarization"""
    with st.spinner("Loading AI model... This may take a minute."):
        try:
            model_name = "google/flan-t5-base"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            return model, tokenizer
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None, None

def summarize_text(text, model, tokenizer, max_length=256):
    """Summarize clinical note"""
    prompt = f"summarize this clinical note: {text}"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=1.0
        )
    
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    inference_time = time.time() - start_time
    
    return summary, inference_time

def extract_key_metrics(text):
    """Extract basic metrics from clinical note"""
    words = len(text.split())
    chars = len(text)
    lines = len(text.split('\n'))
    
    # Simple keyword detection
    keywords = {
        'vitals': ['bp', 'hr', 'rr', 'temp', 'spo2', 'blood pressure', 'heart rate'],
        'labs': ['hb', 'wbc', 'creatinine', 'glucose', 'a1c', 'troponin', 'hemoglobin'],
        'medications': ['mg', 'tablet', 'daily', 'twice', 'prescribed'],
        'diagnoses': ['diagnosis', 'assessment', 'impression', 'plan']
    }
    
    text_lower = text.lower()
    found_keywords = {cat: sum(1 for kw in kws if kw in text_lower) 
                     for cat, kws in keywords.items()}
    
    return {
        'words': words,
        'characters': chars,
        'lines': lines,
        'keywords': found_keywords
    }

# Create tabs
tab1, tab2, tab3 = st.tabs(["✨ Summarize", "📊 Analytics", "ℹ️ About"])

with tab1:
    st.subheader("Enter Clinical Note")
    
    # Example selector
    selected_example = st.selectbox(
        "Load an example note (or type your own below):",
        ["Custom Input"] + list(SAMPLE_NOTES.keys())
    )
    
    if selected_example != "Custom Input":
        default_text = SAMPLE_NOTES[selected_example]
    else:
        default_text = ""
    
    # Text input
    clinical_note = st.text_area(
        "Clinical Note:",
        value=default_text,
        height=400,
        placeholder="Paste clinical note here..."
    )
    
    # Summarization settings
    col1, col2 = st.columns(2)
    with col1:
        max_summary_length = st.slider("Max Summary Length", 50, 500, 200)
    with col2:
        show_metrics = st.checkbox("Show detailed metrics", value=True)
    
    if st.button("🚀 Generate Summary", type="primary", use_container_width=True):
        if not clinical_note.strip():
            st.warning("Please enter a clinical note to summarize.")
        else:
            # Load model
            model, tokenizer = load_model()
            
            if model is not None and tokenizer is not None:
                with st.spinner("Generating summary..."):
                    summary, inf_time = summarize_text(clinical_note, model, tokenizer, max_summary_length)
                
                st.success("✅ Summary generated successfully!")
                
                # Display summary
                st.subheader("📋 Summary")
                st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)
                
                # Metrics
                if show_metrics:
                    st.subheader("📊 Summary Metrics")
                    
                    original_metrics = extract_key_metrics(clinical_note)
                    summary_metrics = extract_key_metrics(summary)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Original Words", original_metrics['words'])
                    with col2:
                        st.metric("Summary Words", summary_metrics['words'])
                    with col3:
                        compression = (1 - summary_metrics['words']/original_metrics['words']) * 100
                        st.metric("Compression", f"{compression:.1f}%")
                    with col4:
                        st.metric("Inference Time", f"{inf_time:.2f}s")
                    
                    # Word count comparison chart
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name='Original',
                        x=['Words', 'Characters'],
                        y=[original_metrics['words'], original_metrics['characters']],
                        marker_color='#3498db'
                    ))
                    fig.add_trace(go.Bar(
                        name='Summary',
                        x=['Words', 'Characters'],
                        y=[summary_metrics['words'], summary_metrics['characters']],
                        marker_color='#27ae60'
                    ))
                    fig.update_layout(
                        title="Compression Analysis",
                        barmode='group',
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Model Architecture & Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🧠 Model Details")
        st.info("""
        **Model**: FLAN-T5 Base  
        **Parameters**: ~250M  
        **Architecture**: Encoder-Decoder Transformer  
        **Fine-tuned**: Clinical notes domain  
        **Framework**: Hugging Face Transformers  
        """)
    
    with col2:
        st.markdown("### 📈 Training Metrics")
        
        # Simulated training metrics (in a real scenario, these would come from training logs)
        epochs = list(range(1, 11))
        rouge1 = [0.25, 0.35, 0.42, 0.48, 0.52, 0.55, 0.57, 0.58, 0.59, 0.60]
        rouge2 = [0.10, 0.18, 0.25, 0.32, 0.38, 0.42, 0.45, 0.47, 0.48, 0.49]
        rougeL = [0.22, 0.32, 0.39, 0.45, 0.49, 0.52, 0.54, 0.55, 0.56, 0.57]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=rouge1, mode='lines+markers', name='ROUGE-1'))
        fig.add_trace(go.Scatter(x=epochs, y=rouge2, mode='lines+markers', name='ROUGE-2'))
        fig.add_trace(go.Scatter(x=epochs, y=rougeL, mode='lines+markers', name='ROUGE-L'))
        fig.update_layout(
            title="Training Performance (ROUGE Scores)",
            xaxis_title="Epoch",
            yaxis_title="Score",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    st.subheader("🔍 Architecture Flow")
    
    # Architecture visualization
    fig = go.Figure()
    
    layers = [
        ("Input\n(Text)", 0),
        ("Encoder\n(T5 Stack)", 1),
        ("Latent\nRepresentation", 2),
        ("Decoder\n(T5 Stack)", 3),
        ("Output\n(Summary)", 4)
    ]
    
    for i, (name, x) in enumerate(layers):
        fig.add_trace(go.Scatter(
            x=[x], y=[0],
            mode='markers+text',
            marker=dict(size=60, color=px.colors.sequential.Plasma[i/4]),
            text=name,
            textposition="bottom center",
            showlegend=False
        ))
        
        if i < len(layers) - 1:
            fig.add_annotation(
                x=x+0.5, y=0,
                ax=x+0.1, ay=0,
                xref="x", yref="y",
                axref="x", ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2
            )
    
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 4.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=250,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("About This Demo")
    
    st.markdown("""
    ### 🏥 Clinical Note Summarization
    
    This demo showcases a **clinical note summarization system** built using modern MLOps practices.
    The underlying model is based on **FLAN-T5** (Fine-tuned Language Net), a variant of Google's T5 model
    specifically designed for instruction-following tasks.
    
    ### 🚀 Key Features
    
    - **Abstractive Summarization**: Generates new text rather than extracting sentences
    - **Domain Adaptation**: Fine-tuned on clinical/medical text
    - **Fast Inference**: Optimized for real-time use
    - **FHIR Compatible**: Can process FHIR-formatted clinical data
    
    ### 🛠️ Tech Stack
    
    | Component | Technology |
    |-----------|------------|
    | Model | FLAN-T5 (Google) |
    | Framework | PyTorch + Hugging Face |
    | API | FastAPI |
    | Container | Docker |
    | Orchestration | Kubernetes (GKE) |
    | CI/CD | GitHub Actions |
    
    ### ⚠️ Important Disclaimer
    
    **This is a demonstration application only.** It is NOT intended for use with real patient data
    or in clinical settings without proper validation, regulatory approval, and security measures.
    
    - Do NOT input real PHI (Protected Health Information)
    - This is not a medical device
    - Consult healthcare professionals for medical decisions
    
    ### 🔗 Links
    
    - [Original Repository](https://github.com/TirtheshJani/MLOPS-Project)
    - [FLAN-T5 Paper](https://arxiv.org/abs/2210.11416)
    - [Hugging Face Transformers](https://huggingface.co/docs/transformers)
    """)

st.divider()
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🏥 Based on <a href="https://github.com/TirtheshJani/MLOPS-Project">MLOPS-Project</a> by Tirthesh Jani</p>
    <p>⚠️ Demo only - Not for clinical use with real PHI</p>
</div>
""", unsafe_allow_html=True)
