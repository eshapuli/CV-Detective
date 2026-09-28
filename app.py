import os
import re
import json
from io import BytesIO

import streamlit as st
import pypdf
from sentence_transformers import SentenceTransformer, util
from anthropic import Anthropic

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import mm


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CV Detective",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM UI
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .hero {
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 1.5rem;
        background: linear-gradient(
            135deg,
            #111827 0%,
            #1e293b 50%,
            #312e81 100%
        );
        color: white;
    }

    .hero h1 {
        font-size: 3rem;
        margin-bottom: 0.3rem;
        font-weight: 800;
    }

    .hero p {
        font-size: 1.1rem;
        color: #cbd5e1;
        margin-bottom: 0;
    }

    .section-title {
        font-size: 1.5rem;
        font-weight: 750;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }

    .metric-card {
        padding: 1.2rem;
        border-radius: 16px;
        border: 1px solid rgba(128,128,128,0.25);
        text-align: center;
        min-height: 110px;
    }

    .metric-number {
        font-size: 2rem;
        font-weight: 800;
    }

    .metric-label {
        font-size: 0.9rem;
        opacity: 0.75;
    }

    .badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        margin: 0.2rem 0.2rem 0.2rem 0;
    }

    .required-badge {
        background: #fee2e2;
        color: #991b1b;
    }

    .preferred-badge {
        background: #fef3c7;
        color: #92400e;
    }

    .footer {
        text-align: center;
        opacity: 0.55;
        padding: 2rem 0 1rem 0;
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🕵️ CV Detective</h1>
        <p>
            Turn your CV into a job-specific evidence report,
            identify skill gaps, and build a targeted improvement plan.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SENTENCE TRANSFORMER MODEL
# ============================================================

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()


# ============================================================
# SKILL DICTIONARY
# ============================================================

SKILL_DICTIONARY = [
    "Python",
    "Java",
    "C++",
    "JavaScript",
    "TypeScript",
    "SQL",
    "MySQL",
    "PostgreSQL",
    "MongoDB",
    "Pandas",
    "NumPy",
    "Matplotlib",
    "Scikit-learn",
    "TensorFlow",
    "PyTorch",
    "Machine Learning",
    "Deep Learning",
    "Natural Language Processing",
    "NLP",
    "Git",
    "GitHub",
    "Docker",
    "AWS",
    "Azure",
    "GCP",
    "Linux",
    "Django",
    "Flask",
    "FastAPI",
    "React",
    "HTML",
    "CSS",
    "REST API",
    "APIs",
    "Data Analysis",
    "Data Science",
    "Problem-solving",
    "Communication",
    "Teamwork",
    "Leadership"
]


# ============================================================
# IMPROVEMENT PLAN
# ============================================================

def generate_improvement_plan(skill, status):

    if status == "Gap":

        return (
            f"Build evidence for {skill}. Learn the fundamentals, "
            f"create a small project using it, and add a concrete "
            f"achievement or implementation to your CV."
        )

    if status == "Mentioned":

        return (
            f"Your CV mentions {skill}, but does not clearly "
            f"demonstrate how you used it. Add a project, task, "
            f"or measurable outcome."
        )

    if status == "Partial Match":

        return (
            f"Strengthen the evidence for {skill} by describing "
            f"the specific project, task, tool, or result where "
            f"you applied it."
        )

    return ""


# ============================================================
# SECTION EXTRACTION
# ============================================================

def extract_section(text, start_words, end_words):

    lines = text.splitlines()

    start_index = None

    for i, line in enumerate(lines):

        clean = line.strip().lower().rstrip(":")

        if clean in start_words:

            start_index = i + 1
            break

    if start_index is None:
        return ""

    collected = []

    for line in lines[start_index:]:

        clean = line.strip().lower().rstrip(":")

        if clean in end_words:
            break

        collected.append(line)

    return "\n".join(collected)


# ============================================================
# REQUIREMENT EXTRACTION
# ============================================================

def extract_requirements(job_description):

    required_section = extract_section(
        job_description,
        {
            "requirements",
            "required skills",
            "required qualifications",
            "qualifications"
        },
        {
            "responsibilities",
            "what you will do",
            "about the role",
            "preferred",
            "preferred skills",
            "desired skills",
            "nice to have"
        }
    )

    preferred_section = extract_section(
        job_description,
        {
            "preferred",
            "preferred skills",
            "desired skills",
            "nice to have"
        },
        {
            "responsibilities",
            "what you will do",
            "about the role"
        }
    )

    if not required_section.strip():

        required_section = job_description

    combined_text = (
        required_section
        + "\n"
        + preferred_section
    )

    requirements = []

    for skill in SKILL_DICTIONARY:

        pattern = (
            r"(?<!\w)"
            + re.escape(skill)
            + r"(?!\w)"
        )

        if re.search(
            pattern,
            combined_text,
            re.IGNORECASE
        ):

            requirements.append(skill)

    required = []
    preferred = []

    for skill in requirements:

        pattern = (
            r"(?<!\w)"
            + re.escape(skill)
            + r"(?!\w)"
        )

        if re.search(
            pattern,
            required_section,
            re.IGNORECASE
        ):

            required.append(skill)

        elif re.search(
            pattern,
            preferred_section,
            re.IGNORECASE
        ):

            preferred.append(skill)

    return required, preferred


# ============================================================
# SKILL ANALYSIS
# ============================================================

def analyze_skill(
    skill,
    cv_lines,
    line_embeddings
):

    # --------------------------------------------------------
    # EXACT WHOLE-WORD MATCH
    # --------------------------------------------------------

    pattern = (
        r"(?<!\w)"
        + re.escape(skill)
        + r"(?!\w)"
    )

    exact_lines = [
        line
        for line in cv_lines
        if re.search(
            pattern,
            line,
            re.IGNORECASE
        )
    ]

    # --------------------------------------------------------
    # ACTION WORDS
    # --------------------------------------------------------

    action_words = [
        "built",
        "developed",
        "created",
        "implemented",
        "used",
        "worked",
        "designed",
        "analyzed",
        "developing",
        "using",
        "deployed",
        "programmed",
        "integrated",
        "managed",
        "engineered",
        "maintained",
        "automated",
        "trained",
        "tested",
        "processed",
        "configured",
        "applied"
    ]

    # --------------------------------------------------------
    # EXACT MATCH FOUND
    # --------------------------------------------------------

    if exact_lines:

        usage_line = None

        for line in exact_lines:

            lower_line = line.lower()

            if any(
                word in lower_line
                for word in action_words
            ):

                usage_line = line
                break

        # Strong match
        if usage_line:

            return {
                "skill": skill,
                "status": "Strong Match",
                "evidence": usage_line,
                "evidence_type": "Actual usage detected",
                "similarity": 1.0,
                "why": (
                    "The skill is explicitly mentioned and "
                    "the CV contains an action showing that "
                    "it was used."
                )
            }

        # Mentioned only
        return {
            "skill": skill,
            "status": "Mentioned",
            "evidence": exact_lines[0],
            "evidence_type": "Skill mentioned",
            "similarity": 1.0,
            "why": (
                "The skill appears in the CV, but there is "
                "not enough evidence of actual usage."
            )
        }

    # --------------------------------------------------------
    # SEMANTIC MATCH
    # --------------------------------------------------------

    skill_embedding = model.encode(
        skill,
        convert_to_tensor=True
    )

    similarities = util.cos_sim(
        skill_embedding,
        line_embeddings
    )[0]

    best_index = int(
        similarities.argmax()
    )

    best_score = float(
        similarities[best_index]
    )

    best_line = cv_lines[best_index]

    # --------------------------------------------------------
    # PARTIAL MATCH
    # --------------------------------------------------------

    if best_score >= 0.55:

        return {
            "skill": skill,
            "status": "Partial Match",
            "evidence": best_line,
            "evidence_type": "Semantic evidence",
            "similarity": round(
                best_score,
                2
            ),
            "why": (
                "The CV does not explicitly mention this "
                "skill, but a related statement was found."
            )
        }

    # --------------------------------------------------------
    # WEAK SEMANTIC RELATION
    # --------------------------------------------------------

    if best_score >= 0.45:

        return {
            "skill": skill,
            "status": "Gap",
            "evidence": best_line,
            "evidence_type": "Weak semantic relation",
            "similarity": round(
                best_score,
                2
            ),
            "why": (
                f"The strongest semantic similarity was "
                f"only {best_score:.2f}, below the meaningful "
                "evidence threshold."
            )
        }

    # --------------------------------------------------------
    # REAL GAP
    # --------------------------------------------------------

    return {
        "skill": skill,
        "status": "Gap",
        "evidence": (
            "No meaningful related CV evidence found."
        ),
        "evidence_type": "No evidence",
        "similarity": round(
            best_score,
            2
        ),
        "why": (
            f"The strongest semantic similarity was "
            f"only {best_score:.2f}, below the meaningful "
            "evidence threshold."
        )
    }


# ============================================================
# CLAUDE ANALYSIS
# ============================================================

def ask_claude(
    cv_text,
    job_description,
    results
):

    api_key = os.getenv(
        "ANTHROPIC_API_KEY"
    )

    # --------------------------------------------------------
    # NO API KEY
    # --------------------------------------------------------

    if not api_key:

        return {
            "success": False,
            "message": (
                "Claude API key is not configured. "
                "The local CV Detective analysis is still "
                "fully available."
            )
        }

    try:

        client = Anthropic(
            api_key=api_key
        )

        prompt = f"""
You are an expert technical recruiter and CV coach.

Analyze this CV against this job description.

CV:
{cv_text}

JOB DESCRIPTION:
{job_description}

LOCAL MATCHING RESULTS:
{json.dumps(results, indent=2)}

Provide:

## Overall Assessment

## Strongest Evidence

## Important Missing Skills

## Weakly Evidenced Skills

## Three CV Improvements

## Three Learning or Project Recommendations

## Example Improved CV Bullet

Rules:
- Never invent experience.
- Never claim a skill without CV evidence.
- Clearly distinguish evidence from inference.
- Be concise and actionable.
"""

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1200,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return {
            "success": True,
            "message": response.content[0].text
        }

    except Exception as e:

        error_text = str(e).lower()

        # ----------------------------------------------------
        # CREDIT / BILLING ERROR
        # ----------------------------------------------------

        if (
            "credit" in error_text
            or "balance" in error_text
            or "billing" in error_text
            or "insufficient" in error_text
        ):

            return {
                "success": False,
                "message": (
                    "Claude is configured, but the Anthropic "
                    "account currently has no available API "
                    "credits. Your local CV Detective engine "
                    "continues to work without Claude."
                )
            }

        # ----------------------------------------------------
        # OTHER ERROR
        # ----------------------------------------------------

        return {
            "success": False,
            "message": (
                "Claude could not complete the analysis.\n\n"
                f"API response: {str(e)}"
            )
        }


# ============================================================
# PDF REPORT
# ============================================================

def create_pdf_report(
    required_skills,
    preferred_skills,
    results,
    summary
):

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm
    )

    styles = getSampleStyleSheet()

    title_style = styles["Title"]
    title_style.alignment = TA_CENTER

    story = []

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "CV Detective — Job Match Report",
            title_style
        )
    )

    story.append(
        Spacer(1, 12)
    )

    story.append(
        Paragraph(
            "Evidence-based CV and job analysis.",
            styles["Normal"]
        )
    )

    story.append(
        Spacer(1, 18)
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Match Summary",
            styles["Heading2"]
        )
    )

    summary_data = [
        ["Metric", "Result"],
        [
            "Strong Matches",
            str(summary["strong"])
        ],
        [
            "Mentioned",
            str(summary["mentioned"])
        ],
        [
            "Partial Matches",
            str(summary["partial"])
        ],
        [
            "Gaps",
            str(summary["gaps"])
        ],
        [
            "Overall Evidence Coverage",
            f"{summary['overall_coverage']:.0f}%"
        ],
        [
            "Required Coverage",
            f"{summary['required_coverage']:.0f}%"
        ],
        [
            "Preferred Coverage",
            f"{summary['preferred_coverage']:.0f}%"
        ]
    ]

    table = Table(
        summary_data,
        colWidths=[
            90 * mm,
            55 * mm
        ]
    )

    table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.lightgrey
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),
            (
                "PADDING",
                (0, 0),
                (-1, -1),
                6
            )
        ])
    )

    story.append(table)

    story.append(
        Spacer(1, 18)
    )

    # --------------------------------------------------------
    # REQUIRED
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Required Skills",
            styles["Heading2"]
        )
    )

    if required_skills:

        for skill in required_skills:

            story.append(
                Paragraph(
                    f"• {skill}",
                    styles["Normal"]
                )
            )

    else:

        story.append(
            Paragraph(
                "No required skills detected.",
                styles["Normal"]
            )
        )

    story.append(
        Spacer(1, 12)
    )

    # --------------------------------------------------------
    # PREFERRED
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Preferred Skills",
            styles["Heading2"]
        )
    )

    if preferred_skills:

        for skill in preferred_skills:

            story.append(
                Paragraph(
                    f"• {skill}",
                    styles["Normal"]
                )
            )

    else:

        story.append(
            Paragraph(
                "None specified by the employer.",
                styles["Normal"]
            )
        )

    story.append(
        Spacer(1, 18)
    )

    # --------------------------------------------------------
    # DETAILED ANALYSIS
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Detailed Evidence Analysis",
            styles["Heading2"]
        )
    )

    for result in results:

        story.append(
            Paragraph(
                f"<b>{result['skill']}</b> — "
                f"{result['status']} "
                f"({result['priority']})",
                styles["Normal"]
            )
        )

        story.append(
            Paragraph(
                f"<b>Evidence:</b> "
                f"{result['evidence']}",
                styles["Normal"]
            )
        )

        story.append(
            Paragraph(
                f"<b>Similarity:</b> "
                f"{result['similarity']:.2f}",
                styles["Normal"]
            )
        )

        story.append(
            Paragraph(
                f"<b>Why:</b> "
                f"{result['why']}",
                styles["Normal"]
            )
        )

        story.append(
            Spacer(1, 8)
        )

    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    story.append(
        Spacer(1, 15)
    )

    story.append(
        Paragraph(
            "Coverage represents evidence found in the CV "
            "and is not a hiring probability.",
            styles["Italic"]
        )
    )

    document.build(story)

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">📥 Start Your Investigation</div>',
    unsafe_allow_html=True
)

left, right = st.columns(
    2,
    gap="large"
)

# ------------------------------------------------------------
# CV
# ------------------------------------------------------------

with left:

    st.markdown("### 📄 Candidate CV")

    uploaded_file = st.file_uploader(
        "Upload your CV as PDF",
        type=["pdf"],
        label_visibility="collapsed"
    )

    if uploaded_file:

        st.success(
            f"✓ {uploaded_file.name} uploaded"
        )


# ------------------------------------------------------------
# JOB
# ------------------------------------------------------------

with right:

    st.markdown("### 💼 Target Job")

    job_description = st.text_area(
        "Paste job description",
        height=180,
        placeholder=(
            "Paste the complete job description here..."
        ),
        label_visibility="collapsed"
    )


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze = st.button(
    "🔎 Analyze My CV",
    type="primary",
    use_container_width=True
)


# ============================================================
# MAIN ANALYSIS
# ============================================================

if analyze:

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not uploaded_file:

        st.error(
            "Please upload your CV first."
        )

        st.stop()

    if not job_description.strip():

        st.error(
            "Please paste a job description first."
        )

        st.stop()

    # --------------------------------------------------------
    # ANALYSIS SPINNER
    # --------------------------------------------------------

    with st.spinner(
        "🕵️ Detective is analyzing your CV..."
    ):

        # ----------------------------------------------------
        # READ PDF
        # ----------------------------------------------------

        reader = pypdf.PdfReader(
            uploaded_file
        )

        cv_text = ""

        for page in reader.pages:

            text = page.extract_text()

            if text:

                cv_text += text + "\n"

        cv_lines = [
            line.strip()
            for line in cv_text.splitlines()
            if line.strip()
        ]

        if not cv_lines:

            st.error(
                "No readable text was found in this PDF. "
                "Please upload a text-based PDF."
            )

            st.stop()

        # ----------------------------------------------------
        # REQUIREMENTS
        # ----------------------------------------------------

        required_skills, preferred_skills = (
            extract_requirements(
                job_description
            )
        )

        # ----------------------------------------------------
        # EMBEDDINGS
        # ----------------------------------------------------

        line_embeddings = model.encode(
            cv_lines,
            convert_to_tensor=True
        )

        # ----------------------------------------------------
        # REQUIRED RESULTS
        # ----------------------------------------------------

        required_results = []

        for skill in required_skills:

            result = analyze_skill(
                skill,
                cv_lines,
                line_embeddings
            )

            result["priority"] = "Required"

            required_results.append(
                result
            )

        # ----------------------------------------------------
        # PREFERRED RESULTS
        # ----------------------------------------------------

        preferred_results = []

        for skill in preferred_skills:

            result = analyze_skill(
                skill,
                cv_lines,
                line_embeddings
            )

            result["priority"] = "Preferred"

            preferred_results.append(
                result
            )

        results = (
            required_results
            + preferred_results
        )

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        strong = sum(
            1
            for r in results
            if r["status"] == "Strong Match"
        )

        mentioned = sum(
            1
            for r in results
            if r["status"] == "Mentioned"
        )

        partial = sum(
            1
            for r in results
            if r["status"] == "Partial Match"
        )

        gaps = sum(
            1
            for r in results
            if r["status"] == "Gap"
        )

        total = len(results)

        if total:

            coverage = (
                (
                    strong
                    + mentioned
                    + partial
                )
                / total
            ) * 100

        else:

            coverage = 0

        # ----------------------------------------------------
        # REQUIRED COVERAGE
        # ----------------------------------------------------

        required_total = len(
            required_results
        )

        required_covered = sum(
            1
            for r in required_results
            if r["status"] != "Gap"
        )

        if required_total:

            required_coverage = (
                required_covered
                / required_total
            ) * 100

        else:

            required_coverage = 0

        # ----------------------------------------------------
        # PREFERRED COVERAGE
        # ----------------------------------------------------

        preferred_total = len(
            preferred_results
        )

        preferred_covered = sum(
            1
            for r in preferred_results
            if r["status"] != "Gap"
        )

        if preferred_total:

            preferred_coverage = (
                preferred_covered
                / preferred_total
            ) * 100

        else:

            preferred_coverage = 0

    # ========================================================
    # DETECTIVE DASHBOARD
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">🕵️ Detective Report</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # STATUS CARDS
    # --------------------------------------------------------

    cols = st.columns(4)

    metrics = [
        ("🟢", strong, "Strong"),
        ("🟡", mentioned, "Mentioned"),
        ("🟠", partial, "Partial"),
        ("🔴", gaps, "Gaps")
    ]

    for col, (icon, value, label) in zip(
        cols,
        metrics
    ):

        with col:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-number">
                        {icon} {value}
                    </div>
                    <div class="metric-label">
                        {label}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.write("")

    # --------------------------------------------------------
    # COVERAGE
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Overall Evidence Coverage",
        f"{coverage:.0f}%"
    )

    c2.metric(
        "Required Coverage",
        f"{required_coverage:.0f}%"
    )

    c3.metric(
        "Preferred Coverage",
        f"{preferred_coverage:.0f}%"
    )

    st.caption(
        "Coverage measures requirements with supporting "
        "CV evidence. It is not a hiring probability."
    )

    # ========================================================
    # REQUIREMENTS
    # ========================================================

    st.markdown(
        '<div class="section-title">🎯 Requirements</div>',
        unsafe_allow_html=True
    )

    req_col, pref_col = st.columns(2)

    # --------------------------------------------------------
    # REQUIRED
    # --------------------------------------------------------

    with req_col:

        st.markdown(
            """
            <div style="
                font-size: 1.45rem;
                font-weight: 750;
                margin-bottom: 12px;
            ">
                🔴 Required
            </div>
            """,
            unsafe_allow_html=True
        )

        if required_skills:

            for skill in required_skills:

                st.markdown(
                    f"""
                    <span class="badge required-badge">
                        {skill}
                    </span>
                    """,
                    unsafe_allow_html=True
                )

        else:

            st.info(
                "No required skills detected."
            )

    # --------------------------------------------------------
    # PREFERRED
    # --------------------------------------------------------

    with pref_col:

        st.markdown(
            """
            <div style="
                font-size: 1.45rem;
                font-weight: 750;
                margin-bottom: 12px;
            ">
                🟡 Preferred
            </div>
            """,
            unsafe_allow_html=True
        )

        if preferred_skills:

            for skill in preferred_skills:

                st.markdown(
                    f"""
                    <span class="badge preferred-badge">
                        {skill}
                    </span>
                    """,
                    unsafe_allow_html=True
                )

        else:

            st.info(
                "None specified by the employer."
            )

    # ========================================================
    # EVIDENCE ANALYSIS
    # ========================================================

    st.markdown(
        '<div class="section-title">🔍 Evidence Analysis</div>',
        unsafe_allow_html=True
    )

    for result in results:

        status = result["status"]

        if status == "Strong Match":

            icon = "🟢"

        elif status == "Mentioned":

            icon = "🟡"

        elif status == "Partial Match":

            icon = "🟠"

        else:

            icon = "🔴"

        with st.expander(
            f"{icon} {result['skill']}  •  "
            f"{status}  •  {result['priority']}"
        ):

            st.markdown(
                f"**Evidence**  \n"
                f"{result['evidence']}"
            )

            st.write(
                f"**Evidence type:** "
                f"{result['evidence_type']}"
            )

            st.progress(
                min(
                    result["similarity"],
                    1.0
                )
            )

            st.caption(
                f"Semantic similarity: "
                f"{result['similarity']:.2f}"
            )

            st.info(
                result["why"]
            )

    # ========================================================
    # ACTION PLAN
    # ========================================================

    st.markdown(
        '<div class="section-title">🛠️ Your Action Plan</div>',
        unsafe_allow_html=True
    )

    improvement_items = [
        result
        for result in results
        if result["status"] in [
            "Gap",
            "Mentioned",
            "Partial Match"
        ]
    ]

    if improvement_items:

        for index, result in enumerate(
            improvement_items,
            start=1
        ):

            st.markdown(
                f"### {index}. {result['skill']}"
            )

            st.write(
                generate_improvement_plan(
                    result["skill"],
                    result["status"]
                )
            )

    else:

        st.success(
            "No major improvement gaps detected."
        )

    # ========================================================
    # CLAUDE
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">🤖 Claude Career Detective</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Claude is an optional reasoning layer that can "
        "turn the structured evidence into deeper career advice."
    )

    st.info(
        "Your local CV matching engine works without Claude. "
        "Claude requires an Anthropic API key and available API credits."
    )

    if st.button(
        "🤖 Ask Claude for Deeper Analysis",
        use_container_width=True
    ):

        with st.spinner(
            "Claude is analyzing the evidence..."
        ):

            claude_result = ask_claude(
                cv_text,
                job_description,
                results
            )

        if claude_result["success"]:

            st.success(
                "Claude analysis completed."
            )

            st.markdown(
                "### 🧠 Claude's Analysis"
            )

            st.markdown(
                claude_result["message"]
            )

        else:

            st.warning(
                claude_result["message"]
            )

            st.info(
                "💡 The local CV Detective analysis "
                "continues to work normally."
            )

    # ========================================================
    # EXPORT
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">📥 Export Results</div>',
        unsafe_allow_html=True
    )

    summary = {
        "strong": strong,
        "mentioned": mentioned,
        "partial": partial,
        "gaps": gaps,
        "overall_coverage": coverage,
        "required_coverage": required_coverage,
        "preferred_coverage": preferred_coverage
    }

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    pdf_report = create_pdf_report(
        required_skills,
        preferred_skills,
        results,
        summary
    )

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    output = {
        "required_skills": required_skills,
        "preferred_skills": preferred_skills,
        "results": results,
        "summary": summary
    }

    export_col1, export_col2 = st.columns(2)

    with export_col1:

        st.download_button(
            "📄 Download PDF Report",
            data=pdf_report,
            file_name="CV_Detective_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    with export_col2:

        st.download_button(
            "🧾 Download JSON",
            data=json.dumps(
                output,
                indent=2
            ),
            file_name="CV_Detective_Analysis.json",
            mime="application/json",
            use_container_width=True
        )

    # ========================================================
    # DEBUG / RAW DATA
    # ========================================================

    with st.expander(
        "📄 View Extracted CV Text"
    ):

        st.text(cv_text)

    with st.expander(
        "💼 View Job Description"
    ):

        st.text(job_description)

    with st.expander(
        "🧾 View Structured JSON"
    ):

        st.json(output)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🕵️ CV Detective · Evidence-based CV analysis
    </div>
    """,
    unsafe_allow_html=True
)